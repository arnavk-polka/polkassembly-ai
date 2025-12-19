"""
StaticProvider - Unified interface for static knowledge retrieval.

This is the main abstraction layer over Chroma for static search.
All static search should go through this provider to ensure:
- Consistent result shapes
- Proper chunk identity/hashing
- Future DKG provenance support

Usage:
    provider = StaticProvider(
        openai_api_key=Config.OPENAI_API_KEY,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY
    )
    
    results = provider.search_docs("What is staking?", k=5)
    for result in results:
        print(result.content, result.chunk_id, result.dkg_match)
"""

import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
import openai
import numpy as np

from .types import (
    StaticSearchResult,
    StaticSearchFilters,
    StaticSourceType,
    DKGMatch,
)
from .dkg_index import load_dkg_chunk_index
from ..config import Config

logger = logging.getLogger(__name__)


class StaticProvider:
    """
    Unified provider for static knowledge search.
    
    Wraps Chroma collections and provides a clean interface that:
    - Returns StaticSearchResult objects with consistent shape
    - Supports filtering by source, tags, etc.
    - Leaves hooks for future DKG integration
    
    The provider manages two collections:
    - docs: Wiki pages, official docs, Polkassembly docs
    - aag: AAG video transcript segments
    """
    
    DOCS_COLLECTION = "static_docs"
    AAG_COLLECTION = "static_aag"
    
    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-ada-002",
        docs_collection_name: Optional[str] = None,
        aag_collection_name: Optional[str] = None,
    ):
        """
        Initialize the StaticProvider.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            chroma_persist_directory: Path to Chroma persistence directory
            embedding_model: OpenAI embedding model to use
            docs_collection_name: Override default docs collection name
            aag_collection_name: Override default AAG collection name
        """
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chroma_persist_directory = chroma_persist_directory
        
        openai.api_key = openai_api_key
        
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self._docs_collection_name = docs_collection_name or self.DOCS_COLLECTION
        self._aag_collection_name = aag_collection_name or self.AAG_COLLECTION
        
        self._docs_collection = None
        self._aag_collection = None
        
        self._init_collections()
    
    def _init_collections(self):
        """Initialize or get Chroma collections."""
        try:
            self._docs_collection = self.chroma_client.get_or_create_collection(
                name=self._docs_collection_name,
                metadata={"description": "Static documentation chunks"}
            )
            logger.info(f"Docs collection '{self._docs_collection_name}': {self._docs_collection.count()} chunks")
        except Exception as e:
            logger.error(f"Failed to init docs collection: {e}")
            raise
        
        try:
            self._aag_collection = self.chroma_client.get_or_create_collection(
                name=self._aag_collection_name,
                metadata={"description": "AAG video transcript segments"}
            )
            logger.info(f"AAG collection '{self._aag_collection_name}': {self._aag_collection.count()} chunks")
        except Exception as e:
            logger.error(f"Failed to init AAG collection: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        embedding = response.data[0].embedding
        normalized = np.array(embedding) / np.linalg.norm(embedding)
        return normalized.tolist()
    
    def _chroma_results_to_static_results(
        self,
        chroma_results: Dict[str, Any],
        source_type: str
    ) -> List[StaticSearchResult]:
        """
        Convert Chroma query results to StaticSearchResult objects.
        """
        results = []
        
        if not chroma_results.get('documents') or not chroma_results['documents'][0]:
            return results
        
        documents = chroma_results['documents'][0]
        metadatas = chroma_results['metadatas'][0]
        distances = chroma_results['distances'][0]
        ids = chroma_results['ids'][0]
        
        dkg_index = load_dkg_chunk_index(
            asset_doc_path="dkg_asset_doc.json",
            docs_export_path="dkg_docs_export.jsonl",
        )

        for i, (doc, meta, dist, chroma_id) in enumerate(zip(documents, metadatas, distances, ids)):
            score = 1 - (dist / 2)
            
            chunk_id = meta.get('chunk_id', chroma_id)
            chunk_hash = meta.get('chunk_hash', '')
            source = meta.get('source', source_type)

            dkg_match = None
            if chunk_id and chunk_hash:
                lookup_key = (chunk_id, chunk_hash)
                if lookup_key in dkg_index:
                    ual_data = dkg_index[lookup_key]
                    dkg_match = DKGMatch(
                        asset_ual=ual_data.get("ual", ""),
                        chunk_id=chunk_id,
                        chunk_hash=chunk_hash,
                    )
                    logger.info(f"✓ Created dkg_match for chunk_id={chunk_id[:50]}..., ual={ual_data.get('ual', '')[:50]}...")
                else:
                    logger.debug(f"✗ (chunk_id, chunk_hash) pair not found in dkg_index for {chunk_id[:50]}...")
            
            result = StaticSearchResult(
                id=chunk_id,
                score=score,
                content=doc,
                source=source,
                chunk_id=chunk_id,
                chunk_hash=chunk_hash,
                metadata=meta,
                dkg_match=dkg_match
            )
            results.append(result)
        
        return results
    
    def search_docs(
        self,
        query: str,
        k: int = 5,
        filters: Optional[StaticSearchFilters] = None
    ) -> List[StaticSearchResult]:
        """
        Search documentation chunks.
        
        Args:
            query: The search query
            k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of StaticSearchResult objects, ordered by relevance
        """
        if not Config.SEARCH_STATIC_DATA:
            logger.info("Static search is disabled")
            return []
        
        try:
            query_embedding = self._generate_embedding(query)
            
            where_clause = filters.to_chroma_where() if filters else None
            
            chroma_results = self._docs_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            results = self._chroma_results_to_static_results(chroma_results, "docs")
            
            if filters and filters.min_score:
                results = [r for r in results if r.score >= filters.min_score]
            
            logger.info(f"search_docs: query='{query[:50]}...', found={len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in search_docs: {e}")
            return []
    
    def search_aag_segments(
        self,
        query: str,
        k: int = 5,
        filters: Optional[StaticSearchFilters] = None
    ) -> List[StaticSearchResult]:
        """
        Search AAG (video transcript) segments.
        
        Args:
            query: The search query
            k: Number of results to return
            filters: Optional filters (video_id, speaker, language, etc.)
            
        Returns:
            List of StaticSearchResult objects, ordered by relevance
            
        Note:
            AAG results include additional metadata:
            - video_id: YouTube video identifier
            - summary: TL;DR of this segment
            - speaker: Who is speaking (if known)
            - start_second, end_second: Timestamp info
        """
        if not Config.SEARCH_STATIC_DATA:
            logger.info("Static search is disabled")
            return []
        
        try:
            query_embedding = self._generate_embedding(query)
            
            where_clause = filters.to_chroma_where() if filters else None
            
            chroma_results = self._aag_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            results = self._chroma_results_to_static_results(chroma_results, "aag")
            
            if filters and filters.min_score:
                results = [r for r in results if r.score >= filters.min_score]
            
            logger.info(f"search_aag_segments: query='{query[:50]}...', found={len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in search_aag_segments: {e}")
            return []
    
    def search_all(
        self,
        query: str,
        k: int = 5,
        filters: Optional[StaticSearchFilters] = None
    ) -> List[StaticSearchResult]:
        """
        Search both docs and AAG segments, merged by score.
        
        Args:
            query: The search query
            k: Total number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of StaticSearchResult objects from both sources, 
            sorted by score descending
        """
        doc_results = self.search_docs(query, k=k, filters=filters)
        aag_results = self.search_aag_segments(query, k=k, filters=filters)
        
        all_results = doc_results + aag_results
        all_results.sort(key=lambda r: r.score, reverse=True)
        
        return all_results[:k]
    
    def search_similar_chunks(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Legacy compatibility method.
        
        This method matches the signature of EmbeddingManager.search_similar_chunks()
        so that StaticProvider can be used as a drop-in replacement.
        
        Args:
            query: The search query
            n_results: Number of results
            filter_metadata: Chroma-style filter dict
            
        Returns:
            Results in legacy format: [{content, metadata, similarity_score, source}, ...]
        """
        filters = None
        if filter_metadata:
            filters = StaticSearchFilters(
                source=filter_metadata.get('source'),
                doc_id=filter_metadata.get('doc_id'),
                video_id=filter_metadata.get('video_id'),
            )
        
        results = self.search_docs(query, k=n_results, filters=filters)
        
        return [r.to_legacy_format() for r in results]
    
    def get_docs_collection(self):
        """Get the underlying docs Chroma collection for direct access."""
        return self._docs_collection
    
    def get_aag_collection(self):
        """Get the underlying AAG Chroma collection for direct access."""
        return self._aag_collection
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about both collections."""
        return {
            "docs_count": self._docs_collection.count(),
            "aag_count": self._aag_collection.count(),
            "docs_collection": self._docs_collection_name,
            "aag_collection": self._aag_collection_name,
            "embedding_model": self.embedding_model,
        }

    def collection_exists(self) -> bool:
        """
        Legacy compatibility method used by API server.
        
        Returns True if there is at least one document in the docs collection.
        """
        try:
            return self._docs_collection.count() > 0
        except Exception:
            return False
    
    def clear_docs_collection(self):
        """Clear all data from docs collection."""
        self.chroma_client.delete_collection(self._docs_collection_name)
        self._docs_collection = self.chroma_client.create_collection(
            name=self._docs_collection_name,
            metadata={"description": "Static documentation chunks"}
        )
        logger.info(f"Cleared docs collection '{self._docs_collection_name}'")
    
    def clear_aag_collection(self):
        """Clear all data from AAG collection."""
        self.chroma_client.delete_collection(self._aag_collection_name)
        self._aag_collection = self.chroma_client.create_collection(
            name=self._aag_collection_name,
            metadata={"description": "AAG video transcript segments"}
        )
        logger.info(f"Cleared AAG collection '{self._aag_collection_name}'")


