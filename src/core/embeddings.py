import openai
import chromadb
import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings
import numpy as np
from .config import Config
from .errors import is_insufficient_quota_error, get_quota_error_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manage OpenAI embeddings and ChromaDB operations"""
    
    def __init__(self, 
                 openai_api_key: str,
                 embedding_model: str = "text-embedding-ada-002",
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "polkadot_embeddings"):
        
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        
        openai.api_key = self.openai_api_key
        
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = None
        self._init_collection()
    
    def _init_collection(self):
        """Initialize or get the ChromaDB collection"""
        try:

            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}' with {self.collection.count()} documents")
        except Exception:

            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Polkadot documentation embeddings"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = openai.embeddings.create(
                            model=self.embedding_model,
                            input=batch
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        embeddings.extend(batch_embeddings)
                        
                        logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                        break
                        
                    except openai.RateLimitError as e:
                        if is_insufficient_quota_error(e):
                            logger.error(f"Insufficient quota error generating embeddings: {e}")
                            raise
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            raise e
                    except Exception as e:
                        if is_insufficient_quota_error(e):
                            logger.error(f"Insufficient quota error generating embeddings: {e}")
                            raise
                        logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        else:
                            raise e
                
                time.sleep(0.1)
                
            except Exception as e:
                if is_insufficient_quota_error(e):
                    logger.error(f"Insufficient quota error in embeddings batch: {e}")
                    raise
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                embeddings.extend([[0.0] * 1536] * len(batch))  # ada-002 has 1536 dimensions
        
        def _normalize(vec):
            normalized = vec / np.linalg.norm(vec)
            return normalized.tolist() if isinstance(normalized, np.ndarray) else normalized
        
        return [_normalize(e) for e in embeddings]
    
    def add_chunks_to_collection(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add chunks with embeddings to ChromaDB collection
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            texts = [chunk['content'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.generate_embeddings(texts)
            
            if len(embeddings) != len(chunks):
                logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
                return False
            
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                documents.append(chunk['content'])
                
                metadata = {}
                for key, value in chunk['metadata'].items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            logger.info(f"Adding {len(chunks)} chunks to ChromaDB collection...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to collection: {e}")
            return False
    
    def search_similar_chunks(self, 
                            query: str, 
                            n_results: int = 5,
                            filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar chunks with content, metadata, and similarity scores
        """
        try:
            is_static = self.collection_name == Config.CHROMA_COLLECTION_NAME
            is_dynamic = self.collection_name == Config.CHROMA_DYNAMIC_COLLECTION_NAME
            is_other = not is_static and not is_dynamic
            
            if is_static and not Config.SEARCH_STATIC_DATA:
                logger.info("Static search is disabled")
                return []
            if is_dynamic and not Config.SEARCH_DYNAMIC_DATA:
                logger.info("Dynamic search is disabled")
                return []

            query_embedding = self.generate_embeddings([query])[0]
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            results = []
            
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            if search_results['documents'] and len(search_results['documents']) > 0:
                source_label = 'static' if is_static else ('dynamic' if is_dynamic else self.collection_name)
                for i in range(len(search_results['documents'][0])):
                    result = {
                        'content': search_results['documents'][0][i],
                        'metadata': search_results['metadatas'][0][i],
                        'similarity_score': 1 - (search_results['distances'][0][i] / 2),
                        'source': source_label
                    }
                    results.append(result)
                logger.info(f"Found {len(results)} chunks from {source_label} collection")
            
            return results
            
        except Exception as e:
            if is_insufficient_quota_error(e):
                logger.error(f"Insufficient quota error searching chunks: {e}")
                raise
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            sample_results = self.collection.get(limit=10, include=["metadatas"])
            
            source_counts = {}
            if sample_results['metadatas']:
                all_results = self.collection.get(include=["metadatas"])
                for metadata in all_results['metadatas']:
                    source = metadata.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
            
            stats = {
                'total_chunks': count,
                'chunks_by_source': source_counts,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            
            self._init_collection()
            logger.info(f"Recreated empty collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def collection_exists(self) -> bool:
        """Check if collection exists and has data"""
        try:
            return self.collection is not None and self.collection.count() > 0
        except Exception:
            return False 