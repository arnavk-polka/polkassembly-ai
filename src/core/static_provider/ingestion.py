"""
Ingestion pipeline for static data into Chroma.

This module handles:
- Loading raw content from various sources (docs, AAG transcripts)
- Chunking with proper offset tracking
- Computing stable chunk_id and chunk_hash
- Upserting into the appropriate Chroma collection

Usage:
    ingester = StaticIngester(
        openai_api_key=Config.OPENAI_API_KEY,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY
    )
    
    ingester.ingest_docs_from_directory("data/static_sources/polkadot_wiki")
    
    ingester.ingest_aag_transcripts([
        {"video_id": "abc123", "transcript": "...", "metadata": {...}}
    ])
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import openai
import numpy as np
import tiktoken

from .types import StaticSourceType
from .hashing import (
    compute_chunk_identity,
    normalize_text,
    _get_source_prefix,
)
from .provider import StaticProvider
from ..config import Config

logger = logging.getLogger(__name__)


class StaticIngester:
    """
    Handles ingestion of static data into Chroma collections.
    
    Key features:
    - Character-offset-based chunking with overlap
    - Stable chunk_id and chunk_hash computation
    - Metadata preservation from source files
    - Batch embedding generation
    """
    
    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the ingester.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            chroma_persist_directory: Path to Chroma persistence
            embedding_model: OpenAI embedding model
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        openai.api_key = openai_api_key
        
        self.provider = StaticProvider(
            openai_api_key=openai_api_key,
            chroma_persist_directory=chroma_persist_directory,
            embedding_model=embedding_model,
        )
        
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 4
    
    def _extract_semantic_boundaries(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract semantic boundaries (headings/sections) from text.
        
        Returns list of (section_id, start_char, end_char) tuples.
        section_id is a path-like identifier (e.g., "introduction", "staking/basics").
        
        Falls back to simple paragraph-based splitting if no clear headings found.
        """
        sections = []
        lines = text.split('\n')
        current_section = "root"
        current_path = []
        section_start = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                continue
            
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                heading_text = stripped.lstrip('#').strip()
                
                if heading_text:
                    heading_slug = heading_text.lower().replace(' ', '_').replace('/', '_')
                    heading_slug = ''.join(c for c in heading_slug if c.isalnum() or c in ('_', '-'))
                    
                    current_path = current_path[:level-1] + [heading_slug]
                    current_section = '/'.join(current_path) if current_path else "root"
                    
                    char_pos = sum(len(l) + 1 for l in lines[:i])
                    if section_start < char_pos:
                        sections.append((current_section, section_start, char_pos))
                    section_start = char_pos
        
        if section_start < len(text):
            sections.append((current_section, section_start, len(text)))
        
        if not sections:
            sections = [("root", 0, len(text))]
        
        return sections
    
    def _chunk_with_semantic_boundaries(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Tuple[str, int, int, str]]:
        """
        Chunk text using semantic boundaries, then sub-chunk within sections.
        
        Returns list of (chunk_text, start_offset, end_offset, chunking_key) tuples.
        chunking_key format: "section_id:chunk_index"
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        if not text or not text.strip():
            return []
        
        sections = self._extract_semantic_boundaries(text)
        all_chunks = []
        
        for section_id, section_start, section_end in sections:
            section_text = text[section_start:section_end]
            
            if not section_text.strip():
                continue
            
            tokens = self.encoding.encode(section_text)
            if len(tokens) <= chunk_size:
                chunking_key = f"{section_id}:0"
                all_chunks.append((section_text, section_start, section_end, chunking_key))
                continue
            
            chunk_index = 0
            token_start = 0
            
            while token_start < len(tokens):
                token_end = min(token_start + chunk_size, len(tokens))
                chunk_tokens = tokens[token_start:token_end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                text_before = self.encoding.decode(tokens[:token_start])
                char_start = section_start + len(text_before)
                char_end = char_start + len(chunk_text)
                
                if chunk_text.strip():
                    chunking_key = f"{section_id}:{chunk_index}"
                    all_chunks.append((chunk_text, char_start, char_end, chunking_key))
                    chunk_index += 1
                
                if token_end >= len(tokens):
                    break
                
                token_start = token_end - chunk_overlap
                if token_start >= token_end:
                    token_start = token_end
        
        return all_chunks
    
    def _chunk_text_with_offsets(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks, tracking character offsets.
        
        Args:
            text: The full text to chunk
            chunk_size: Override default chunk size (in tokens)
            chunk_overlap: Override default overlap (in tokens)
            
        Returns:
            List of (chunk_text, start_offset, end_offset) tuples
            Offsets are character-based.
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        if not text or not text.strip():
            return []
        
        tokens = self.encoding.encode(text)
        if len(tokens) <= chunk_size:
            return [(text, 0, len(text))]
        
        chunks = []
        token_start = 0
        
        while token_start < len(tokens):
            token_end = min(token_start + chunk_size, len(tokens))
            chunk_tokens = tokens[token_start:token_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            text_before = self.encoding.decode(tokens[:token_start])
            char_start = len(text_before)
            char_end = char_start + len(chunk_text)
            
            if chunk_text.strip():
                chunks.append((chunk_text, char_start, char_end))
            
            if token_end >= len(tokens):
                break
            
            token_start = token_end - chunk_overlap
            if token_start >= token_end:
                token_start = token_end
        
        return chunks
    
    def _generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                for item in response.data:
                    embedding = np.array(item.embedding)
                    normalized = embedding / np.linalg.norm(embedding)
                    all_embeddings.append(normalized.tolist())
                
                logger.info(f"Generated embeddings batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                all_embeddings.extend([[0.0] * 1536] * len(batch))
        
        return all_embeddings
    
    def _parse_doc_file(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a documentation file, extracting content and metadata.
        
        Expected file format:
        Title: <title>
        URL: <url>
        Description: <description>
        Type: <type>
        ---
        <content>
        
        Returns:
            Tuple of (content, metadata)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        if not raw_content.strip():
            return "", {}
        
        lines = raw_content.split('\n')
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
        }
        content_start = 0
        
        for i, line in enumerate(lines[:15]):
            if line.startswith('Title: '):
                metadata['title'] = line.replace('Title: ', '').strip()
                content_start = max(content_start, i + 1)
            elif line.startswith('URL: '):
                metadata['source_url'] = line.replace('URL: ', '').strip()
                content_start = max(content_start, i + 1)
            elif line.startswith('Description: '):
                metadata['description'] = line.replace('Description: ', '').strip()
                content_start = max(content_start, i + 1)
            elif line.startswith('Type: '):
                metadata['doc_type'] = line.replace('Type: ', '').strip()
                content_start = max(content_start, i + 1)
            elif line.strip() == '---':
                content_start = i + 1
                break
            elif line.strip() == '' and content_start > 0:
                content_start = i + 1
                break
        
        if 'title' not in metadata:
            metadata['title'] = file_path.stem
        
        content = '\n'.join(lines[content_start:]).strip()
        return content, metadata
    
    def _determine_source_type(self, subdir_name: str) -> str:
        """Determine the source type from subdirectory name."""
        name_lower = subdir_name.lower()
        
        if 'wiki' in name_lower:
            return StaticSourceType.POLKA_WIKI.value
        elif 'polkassembly' in name_lower or 'pa_' in name_lower:
            return StaticSourceType.POLKASSEMBLY_DOC.value
        elif 'network' in name_lower:
            return StaticSourceType.POLKADOT_NETWORK.value
        else:
            return StaticSourceType.UNKNOWN.value
    
    def _get_doc_key(self, file_path: Path, subdir_name: str) -> str:
        """
        Generate stable doc_key from file path.
        
        Examples:
        - polkadot_wiki/staking.txt -> "staking"
        - pa_docs/governance.md -> "governance"
        """
        return file_path.stem.replace(' ', '_').replace(':', '_')
    
    def ingest_docs_from_directory(
        self,
        data_dir: str,
        clear_existing: bool = False,
        subdirectory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest documentation files from a directory.
        
        Args:
            data_dir: Root directory containing subdirectories with .txt files
            clear_existing: If True, clear the docs collection first
            subdirectory: If specified, only process this subdirectory
            
        Returns:
            Summary dict with counts and any errors
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"Data directory not found: {data_path}")
            return {"error": f"Directory not found: {data_path}"}
        
        if clear_existing:
            self.provider.clear_docs_collection()
        
        if subdirectory:
            subdirs = [data_path / subdirectory]
        else:
            subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        logger.info(f"Processing {len(subdirs)} subdirectories from {data_path}")
        
        all_chunks = []
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "subdirs_processed": 0,
            "errors": [],
        }
        
        for subdir in subdirs:
            source_type = self._determine_source_type(subdir.name)
            txt_files = list(subdir.glob("**/*.txt"))
            
            logger.info(f"Processing {subdir.name}: {len(txt_files)} files, source_type={source_type}")
            
            for file_path in txt_files:
                try:
                    content, file_metadata = self._parse_doc_file(file_path)
                    
                    if not content:
                        stats["files_skipped"] += 1
                        continue
                    
                    doc_id = f"{subdir.name}_{file_path.stem}"
                    doc_key = self._get_doc_key(file_path, subdir.name)
                    source_prefix = _get_source_prefix(source_type)
                    
                    chunks_with_keys = self._chunk_with_semantic_boundaries(content)
                    
                    for chunk_text, start_offset, end_offset, chunking_key in chunks_with_keys:
                        chunk_id, chunk_hash, normalized = compute_chunk_identity(
                            source=source_prefix,
                            doc_key=doc_key,
                            chunking_key=chunking_key,
                            raw_text=chunk_text,
                            source_type=source_type
                        )
                        
                        chunk_metadata = {
                            "chunk_id": chunk_id,
                            "chunk_hash": chunk_hash,
                            "source": source_type,
                            "source_prefix": source_prefix,
                            "doc_id": doc_id,
                            "doc_key": doc_key,
                            "chunking_key": chunking_key,
                            "title": file_metadata.get("title", file_path.stem),
                            "source_url": file_metadata.get("source_url", ""),
                            "description": file_metadata.get("description", ""),
                            "subdirectory": subdir.name,
                            "start_offset": start_offset,
                            "end_offset": end_offset,
                            "version": 1,
                            "is_latest": True,
                            "created_at": datetime.now().isoformat(),
                            "data_type": "static",
                        }
                        
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "content": normalized,
                            "metadata": chunk_metadata,
                        })
                    
                    stats["files_processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    stats["errors"].append(str(e))
                    stats["files_skipped"] += 1
            
            stats["subdirs_processed"] += 1
        
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            texts = [c["content"] for c in all_chunks]
            embeddings = self._generate_embeddings_batch(texts)
            
            ids = [c["chunk_id"] for c in all_chunks]
            documents = [c["content"] for c in all_chunks]
            metadatas = []
            for c in all_chunks:
                meta = {}
                for k, v in c["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)
            
            logger.info(f"Upserting {len(all_chunks)} chunks to docs collection...")
            self.provider.get_docs_collection().upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            
            stats["chunks_created"] = len(all_chunks)
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def ingest_aag_transcripts(
        self,
        transcripts: List[Dict[str, Any]],
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest AAG video transcripts.
        
        Args:
            transcripts: List of transcript dicts with:
                - video_id: YouTube video ID
                - transcript: Full transcript text
                - metadata: Optional dict with:
                    - title: Video title
                    - speaker: Speaker name if known
                    - language: Language code (default "en")
                    - video_url: Full video URL
                    - segments: Optional list of {start, end, text, summary}
            clear_existing: If True, clear AAG collection first
            
        Returns:
            Summary dict with counts and errors
        """
        if clear_existing:
            self.provider.clear_aag_collection()
        
        all_chunks = []
        stats = {
            "videos_processed": 0,
            "videos_skipped": 0,
            "chunks_created": 0,
            "errors": [],
        }
        
        for transcript_data in transcripts:
            try:
                video_id = transcript_data.get("video_id")
                if not video_id:
                    stats["videos_skipped"] += 1
                    continue
                
                transcript_text = transcript_data.get("transcript", "")
                metadata = transcript_data.get("metadata", {})
                segments = transcript_data.get("segments")
                
                if segments:
                    for seg in segments:
                        seg_text = seg.get("text", "")
                        if not seg_text:
                            continue
                        
                        start_second = seg.get("start", 0)
                        end_second = seg.get("end", 0)
                        chunking_key = f"{int(start_second)}-{int(end_second)}"
                        
                        chunk_id, chunk_hash, normalized = compute_chunk_identity(
                            source="aag",
                            doc_key=video_id,
                            chunking_key=chunking_key,
                            raw_text=seg_text,
                            source_type="aag"
                        )
                        
                        chunk_metadata = {
                            "chunk_id": chunk_id,
                            "chunk_hash": chunk_hash,
                            "source": StaticSourceType.AAG.value,
                            "video_id": video_id,
                            "video_title": metadata.get("title", ""),
                            "video_url": metadata.get("video_url", f"https://youtube.com/watch?v={video_id}"),
                            "speaker": metadata.get("speaker", ""),
                            "language": metadata.get("language", "en"),
                            "summary": seg.get("summary", ""),
                            "start_second": start_second,
                            "end_second": end_second,
                            "version": 1,
                            "is_latest": True,
                            "created_at": datetime.now().isoformat(),
                            "data_type": "aag",
                        }
                        
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "content": normalized,
                            "metadata": chunk_metadata,
                        })
                
                else:
                    if not transcript_text:
                        stats["videos_skipped"] += 1
                        continue
                    
                    chunks_with_keys = self._chunk_with_semantic_boundaries(transcript_text)
                    
                    for chunk_text, start_offset, end_offset, chunking_key in chunks_with_keys:
                        chunk_id, chunk_hash, normalized = compute_chunk_identity(
                            source="aag",
                            doc_key=video_id,
                            chunking_key=chunking_key,
                            raw_text=chunk_text,
                            source_type="aag"
                        )
                        
                        chunk_metadata = {
                            "chunk_id": chunk_id,
                            "chunk_hash": chunk_hash,
                            "source": StaticSourceType.AAG.value,
                            "video_id": video_id,
                            "video_title": metadata.get("title", ""),
                            "video_url": metadata.get("video_url", f"https://youtube.com/watch?v={video_id}"),
                            "speaker": metadata.get("speaker", ""),
                            "language": metadata.get("language", "en"),
                            "summary": "",
                            "start_offset": start_offset,
                            "end_offset": end_offset,
                            "version": 1,
                            "is_latest": True,
                            "created_at": datetime.now().isoformat(),
                            "data_type": "aag",
                        }
                        
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "content": normalized,
                            "metadata": chunk_metadata,
                        })
                
                stats["videos_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing video {transcript_data.get('video_id', 'unknown')}: {e}")
                stats["errors"].append(str(e))
                stats["videos_skipped"] += 1
        
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} AAG chunks...")
            texts = [c["content"] for c in all_chunks]
            embeddings = self._generate_embeddings_batch(texts)
            
            ids = [c["chunk_id"] for c in all_chunks]
            documents = [c["content"] for c in all_chunks]
            metadatas = []
            for c in all_chunks:
                meta = {}
                for k, v in c["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)
            
            logger.info(f"Upserting {len(all_chunks)} chunks to AAG collection...")
            self.provider.get_aag_collection().upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            
            stats["chunks_created"] = len(all_chunks)
        
        logger.info(f"AAG ingestion complete: {stats}")
        return stats
    
    def ingest_single_doc(
        self,
        doc_id: str,
        title: str,
        content: str,
        source_type: str = "unknown",
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single document programmatically.
        
        Useful for testing or adding individual documents.
        
        Args:
            doc_id: Unique identifier for the document
            title: Document title
            content: Full document content
            source_type: Source type string
            source_url: Optional URL
            tags: Optional tags list
            extra_metadata: Any additional metadata
            
        Returns:
            Summary dict
        """
        source_prefix = _get_source_prefix(source_type)
        chunks_with_keys = self._chunk_with_semantic_boundaries(content)
        all_chunks = []
        
        for chunk_text, start_offset, end_offset, chunking_key in chunks_with_keys:
            chunk_id, chunk_hash, normalized = compute_chunk_identity(
                source=source_prefix,
                doc_key=doc_id,
                chunking_key=chunking_key,
                raw_text=chunk_text,
                source_type=source_type
            )
            
            chunk_metadata = {
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "source": source_type,
                "doc_id": doc_id,
                "title": title,
                "source_url": source_url or "",
                "tags": ",".join(tags) if tags else "",
                "version": 1,
                "is_latest": True,
                "created_at": datetime.now().isoformat(),
                "data_type": "static",
            }
            
            if extra_metadata:
                for k, v in extra_metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        chunk_metadata[k] = v
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "content": normalized,
                "metadata": chunk_metadata,
            })
        
        if all_chunks:
            texts = [c["content"] for c in all_chunks]
            embeddings = self._generate_embeddings_batch(texts)
            
            ids = [c["chunk_id"] for c in all_chunks]
            documents = [c["content"] for c in all_chunks]
            metadatas = [c["metadata"] for c in all_chunks]
            
            self.provider.get_docs_collection().upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        
        return {
            "doc_id": doc_id,
            "chunks_created": len(all_chunks),
        }



