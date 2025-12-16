"""
Type definitions for the StaticProvider abstraction.

This module defines the canonical result shape that all static search results
must conform to, regardless of whether they come from docs, AAG segments, or
(in the future) DKG.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class StaticSourceType(str, Enum):
    """Known static source types."""
    POLKA_WIKI = "polka_wiki"
    POLKASSEMBLY_DOC = "polkassembly_doc"
    POLKADOT_NETWORK = "polkadot_network"
    AAG = "aag"
    UNKNOWN = "unknown"


@dataclass
class DKGMatch:
    """
    Placeholder for future DKG provenance.
    
    When DKG integration is implemented, this will contain:
    - asset_ual: The unique asset locator on DKG
    - chunk_id: The chunk identifier within the DKG asset
    - chunk_hash: The hash of the chunk for verification
    
    For now, this is always None in results.
    """
    asset_ual: str
    chunk_id: str
    chunk_hash: str


@dataclass
class DocMetadata:
    """
    Metadata specific to documentation chunks (wiki, Polkassembly, etc.)
    """
    doc_id: str
    title: str
    source_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    subdirectory: Optional[str] = None


@dataclass
class AAGMetadata:
    """
    Metadata specific to AAG (video transcript) segments.
    """
    video_id: str
    summary: Optional[str] = None
    speaker: Optional[str] = None
    language: str = "en"
    start_second: Optional[float] = None
    end_second: Optional[float] = None
    video_title: Optional[str] = None
    video_url: Optional[str] = None


@dataclass
class ChunkIdentity:
    """
    Core identity fields that MUST be present on every chunk.
    
    These fields enable:
    - Stable identification across re-ingestion
    - Matching against DKG assets
    - Deduplication and versioning
    """
    chunk_id: str
    chunk_hash: str
    version: int = 1
    is_latest: bool = True


@dataclass
class StaticSearchResult:
    """
    The canonical result shape for all static search results.
    
    This is what the StaticProvider returns, and what all consumers should depend on.
    The shape is designed to:
    - Unify docs and AAG segments under one interface
    - Support future DKG provenance annotation
    - Carry all metadata needed for display and attribution
    """
    id: str
    score: float
    content: str
    source: str
    chunk_id: str
    chunk_hash: str
    metadata: Dict[str, Any]
    dkg_match: Optional[DKGMatch] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "chunk_hash": self.chunk_hash,
            "metadata": self.metadata,
            "dkg_match": {
                "asset_ual": self.dkg_match.asset_ual,
                "chunk_id": self.dkg_match.chunk_id,
                "chunk_hash": self.dkg_match.chunk_hash,
            } if self.dkg_match else None
        }
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to the legacy format expected by existing handlers.
        
        This maintains backward compatibility with code that expects:
        {content, metadata, similarity_score, source}
        """
        dkg_payload: Optional[Dict[str, Any]] = None
        if self.dkg_match:
            dkg_payload = {
                "asset_ual": self.dkg_match.asset_ual,
                "chunk_id": self.dkg_match.chunk_id,
                "chunk_hash": self.dkg_match.chunk_hash,
            }
        
        return {
            "content": self.content,
            "metadata": {
                **self.metadata,
                "chunk_id": self.chunk_id,
                "chunk_hash": self.chunk_hash,
                "dkg_match": dkg_payload,
            },
            "similarity_score": self.score,
            "source": self.source,
        }


@dataclass
class StaticSearchFilters:
    """
    Filters that can be applied to static search.
    """
    source: Optional[str] = None
    doc_id: Optional[str] = None
    video_id: Optional[str] = None
    speaker: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    min_score: Optional[float] = None
    
    def to_chroma_where(self) -> Optional[Dict[str, Any]]:
        """
        Convert filters to Chroma 'where' clause format.
        
        Returns None if no filters are set.
        """
        conditions = []
        
        if self.source:
            conditions.append({"source": self.source})
        if self.doc_id:
            conditions.append({"doc_id": self.doc_id})
        if self.video_id:
            conditions.append({"video_id": self.video_id})
        if self.speaker:
            conditions.append({"speaker": self.speaker})
        if self.language:
            conditions.append({"language": self.language})
        
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


