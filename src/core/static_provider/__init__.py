"""
StaticProvider - Unified abstraction for static knowledge retrieval.

This module provides:
- StaticProvider: Main search interface over Chroma
- StaticIngester: Ingestion pipeline for docs and AAG segments
- Type definitions for consistent result shapes
- Hashing utilities for stable chunk identity

Example usage:

    from src.core.static_provider import StaticProvider, StaticIngester
    
    provider = StaticProvider(openai_api_key="...")
    results = provider.search_docs("What is staking?", k=5)
    
    ingester = StaticIngester(openai_api_key="...")
    ingester.ingest_docs_from_directory("data/static_sources")

DKG Integration (TODO):
    Results include a `dkg_match` field that is always None for now.
    When DKG is integrated:
    1. After Chroma search, look up chunk_id/chunk_hash in DKG
    2. If match found, populate dkg_match with {asset_ual, chunk_id, chunk_hash}
    3. UI can then show "Verified on DKG" badge for matched chunks
"""

from .types import (
    StaticSearchResult,
    StaticSearchFilters,
    StaticSourceType,
    DKGMatch,
    DocMetadata,
    AAGMetadata,
    ChunkIdentity,
)

from .hashing import (
    normalize_text,
    generate_chunk_id,
    generate_chunk_hash,
    compute_chunk_identity,
    verify_chunk_hash,
)

from .provider import StaticProvider

from .ingestion import StaticIngester

__all__ = [
    "StaticProvider",
    "StaticIngester",
    "StaticSearchResult",
    "StaticSearchFilters",
    "StaticSourceType",
    "DKGMatch",
    "DocMetadata",
    "AAGMetadata",
    "ChunkIdentity",
    "normalize_text",
    "generate_chunk_id",
    "generate_chunk_hash",
    "compute_chunk_identity",
    "verify_chunk_hash",
]



