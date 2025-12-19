"""
Chunk identity and hashing utilities.

This module provides deterministic, stable chunk identification and hashing.
These functions MUST be used consistently across:
- Initial ingestion into Chroma
- Re-ingestion / updates
- Future DKG asset creation

The design ensures that:
- Same source + same content = same chunk_id and chunk_hash
- Formatting noise (extra spaces, line endings) doesn't break identity
- Content changes result in different hashes

NEW SCHEME (deterministic, source-scoped):
- chunk_id = <source>:<doc_key>:<chunking_key>
- chunk_hash = SHA256(chunk_id + "||" + normalized_text)
- Mapping: (chunk_id, chunk_hash) -> UAL
"""

import hashlib
import re
from typing import Tuple, Optional


def normalize_text(text: str) -> str:
    """
    Apply canonical normalization to chunk text before hashing.
    
    FROZEN RULES (never change without migration):
    1. Trim leading/trailing whitespace
    2. Convert Windows newlines to \n
    3. Collapse repeated spaces/tabs to single spaces
    4. Collapse \n{3,} to \n\n
    5. Strip each line, then rejoin with \n
    6. Preserve case (content meaning matters)
    
    This normalization is intentionally simple and stable.
    Do NOT change these rules without re-indexing all data.
    
    Args:
        text: Raw chunk text
        
    Returns:
        Normalized text suitable for hashing
    """
    if not text:
        return ""
    
    normalized = text.strip()
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'[ \t]+', ' ', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    normalized = '\n'.join(line.strip() for line in normalized.split('\n'))
    
    return normalized


def _get_source_prefix(source_type: str) -> str:
    """
    Map source type to deterministic source prefix.
    
    Source prefixes (frozen):
    - wiki: PolkaWiki pages
    - aag: AAG video transcripts
    - pa_md: Polkassembly markdown docs
    - pa_doc: Polkassembly official docs
    - network: Polkadot Network docs
    """
    mapping = {
        "polka_wiki": "wiki",
        "polkassembly_doc": "pa_md",
        "polkadot_network": "network",
        "aag": "aag",
        "doc": "wiki",  # default fallback
    }
    return mapping.get(source_type.lower(), "unknown")


def generate_chunk_id(
    source: str,
    doc_key: str,
    chunking_key: str,
    source_type: Optional[str] = None
) -> str:
    """
    Generate deterministic chunk ID using new scheme.
    
    Format: <source>:<doc_key>:<chunking_key>
    
    Examples:
    - wiki:staking:introduction:0
    - wiki:staking:staking/basics:1
    - aag:YT_abc123:230-420
    - pa_md:governance_guide:overview:0
    
    Args:
        source: Source prefix (e.g., "wiki", "aag", "pa_md")
               If None, will be derived from source_type
        doc_key: Stable document identifier (e.g., page slug, video ID)
        chunking_key: Chunking identifier (e.g., "section_id:chunk_index" or "start-end")
        source_type: Optional source type string (used if source is None)
        
    Returns:
        Deterministic chunk ID string
        
    Important:
    - All components are sanitized (no colons except as separators)
    - Source prefix prevents cross-source collisions
    - chunking_key should be stable across re-chunking (use semantic boundaries)
    """
    if source is None:
        source = _get_source_prefix(source_type or "doc")
    
    safe_source = source.replace(':', '_').replace(' ', '_')
    safe_doc_key = doc_key.replace(':', '_').replace(' ', '_')
    safe_chunking_key = chunking_key.replace(' ', '_')
    
    return f"{safe_source}:{safe_doc_key}:{safe_chunking_key}"




def generate_chunk_hash(
    chunk_id: str,
    normalized_text: str
) -> str:
    """
    Generate a stable hash for a chunk.
    
    The hash depends on BOTH:
    - The chunk's identity (chunk_id)
    - The chunk's normalized content
    
    This means:
    - Same position + same content = same hash
    - Same position + different content = different hash
    - Different position + same content = different hash
    
    Args:
        chunk_id: The chunk's stable identifier
        normalized_text: Text after applying normalize_text()
        
    Returns:
        SHA-256 hex digest (64 chars)
    """
    hash_input = f"{chunk_id}||{normalized_text}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def compute_chunk_identity(
    source: str,
    doc_key: str,
    chunking_key: str,
    raw_text: str,
    source_type: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Compute full chunk identity in one call (new scheme).
    
    This is the primary entry point for chunk identity computation.
    Use this when ingesting new chunks.
    
    Args:
        source: Source prefix (e.g., "wiki", "aag") or None to derive from source_type
        doc_key: Stable document identifier (e.g., page slug, video ID)
        chunking_key: Chunking identifier (e.g., "section_id:chunk_index" or "start-end")
        raw_text: The raw chunk text (will be normalized)
        source_type: Optional source type string (used if source is None)
        
    Returns:
        Tuple of (chunk_id, chunk_hash, normalized_text)
        
    Example:
        chunk_id, chunk_hash, normalized = compute_chunk_identity(
            source="wiki",
            doc_key="staking",
            chunking_key="introduction:0",
            raw_text="  Some text with extra   spaces  ",
        )
    """
    chunk_id = generate_chunk_id(source, doc_key, chunking_key, source_type)
    normalized = normalize_text(raw_text)
    chunk_hash = generate_chunk_hash(chunk_id, normalized)
    
    return chunk_id, chunk_hash, normalized




def verify_chunk_hash(
    chunk_id: str,
    content: str,
    expected_hash: str
) -> bool:
    """
    Verify that a chunk's content matches its stored hash.
    
    Useful for:
    - Detecting content drift
    - Validating DKG matches
    - Integrity checks
    
    Args:
        chunk_id: The chunk's identifier
        content: The chunk's current content (will be normalized)
        expected_hash: The hash to verify against
        
    Returns:
        True if hash matches, False otherwise
    """
    normalized = normalize_text(content)
    computed_hash = generate_chunk_hash(chunk_id, normalized)
    return computed_hash == expected_hash





