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
"""

import hashlib
import re
from typing import Tuple


def normalize_text(text: str) -> str:
    """
    Apply canonical normalization to chunk text before hashing.
    
    Rules (applied in order):
    1. Strip leading/trailing whitespace
    2. Normalize all line endings to \n
    3. Collapse multiple consecutive whitespace chars to single space
    4. Preserve case (we don't lowercase - content meaning matters)
    
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


def generate_chunk_id(
    source_id: str,
    start_offset: int,
    end_offset: int,
    source_type: str = "doc"
) -> str:
    """
    Generate a stable, deterministic chunk ID.
    
    Format: {source_type}:{source_id}:{start}-{end}
    
    Examples:
    - doc:polka_wiki_staking:0-1000
    - aag:YT_abc123:230-420
    
    Args:
        source_id: Unique identifier for the source document/video
        start_offset: Character offset where this chunk starts in the source
        end_offset: Character offset where this chunk ends in the source
        source_type: Type prefix ("doc" or "aag")
        
    Returns:
        Stable chunk ID string
        
    Important:
    - source_id should be sanitized (no colons, safe chars only)
    - Offsets are character-based for docs, second-based for AAG
    """
    safe_source_id = source_id.replace(':', '_').replace(' ', '_')
    return f"{source_type}:{safe_source_id}:{start_offset}-{end_offset}"


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
    source_id: str,
    start_offset: int,
    end_offset: int,
    raw_text: str,
    source_type: str = "doc"
) -> Tuple[str, str, str]:
    """
    Compute full chunk identity in one call.
    
    This is the primary entry point for chunk identity computation.
    Use this when ingesting new chunks.
    
    Args:
        source_id: Unique identifier for source document/video
        start_offset: Start position in source
        end_offset: End position in source
        raw_text: The raw chunk text (will be normalized)
        source_type: "doc" or "aag"
        
    Returns:
        Tuple of (chunk_id, chunk_hash, normalized_text)
        
    Example:
        chunk_id, chunk_hash, normalized = compute_chunk_identity(
            source_id="polka_wiki_staking",
            start_offset=0,
            end_offset=1000,
            raw_text="  Some text with extra   spaces  ",
            source_type="doc"
        )
    """
    chunk_id = generate_chunk_id(source_id, start_offset, end_offset, source_type)
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



