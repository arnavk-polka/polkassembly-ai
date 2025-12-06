#!/usr/bin/env python3
"""
Chunks Reranker - Logic to prioritize chunks based on content quality and image presence
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def is_polkassembly_chunk(chunk):
    """Check if chunk is from Polkassembly source."""
    raw_metadata = chunk.get("metadata", {})
    metadata_lower = {k.lower(): str(v).lower() for k, v in raw_metadata.items()}
    content = chunk.get("content", "").lower()
    
    return (
        metadata_lower.get("source", "") == "polkassembly" or
        "pa_docs" in str(raw_metadata).lower() or
        "polkassembly" in str(raw_metadata).lower() or
        "polkassembly" in content
    )

def metadata_priority_score(chunk):
    """Assigns bias bonuses based on domain metadata. Polkassembly chunks get strong boost."""
    raw_metadata = chunk.get("metadata", {})
    metadata_lower = {k.lower(): str(v).lower() for k, v in raw_metadata.items()}
    content = chunk.get("content", "").lower()

    score = 0.0

    if is_polkassembly_chunk(chunk):
        score += 5.0

    if "pa_docs" in str(raw_metadata).lower():
        score += 2.0

    if "polkassembly" in content or "polkassembly" in str(raw_metadata).lower():
        score += 1.0

    if "s3.amazonaws.com" in content:
        score += 0.10

    return score

def keyword_filter(query, chunks):
    words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
    if not words:
        return chunks

    filtered = []
    polkassembly_chunks = []
    other_chunks = []
    
    for chunk in chunks:
        text = chunk.get("content", "").lower()
        overlap = sum(1 for w in words if w in text)
        chunk["keyword_score"] = overlap
        
        if is_polkassembly_chunk(chunk):
            polkassembly_chunks.append(chunk)
        else:
            other_chunks.append(chunk)

    strong_other = [c for c in other_chunks if c["keyword_score"] > 0]
    
    if strong_other:
        return polkassembly_chunks + strong_other
    else:
        return polkassembly_chunks + other_chunks

def final_rerank(query, chunks):
    """
    Full hybrid reranking after semantic scores are assigned:
    final_score = semantic + keyword_weight + metadata_bias
    """
    ranked = []

    for c in chunks:
        semantic = c.get("semantic_score", 0)
        keyword  = c.get("keyword_score", 0)
        metadata = metadata_priority_score(c)

        final = (
            semantic
            + keyword * 0.40
            + metadata
        )

        c["final_score"] = final
        ranked.append(c)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked

def prioritize_polkassembly_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prioritize Polkassembly chunks before semantic reranking.
    Ensures they are not removed even if they have lower similarity scores.
    """
    polkassembly_chunks = []
    other_chunks = []
    
    for chunk in chunks:
        if is_polkassembly_chunk(chunk):
            polkassembly_chunks.append(chunk)
        else:
            other_chunks.append(chunk)
    
    return polkassembly_chunks + other_chunks

def boost_polkassembly_semantic_scores(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Boost semantic_score for Polkassembly chunks after semantic reranking.
    Ensures they rank higher in final scoring.
    """
    for chunk in chunks:
        if is_polkassembly_chunk(chunk) and "semantic_score" in chunk:
            chunk["semantic_score"] = chunk["semantic_score"] + 5.0
    
    return chunks

def rerank_static_chunks(query: str, static_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply keyword filtering to static chunks.
    Semantic reranking and final hybrid scoring happen later in the pipeline.
    
    Args:
        query: Search query string
        static_chunks: List of static chunk dictionaries
    
    Returns:
        List of keyword-filtered chunks with keyword_score assigned
    """
    return keyword_filter(query, static_chunks)
