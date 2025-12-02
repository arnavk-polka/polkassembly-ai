#!/usr/bin/env python3
"""
Chunks Reranker - Logic to prioritize chunks based on content quality and image presence
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def metadata_priority_score(chunk):
    """Assigns bias bonuses based on domain metadata."""
    metadata = {k: str(v).lower() for k,v in chunk.get("metadata", {}).items()}
    content  = chunk.get("content", "").lower()

    score = 0.0

    if "pa_docs" in str(metadata):
        score += 0.30      

    if "polkassembly" in content or "polkassembly" in str(metadata):
        score += 0.20      

    if "s3.amazonaws.com" in content:
        score += 0.10      

    return score

def keyword_filter(query, chunks):
    words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
    if not words:
        return chunks

    filtered = []
    for chunk in chunks:
        text = chunk.get("content", "").lower()
        overlap = sum(1 for w in words if w in text)
        chunk["keyword_score"] = overlap
        filtered.append(chunk)

    strong = [c for c in filtered if c["keyword_score"] > 0]
    return strong if strong else filtered

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
