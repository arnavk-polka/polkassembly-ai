import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sources = []
    seen_urls = set()
    
    sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    for chunk in sorted_chunks[:3]:
        metadata = chunk.get('metadata', {})
        
        source = {
            'title': metadata.get('title', 'Unknown Title'),
            'url': metadata.get('url', ''),
            'source_type': metadata.get('source', 'unknown'),
            'similarity_score': chunk.get('similarity_score', 0.0)
        }
        
        if source['url'] and source['url'] not in seen_urls:
            sources.append(source)
            seen_urls.add(source['url'])
        elif not source['url'] and len(sources) < 2:
            sources.append(source)
        
        if len(sources) >= 2 and any(s['url'] for s in sources):
            break
    
    return sources

def estimate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    
    similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in chunks]
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    
    chunk_bonus = min(len(chunks) * 0.1, 0.3)
    
    confidence = min(avg_similarity + chunk_bonus, 1.0)
    return round(confidence, 2)

