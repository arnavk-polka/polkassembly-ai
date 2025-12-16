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

        dkg_match = metadata.get('dkg_match')
        chunk_id_from_meta = metadata.get('chunk_id')
        chunk_hash_from_meta = metadata.get('chunk_hash')
        
        logger.debug(f"extract_sources: chunk metadata keys={list(metadata.keys())}, dkg_match={dkg_match}, chunk_id={chunk_id_from_meta}")
        
        if dkg_match and isinstance(dkg_match, dict):
            asset_ual = dkg_match.get('asset_ual')
            chunk_id = dkg_match.get('chunk_id')
            chunk_hash = dkg_match.get('chunk_hash')
            if asset_ual:
                source['dkg_asset_ual'] = asset_ual
                logger.info(f"extract_sources: Found dkg_asset_ual={asset_ual} for chunk_id={chunk_id}")
            if chunk_id:
                source['chunk_id'] = chunk_id
            if chunk_hash:
                source['chunk_hash'] = chunk_hash
        elif chunk_id_from_meta:
            source['chunk_id'] = chunk_id_from_meta
            if chunk_hash_from_meta:
                source['chunk_hash'] = chunk_hash_from_meta
            logger.debug(f"extract_sources: Using chunk_id from metadata (no dkg_match): {chunk_id_from_meta}")
        
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

