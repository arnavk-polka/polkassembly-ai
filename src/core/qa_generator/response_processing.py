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

def generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
    try:
        if not chunks:
            return "No relevant information found."
        
        from .context_processing import create_context_from_chunks
        context = create_context_from_chunks(chunks, max_context_length=2000)
        
        from ...prompts.summary_prompt import PROMPT_TEMPLATE as summary_prompt_template
        summary_prompt = summary_prompt_template.format(context=context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": summary_prompt.format(context=context)}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        summary = response.choices[0].message.content.strip()
        
        from .context_processing import clean_example_urls
        summary = clean_example_urls(summary)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Unable to generate summary."

