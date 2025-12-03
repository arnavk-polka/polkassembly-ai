import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def create_context_from_chunks(chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
    context_parts = []
    current_length = 0
    
    for i, chunk in enumerate(chunks):
        source_info = ""
        metadata = chunk.get('metadata', {})
        
        if metadata.get('title'):
            source_info += f"Title: {metadata['title']}\n"
        if metadata.get('url'):
            source_info += f"URL: {metadata['url']}\n"
        if metadata.get('source'):
            source_info += f"Source: {metadata['source']}\n"
        
        chunk_text = f"--- Document {i+1} ---\n{source_info}\nContent:\n{chunk['content']}\n\n"
        
        if current_length + len(chunk_text) > max_context_length:
            break
        
        context_parts.append(chunk_text)
        current_length += len(chunk_text)
    
    return ''.join(context_parts)

def remove_double_asterisks(text):
    return text.replace("**", "").replace("-", "")

def clean_example_urls(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    for line in lines:
        should_include_line = True
        
        image_matches = re.findall(image_pattern, line)
        
        if image_matches:
            for alt_text, url in image_matches:
                if 'https://polkassembly-ai.s3.us-east-1.amazonaws.com' not in url:
                    should_include_line = False
                    logger.info(f"Removed image line with invalid URL: {line.strip()}")
                    break
        
        if should_include_line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

