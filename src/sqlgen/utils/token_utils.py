import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
except ImportError:
    tiktoken = None

def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Count tokens in text using tiktoken or approximate counting"""
    if tiktoken:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error with tiktoken: {e}, using approximate counting")
    
    return len(text) // 4

def trim_prompt_to_fit_tokens(system_prompt: str, max_tokens: int = 20000, completion_tokens: int = 1000, buffer_tokens: int = 500) -> str:
    """Trim the system prompt to fit within token limits"""
    current_tokens = count_tokens(system_prompt)
    
    if current_tokens <= max_tokens:
        logger.info(f"Token analysis - Current: {current_tokens}, Max: {max_tokens} - No trimming needed")
        return system_prompt
    
    target_tokens = int(max_tokens * 0.95)
    logger.info(f"Token analysis - Current: {current_tokens}, Max: {max_tokens}, Target: {target_tokens} - Trimming needed")
    
    target_length = int(len(system_prompt) * (target_tokens / current_tokens))
    lines = system_prompt.split('\n')
    
    essential_sections = [
        'COLUMN SELECTION STRATEGY:',
        'EXAMPLE QUERIES:',
        'CRITICAL NULL VALUE HANDLING:',
        'NaN VALUE HANDLING:',
        'Very very Important Rule:'
    ]
    
    trimmed_lines = []
    current_length = 0
    
    for line in lines:
        if any(section in line for section in essential_sections):
            section_start = lines.index(line)
            for i in range(section_start, min(section_start + 10, len(lines))):
                if lines[i] not in trimmed_lines:
                    test_length = current_length + len(lines[i]) + 1
                    if test_length < target_length:
                        trimmed_lines.append(lines[i])
                        current_length = test_length
                    else:
                        break
    
    for line in lines:
        if line not in trimmed_lines and current_length + len(line) + 1 < target_length:
            if len(line) < 200:
                trimmed_lines.append(line)
                current_length += len(line) + 1
    
    trimmed_prompt = '\n'.join(trimmed_lines)
    final_tokens = count_tokens(trimmed_prompt)
    logger.info(f"Prompt trimmed: {current_tokens} -> {final_tokens} tokens (target: {target_tokens})")
    
    return trimmed_prompt

