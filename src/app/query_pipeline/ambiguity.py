"""
Query ambiguity detection using LLM.
"""

import logging
from typing import Dict, Any, Optional, List

from src.core.errors import is_insufficient_quota_error

logger = logging.getLogger(__name__)


async def is_query_truly_ambiguous(query: str, qa_generator, sql_queries: Optional[List[str]] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Determine if a query truly needs clarification using LLM.
    
    Only marks as ambiguous if a REQUIRED parameter is missing (e.g., referenda ID when asking for details).
    Does NOT mark as ambiguous for missing optional parameters (e.g., network for generic listing queries).
    
    Uses GPT-3.5-turbo for fast, context-aware ambiguity detection.
    
    Args:
        query: The user query to check
        qa_generator: QA generator instance with LLM access
        sql_queries: Optional list of SQL queries to analyze
        conversation_history: Optional conversation history for context
    
    Returns:
        True if query truly needs clarification (missing required parameter), False otherwise
    """
    if not qa_generator or not hasattr(qa_generator, 'client'):
        logger.warning("No LLM client available for ambiguity check, defaulting to False")
        return False
    
    sql_context = ""
    if sql_queries:
        sql_str = ' '.join(sql_queries) if isinstance(sql_queries, list) else str(sql_queries)
        sql_context = f"\n\nSQL Query Generated: {sql_str[:200]}"
    
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent_messages = conversation_history[-6:]
        context_parts = []
        for msg in recent_messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '') or msg.get('message', '')
                if content and len(str(content).strip()) > 5:
                    role_display = role if role else 'user'
                    content_str = str(content)[:500]
                    context_parts.append(f"{role_display}: {content_str}")
        if context_parts:
            conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand what the user is referring to in their current query."
            logger.info(f"Ambiguity check - Including conversation history with {len(context_parts)} messages for query: '{query[:50]}'")
        else:
            logger.info(f"Ambiguity check - No conversation history available for query: '{query[:50]}'")
    
    from ...prompts.ambiguity_checker_prompt import PROMPT_TEMPLATE as ambiguity_prompt_template
    ambiguity_prompt = ambiguity_prompt_template.format(
        query=query,
        conversation_context=conversation_context
    )
    
    try:
        model_to_use = "gpt-4o-mini"
        
        response = qa_generator.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a query ambiguity detector. Respond with ONLY 'true' or 'false' (lowercase, one word)."},
                {"role": "user", "content": ambiguity_prompt}
            ],
            temperature=0.0,
            max_tokens=3
        )
        raw_answer = response.choices[0].message.content or ""
        answer = raw_answer.strip().lower()
        
        first_word = answer.split()[0] if answer.split() else ""
        is_ambiguous = first_word == "true"
        
        logger.info(f"Ambiguity check - Query: '{query[:50]}', Model: {model_to_use}, Raw response: '{raw_answer}', Parsed: '{answer}', First word: '{first_word}', Is ambiguous: {is_ambiguous}")
        
        return is_ambiguous
    except Exception as e:
        if is_insufficient_quota_error(e):
            logger.error(f"Insufficient quota error in LLM ambiguity check: {e}")
            raise
        logger.error(f"Error in LLM ambiguity check: {e}, defaulting to False")
        return False

