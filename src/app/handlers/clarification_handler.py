"""
Clarification detection and query combination logic.
"""

import logging
from typing import Dict, Any, Optional, List

from ..pipeline.utils import log_step
from ..pipeline.routing import route_query_llm

logger = logging.getLogger(__name__)


async def combine_query_with_clarification(
    original_query: str,
    clarification_response: str,
    clarification_question: str,
    qa_generator,
    log_step,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Intelligently combine the original query with the user's clarification response
    using an LLM to create a better, more coherent combined query.
    
    Args:
        original_query: The original ambiguous query
        clarification_response: The user's response to the clarification question
        clarification_question: The clarification question that was asked (needed to understand context)
        qa_generator: QA generator instance
        log_step: Logging function
        conversation_history: Optional conversation history for additional context
    
    Returns:
        A combined query that intelligently merges the original query and clarification
    """
    log_step("query_combination_start", {
        "original_query": original_query[:100],
        "clarification_question": clarification_question[:100],
        "clarification_response": clarification_response[:100]
    })
    
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent_messages = conversation_history[-6:]
        context_parts = []
        for msg in recent_messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '')
                if content and len(content) > 5:
                    role_display = role if role else 'user'
                    context_parts.append(f"{role_display}: {content[:150]}")
        if context_parts:
            conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand the full context of what the user is asking about."
    
    from ...prompts.query_combination_prompt import PROMPT_TEMPLATE as combination_prompt_template
    combination_prompt = combination_prompt_template.format(
        original_query=original_query,
        clarification_question=clarification_question,
        clarification_response=clarification_response,
        conversation_context=conversation_context
    )
    
    try:
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("query_combination_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(combination_prompt)
            combined_query = response.strip()
            
            if combined_query.startswith('"') and combined_query.endswith('"'):
                combined_query = combined_query[1:-1]
            elif combined_query.startswith("'") and combined_query.endswith("'"):
                combined_query = combined_query[1:-1]
            
            log_step("query_combination_complete", {
                "combined_query": combined_query[:100]
            })
            
            return combined_query
        else:
            if hasattr(qa_generator, 'client'):
                system_prompt = """You are a query combination assistant. You combine user queries with clarification responses to create coherent, natural queries."""
                
                response = qa_generator.client.chat.completions.create(
                    model=qa_generator.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combination_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                combined_query = response.choices[0].message.content.strip()
                
                if combined_query.startswith('"') and combined_query.endswith('"'):
                    combined_query = combined_query[1:-1]
                elif combined_query.startswith("'") and combined_query.endswith("'"):
                    combined_query = combined_query[1:-1]
                
                return combined_query
        
        log_step("query_combination_fallback", {"note": "Using simple concatenation fallback"})
        return f"{original_query} {clarification_response}".strip()
        
    except Exception as e:
        log_step("query_combination_error", {"error": str(e)}, "error")
        return f"{original_query} {clarification_response}".strip()


async def detect_and_handle_clarification_response(
    userMessage: str,
    conversationHistory: Optional[List[Dict[str, Any]]],
    qa_generator,
    log_step
) -> Optional[Dict[str, Any]]:
    """
    Detect if the current user message is a response to a clarification question.
    Uses simple conversation history pattern: if last assistant message is a question,
    and we can find the original user query before it, treat as clarification.
    
    Args:
        userMessage: The current user message
        conversationHistory: Previous conversation messages in format [{'role': 'user/assistant', 'content': '...'}, ...]
        qa_generator: QA generator instance
        log_step: Logging function
        
    Returns:
        None if not a clarification response, or dict with:
        - combined_query: Original query + clarification response
        - original_query: The original query that needed clarification
        - original_route: The original route (determined by re-routing)
        - clarification_response: The user's clarification response
    """
    if not conversationHistory or len(conversationHistory) < 2:
        log_step("clarification_check_no_history", {"note": "Not enough conversation history"}, "debug")
        return None
    
    last_entry = conversationHistory[-1]
    if isinstance(last_entry, dict):
        last_role = last_entry.get('role', '')
        last_content = last_entry.get('content', '') or last_entry.get('response', '')
    elif hasattr(last_entry, 'role'):
        last_role = last_entry.role
        last_content = getattr(last_entry, 'content', None) or getattr(last_entry, 'response', None)
    else:
        log_step("clarification_check_invalid_format", {"entry_type": str(type(last_entry))}, "debug")
        return None
    
    if last_role != 'assistant' or not last_content or not last_content.strip().endswith('?'):
        log_step("clarification_check_not_question", {
            "last_role": last_role,
            "is_question": last_content.strip().endswith('?') if last_content else False
        }, "debug")
        return None
    
    user_message_lower = userMessage.lower().strip()
    word_count = len(user_message_lower.split())
    question_words = ['how', 'what', 'when', 'where', 'who', 'why', 'which', 'can', 'could', 'should', 'would', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'will', 'show', 'tell', 'explain', 'list', 'find']
    is_question_word = any(user_message_lower.startswith(qw + ' ') or user_message_lower.startswith(qw + '?') for qw in question_words)
    has_question_mark = '?' in userMessage
    
    if is_question_word or word_count > 5 or has_question_mark:
        log_step("clarification_check_full_question", {
            "user_message": userMessage[:100],
            "word_count": word_count,
            "is_question_word": is_question_word,
            "has_question_mark": has_question_mark,
            "note": "User message looks like a full question, not a clarification response"
        }, "debug")
        return None
    
    original_query = None
    for i in range(len(conversationHistory) - 2, -1, -1):
        entry = conversationHistory[i]
        if isinstance(entry, dict):
            role = entry.get('role', '')
            if role == 'user':
                content = entry.get('content', '') or entry.get('query', '')
                if content:
                    original_query = content
                    break
        elif hasattr(entry, 'query'):
            original_query = entry.query
            break
    
    if not original_query:
        log_step("clarification_check_no_original_query", {
            "note": "Found clarification question but couldn't find original user query"
        }, "debug")
        return None
    
    log_step("clarification_detected", {
        "original_query": original_query[:100],
        "clarification_question": last_content[:100],
        "clarification_response": userMessage[:100]
    }, "info")
    
    route_result = await route_query_llm(
        original_query,
        conversationHistory[:-1] if conversationHistory else None,
        qa_generator
    )
    original_route = route_result.get("route", "dynamic")
    original_router_confidence = route_result.get("confidence", 0.8)
    
    return {
        'combined_query': userMessage,
        'original_query': original_query,
        'original_route': original_route,
        'original_router_confidence': original_router_confidence,
        'clarification_response': userMessage
    }

