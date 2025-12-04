"""
Clarification handler for low-confidence queries.
"""

from typing import Dict, Any, Optional, List
import logging
from src.core.errors import is_insufficient_quota_error, get_quota_error_message

logger = logging.getLogger(__name__)


def _build_route_context_section(route: Optional[str]) -> str:
    """Build the route context section to embed in the clarification prompt."""
    normalized_route = (route or "undetermined").lower()
    
    route_contexts = {
        "dynamic": """
This is a dynamic/on-chain data query. Common ambiguities include:
- Network selection (Polkadot vs Kusama)
- Proposal/referendum type or status
- Time period or date range
- Specific filters (active, passed, rejected, etc.)

For queries about proposals, referenda, votes, or treasury data, the most common ambiguity is which network (Polkadot or Kusama).
""",
        "static": """
This is a static/educational query. Common ambiguities include:
- Specific topic or concept within the broader subject
- Level of detail needed (overview vs deep dive)
- Specific use case or scenario
- Unclear terminology or acronyms
""",
        "hybrid": """
This is a hybrid query needing both explanation and data. Common ambiguities include:
- Network selection (Polkadot or Kusama) for the data portion
- Scope of explanation vs data requested
"""
    }
    
    return route_contexts.get(
        normalized_route,
        """
The route for this query has not been determined yet. Focus on clarifying:
- Whether the user is referring to a specific proposal/referendum/bounty or asking generally
- Any missing identifiers (ID numbers, links, titles)
- The exact topic or scope they care about if they're being vague
"""
    )


def _build_conversation_context(conversation_history: Optional[List[Dict[str, Any]]]) -> str:
    """Extract and format conversation history for context."""
    if not conversation_history or len(conversation_history) == 0:
        return ""
    
    recent_messages = conversation_history[-6:]
    context_parts = []
    
    for msg in recent_messages:
        if not isinstance(msg, dict):
            continue
        
        role = msg.get('role', '')
        content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '')
        
        if content and len(content) > 5:
            role_display = role if role else 'user'
            context_parts.append(f"{role_display}: {content[:200]}")
    
    if not context_parts:
        return ""
    
    return f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand what the user is referring to in their query."


def _clean_llm_response(response: str) -> str:
    """Remove quotes if LLM wrapped the response."""
    response = response.strip()
    if (response.startswith('"') and response.endswith('"')) or \
       (response.startswith("'") and response.endswith("'")):
        return response[1:-1]
    return response


async def _call_llm_for_clarification(
    clarification_prompt: str,
    qa_generator,
    log_step
) -> str:
    """Call LLM to generate clarification question."""
    if qa_generator.gemini_client:
        model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
        log_step("clarification_llm_call", {"model": model_name})
        
        response = qa_generator.gemini_client.get_response(clarification_prompt)
        return _clean_llm_response(response)
    
    elif hasattr(qa_generator, 'client'):
        system_prompt = """You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly. 
You help clarify ambiguous user queries by asking one specific question. Always use the exact same terminology the user used in their query."""
        
        response = qa_generator.client.chat.completions.create(
            model=qa_generator.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clarification_prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return _clean_llm_response(response.choices[0].message.content)
    
    else:
        raise Exception("No LLM client available")


def _get_model_name(qa_generator) -> str:
    """Extract model name from qa_generator."""
    if qa_generator.gemini_client:
        return getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
    if hasattr(qa_generator, 'model'):
        return qa_generator.model
    return 'fallback'


async def generate_clarification_question(
    query: str,
    route: Optional[str],
    router_confidence: float,
    qa_generator,
    log_step,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate a clarifying question when retrieval confidence is low.
    
    Args:
        query: The original user query
        route: The route that was selected (for logging only)
        router_confidence: The router's confidence level
        qa_generator: QA generator instance
        log_step: Logging function
        conversation_history: Optional conversation history for context
    
    Returns:
        Dictionary with clarification question and metadata
    """
    log_step("clarification_start", {
        "query_preview": query[:100],
        "route": route or "undetermined",
        "router_confidence": router_confidence
    })
    
    normalized_route = (route or "undetermined").lower()
    route_context = _build_route_context_section(route)
    conversation_context = _build_conversation_context(conversation_history)
    
    from src.prompts.clarification_prompt import PROMPT_TEMPLATE as clarification_prompt_template
    clarification_prompt = clarification_prompt_template.format(
        query=query,
        normalized_route=normalized_route,
        route_context=route_context,
        conversation_context=conversation_context
    )
    
    try:
        clarification_question = await _call_llm_for_clarification(
            clarification_prompt,
            qa_generator,
            log_step
        )
    except Exception as e:
        if is_insufficient_quota_error(e):
            log_step("clarification_llm_error", {"error": str(e), "quota_error": True}, "error")
            return {
                'answer': get_quota_error_message(),
                'sources': [],
                'confidence': router_confidence,
                'follow_up_questions': [],
                'context_used': False,
                'model_used': 'error',
                'chunks_used': 0,
                'search_method': 'quota_error'
            }
        
        log_step("clarification_llm_error", {"error": str(e)}, "error")
        raise
    
    log_step("clarification_complete", {
        "question": clarification_question[:100],
        "method": "llm_generated"
    })
    
    model_name = _get_model_name(qa_generator)
    
    return {
        'answer': clarification_question,
        'sources': [],
        'confidence': router_confidence,
        'follow_up_questions': [],
        'context_used': False,
        'model_used': model_name,
        'chunks_used': 0,
        'search_method': 'clarification',
        'requires_clarification': True,
        'clarification_question': clarification_question
    }

