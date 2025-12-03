"""
Query routing logic to determine the appropriate route.
"""

import logging
import math
import os
from typing import Dict, Any, Optional, List

from src.core.errors import is_insufficient_quota_error
from .utils import log_step

logger = logging.getLogger(__name__)


def get_router_confidence_from_logprobs(choice) -> float:
    """Convert the logprob of the predicted route token into probability."""
    try:
        route_text = (choice.message.content or "").strip().lower()
        if not route_text:
            return 0.0
        logprobs = getattr(choice, "logprobs", None)
        if not logprobs:
            return 0.0
        content_tokens = getattr(logprobs, "content", None)
        if not content_tokens:
            return 0.0
        def _token_field(token_info, field):
            if isinstance(token_info, dict):
                return token_info.get(field)
            return getattr(token_info, field, None)
        logprob_value = None
        for token_info in content_tokens:
            token = (_token_field(token_info, "token") or "").strip().lower()
            if token == route_text:
                logprob_value = _token_field(token_info, "logprob")
                break
        if logprob_value is None:
            logprob_value = _token_field(content_tokens[-1], "logprob")
        if logprob_value is None:
            return 0.0
        return float(math.exp(logprob_value))
    except Exception:
        return 0.0


def fallback_route_inference(query_lower: str) -> str:
    dynamic_keywords = ['proposal', 'referendum', 'referenda', 'bounty', 'treasury', 'voter', 'vote', 'show me', 'list', 'find', 'get', 'count', 'how many', 'specific', 'address']
    static_keywords = ['how to', 'how can i', 'what is', 'how does', 'explain', 'tutorial', 'guide', 'delegate', 'delegation', 'concept', 'definition', 'identity', 'verified', 'judgement', 'seems like', 'i have no', 'why don\'t i', 'why can\'t i', 'it seems', 'i don\'t see']
    is_person_query = query_lower.startswith('who is ') and len(query_lower.split()) <= 4
    governance_who_is = any(term in query_lower for term in ['delegate', 'curator', 'proposer', 'beneficiary', 'ambassador'])
    if any(keyword in query_lower for keyword in dynamic_keywords):
        return "dynamic"
    if is_person_query and not governance_who_is:
        return "generic"
    if any(phrase in query_lower for phrase in static_keywords) or (query_lower.startswith('who is ') and governance_who_is):
        return "static"
    if any(word in query_lower for word in ['hi', 'hello', 'hey', 'greetings']):
        return "generic"
    return "static"


async def route_query_llm(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    qa_generator
) -> Dict[str, Any]:
    """
    Route query using LLM to determine the appropriate route.
    
    Returns:
        {
            "route": "static" | "dynamic" | "hybrid" | "generic",
            "confidence": float (0.0-1.0)
        }
    """
    log_step("router_llm_start", {"query_preview": query[:100]})
    
    try:
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
                        context_parts.append(f"{role_display}: {content[:100]}")
            if context_parts:
                conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand what the user is referring to in their query."
        
        from ...prompts.routing_prompt import PROMPT_TEMPLATE as routing_prompt_template
        routing_prompt = routing_prompt_template.format(
            query=query,
            conversation_context=conversation_context
        )
        
        if getattr(qa_generator, "client", None):
            try:
                router_model = os.getenv("ROUTER_MODEL", "gpt-4.1")
                response = qa_generator.client.chat.completions.create(
                    model=router_model,
                    messages=[{"role": "user", "content": routing_prompt}],
                    temperature=0.0,
                    max_tokens=4,
                    logprobs=True,
                    top_logprobs=5
                )
                choice = response.choices[0]
                route_text = (choice.message.content or "").strip().lower()
                if route_text.endswith('.'):
                    route_text = route_text[:-1]
                allowed_routes = ['static', 'dynamic', 'hybrid', 'generic']
                if route_text not in allowed_routes:
                    route_text = None
                probability = get_router_confidence_from_logprobs(choice)
                certainty = (
                    "high" if probability >= 0.85 else
                    "medium" if probability >= 0.60 else
                    "low"
                )
                log_step("router_llm_complete", {
                    "route": route_text,
                    "probability": probability,
                    "certainty": certainty,
                    "model": router_model
                })
                if route_text and probability >= 0.60:
                    return {
                        "route": route_text,
                        "confidence": probability
                    }
                fallback_route = fallback_route_inference(query.lower())
                log_step("router_llm_low_confidence_fallback", {
                    "fallback_route": fallback_route,
                    "probability": probability
                }, "warning")
                return {
                    "route": fallback_route,
                    "confidence": probability
                }
            except Exception as e:
                if is_insufficient_quota_error(e):
                    log_step("router_llm_error", {"error": str(e), "quota_error": True}, "error")
                    raise
                log_step("router_llm_error", {"error": str(e)}, "error")
        else:
            log_step("router_llm_fallback", {"reason": "no_openai_client"}, "warning")
        
        fallback_route = fallback_route_inference(query.lower())
        log_step("router_llm_parse_error", {
            "route": fallback_route,
            "reason": "llm_response_unavailable"
        }, "error")
        return {
            "route": fallback_route,
            "confidence": 0.5
        }
            
    except Exception as e:
        if is_insufficient_quota_error(e):
            log_step("router_llm_error", {"error": str(e), "quota_error": True}, "error")
            raise
        log_step("router_llm_error", {"error": str(e)}, "error")
        return {
            "route": "static",
            "confidence": 0.3
        }

