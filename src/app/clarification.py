"""
Clarification handler for low-confidence queries.
"""

from typing import Dict, Any, Optional, List
import logging
from src.core.errors import is_insufficient_quota_error, get_quota_error_message

logger = logging.getLogger(__name__)


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
        route: The route that was selected
        router_confidence: The router's confidence level
        qa_generator: QA generator instance
        log_step: Logging function
    
    Returns:
        Dictionary with clarification question and metadata
    """
    log_step("clarification_start", {
        "query_preview": query[:100],
        "route": route or "undetermined",
        "router_confidence": router_confidence
    })
    
    # Build context-aware clarification prompt based on route
    route_context = ""
    normalized_route = (route or "undetermined").lower()
    
    if normalized_route == "dynamic":
        route_context = """
This is a dynamic/on-chain data query. Common ambiguities include:
- Network selection (Polkadot vs Kusama)
- Proposal/referendum type or status
- Time period or date range
- Specific filters (active, passed, rejected, etc.)

For queries about proposals, referenda, votes, or treasury data, the most common ambiguity is which network (Polkadot or Kusama).
"""
    elif normalized_route == "static":
        route_context = """
This is a static/educational query. Common ambiguities include:
- Specific topic or concept within the broader subject
- Level of detail needed (overview vs deep dive)
- Specific use case or scenario
- Unclear terminology or acronyms
"""
    elif normalized_route == "hybrid":
        route_context = """
This is a hybrid query needing both explanation and data. Common ambiguities include:
- Network selection (Polkadot or Kusama) for the data portion
- Scope of explanation vs data requested
"""
    else:
        route_context = """
The route for this query has not been determined yet. Focus on clarifying:
- Whether the user is referring to a specific proposal/referendum/bounty or asking generally
- Any missing identifiers (ID numbers, links, titles)
- The exact topic or scope they care about if they’re being vague
"""
    
    # Build conversation context if available
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent_messages = conversation_history[-6:]  # Last 6 messages for context
        context_parts = []
        for msg in recent_messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '')
                if content and len(content) > 5:
                    role_display = role if role else 'user'
                    context_parts.append(f"{role_display}: {content[:200]}")
        if context_parts:
            conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand what the user is referring to in their query."
    
    # Use LLM to dynamically generate context-aware clarification
    clarification_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

The user asked: "{query}"

This query was routed to the "{normalized_route}" category. The query is ambiguous and needs clarification.

{route_context}{conversation_context}

CRITICAL INSTRUCTIONS:
- Analyze the query type FIRST:
  * If it's asking "what is X" or "explain X" or defining a term/concept → Ask what they mean by that term (e.g., "Can you explain what you mean by that?" or "What specifically are you referring to?")
  * If it's asking for data/listings (show, list, find, get) → Ask which network (Polkadot or Kusama) if not specified
  * If it's asking about votes without ID → Ask which specific proposal/referendum
- You MUST ask ONE specific clarifying question based on the EXACT terms used in the user's query
- DO NOT default to network questions for concept/definition queries
- Be DIRECT and SPECIFIC - use the SAME terminology the user used
- Match the query's language and terminology exactly
- Be natural and conversational

EXAMPLES:
- Query: "what is pop" → Response: "Can you explain what you mean by that? Are you referring to a specific term or concept?"
- Query: "what is XCM" → Response: "Can you clarify what you're asking about? Are you looking for an explanation of XCM?"
- Query: "show me proposals" → Response: "Are you looking for proposals on Polkadot or Kusama network?"
- Query: "show me active referenda" → Response: "Are you looking for referenda on Polkadot or Kusama network?"
- Query: "list referenda" → Response: "Are you looking for referenda on Polkadot or Kusama network?"
- Query: "what are the votes" → Response: "Which proposal or referendum are you asking about? Please provide the ID or title."
- Query: "treasury data" → Response: "Are you asking about Polkadot or Kusama treasury proposals?"
- Query: "show me bounties" → Response: "Are you looking for bounties on Polkadot or Kusama network?"
- Query: "explain governance" → Response: "What specific aspect of governance would you like me to explain?"

Now, for the query "{query}", respond with ONLY the clarifying question (no explanations, no extra text). Use the exact same terminology the user used:
"""
    
    try:
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("clarification_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(clarification_prompt)
            clarification_question = response.strip()
            
            # Clean up the response - remove quotes if LLM wrapped it
            if clarification_question.startswith('"') and clarification_question.endswith('"'):
                clarification_question = clarification_question[1:-1]
            elif clarification_question.startswith("'") and clarification_question.endswith("'"):
                clarification_question = clarification_question[1:-1]
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
            clarification_question = response.choices[0].message.content.strip()
        else:
            raise Exception("No LLM client available")
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
        # Minimal fallback - still try to be specific
        if normalized_route == "dynamic":
            clarification_question = "Are you looking for this information on Polkadot or Kusama network?"
        else:
            clarification_question = "Could you please provide more details about what you're looking for?"
    
    # No need for markers - conversation history pattern is sufficient
    # The detection function will identify clarifications by checking if last assistant message is a question
    clarification_question_with_marker = clarification_question
    
    log_step("clarification_complete", {
        "question": clarification_question[:100],
        "method": "llm_generated"
    })
    
    model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini') if qa_generator.gemini_client else (qa_generator.model if hasattr(qa_generator, 'model') else 'fallback')
    
    return {
        'answer': clarification_question_with_marker,
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

