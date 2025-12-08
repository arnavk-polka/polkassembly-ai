"""
Internet search fallback for low-confidence queries with no data.
"""

from typing import Dict, Any, Optional, List
import logging
import os
import requests
from src.utils.error_handler import is_insufficient_quota_error, get_quota_error_message

logger = logging.getLogger(__name__)


def _build_contextual_search_query_heuristic(query: str, route: Optional[str]) -> str:
    """Fallback heuristic for contextual search query generation."""
    base = (query or "").strip()
    lowered = base.lower()
    keywords: List[str] = []
    
    route = (route or "").lower()
    if route == "dynamic":
        keywords.extend([
            "Polkadot OpenGov on-chain data",
            "Polkassembly referenda",
            "Kusama governance"
        ])
    elif route == "static":
        keywords.extend([
            "Polkadot governance documentation",
            "Polkassembly guide"
        ])
    else:
        keywords.extend([
            "Polkadot",
            "Kusama",
            "Polkassembly"
        ])
    
    if any(term in lowered for term in ["referenda", "referendum", "proposal", "bounty", "treasury", "track", "motion", "tip"]):
        keywords.append("OpenGov")
    if "vote" in lowered or "voting" in lowered or "delegate" in lowered:
        keywords.append("governance votes")
    if "kusama" in lowered:
        keywords.append("Kusama network")
    if "polkadot" in lowered:
        keywords.append("Polkadot network")
    
    # Deduplicate while preserving order
    seen = set()
    context_tokens = []
    for word in keywords:
        if word and word not in seen:
            seen.add(word)
            context_tokens.append(word)
    
    contextual_suffix = " ".join(context_tokens)
    if contextual_suffix:
        return f"{base} {contextual_suffix}".strip()
    return base


async def _build_contextual_search_query(query: str, route: Optional[str], qa_generator, log_step, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Use GPT-3.5 Turbo to enrich the search query with Polkassembly/OpenGov context.
    Falls back to heuristic builder if the LLM call fails or client unavailable.
    """
    if not qa_generator or not getattr(qa_generator, "client", None):
        log_step("contextual_query_llm_skipped", {"reason": "no_openai_client"}, "warning")
        return _build_contextual_search_query_heuristic(query, route)
    
    # Build conversation context if available
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent_messages = conversation_history[-6:]  # Last 6 messages for context
        context_parts = []
        for msg in recent_messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '') or msg.get('message', '')
                if content and len(str(content).strip()) > 5:
                    role_display = role if role else 'user'
                    content_str = str(content)[:300]  # Limit to 300 chars per message
                    context_parts.append(f"{role_display}: {content_str}")
        if context_parts:
            conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand what the user is really asking about."
    
    system_prompt = (
        "You rewrite user queries into precise web-search strings focused on Polkadot/Kusama governance. "
        "Return ONLY the rewritten query without commentary."
    )
    user_prompt = f"""
Original user query: "{query}"
Route category: "{route or 'unknown'}"{conversation_context}

Task:
- Add context so the search targets Polkassembly, Polkadot/Kusama OpenGov, referendum/voting/bounty data, etc.
- Include network names (Polkadot, Kusama) or terms like "Polkassembly", "OpenGov", "referendum" when relevant.
- If conversation history is provided, use it to understand the full context of what the user is asking about.
- Include specific topics, proposal names, track names, or other details mentioned in the conversation history.
- If the user already mentions unrelated topics, keep them but still bias towards blockchain governance sources.
- Output ONE enhanced search query string. No explanations, no quotes.
"""
    try:
        response = qa_generator.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=50
        )
        enhanced_query = (response.choices[0].message.content or "").strip()
        if not enhanced_query:
            raise ValueError("Empty response from GPT-3.5 for contextual query")
        log_step("contextual_query_llm_success", {"query": enhanced_query[:150]})
        return enhanced_query
    except Exception as e:
        if is_insufficient_quota_error(e):
            log_step("contextual_query_llm_error", {"error": str(e), "quota_error": True}, "error")
            raise
        log_step("contextual_query_llm_error", {"error": str(e)}, "error")
        return _build_contextual_search_query_heuristic(query, route)


async def generate_internet_search_response(
    query: str,
    qa_generator,
    log_step,
    route: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    sql_query: Optional[str] = None,
    validator_reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate response using LLM when no data is available.
    
    Args:
        query: The user's query
        qa_generator: QA generator instance
        log_step: Logging function
        route: The route category (dynamic, static, etc.)
        conversation_history: Optional conversation history for context
    
    Returns:
        Dictionary with answer from LLM and metadata
    """
    log_step("internet_fallback_start", {
        "query_preview": query[:100],
        "route": route
    })
    
    try:
        # Build conversation context for the answer generation
        conversation_context_for_answer = ""
        if conversation_history and len(conversation_history) > 0:
            recent_messages = conversation_history[-6:]  # Last 6 messages for context
            context_parts = []
            for msg in recent_messages:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '') or msg.get('message', '')
                    if content and len(str(content).strip()) > 5:
                        role_display = role if role else 'user'
                        content_str = str(content)[:300]  # Limit to 300 chars per message
                        context_parts.append(f"{role_display}: {content_str}")
            if context_parts:
                conversation_context_for_answer = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse this conversation history to understand the full context of what the user is asking about."
        
        # Do NOT surface SQL to the model to avoid leaking internals to users
        sql_context = ""
        
        validator_context = ""
        if validator_reason:
            validator_context = f"\n\nVALIDATOR NOTE:\n{validator_reason}\n\nThis explains why the SQL query didn't return results."
        
        # Optional Tavily grounding for real snippets
        tavily_context = ""
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                log_step("internet_fallback_tavily_attempt", {"query_preview": query[:80]})
                url = "https://api.tavily.com/search"
                body = {
                    "api_key": tavily_api_key,
                    "query": query,
                    "max_results": 5,
                    "search_depth": "basic",
                }
                resp = requests.post(url, json=body, timeout=12)
                if resp.ok:
                    data = resp.json()
                    results = data.get("results", []) or []
                    snippets: List[str] = []
                    for hit in results[:5]:
                        title = hit.get("title") or hit.get("url")
                        snippet = hit.get("content") or hit.get("snippet") or hit.get("description")
                        if title or snippet:
                            line = f"- {title or ''}: {snippet or ''}".strip(": ").strip()
                            if line:
                                snippets.append(line)
                    if snippets:
                        tavily_context = "\n\nWEB CONTEXT (ONLY USE THESE FACTS, DO NOT INVENT):\n" + "\n".join(snippets)
                        log_step("internet_fallback_tavily_hits", {"hits": len(snippets)})
                    else:
                        log_step("internet_fallback_tavily_no_hits", {"hits": len(results)})
                else:
                    log_step("internet_fallback_tavily_error", {"status": resp.status_code, "text": resp.text[:200]}, "warning")
            except Exception as tavily_err:
                log_step("internet_fallback_tavily_exception", {"error": str(tavily_err)}, "warning")

        llm_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

 A user has asked: "{query}"{conversation_context_for_answer}{sql_context}{validator_context}{tavily_context}

Based on your knowledge about Polkadot, Kusama, blockchain governance, and related topics, provide a helpful answer that directly addresses the user's question.

CRITICAL WEB CONTEXT HANDLING (MUST FOLLOW):
- If WEB CONTEXT is provided but it is NOT relevant to Polkadot/Kusama governance (e.g., U.S. Treasury, unrelated topics, different blockchains), you MUST COMPLETELY IGNORE IT.
- DO NOT mention the web context, DO NOT reference it, DO NOT say "the provided web context discusses..." or similar phrases.
- DO NOT explain what the web context contains if it's irrelevant.
- Simply state: "I couldn't find any information on [topic] for [time period/query]." and STOP.
- Example of WRONG response: "I couldn't find any information... The provided web context discusses the U.S. Treasury..." (DO NOT DO THIS)
- Example of CORRECT response: "I couldn't find any information on treasury payouts on Polkadot or Kusama for September 2025." (STOP HERE, do not mention web context)
- Only use web context if it DIRECTLY relates to Polkadot/Kusama governance and answers the user's question.

Important guidelines:
- Directly answer the question: "{query}"
- Do NOT mention any SQL queries, attempted SQL, internal pipelines, or filters
- Focus on Polkadot/Kusama/blockchain governance topics if relevant
- Keep the response concise and informative
- If conversation history is provided, use it to understand the full context
- If you don't know the answer, acknowledge the question and explain what information would be needed
- DO NOT start with greetings like "Hello" or "As Klara" - just provide the answer directly
- DO NOT start with meta phrases like "Here's what I found" or similar; start directly with the information.
- Make sure your response clearly relates to the question asked
- CRITICAL: NEVER invent, speculate, or make up any facts. If you cannot find concrete data, say so plainly (e.g., "I couldn't find any on-chain data for that") and stop; do not fill in with guesses.
- CRITICAL: NEVER generate placeholder data, dummy data, example data, or fake data. Do NOT use placeholders like "[Proposal Hash 1]", "[Short Description]", "[Amount in DOT]", or any other bracketed placeholder text. Only provide real, factual information. If you don't have specific data to share, explain that you couldn't find the specific information requested rather than making up examples.
- CRITICAL DATE HANDLING: If the query mentions a date (e.g., "October 2025", "in 2025", "last month"), treat it as a FILTER requirement, not a validation check. The user is asking for data FROM that time period. Do NOT say the date is "in the future" or "not available" - instead, explain that no data was found matching those specific filters. Dates in queries are filters to apply to the data, not validation checks about whether the date is valid.

Provide your answer:
"""
        
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("internet_fallback_llm_call", {"model": model_name})
            
            try:
                response = qa_generator.gemini_client.get_response(llm_prompt)
                answer_text = response.strip()
                
                formatted_answer = answer_text
                
                log_step("internet_fallback_complete", {
                    "response_length": len(formatted_answer),
                    "model": model_name
                })
                
                return {
                    'answer': formatted_answer,
                    'sources': [],
                    'confidence': 0.5,
                    'follow_up_questions': [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ],
                    'context_used': bool(conversation_history),
                    'model_used': model_name,
                    'chunks_used': 0,
                    'search_method': 'internet_fallback_llm',
                    'internet_fallback': True
                }
            except Exception as gemini_error:
                log_step("internet_fallback_gemini_error", {"error": str(gemini_error)}, "error")
                logger.warning(f"Gemini fallback failed: {gemini_error}, falling back to OpenAI")
                # Fall through to OpenAI fallback
        
        if hasattr(qa_generator, 'client') and qa_generator.client:
            system_prompt = """You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly. 
You help users with questions about Polkadot and Kusama governance. 
Always reference the user's question in your response to show you understand what was asked.
DO NOT start responses with greetings like "Hello" or "As Klara" - just provide the answer directly.
CRITICAL: NEVER mention that you cannot access data, don't have access to data, cannot directly access data, or lack access to real-time data. This is a Polkassembly product with full access to Polkadot and Kusama governance data. Answer questions directly as if you have access to all relevant data.
CRITICAL: NEVER generate placeholder data, dummy data, example data, or fake data. Do NOT use placeholders like "[Proposal Hash 1]", "[Short Description]", "[Amount in DOT]", or any other bracketed placeholder text. Only provide real, factual information. If you don't have specific data to share, explain that you couldn't find the specific information requested rather than making up examples.
CRITICAL DATE HANDLING: If the query mentions a date (e.g., "October 2025", "in 2025", "last month"), treat it as a FILTER requirement, not a validation check. The user is asking for data FROM that time period. Do NOT say the date is "in the future" or "not available" - instead, explain that no data was found matching those specific filters. Dates in queries are filters to apply to the data, not validation checks about whether the date is valid.
CRITICAL WEB CONTEXT HANDLING (MUST FOLLOW): If WEB CONTEXT is provided but it is NOT relevant to Polkadot/Kusama governance (e.g., U.S. Treasury, unrelated topics, different blockchains), you MUST COMPLETELY IGNORE IT. DO NOT mention the web context, DO NOT reference it, DO NOT say "the provided web context discusses..." or similar phrases. DO NOT explain what the web context contains if it's irrelevant. Simply state: "I couldn't find any information on [topic] for [time period/query]." and STOP. Example of WRONG response: "I couldn't find any information... The provided web context discusses the U.S. Treasury..." (DO NOT DO THIS). Example of CORRECT response: "I couldn't find any information on treasury payouts on Polkadot or Kusama for September 2025." (STOP HERE, do not mention web context). Only use web context if it DIRECTLY relates to Polkadot/Kusama governance and answers the user's question."""
            
            log_step("internet_fallback_llm_call", {"model": qa_generator.model})
            
            try:
                response = qa_generator.client.chat.completions.create(
                    model=qa_generator.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
                answer_text = answer.strip()
                if tavily_context:
                    formatted_answer = "No on-chain data matched; here is what I found via internet search:\n" + answer_text
                else:
                    formatted_answer = answer_text
                
                log_step("internet_fallback_complete", {
                    "response_length": len(formatted_answer),
                    "model": qa_generator.model
                })
                
                return {
                    'answer': formatted_answer,
                    'sources': [],
                    'confidence': 0.5,
                    'follow_up_questions': [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ],
                    'context_used': bool(conversation_history),
                    'model_used': qa_generator.model,
                    'chunks_used': 0,
                    'search_method': 'internet_fallback_openai',
                    'internet_fallback': True
                }
            except Exception as openai_error:
                if is_insufficient_quota_error(openai_error):
                    log_step("internet_fallback_openai_error", {"error": str(openai_error), "quota_error": True}, "error")
                    return {
                        'answer': get_quota_error_message(),
                        'sources': [],
                        'confidence': 0.0,
                        'follow_up_questions': [],
                        'context_used': bool(conversation_history),
                        'model_used': 'error',
                        'chunks_used': 0,
                        'search_method': 'quota_error',
                        'internet_fallback': True
                    }
                log_step("internet_fallback_openai_error", {"error": str(openai_error)}, "error")
                raise
        
        fallback_answer = f"I'm unable to provide an answer for your question: \"{query}\" at this time. Please try rephrasing your question or ask about Polkadot/Kusama governance topics."
        return {
            'answer': fallback_answer,
            'sources': [],
            'confidence': 0.3,
            'follow_up_questions': [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ],
            'context_used': False,
            'model_used': 'fallback',
            'chunks_used': 0,
            'search_method': 'internet_fallback_error',
            'internet_fallback': True
        }
        
    except Exception as e:
        if is_insufficient_quota_error(e):
            log_step("internet_fallback_error", {"error": str(e), "quota_error": True}, "error")
            logger.error(f"Internet fallback quota error: {e}")
            return {
                'answer': get_quota_error_message(),
                'sources': [],
                'confidence': 0.0,
                'follow_up_questions': [],
                'context_used': False,
                'model_used': 'error',
                'chunks_used': 0,
                'search_method': 'quota_error',
                'internet_fallback': True
            }
        log_step("internet_fallback_error", {"error": str(e)}, "error")
        logger.error(f"Internet fallback error: {e}")
        
        fallback_answer = f"I encountered an error while trying to answer your question: \"{query}\". Please try again later or rephrase your question."
        return {
            'answer': fallback_answer,
            'sources': [],
            'confidence': 0.3,
            'follow_up_questions': [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ],
            'context_used': False,
            'model_used': 'error_fallback',
            'chunks_used': 0,
            'search_method': 'internet_fallback_error',
            'internet_fallback': True
        }

