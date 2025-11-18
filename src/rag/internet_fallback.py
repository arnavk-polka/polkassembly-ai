"""
Internet search fallback for low-confidence queries with no data.
"""

from typing import Dict, Any, Optional, List
import logging
import os

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
            conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nIMPORTANT: Use the conversation history to understand what the user is really asking about. If the conversation mentions specific topics, proposals, referenda, tracks, or networks, include those in the search query. For example, if the conversation was about 'vitro connect referenda' and the current query is 'Yes it is a kusama referenda', the search should include 'vitro connect' and 'kusama'."
    
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
        log_step("contextual_query_llm_error", {"error": str(e)}, "error")
        return _build_contextual_search_query_heuristic(query, route)


async def generate_internet_search_response(
    query: str,
    qa_generator,
    log_step,
    route: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate response using internet search when no data is available.
    
    Args:
        query: The user's query
        qa_generator: QA generator instance
        log_step: Logging function
        route: The route category (dynamic, static, etc.)
        conversation_history: Optional conversation history for context
    
    Returns:
        Dictionary with answer from internet search and metadata
    """
    contextual_query = await _build_contextual_search_query(query, route, qa_generator, log_step, conversation_history)
    
    log_step("internet_fallback_start", {
        "query_preview": query[:100],
        "route": route,
        "contextual_query": contextual_query[:150]
    })
    
    try:
        from ..utils.web_search import search_tavily
        
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
        
        internet_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

A user has asked: "{query}"{conversation_context_for_answer}

I have no context or data about this query in my system. Please search the internet and provide the best answer you can find.

Important guidelines:
- Provide a helpful, accurate answer based on internet search results
- Focus on Polkadot/Kusama/blockchain governance topics if relevant
- Keep the response concise and informative
- If conversation history is provided, use it to understand the full context of what the user is asking about
- If the query is not related to Polkadot/Kusama, still provide a helpful answer
- Use the search results to inform your response
- DO NOT start with greetings like "Hello" or "As Klara" - just provide the answer directly

Search the internet and provide the best answer you can:
"""
        
        web_search_enabled = os.getenv("WEB_SEARCH", "true").lower() == "true"
        
        if web_search_enabled:
            try:
                log_step("internet_search_tavily", {
                    "contextual_query": contextual_query[:200]
                })
                answer, sources = await search_tavily(contextual_query, min_score=0.5)
                
                if answer:
                    log_step("internet_search_success", {
                        "answer_length": len(answer),
                        "sources_count": len(sources)
                    })
                    
                    formatted_answer = answer.strip()
                    if sources:
                        formatted_answer += "\n\nSources:\n" + "\n".join(
                            f"- {src.get('title', src.get('url', 'Source'))} ({src.get('url', '')})"
                            for src in sources
                        )
                    
                    follow_up_questions = [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ]
                    
                    return {
                        'answer': formatted_answer,
                        'sources': sources,
                        'confidence': 0.6,
                        'follow_up_questions': follow_up_questions,
                        'context_used': False,
                        'model_used': 'tavily_web_search',
                        'chunks_used': 0,
                        'search_method': 'internet_search',
                        'internet_fallback': True
                    }
                else:
                    log_step("internet_search_no_results", {})
            except Exception as e:
                log_step("internet_search_error", {"error": str(e)}, "error")
                logger.error(f"Tavily search failed: {e}")
        
        # Fallback to LLM without web search
        log_step("internet_fallback_llm", {})
        
        llm_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

A user has asked: "{query}"

I have no context or data about this query in my system. Based on your general knowledge, provide the best answer you can.

Important guidelines:
- Provide a helpful, accurate answer based on your knowledge
- Focus on Polkadot/Kusama/blockchain governance topics if relevant
- Keep the response concise and informative
- If you don't know the answer, politely explain that you don't have specific information
- DO NOT start with greetings like "Hello" or "As Klara" - just provide the answer directly

Provide the best answer you can:
"""
        
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("internet_fallback_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(llm_prompt)
            
            formatted_answer = response.strip()
            
            log_step("internet_fallback_complete", {
                "response_length": len(formatted_answer)
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
                'context_used': False,
                'model_used': model_name,
                'chunks_used': 0,
                'search_method': 'internet_fallback_llm',
                'internet_fallback': True
            }
        else:
            if hasattr(qa_generator, 'client'):
                system_prompt = """You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly. 
You help users with questions even when you don't have specific data in your system. 
DO NOT start responses with greetings like "Hello" or "As Klara" - just provide the answer directly."""
                
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
                
                formatted_answer = answer.strip()
                
                return {
                    'answer': formatted_answer,
                    'sources': [],
                    'confidence': 0.5,
                    'follow_up_questions': [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ],
                    'context_used': False,
                    'model_used': qa_generator.model,
                    'chunks_used': 0,
                    'search_method': 'internet_fallback_openai',
                    'internet_fallback': True
                }
        
        fallback_answer = "We do not have direct data for this in our system. I apologize, but I'm unable to provide an answer at this time. Please try rephrasing your question or ask about Polkadot/Kusama governance topics."
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
        log_step("internet_fallback_error", {"error": str(e)}, "error")
        logger.error(f"Internet fallback error: {e}")
        
        fallback_answer = "We do not have direct data for this in our system. I apologize, but I encountered an error while searching for information. Please try again later."
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

