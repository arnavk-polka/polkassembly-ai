"""
Query processing pipeline for the Polkadot AI Chatbot.
Implements the new routing-first architecture with structured logging.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from .confidence import compute_retrieval_confidence
from .greeting_handler import handle_generic_query_llm
from .clarification import generate_clarification_question
from .internet_fallback import generate_internet_search_response

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
    
    # Build conversation context if available
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent_messages = conversation_history[-6:]  # Last 6 messages (3 conversation turns) for context
        context_parts = []
        for msg in recent_messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '')
                if content and len(content) > 5:
                    role_display = role if role else 'user'
                    context_parts.append(f"{role_display}: {content[:150]}")
        if context_parts:
            conversation_context = f"\n\nRecent conversation context (to understand what the original query refers to):\n" + "\n".join(context_parts)
    
    combination_prompt = f"""
You are helping to combine a user's original query with their clarification response.

Original query: "{original_query}"

Clarification question that was asked: "{clarification_question}"

User's clarification response: "{clarification_response}"{conversation_context}

Your task:
- Understand what the user's clarification response means in the context of the clarification question
- Use the conversation context to understand what the original query refers to (e.g., if original query is "what about october", use context to understand it's about "unique voters")
- Create a single, clear, and coherent query that combines both the original intent and the clarification
- Make it natural and well-formed
- Preserve the original intent while incorporating the clarification details
- Do NOT add any explanations or meta-commentary - just output the combined query

Examples:
Original: "show me proposals"
Clarification question: "Are you looking for information on the Polkadot or Kusama network?"
Clarification response: "Polkadot"
Combined: "show me proposals on Polkadot network"

Original: "how many unique voters were there in november 2025"
Clarification question: "Are you looking for information on the Polkadot or Kusama network?"
Clarification response: "both"
Combined: "how many unique voters were there in november 2025 on both Polkadot and Kusama networks"

Original: "what about october"
Context: Previous conversation was about "How many unique voters were there in November 2025?"
Clarification question: "Are you asking about Polkadot or Kusama network in October?"
Clarification response: "Kusama"
Combined: "How many unique voters were there in October 2025 on the Kusama network?"

Now create the combined query:
"""
    
    try:
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("query_combination_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(combination_prompt)
            combined_query = response.strip()
            
            # Remove any quotes if the LLM wrapped the response
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
                
                # Remove any quotes if the LLM wrapped the response
                if combined_query.startswith('"') and combined_query.endswith('"'):
                    combined_query = combined_query[1:-1]
                elif combined_query.startswith("'") and combined_query.endswith("'"):
                    combined_query = combined_query[1:-1]
                
                return combined_query
        
        # Fallback to simple concatenation if LLM is not available
        log_step("query_combination_fallback", {"note": "Using simple concatenation fallback"})
        return f"{original_query} {clarification_response}".strip()
        
    except Exception as e:
        log_step("query_combination_error", {"error": str(e)}, "error")
        # Fallback to simple concatenation on error
        return f"{original_query} {clarification_response}".strip()

def log_step(step_name: str, data: Dict[str, Any], level: str = "info"):
    """Log a pipeline step with structured data"""
    log_data = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        **data
    }
    if level == "info":
        logger.info(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "warning":
        logger.warning(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "error":
        logger.error(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "debug":
        logger.debug(f"[{step_name}] {json.dumps(log_data, default=str)}")


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
    
    # Get the last assistant message (should be the clarification question)
    last_entry = conversationHistory[-1]
    if isinstance(last_entry, dict):
        last_role = last_entry.get('role', '')
        last_content = last_entry.get('content', '') or last_entry.get('response', '')
    elif hasattr(last_entry, 'role'):
        last_role = last_entry.role
        last_content = getattr(last_entry, 'content', None) or getattr(last_entry, 'response', '')
    else:
        log_step("clarification_check_invalid_format", {"entry_type": str(type(last_entry))}, "debug")
        return None
    
    # Check if last message is an assistant question (clarification)
    if last_role != 'assistant' or not last_content or not last_content.strip().endswith('?'):
        log_step("clarification_check_not_question", {
            "last_role": last_role,
            "is_question": last_content.strip().endswith('?') if last_content else False
        }, "debug")
        return None
    
    # Check if user message is a short clarification response (not a full new question)
    # Full questions typically have: question words (how, what, when, etc.), multiple words, or are complete sentences
    user_message_lower = userMessage.lower().strip()
    word_count = len(user_message_lower.split())
    question_words = ['how', 'what', 'when', 'where', 'who', 'why', 'which', 'can', 'could', 'should', 'would', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'will', 'show', 'tell', 'explain', 'list', 'find']
    is_question_word = any(user_message_lower.startswith(qw + ' ') or user_message_lower.startswith(qw + '?') for qw in question_words)
    has_question_mark = '?' in userMessage
    
    # If it looks like a full question (has question words, many words, or question mark), it's NOT a clarification
    if is_question_word or word_count > 5 or has_question_mark:
        log_step("clarification_check_full_question", {
            "user_message": userMessage[:100],
            "word_count": word_count,
            "is_question_word": is_question_word,
            "has_question_mark": has_question_mark,
            "note": "User message looks like a full question, not a clarification response"
        }, "debug")
        return None
    
    # Find the user query that came before this clarification question
    # Look backwards through history: assistant question -> user query (the one we need)
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
    
    # Re-route the original query to determine route and confidence
    route_result = await route_query_llm(
        original_query,
        conversationHistory[:-1] if conversationHistory else None,  # Exclude the clarification question
        qa_generator
    )
    original_route = route_result.get("route", "dynamic")
    original_router_confidence = route_result.get("confidence", 0.8)
    
    # Use LLM to intelligently combine original query with clarification response
    # Pass the clarification question and conversation history so the LLM understands context
    combined_query = await combine_query_with_clarification(
        original_query,
        userMessage,
        last_content,  # The clarification question
        qa_generator,
        log_step,
        conversationHistory  # Pass full conversation history for context
    )
    
    return {
        'combined_query': combined_query,
        'original_query': original_query,
        'original_route': original_route,
        'original_router_confidence': original_router_confidence,
        'clarification_response': userMessage
    }


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
        # Build conversation context for routing
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_messages = conversation_history[-6:]  # Last 6 messages (3 conversation turns) for context
            context_parts = []
            for msg in recent_messages:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '')
                    if content and len(content) > 5:
                        role_display = role if role else 'user'
                        context_parts.append(f"{role_display}: {content[:100]}")
            if context_parts:
                conversation_context = f"\n\nRecent conversation context:\n" + "\n".join(context_parts)
        
        routing_prompt = f"""
You are a query router. Analyze this user query and determine the best route for answering it.

Query: "{query}"{conversation_context}

Available Routes:
1. "static" - For procedural, educational, or informational questions:
   - Questions about what you CAN or CANNOT do (e.g., "can I cancel", "is it possible to", "can I still")
   - Questions about HOW to do something (e.g., "how to cancel", "how do I", "how can I")
   - Governance/OpenGov concepts and explanations
   - Ambassador Programme information
   - Parachains & AnV explanations
   - Hyperbridge, JAM definitions
   - Dashboard information
   - Wiki pages, how-to guides, tutorials
   - Governance-related "what is", "who is", "how does" questions (e.g., "Who is the most trustable delegate", "What is OpenGov", "What is a delegate")
   - Questions asking for explanations, definitions, or conceptual information about governance/blockchain
   - "How to" questions about using Polkassembly features
   - Questions about processes, rules, or procedures
   - Questions about delegates, delegation concepts, or how delegation works

2. "dynamic" - For queries requesting specific on-chain DATA:
   - "Show me", "list", "find", "get" queries for proposals/referenda/bounties
   - Questions asking for specific proposal information (title, content, status, dates, network)
   - Proposal metadata (type, proposer, beneficiary, amounts, curator)
   - Questions about specific proposal IDs (e.g., "Who is the curator of 1671", "What is the status of proposal 123")
   - Questions about blockchain addresses (e.g., "Who is 0x163830...", "What proposals did [address] make", "Show me proposals by [address]")
   - Voting data (voter information, voting power, decisions)
   - Proposal filtering by ID, dates, network, type, status
   - Aggregations, counts, or summaries of on-chain data
   - Questions asking to RETRIEVE or DISPLAY specific data from the blockchain
   - Questions asking for specific delegate addresses, vote counts, or on-chain delegate metrics
   - URLs to pages (e.g., "http://polkadot.polkassembly.io/referenda/1781" = very specific query for referenda 1781 on Polkadot)
     * URLs contain specific proposal/referenda IDs and network information - these are HIGHLY SPECIFIC queries
     * Extract the referenda/proposal ID and network from the URL (polkadot.polkassembly.io = Polkadot, kusama.polkassembly.io = Kusama)

3. "hybrid" - For queries that need both static context and dynamic data:
   - Questions that require explaining concepts AND showing specific data
   - Example: "What is OpenGov and show me recent proposals"

4. "generic" - For queries that don't fit the above categories:
   - Greetings (hi, hello, hey, greetings, etc.)
   - Casual conversation and small talk
   - Questions completely outside Polkadot/blockchain domain
   - Requests for general help or introduction
   - Ambiguous or unclear queries that can't be categorized
   - General knowledge questions about people (e.g., "Who is Gavin Wood", "Who is Satoshi Nakamoto")
   - Questions about individuals that require web search or general knowledge

CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no text before or after. Just the JSON object.

Example responses:
{{"route": "static", "confidence": 0.9}}  // e.g., "Who is the most trustable delegate", "What is OpenGov"
{{"route": "dynamic", "confidence": 0.95}}  // e.g., "Show me proposals from last month", "Who is the curator of 1671", "Who is 0x163830...", "http://polkadot.polkassembly.io/referenda/1781"
{{"route": "generic", "confidence": 0.7}}  // e.g., "Hello", "Hi there", "Who is Gavin Wood"

Now respond with ONLY the JSON for this query:
"""
        
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("router_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(routing_prompt)
            
            result = None
            
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                if not result:
                    json_match = re.search(r'\{[^{}]*"route"[^{}]*"confidence"[^{}]*\}', response)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass
                
                if not result:
                    route_match = re.search(r'"route"\s*:\s*"([^"]+)"', response, re.IGNORECASE)
                    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', response, re.IGNORECASE)
                    if route_match:
                        route_str = route_match.group(1).lower()
                        confidence_val = float(confidence_match.group(1)) if confidence_match else 0.5
                        if route_str in ['static', 'dynamic', 'hybrid', 'generic']:
                            result = {"route": route_str, "confidence": confidence_val}
            
            if result:
                route = result.get('route', 'static').lower()
                confidence = float(result.get('confidence', 0.5))
                
                if route not in ['static', 'dynamic', 'hybrid', 'generic']:
                    log_step("router_llm_invalid_route", {"route": route}, "warning")
                    route = 'static'
                    confidence = 0.5
                
                confidence = max(0.0, min(1.0, confidence))
                
                log_step("router_llm_complete", {
                    "route": route,
                    "confidence": confidence
                })
                
                return {
                    "route": route,
                    "confidence": confidence
                }
            else:
                log_step("router_llm_parse_error", {
                    "error": "Could not extract JSON from response",
                    "response_preview": response[:200]
                }, "error")
                
                query_lower = query.lower()
                
                dynamic_keywords = ['proposal', 'referendum', 'bounty', 'treasury', 'voter', 'vote', 'show me', 'list', 'find', 'get', 'count', 'how many', 'specific', 'address']
                static_keywords = ['how to', 'how can i', 'what is', 'how does', 'explain', 'tutorial', 'guide', 'delegate', 'delegation', 'concept', 'definition']
                # Check for "who is" - if it's a person name (not governance concept), route to generic
                is_person_query = query_lower.startswith('who is ') and len(query.split()) <= 4  # Simple heuristic: "who is [name]" is likely a person
                governance_who_is = any(term in query_lower for term in ['delegate', 'curator', 'proposer', 'beneficiary', 'ambassador'])
                
                if any(keyword in query_lower for keyword in dynamic_keywords):
                    route = "dynamic"
                elif is_person_query and not governance_who_is:
                    route = "generic"  # "Who is [person name]" -> generic for web search
                elif any(phrase in query_lower for phrase in static_keywords) or (query_lower.startswith('who is ') and governance_who_is):
                    route = "static"
                elif any(word in query_lower for word in ['hi', 'hello', 'hey', 'greetings']):
                    route = "generic"
                else:
                    route = "static"
                
                log_step("router_llm_fallback_inference", {
                    "route": route,
                    "reason": "json_parse_failed_using_query_analysis"
                }, "warning")
                
                return {
                    "route": route,
                    "confidence": 0.5
                }
        else:
            log_step("router_llm_fallback", {"reason": "no_gemini_client"}, "warning")
            return {
                "route": "static",
                "confidence": 0.5
            }
            
    except Exception as e:
        log_step("router_llm_error", {"error": str(e)}, "error")
        return {
            "route": "static",
            "confidence": 0.3
        }




async def processUserQuery(
    userMessage: str,
    conversationHistory: Optional[List[Dict[str, Any]]],
    static_embedding_manager,
    dynamic_embedding_manager,
    qa_generator,
    max_chunks: int = 5,
    custom_prompt: Optional[str] = None,
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    Main entry point for processing user queries.
    Implements the new routing-first architecture.
    
    Args:
        userMessage: The user's query
        conversationHistory: Previous conversation messages
        static_embedding_manager: Manager for static embeddings
        dynamic_embedding_manager: Manager for dynamic embeddings
        qa_generator: QA generator instance
        max_chunks: Maximum number of chunks to retrieve
        custom_prompt: Optional custom system prompt
        user_id: User identifier
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    pipeline_start = datetime.now()
    log_step("pipeline_start", {
        "user_id": user_id,
        "query_preview": userMessage[:100],
        "has_history": bool(conversationHistory)
    })
    
    try:
        # Check if this is a clarification response
        clarification_info = await detect_and_handle_clarification_response(
            userMessage,
            conversationHistory,
            qa_generator,
            log_step
        )
        
        is_clarification_followup = clarification_info is not None
        original_query_for_route = userMessage
        combined_query = userMessage
        stored_route = None
        stored_router_confidence = None
        is_voting_data = False  # Track if query is for voting_data table
        
        if is_clarification_followup:
            log_step("clarification_followup_detected", {
                "original_query": clarification_info['original_query'],
                "original_route": clarification_info.get('original_route', 'unknown'),
                "original_router_confidence": clarification_info.get('original_router_confidence', 'unknown'),
                "clarification_response": clarification_info['clarification_response'],
                "combined_query_preview": clarification_info['combined_query'][:100]
            })
            # Use the stored original route and router confidence (don't re-route)
            stored_route = clarification_info.get('original_route')
            stored_router_confidence = clarification_info.get('original_router_confidence', 0.7)
            original_query_for_route = clarification_info['original_query']
            combined_query = clarification_info['combined_query']
        
        # Analyze query with memory FIRST, then route on the analyzed query
        # Use combined query if this is a clarification followup, otherwise use original
        if is_clarification_followup:
            analyzed_query = combined_query
        else:
            analyzed_query = userMessage
        
        if conversationHistory and qa_generator and not is_clarification_followup:
            log_step("query_analysis_start", {})
            try:
                # Use analyzed_query (which may be combined) for memory analysis
                analyzed_query = qa_generator.analyze_query_with_memory(
                    analyzed_query,
                    conversationHistory
                )
                log_step("query_analysis_complete", {
                    "original": userMessage[:100],
                    "analyzed": analyzed_query[:100],
                    "is_clarification_followup": is_clarification_followup
                })
            except Exception as e:
                log_step("query_analysis_error", {"error": str(e)}, "error")
                # Keep the analyzed_query (combined or original) on error
                pass
        
        # Use stored route if available (from clarification followup), otherwise route normally
        if stored_route and stored_route in ['static', 'dynamic', 'hybrid', 'generic']:
            log_step("routing_using_stored_route", {
                "is_clarification_followup": is_clarification_followup,
                "stored_route": stored_route,
                "stored_router_confidence": stored_router_confidence,
                "combined_query_preview": combined_query[:100]
            })
            route = stored_route
            confidence = stored_router_confidence
            # Use the combined query as the analyzed query - don't re-route
            analyzed_query = combined_query
        else:
            log_step("routing_start", {
                "is_clarification_followup": is_clarification_followup,
                "query_for_routing": analyzed_query[:100]
            })
            route_result = await route_query_llm(
                analyzed_query,
                conversationHistory,
                qa_generator
            )
            route = route_result["route"]
            confidence = route_result["confidence"]
        log_step("routing_complete", {
            "route": route,
            "confidence": confidence,
            "is_clarification_followup": is_clarification_followup
        })
        
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        
        if route == "static":
            log_step("static_route_start", {})
            
            # Retrieve more chunks initially to ensure Polkassembly docs are included
            # Polkassembly docs may have lower similarity scores but should be prioritized
            initial_chunks_to_retrieve = max(max_chunks * 2, 10)
            static_chunks = static_embedding_manager.search_similar_chunks(
                query=analyzed_query,
                n_results=initial_chunks_to_retrieve
            )
            from .chunks_reranker import rerank_static_chunks
            static_chunks = rerank_static_chunks(static_chunks)
            # Limit to max_chunks after reranking
            static_chunks = static_chunks[:max_chunks]
            log_step("static_retrieval_complete", {"chunks_count": len(static_chunks)})
            
            retrieval_confidence, semantic_completeness = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                static_chunks=static_chunks,
                query=analyzed_query,
                qa_generator=qa_generator
            )
            
            # Use semantic_completeness from compute_retrieval_confidence (or default to 0.5)
            if semantic_completeness is None:
                semantic_completeness = 0.5
            
            static_similarity = 0.0
            if static_chunks and len(static_chunks) > 0:
                similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in static_chunks]
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
                max_similarity = max(similarity_scores) if similarity_scores else 0.0
                static_similarity = (avg_similarity * 0.5 + max_similarity * 0.5)
            
            chunk_count_factor = min(len(static_chunks) / 5.0, 1.0) if static_chunks else 0.0
            final_static_confidence = retrieval_confidence
            
            # Decision logic for static route
            decision = None
            if final_static_confidence >= 0.45:
                decision = "ANSWER"
                qa_result = await qa_generator.generate_answer(
                    query=analyzed_query,
                    chunks=static_chunks,
                    custom_prompt=custom_prompt,
                    user_id=user_id,
                    conversation_history=conversationHistory,
                    route=route,
                    route_confidence=confidence
                )
                
                log_step("static_confidence_decision", {
                    "route": "static",
                    "routerConfidence": confidence,
                    "semanticCompleteness": semantic_completeness,
                    "staticSimilarity": static_similarity,
                    "chunkCountFactor": chunk_count_factor,
                    "finalStaticConfidence": final_static_confidence,
                    "decision": decision
                })
                
                log_step("static_route_complete", {
                    "chunks_used": qa_result.get('chunks_used', 0),
                    "search_method": qa_result.get('search_method', 'unknown'),
                    "retrieval_confidence": retrieval_confidence
                })
            elif 0.35 <= final_static_confidence < 0.45:
                decision = "AMBIGUITY_FLOW"
                if not is_clarification_followup:
                    log_step("static_confidence_decision", {
                        "route": "static",
                        "routerConfidence": confidence,
                        "semanticCompleteness": semantic_completeness,
                        "staticSimilarity": static_similarity,
                        "chunkCountFactor": chunk_count_factor,
                        "finalStaticConfidence": final_static_confidence,
                        "decision": decision
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = retrieval_confidence
                    clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                    clarification_result['original_query'] = userMessage
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": clarification_result['processing_time_ms'],
                        "requires_clarification": True,
                        "success": True
                    })
                    
                    return clarification_result
                else:
                    # Already a clarification followup - proceed with answer
                    qa_result = await qa_generator.generate_answer(
                        query=analyzed_query,
                        chunks=static_chunks,
                        custom_prompt=custom_prompt,
                        user_id=user_id,
                        conversation_history=conversationHistory,
                        route=route,
                        route_confidence=confidence
                    )
                    
                    log_step("static_confidence_decision", {
                        "route": "static",
                        "routerConfidence": confidence,
                        "semanticCompleteness": semantic_completeness,
                        "staticSimilarity": static_similarity,
                        "chunkCountFactor": chunk_count_factor,
                        "finalStaticConfidence": final_static_confidence,
                        "decision": decision,
                        "note": "Proceeding despite medium confidence as this is already a clarification followup"
                    })
                    
                    log_step("static_route_complete", {
                        "chunks_used": qa_result.get('chunks_used', 0),
                        "search_method": qa_result.get('search_method', 'unknown'),
                        "retrieval_confidence": retrieval_confidence
                    })
            else:
                if static_chunks and len(static_chunks) > 0:
                    decision = "AMBIGUITY_FLOW"
                    if not is_clarification_followup:
                        log_step("static_confidence_decision", {
                            "route": "static",
                            "routerConfidence": confidence,
                            "semanticCompleteness": semantic_completeness,
                            "staticSimilarity": static_similarity,
                            "chunkCountFactor": chunk_count_factor,
                            "finalStaticConfidence": final_static_confidence,
                            "decision": decision,
                            "note": "Low confidence but chunks found - using ambiguity flow instead of internet fallback"
                        })
                        
                        clarification_result = await generate_clarification_question(
                            query=userMessage,
                            route=route,
                            router_confidence=confidence,
                            qa_generator=qa_generator,
                            log_step=log_step
                        )
                        
                        clarification_result['route'] = route
                        clarification_result['route_confidence'] = confidence
                        clarification_result['retrievalConfidence'] = retrieval_confidence
                        clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                        clarification_result['original_query'] = userMessage
                        
                        log_step("pipeline_complete", {
                            "route": route,
                            "confidence": confidence,
                            "retrieval_confidence": retrieval_confidence,
                            "processing_time_ms": clarification_result['processing_time_ms'],
                            "requires_clarification": True,
                            "success": True
                        })
                        
                        return clarification_result
                    else:
                        qa_result = await qa_generator.generate_answer(
                            query=analyzed_query,
                            chunks=static_chunks,
                            custom_prompt=custom_prompt,
                            user_id=user_id,
                            conversation_history=conversationHistory,
                            route=route,
                            route_confidence=confidence
                        )
                        
                        log_step("static_confidence_decision", {
                            "route": "static",
                            "routerConfidence": confidence,
                            "semanticCompleteness": semantic_completeness,
                            "staticSimilarity": static_similarity,
                            "chunkCountFactor": chunk_count_factor,
                            "finalStaticConfidence": final_static_confidence,
                            "decision": decision,
                            "note": "Low confidence but chunks found - proceeding with answer as clarification followup"
                        })
                        
                        log_step("static_route_complete", {
                            "chunks_used": qa_result.get('chunks_used', 0),
                            "search_method": qa_result.get('search_method', 'unknown'),
                            "retrieval_confidence": retrieval_confidence
                        })
                else:
                    decision = "FALLBACK_FLOW"
                    log_step("static_confidence_decision", {
                        "route": "static",
                        "routerConfidence": confidence,
                        "semanticCompleteness": semantic_completeness,
                        "staticSimilarity": static_similarity,
                        "chunkCountFactor": chunk_count_factor,
                        "finalStaticConfidence": final_static_confidence,
                        "decision": decision,
                        "note": "No chunks found - falling back to internet search"
                    })
                    
                    internet_result = await generate_internet_search_response(
                        query=analyzed_query,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    internet_result['route'] = route
                    internet_result['route_confidence'] = confidence
                    internet_result['retrievalConfidence'] = retrieval_confidence
                    internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": internet_result['processing_time_ms'],
                        "internet_fallback": True,
                        "success": True
                    })
                    
                    return internet_result
            
        elif route == "dynamic":
            log_step("dynamic_route_start", {})
            
            # Check which table will be used - skip clarification for voting_data
            # Use analyzed query for table selection to get full context (e.g., "how about in july" -> "How many unique voters were there in July 2025")
            selected_table = qa_generator._determine_table_from_query(analyzed_query)
            is_voting_data = selected_table == "voting_data" if selected_table else False
            
            log_step("table_selection_check", {
                "selected_table": selected_table,
                "skip_clarification": is_voting_data
            })
            
            from .confidence import getSemanticCompletenessScore
            
            # Skip semantic completeness check and clarification for voting_data queries
            # Voting data is in a separate DB and cannot filter by network
            semantic_score = None
            if not is_voting_data:
                semantic_score = await getSemanticCompletenessScore(analyzed_query, qa_generator)
                log_step("semantic_completeness_check", {
                    "semantic_score": semantic_score,
                    "threshold": 0.35
                })
                
                if semantic_score < 0.35:
                    log_step("semantic_completeness_low", {
                        "semantic_score": semantic_score,
                        "action": "triggering_ambiguity_flow"
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = 0.0
                    clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                    clarification_result['original_query'] = userMessage
                    clarification_result['semanticCompleteness'] = semantic_score
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": 0.0,
                        "processing_time_ms": clarification_result['processing_time_ms'],
                        "requires_clarification": True,
                        "semantic_completeness": semantic_score,
                        "success": True
                    })
                    
                    return clarification_result
            else:
                log_step("semantic_completeness_skipped", {
                    "reason": "voting_data table - network filtering not possible"
                })
                # Set a high score for voting_data since we skip the check
                semantic_score = 1.0
            
            qa_result = await qa_generator.generate_answer(
                query=analyzed_query,
                chunks=[],
                custom_prompt=custom_prompt,
                user_id=user_id,
                conversation_history=conversationHistory,
                route=route,
                route_confidence=confidence,
                dynamic_embedding_manager=dynamic_embedding_manager
            )
            
            # Handle SQL precision too low (requires clarification)
            # Skip for voting_data - network filtering not possible, so clarification won't help
            if qa_result.get('requires_clarification', False) and qa_result.get('search_method') == 'sql_precision_too_low':
                sql_precision = qa_result.get('sql_precision', 0.0)
                
                if is_voting_data:
                    log_step("sql_precision_too_low_skipped", {
                        "sql_precision": sql_precision,
                        "reason": "voting_data table - network filtering not possible, proceeding anyway"
                    })
                    # Clear the requires_clarification flag so it proceeds
                    qa_result['requires_clarification'] = False
                else:
                    log_step("sql_precision_too_low", {
                        "sql_precision": sql_precision,
                        "threshold": 0.3,
                        "action": "triggering_ambiguity_flow"
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = 0.0
                    clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                    clarification_result['original_query'] = userMessage
                    clarification_result['sqlPrecision'] = sql_precision
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": 0.0,
                        "processing_time_ms": clarification_result['processing_time_ms'],
                        "requires_clarification": True,
                        "sql_precision": sql_precision,
                        "success": True
                    })
                    
                    return clarification_result
            
            # Handle no results (requires fallback)
            if qa_result.get('requires_fallback', False) and qa_result.get('search_method') == 'no_results':
                log_step("no_results_fallback", {
                    "result_count": qa_result.get('result_count', 0),
                    "action": "triggering_fallback_flow"
                })
                
                internet_result = await generate_internet_search_response(
                    query=analyzed_query,
                    qa_generator=qa_generator,
                    log_step=log_step
                )
                
                internet_result['route'] = route
                internet_result['route_confidence'] = confidence
                internet_result['retrievalConfidence'] = 0.0
                internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                
                log_step("pipeline_complete", {
                    "route": route,
                    "confidence": confidence,
                    "retrieval_confidence": 0.0,
                    "processing_time_ms": internet_result['processing_time_ms'],
                    "internet_fallback": True,
                    "success": True
                })
                
                return internet_result
            
            # Skip ambiguity check for voting_data - network filtering not possible
            is_ambiguous_query = False
            sql_has_network_filter = False
            if not is_voting_data:
                # Check if query is ambiguous (missing network specification)
                # Ambiguous if: user didn't specify network AND (SQL filtered by network OR SQL returned both networks)
                # Check both original and analyzed query for network mentions
                user_query_lower = userMessage.lower()
                analyzed_query_lower = analyzed_query.lower()
                
                # Check for explicit network mentions
                has_network_in_user_query = any(network in user_query_lower for network in ['polkadot', 'kusama', 'dot', 'ksm'])
                has_network_in_analyzed = any(network in analyzed_query_lower for network in ['polkadot', 'kusama', 'dot', 'ksm'])
                
                # Check if user explicitly asked for "both" networks
                explicitly_both_networks = (
                    'both' in user_query_lower and 
                    ('polkadot' in user_query_lower or 'kusama' in user_query_lower) and
                    ('polkadot' in analyzed_query_lower or 'kusama' in analyzed_query_lower)
                )
                
                # User specified network if it's in either query OR they explicitly asked for both
                has_network_specified = has_network_in_user_query or has_network_in_analyzed or explicitly_both_networks
                
                # Check if SQL query contains network filter
                sql_queries = qa_result.get('sql_query', [])
                sql_has_both_networks = False
                is_aggregate_query = False
                if sql_queries:
                    if isinstance(sql_queries, list):
                        sql_query_str = ' '.join(sql_queries).lower()
                    else:
                        sql_query_str = str(sql_queries).lower()
                    sql_has_network_filter = 'source_network' in sql_query_str and ('polkadot' in sql_query_str or 'kusama' in sql_query_str)
                    # Check if SQL explicitly filters for both networks
                    sql_has_both_networks = 'source_network' in sql_query_str and 'polkadot' in sql_query_str and 'kusama' in sql_query_str
                    # Check if SQL contains aggregate functions (COUNT, SUM, AVG, MAX, MIN)
                    # Aggregate queries are asking for summary statistics, so it's reasonable to return data from both networks
                    is_aggregate_query = any(func in sql_query_str for func in ['count(', 'sum(', 'avg(', 'max(', 'min(', 'count(*)'])
            
                # Check if results contain multiple networks (indicating no network filter was applied)
                # This is a proxy check - if SQL didn't filter by network, results likely contain both
                results_have_multiple_networks = False
                if qa_result.get('success', False) and qa_result.get('result_count', 0) > 0:
                    # If SQL doesn't have network filter, assume it returned both networks
                    results_have_multiple_networks = not sql_has_network_filter
                
                # Query is ambiguous if:
                # 1. User didn't specify network (and didn't ask for both), AND
                # 2. (SQL filtered by a network user didn't specify OR SQL returned both networks), AND
                # 3. SQL got results
                # NOT ambiguous if: user asked for both networks and SQL has both networks
                # NOT ambiguous if: SQL is an aggregate query (COUNT, SUM, etc.) - aggregate queries are fine without network specification
                is_ambiguous_query = (
                    not has_network_specified and 
                    (sql_has_network_filter or results_have_multiple_networks) and
                    qa_result.get('success', False) and 
                    qa_result.get('result_count', 0) > 0 and
                    not is_aggregate_query
                ) and not (explicitly_both_networks and sql_has_both_networks)
            else:
                log_step("ambiguity_check_skipped", {
                    "reason": "voting_data table - network filtering not possible"
                })
            
            if is_ambiguous_query:
                log_step("ambiguous_query_detected", {
                    "query": analyzed_query,
                    "user_query": userMessage,
                    "sql_has_network_filter": sql_has_network_filter,
                    "note": "Query is ambiguous - SQL filtered by network user didn't specify"
                })
            
            retrieval_confidence, _ = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                sql_result_count=qa_result.get('result_count', 0),
                sql_success=qa_result.get('success', False),
                is_ambiguous_query=is_ambiguous_query,
                query=analyzed_query,
                sql_query=qa_result.get('sql_query', []),
                qa_generator=qa_generator
            )
            
            result_count = qa_result.get('result_count', 0)
            final_confidence = retrieval_confidence
            
            # Updated decision logic for dynamic route
            decision = None
            if result_count > 0:
                if final_confidence >= 0.65:
                    decision = "ANSWER"
                else:
                    # Skip ambiguity flow for voting_data - network filtering not possible
                    if is_voting_data:
                        decision = "ANSWER"
                        log_step("voting_data_low_confidence_override", {
                            "final_confidence": final_confidence,
                            "reason": "voting_data table - network filtering not possible, returning answer anyway"
                        })
                    else:
                        decision = "AMBIGUITY_FLOW"
                        if not is_clarification_followup:
                            clarification_result = await generate_clarification_question(
                                query=userMessage,
                                route=route,
                                router_confidence=confidence,
                                qa_generator=qa_generator,
                                log_step=log_step
                            )
                            
                            clarification_result['route'] = route
                            clarification_result['route_confidence'] = confidence
                            clarification_result['retrievalConfidence'] = retrieval_confidence
                            clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                            clarification_result['original_query'] = userMessage
                            clarification_result['semanticCompleteness'] = semantic_score
                            clarification_result['sqlPrecision'] = qa_result.get('sql_precision', None)
                            
                            log_step("dynamic_decision_log", {
                                "route": "dynamic",
                                "semanticCompleteness": semantic_score,
                                "sqlPrecision": qa_result.get('sql_precision', None),
                                "resultCount": result_count,
                                "finalConfidence": final_confidence,
                                "decision": decision
                            })
                            
                            log_step("pipeline_complete", {
                                "route": route,
                                "confidence": confidence,
                                "retrieval_confidence": retrieval_confidence,
                                "processing_time_ms": clarification_result['processing_time_ms'],
                                "requires_clarification": True,
                                "success": True
                            })
                            
                            return clarification_result
            else:
                decision = "FALLBACK_FLOW"
                internet_result = await generate_internet_search_response(
                    query=analyzed_query,
                    qa_generator=qa_generator,
                    log_step=log_step
                )
                
                internet_result['route'] = route
                internet_result['route_confidence'] = confidence
                internet_result['retrievalConfidence'] = retrieval_confidence
                internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                
                log_step("dynamic_decision_log", {
                    "route": "dynamic",
                    "semanticCompleteness": semantic_score,
                    "sqlPrecision": qa_result.get('sql_precision', None),
                    "resultCount": result_count,
                    "finalConfidence": final_confidence,
                    "decision": decision
                })
                
                log_step("pipeline_complete", {
                    "route": route,
                    "confidence": confidence,
                    "retrieval_confidence": retrieval_confidence,
                    "processing_time_ms": internet_result['processing_time_ms'],
                    "internet_fallback": True,
                    "success": True
                })
                
                return internet_result
            
            log_step("dynamic_route_complete", {
                "success": qa_result.get('success', False),
                "result_count": result_count,
                "search_method": qa_result.get('search_method', 'unknown'),
                "retrieval_confidence": retrieval_confidence,
                "is_ambiguous_query": is_ambiguous_query
            })
            
            log_step("dynamic_decision_log", {
                "route": "dynamic",
                "semanticCompleteness": semantic_score,
                "sqlPrecision": qa_result.get('sql_precision', None),
                "resultCount": result_count,
                "finalConfidence": final_confidence,
                "decision": decision
            })
            
            qa_result['semanticCompleteness'] = semantic_score
            qa_result['sqlPrecision'] = qa_result.get('sql_precision', None)
            
        elif route == "hybrid":
            log_step("hybrid_route_start", {})
            
            # Retrieve more chunks initially to ensure Polkassembly docs are included
            initial_chunks_to_retrieve = max(max_chunks * 2, 10)
            static_chunks = static_embedding_manager.search_similar_chunks(
                query=analyzed_query,
                n_results=initial_chunks_to_retrieve
            )
            from .chunks_reranker import rerank_static_chunks
            static_chunks = rerank_static_chunks(static_chunks)
            # Limit to max_chunks after reranking
            static_chunks = static_chunks[:max_chunks]
            log_step("hybrid_static_retrieval_complete", {"chunks_count": len(static_chunks)})
            
            hybrid_static_available = len(static_chunks) > 0 if static_chunks else False
            
            qa_result = await qa_generator.generate_answer(
                query=analyzed_query,
                chunks=static_chunks,
                custom_prompt=custom_prompt,
                user_id=user_id,
                conversation_history=conversationHistory,
                route=route,
                route_confidence=confidence,
                dynamic_embedding_manager=dynamic_embedding_manager
            )
            
            hybrid_dynamic_available = qa_result.get('success', False) and qa_result.get('result_count', 0) > 0
            
            # Check if query is ambiguous (missing network specification for dynamic part)
            user_query_lower = userMessage.lower()
            has_network_in_user_query = any(network in user_query_lower for network in ['polkadot', 'kusama', 'dot', 'ksm'])
            
            # Check if SQL query contains network filter
            sql_queries = qa_result.get('sql_query', [])
            sql_has_network_filter = False
            if sql_queries:
                if isinstance(sql_queries, list):
                    sql_query_str = ' '.join(sql_queries).lower()
                else:
                    sql_query_str = str(sql_queries).lower()
                sql_has_network_filter = 'source_network' in sql_query_str and ('polkadot' in sql_query_str or 'kusama' in sql_query_str)
            
            # Check if results contain multiple networks (indicating no network filter was applied)
            results_have_multiple_networks = False
            if qa_result.get('success', False) and qa_result.get('result_count', 0) > 0:
                # If SQL doesn't have network filter, assume it returned both networks
                results_have_multiple_networks = not sql_has_network_filter
            
            # Query is ambiguous if:
            # 1. User didn't specify network, AND
            # 2. (SQL filtered by a network user didn't specify OR SQL returned both networks), AND
            # 3. SQL got results
            is_ambiguous_query = (
                not has_network_in_user_query and 
                (sql_has_network_filter or results_have_multiple_networks) and
                qa_result.get('success', False) and 
                qa_result.get('result_count', 0) > 0
            )
            
            if is_ambiguous_query:
                log_step("ambiguous_query_detected_hybrid", {
                    "query": analyzed_query,
                    "user_query": userMessage,
                    "sql_has_network_filter": sql_has_network_filter,
                    "note": "Hybrid query is ambiguous - SQL filtered by network user didn't specify"
                })
            
            retrieval_confidence, _ = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                static_chunks=static_chunks,
                sql_result_count=qa_result.get('result_count', 0),
                sql_success=qa_result.get('success', False),
                hybrid_static_available=hybrid_static_available,
                hybrid_dynamic_available=hybrid_dynamic_available,
                is_ambiguous_query=is_ambiguous_query,
                query=analyzed_query,
                sql_query=qa_result.get('sql_query', []),
                qa_generator=qa_generator
            )
            
            log_step("hybrid_route_complete", {
                "chunks_used": qa_result.get('chunks_used', 0),
                "search_method": qa_result.get('search_method', 'unknown'),
                "retrieval_confidence": retrieval_confidence,
                "hybrid_static_available": hybrid_static_available,
                "hybrid_dynamic_available": hybrid_dynamic_available,
                "is_ambiguous_query": is_ambiguous_query
            })
            
        elif route == "generic":
            log_step("generic_route_start", {})
            
            if qa_generator.memory_manager and qa_generator.memory_manager.enabled:
                try:
                    qa_generator.memory_manager.add_user_query(analyzed_query, user_id)
                except Exception as e:
                    log_step("memory_add_error", {"error": str(e)}, "warning")
            
            qa_result = await handle_generic_query_llm(
                analyzed_query,
                conversationHistory,
                qa_generator,
                log_step
            )
            
            if qa_generator.memory_manager and qa_generator.memory_manager.enabled:
                try:
                    qa_generator.memory_manager.add_assistant_response(
                        qa_result.get('answer', ''),
                        user_id
                    )
                except Exception as e:
                    log_step("memory_add_error", {"error": str(e)}, "warning")
            
            log_step("generic_route_complete", {
                "search_method": qa_result.get('search_method', 'unknown')
            })
        
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        
        # Static route handles its own decision logic, so skip unified threshold logic
        if route == "static":
            qa_result['route'] = route
            qa_result['route_confidence'] = confidence
            qa_result['retrievalConfidence'] = retrieval_confidence
            qa_result['processing_time_ms'] = processing_time
            if is_clarification_followup and clarification_info:
                qa_result['is_clarification_followup'] = True
                qa_result['original_query'] = clarification_info['original_query']
            
            log_step("pipeline_complete", {
                "route": route,
                "confidence": confidence,
                "retrieval_confidence": retrieval_confidence,
                "processing_time_ms": processing_time,
                "is_clarification_followup": is_clarification_followup,
                "success": True
            })
            
            return qa_result
        
        # retrieval_confidence may have been calculated in route-specific blocks above
        # Check if it exists and is not None, otherwise calculate it
        if route != "generic":
            try:
                # Check if retrieval_confidence was already calculated in route block
                if retrieval_confidence is None:
                    raise NameError("retrieval_confidence is None")
            except NameError:
                # retrieval_confidence doesn't exist or is None, calculate it
                # Static route already calculated it above, so this should only happen for dynamic/hybrid
                if route == "dynamic":
                    retrieval_confidence, _ = await compute_retrieval_confidence(
                        route=route,
                        router_confidence=confidence,
                        sql_result_count=qa_result.get('result_count', 0),
                        sql_success=qa_result.get('success', False),
                        query=analyzed_query,
                        sql_query=qa_result.get('sql_query', []),
                        qa_generator=qa_generator
                    )
                elif route == "hybrid":
                    hybrid_static_available = len(static_chunks) > 0 if 'static_chunks' in locals() and static_chunks else False
                    hybrid_dynamic_available = qa_result.get('success', False) and qa_result.get('result_count', 0) > 0
                    retrieval_confidence, _ = await compute_retrieval_confidence(
                        route=route,
                        router_confidence=confidence,
                        static_chunks=static_chunks if 'static_chunks' in locals() else None,
                        sql_result_count=qa_result.get('result_count', 0),
                        sql_success=qa_result.get('success', False),
                        hybrid_static_available=hybrid_static_available,
                        hybrid_dynamic_available=hybrid_dynamic_available
                    )
                log_step("retrieval_confidence_calculated", {
                    "retrieval_confidence": retrieval_confidence,
                    "route": route
                })
            else:
                # retrieval_confidence was already calculated in route block, use it
                log_step("using_existing_retrieval_confidence", {
                    "retrieval_confidence": retrieval_confidence,
                    "route": route
                })
            
            qa_result['retrievalConfidence'] = retrieval_confidence
            
            # Ensure retrieval_confidence is not None before comparison
            if retrieval_confidence is None:
                log_step("retrieval_confidence_none", {
                    "route": route,
                    "note": "Setting retrieval_confidence to 0.0 as fallback"
                }, "warning")
                retrieval_confidence = 0.0
                qa_result['retrievalConfidence'] = 0.0
            
            # New threshold logic: 0.65 for ANSWER, 0.35 for AMBIGUITY_FLOW, below 0.35 for FALLBACK_FLOW
            # BUT: If we have data but confidence is low, go to AMBIGUITY_FLOW instead of FALLBACK_FLOW
            final_confidence = retrieval_confidence
            
            # Check if we have data available (static route already handled its own logic)
            has_data = False
            if route == "dynamic":
                has_data = qa_result.get('success', False) and qa_result.get('result_count', 0) > 0
            elif route == "hybrid":
                has_data = (
                    (len(static_chunks) > 0 if 'static_chunks' in locals() and static_chunks else False) or
                    (qa_result.get('success', False) and qa_result.get('result_count', 0) > 0)
                )
            
            if final_confidence >= 0.65:
                # ANSWER: High confidence, proceed with answer
                log_step("confidence_high_answer", {
                    "final_confidence": final_confidence,
                    "threshold": 0.65,
                    "action": "ANSWER"
                })
            elif 0.35 <= final_confidence < 0.65:
                # AMBIGUITY_FLOW: Medium confidence, ask for clarification
                # Skip clarification for voting_data - network filtering not possible
                # Skip clarification if this is already a clarification followup (to avoid loops)
                if is_voting_data:
                    log_step("voting_data_medium_confidence_override", {
                        "final_confidence": final_confidence,
                        "reason": "voting_data table - network filtering not possible, returning answer anyway"
                    })
                elif not is_clarification_followup:
                    log_step("confidence_medium_ambiguity", {
                        "final_confidence": final_confidence,
                        "threshold_low": 0.35,
                        "threshold_high": 0.65,
                        "action": "AMBIGUITY_FLOW"
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = retrieval_confidence
                    clarification_result['processing_time_ms'] = processing_time
                    clarification_result['original_query'] = userMessage
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": processing_time,
                        "requires_clarification": True,
                        "success": True
                    })
                    
                    return clarification_result
                else:
                    # Already a clarification followup with medium confidence - log but proceed
                    log_step("clarification_followup_medium_confidence", {
                        "final_confidence": final_confidence,
                        "note": "Proceeding despite medium confidence as this is already a clarification followup"
                    }, "warning")
            else:
                # Low confidence (< 0.35)
                # If we have data but confidence is low (ambiguous query), go to AMBIGUITY_FLOW
                # Only use FALLBACK_FLOW if we truly have no data
                if has_data and not is_clarification_followup:
                    # We have data but query is ambiguous - ask for clarification
                    log_step("confidence_low_but_has_data_ambiguity", {
                        "final_confidence": final_confidence,
                        "has_data": has_data,
                        "threshold": 0.35,
                        "action": "AMBIGUITY_FLOW"
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = retrieval_confidence
                    clarification_result['processing_time_ms'] = processing_time
                    clarification_result['original_query'] = userMessage
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": processing_time,
                        "requires_clarification": True,
                        "success": True
                    })
                    
                    return clarification_result
                else:
                    # FALLBACK_FLOW: Low confidence and no data, use internet search fallback
                    log_step("confidence_low_fallback", {
                        "final_confidence": final_confidence,
                        "has_data": has_data,
                        "threshold": 0.35,
                        "action": "FALLBACK_FLOW"
                    })
                    
                    internet_result = await generate_internet_search_response(
                        query=analyzed_query,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    internet_result['route'] = route
                    internet_result['route_confidence'] = confidence
                    internet_result['retrievalConfidence'] = retrieval_confidence
                    internet_result['processing_time_ms'] = processing_time
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": processing_time,
                        "internet_fallback": True,
                        "success": True
                    })
                    
                    return internet_result
        
        qa_result['route'] = route
        qa_result['route_confidence'] = confidence
        qa_result['processing_time_ms'] = processing_time
        if is_clarification_followup and clarification_info:
            qa_result['is_clarification_followup'] = True
            qa_result['original_query'] = clarification_info['original_query']
        
        log_step("pipeline_complete", {
            "route": route,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "is_clarification_followup": is_clarification_followup,
            "success": True
        })
        
        return qa_result
        
    except Exception as e:
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        log_step("pipeline_error", {
            "error": str(e),
            "processing_time_ms": processing_time
        }, "error")
        
        return {
            'answer': "I encountered an error processing your query. Please try again.",
            'sources': [],
            'chunks_used': 0,
            'search_method': 'error',
            'route': 'generic',
            'route_confidence': 0.0,
            'processing_time_ms': processing_time,
            'error': str(e)
        }

