"""
Query processing pipeline for the Polkadot AI Chatbot.
Implements the new routing-first architecture with structured logging.
"""

import logging
import json
import re
import math
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from .confidence import compute_retrieval_confidence
from .greeting_handler import handle_generic_query_llm
from .clarification import generate_clarification_question
from .internet_fallback import generate_internet_search_response

logger = logging.getLogger(__name__)


async def is_query_truly_ambiguous(query: str, qa_generator, sql_queries: Optional[List[str]] = None) -> bool:
    """
    Determine if a query truly needs clarification using LLM.
    
    Only marks as ambiguous if a REQUIRED parameter is missing (e.g., referenda ID when asking for details).
    Does NOT mark as ambiguous for missing optional parameters (e.g., network for generic listing queries).
    
    Uses GPT-3.5-turbo for fast, context-aware ambiguity detection.
    
    Args:
        query: The user query to check
        qa_generator: QA generator instance with LLM access
        sql_queries: Optional list of SQL queries to analyze
    
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
    
    ambiguity_prompt = f"""You are a STRICT ambiguity checker for a Polkadot/Kusama governance assistant.

Your ONLY job:

Decide if the user's query is missing a REQUIRED identifier for a **single, specific on-chain item** (referendum, proposal, bounty, treasury item, etc.).
You must output ONLY one word: "true" or "false" (lowercase, no punctuation).

User Query:

"{query}"{sql_context}

---

DECISION RULES (follow these in order):

1) IS THIS A LIST / SEARCH / AGGREGATE QUESTION?
   - Examples: "show me proposals", "list treasury proposals", "find bounties",
     "how many voters", "show active referenda", "show proposals about staking".
   - If the query can reasonably be answered by returning a list, a count,
     or a filtered list (by topic, date, track, etc.), then it is NOT ambiguous.
   → In this case, answer "false".

2) IS THE USER ASKING ABOUT A SPECIFIC SINGLE ITEM?
   - Look for language like:
     - "this", "that", "the" + singular noun WITHOUT a topic/filter ("the referendum", "that bounty",
       "this treasury proposal") - these refer to a specific item without identifier
     - CRITICAL: If "the" is followed by a TOPIC/FILTER keyword, it's a LISTING query, NOT a specific item:
       * "tell me about the polkabot.ai referenda" → LISTING (has topic "polkabot.ai")
       * "show me the staking proposals" → LISTING (has topic "staking")
     - If the query has a topic/filter keyword (like "polkabot.ai", "staking", "treasury", etc.),
       it's a LISTING query and is NOT ambiguous

   - If the query is NOT clearly about one specific item, it is NOT ambiguous.
   → In this case, answer "false".

3) IF IT IS ABOUT A SPECIFIC ITEM, DOES IT INCLUDE A CLEAR IDENTIFIER?

   Acceptable identifiers include ANY of:

   - A numeric ID (e.g., 123, 456, 1781)
   - A full Polkassembly URL (which contains the ID
   - An explicit unique title or name that could reasonably identify it
     (e.g., a full proposal title, or a very specific phrase)

   If any of these are present, then the query is NOT ambiguous.
   → In this case, answer "false".

4) ONLY IF ALL OF THE FOLLOWING ARE TRUE, IT IS AMBIGUOUS:

   - The user is clearly asking about ONE specific item (Step 2 = yes)
   - AND they use vague references like "this", "that", "the" WITHOUT a topic/filter keyword
   - AND there is NO numeric ID, NO URL with ID, and NO clear unique identifier
   - AND there is NO topic/filter keyword (like "polkabot.ai", "staking", etc.)
   - AND we cannot reasonably treat it as a list/search query instead
   → ONLY in this case answer "true".

IMPORTANT CONSTRAINTS:

- The network (Polkadot vs Kusama) is ALWAYS OPTIONAL.
  Missing network MUST NEVER make the query ambiguous.

- Listing / searching / counting queries are NEVER ambiguous,
  even if they could be more specific.

- Queries with filters or topics ("about polkabot.ai", "about staking", "in October")
  are NOT ambiguous if they can be answered by a list or count.

- Do NOT try to be helpful or suggest follow-up questions.
  Just decide: is a REQUIRED identifier missing for a single specific item?

EXAMPLES (for your own understanding):

Should be "true" (ambiguous):

- "show me details about this referenda"
- "what are the votes for that proposal"
- "tell me about the treasury proposal"
- "who is the curator of that bounty"

Should be "false" (not ambiguous):

- "show me proposals"
- "list treasury proposals"
- "show me referenda 123"
- "what are the votes for proposal 456"
- "show me active referenda"
- "find bounties"
- "how many voters"
- "show me proposals on Polkadot"
- "show proposals about staking"
- "tell me about the polkabot.ai referenda" (has topic "polkabot.ai", so it's a listing query)
- "show me the staking proposals" (has topic "staking", so it's a listing query)
- "how many unique voters were there in November 2025"
Now, after applying the rules above, respond with ONLY:
true
or
false

(lowercase, no extra text)."""
    
    try:
        # Use GPT-4 for better accuracy on this task
        model_to_use = "gpt-4"
        
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
        
        # Parse the answer - extract first word
        first_word = answer.split()[0] if answer.split() else ""
        is_ambiguous = first_word == "true"
        
        logger.info(f"Ambiguity check - Query: '{query[:50]}', Model: {model_to_use}, Raw response: '{raw_answer}', Parsed: '{answer}', First word: '{first_word}', Is ambiguous: {is_ambiguous}")
        
        return is_ambiguous
    except Exception as e:
        logger.error(f"Error in LLM ambiguity check: {e}, defaulting to False")
        return False


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
    dynamic_keywords = ['proposal', 'referendum', 'bounty', 'treasury', 'voter', 'vote', 'show me', 'list', 'find', 'get', 'count', 'how many', 'specific', 'address']
    static_keywords = ['how to', 'how can i', 'what is', 'how does', 'explain', 'tutorial', 'guide', 'delegate', 'delegation', 'concept', 'definition']
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
   - Questions about HOW to do something (e.g., "how to cancel", "how do i", "how can i", "how to")
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
   - Track definitions or theoretical limits without asking for actual on-chain numbers (e.g., "What is the Medium Spender track?")

2. "dynamic" - For queries requesting specific on-chain DATA:
   - "Show me", "list", "find", "get" queries for proposals/referenda/bounties
   - Questions mentioning numbers (e.g., "Who had the highest voting power in referenda 1232", "What is the status of proposal 123", "Show me the 10 most recent proposals")
   - Questions asking for specific proposal information (title, content, status, dates, network)
   - Proposal metadata (type, proposer, beneficiary, amounts, curator)
   - Questions about specific proposal IDs (e.g., "Who is the curator of 1671", "What is the status of proposal 123")
   - Questions about blockchain addresses (e.g., "Who is 0x163830...", "What proposals did [address] make", "Show me proposals by [address]")
   - Voting data (voter information, voting power, decisions)
   - Proposal filtering by ID, dates, network, type, status
   - Aggregations, counts, or summaries of on-chain data (e.g., "How many referenda were created in June?", "What is the max spend in the Medium Spender track?")
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

Respond with ONLY one word from: static, dynamic, hybrid, generic. No explanations.

Now respond for this query:
"""
        
        if getattr(qa_generator, "client", None):
            try:
                router_model = os.getenv("ROUTER_MODEL", "gpt-4")
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
        is_ambiguous_query = False  # Track if query truly needs clarification (missing required parameter)
        
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
        
        # Run ambiguity check BEFORE routing so we can clarify immediately
        if not is_clarification_followup:
            log_step("ambiguity_check_pre_route_start", {
                "query": analyzed_query[:100]
            })
            is_ambiguous_query = await is_query_truly_ambiguous(analyzed_query, qa_generator, None)
            log_step("ambiguity_check_pre_route_complete", {
                "query": analyzed_query[:100],
                "is_ambiguous": is_ambiguous_query
            })
            
            if is_ambiguous_query:
                log_step("ambiguous_query_detected_pre_route", {
                    "query": analyzed_query,
                    "user_query": userMessage,
                    "note": "Query is ambiguous before routing - returning clarification immediately"
                })
                
                clarification_result = await generate_clarification_question(
                    query=userMessage,
                    route=None,
                    router_confidence=0.0,
                    qa_generator=qa_generator,
                    log_step=log_step
                )
                
                clarification_result['route'] = 'ambiguous_pre_route'
                clarification_result['route_confidence'] = 0.0
                clarification_result['retrievalConfidence'] = 0.0
                clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                clarification_result['original_query'] = userMessage
                
                log_step("pipeline_complete", {
                    "route": "ambiguous_pre_route",
                    "confidence": 0.0,
                    "retrieval_confidence": 0.0,
                    "processing_time_ms": clarification_result['processing_time_ms'],
                    "requires_clarification": True,
                    "success": True
                })
                
                return clarification_result
            else:
                is_ambiguous_query = False
        
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
            
            selected_table = qa_generator._determine_table_from_query(analyzed_query)
            log_step("table_selection_check", {
                "selected_table": selected_table
            })
            
            # Proceed with SQL generation and answer (query is not ambiguous)
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
            
            # Extract validator and result information
            validator_verdict = qa_result.get('validator_verdict')
            validator_reason = qa_result.get('validator_reason')
            requires_clarification = qa_result.get('requires_clarification', False)
            requires_fallback = qa_result.get('requires_fallback', False)
            result_count = qa_result.get('result_count', 0)
            sql_precision = qa_result.get('sql_precision')
            
            # Decision logic based on validator_verdict as primary signal
            decision = None
            
            # Case 1: validator_verdict == "bad" - treat as hard error, need clarification
            if validator_verdict == "bad":
                if not is_ambiguous_query and not is_clarification_followup:
                    decision = "CLARIFICATION"
                    log_step("dynamic_validator_bad", {
                        "validator_verdict": validator_verdict,
                        "validator_reason": validator_reason,
                        "result_count": result_count,
                        "requires_clarification": requires_clarification,
                        "requires_fallback": requires_fallback,
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
                    clarification_result['retrievalConfidence'] = 0.0
                    clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                    clarification_result['original_query'] = userMessage
                    clarification_result['validator_verdict'] = validator_verdict
                    clarification_result['validator_reason'] = validator_reason
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": 0.0,
                        "processing_time_ms": clarification_result['processing_time_ms'],
                        "requires_clarification": True,
                        "validator_verdict": validator_verdict,
                        "success": True
                    })
                    
                    return clarification_result
                else:
                    # Already ambiguous or clarification followup - proceed with answer anyway
                    decision = "ANSWER"
            
            # Case 2: validator_verdict == "empty" - no results, trigger fallback
            elif validator_verdict == "empty" or (result_count == 0 and not qa_result.get('success', False)):
                decision = "FALLBACK"
                log_step("dynamic_validator_empty", {
                    "validator_verdict": validator_verdict,
                    "validator_reason": validator_reason,
                    "result_count": result_count,
                    "requires_clarification": requires_clarification,
                    "requires_fallback": requires_fallback,
                    "decision": decision
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
                internet_result['validator_verdict'] = validator_verdict
                internet_result['validator_reason'] = validator_reason
                
                log_step("pipeline_complete", {
                    "route": route,
                    "confidence": confidence,
                    "retrieval_confidence": 0.0,
                    "processing_time_ms": internet_result['processing_time_ms'],
                    "internet_fallback": True,
                    "validator_verdict": validator_verdict,
                    "success": True
                })
                
                return internet_result
            
            # Case 3: validator_verdict == "partial" - proceed with answer, log partial match
            elif validator_verdict == "partial":
                decision = "ANSWER"
                log_step("dynamic_validator_partial", {
                    "validator_verdict": validator_verdict,
                    "validator_reason": validator_reason,
                    "result_count": result_count,
                    "requires_clarification": requires_clarification,
                    "requires_fallback": requires_fallback,
                    "decision": decision,
                    "note": "Results may be incomplete, proceeding with answer"
                })
            
            # Case 4: validator_verdict == "good" - normal success path
            elif validator_verdict == "good":
                decision = "ANSWER"
                log_step("dynamic_validator_good", {
                    "validator_verdict": validator_verdict,
                    "validator_reason": validator_reason,
                    "result_count": result_count,
                    "requires_clarification": requires_clarification,
                    "requires_fallback": requires_fallback,
                    "decision": decision
                })
            
            # Case 5: validator_verdict is missing (backward compatibility)
            else:
                # Fall back to existing requires_clarification / requires_fallback behavior
                if requires_clarification:
                    decision = "CLARIFICATION"
                    log_step("dynamic_fallback_clarification", {
                        "validator_verdict": validator_verdict,
                        "validator_reason": validator_reason,
                        "result_count": result_count,
                        "requires_clarification": requires_clarification,
                        "requires_fallback": requires_fallback,
                        "decision": decision,
                        "note": "Using requires_clarification flag (validator_verdict missing)"
                    })
                    
                    if not is_ambiguous_query and not is_clarification_followup:
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
                        clarification_result['validator_verdict'] = validator_verdict
                        clarification_result['validator_reason'] = validator_reason
                        
                        log_step("pipeline_complete", {
                            "route": route,
                            "confidence": confidence,
                            "retrieval_confidence": 0.0,
                            "processing_time_ms": clarification_result['processing_time_ms'],
                            "requires_clarification": True,
                            "validator_verdict": validator_verdict,
                            "success": True
                        })
                        
                        return clarification_result
                elif requires_fallback or result_count == 0:
                    decision = "FALLBACK"
                    log_step("dynamic_fallback_no_results", {
                        "validator_verdict": validator_verdict,
                        "validator_reason": validator_reason,
                        "result_count": result_count,
                        "requires_clarification": requires_clarification,
                        "requires_fallback": requires_fallback,
                        "decision": decision,
                        "note": "Using requires_fallback flag or result_count == 0 (validator_verdict missing)"
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
                    internet_result['validator_verdict'] = validator_verdict
                    internet_result['validator_reason'] = validator_reason
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": 0.0,
                        "processing_time_ms": internet_result['processing_time_ms'],
                        "internet_fallback": True,
                        "validator_verdict": validator_verdict,
                        "success": True
                    })
                    
                    return internet_result
                else:
                    decision = "ANSWER"
                    log_step("dynamic_fallback_answer", {
                        "validator_verdict": validator_verdict,
                        "validator_reason": validator_reason,
                        "result_count": result_count,
                        "requires_clarification": requires_clarification,
                        "requires_fallback": requires_fallback,
                        "decision": decision,
                        "note": "Proceeding with answer (validator_verdict missing, no flags set)"
                    })
            
            # Calculate confidence for logging/metrics
            retrieval_confidence, _ = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                sql_result_count=result_count,
                sql_success=qa_result.get('success', False),
                is_ambiguous_query=False,
                query=analyzed_query,
                sql_query=qa_result.get('sql_query', []),
                qa_generator=qa_generator
            )
            
            # Log final decision summary
            log_step("dynamic_route_decision", {
                "validator_verdict": validator_verdict,
                "validator_reason": validator_reason,
                "result_count": result_count,
                "requires_clarification": requires_clarification,
                "requires_fallback": requires_fallback,
                "decision": decision,
                "sql_precision": sql_precision
            })
            
            qa_result['sqlPrecision'] = sql_precision
            qa_result['retrievalConfidence'] = retrieval_confidence
            qa_result['route'] = route
            qa_result['route_confidence'] = confidence
            
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
            
            retrieval_confidence, _ = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                static_chunks=static_chunks,
                sql_result_count=qa_result.get('result_count', 0),
                sql_success=qa_result.get('success', False),
                hybrid_static_available=hybrid_static_available,
                hybrid_dynamic_available=hybrid_dynamic_available,
                is_ambiguous_query=False,  # Already checked, so set to False
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
            
            # Ambiguity is already checked early in the route-specific blocks
            # At this point, if we reach here, the query is not ambiguous
            # Just proceed with the answer (confidence is only for logging/metrics)
        
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

