"""
Confidence computation utilities for retrieval quality assessment.
"""

import re
import math
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _read_field(obj, field):
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _get_token_probability(choice, target_token: str) -> Optional[float]:
    logprobs = getattr(choice, "logprobs", None)
    if not logprobs:
        return None
    content = getattr(logprobs, "content", None)
    if not content:
        return None
    for token_info in content:
        token = (_read_field(token_info, "token") or "").strip().lower()
        if not token:
            continue
        logprob = _read_field(token_info, "logprob")
        if logprob is None:
            continue
        if token == target_token:
            return float(math.exp(logprob))
    token_info = content[0]
    logprob = _read_field(token_info, "logprob")
    if logprob is None:
        return None
    return float(math.exp(logprob))


async def getSemanticCompletenessScore(query: str, qa_generator) -> float:
    """
    Rate how specific a query is for SQL generation from 0.0 (very vague) to 1.0 (highly specific).
    
    Args:
        query: The user query to evaluate
        qa_generator: QA generator instance with LLM access
    
    Returns:
        Float score between 0.0 and 1.0
    """
    prompt = f"""Decide if this query is SPECIFIC (good for SQL) or VAGUE (needs clarification).
Respond with only one word: SPECIFIC or VAGUE.

CRITICAL: URL HANDLING:
- If the query is a URL (e.g., "http://polkadot.polkassembly.io/referenda/1781"), this is HIGHLY SPECIFIC (score 0.9-1.0)
- URLs contain specific proposal/referenda IDs and network information
- Extract: polkadot.polkassembly.io/referenda/1781 = referenda 1781 on Polkadot network (very specific)
- Extract: kusama.polkassembly.io/referenda/123 = referenda 123 on Kusama network (very specific)
- URLs are NOT vague - they point to specific on-chain data

Query: {query}"""
    
    try:
        if not hasattr(qa_generator, 'client'):
            logger.warning("No OpenAI client available for semantic completeness scoring")
            return 0.5

        response_obj = qa_generator.client.chat.completions.create(
            model=qa_generator.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2,
            logprobs=True,
            top_logprobs=5
        )
        choice = response_obj.choices[0]
        answer = (choice.message.content or "").strip().lower().strip('.')
        probability = _get_token_probability(choice, answer)
        if probability is None:
            probability = 0.0
        if answer == "specific":
            score = probability
        elif answer == "vague":
            score = 1.0 - probability
        else:
            score = probability
        score = max(0.0, min(1.0, score))
        logger.info("semantic_specificity_result", {
            "answer": answer,
            "probability": probability,
            "score": score
        })
        return score
    except Exception as e:
        logger.error(f"Error computing semantic completeness score: {e}")
        return 0.5


async def getSemanticCompletenessScoreForStatic(query: str, qa_generator) -> float:
    """
    Rate how specific a query is for static documentation lookup from 0.0 (very vague) to 1.0 (highly specific).
    
    Args:
        query: The user query to evaluate
        qa_generator: QA generator instance with LLM access
    
    Returns:
        Float score between 0.0 and 1.0
    """
    prompt = f"""Rate how specific this query is for static documentation lookup from 0.0 (very vague) to 1.0 (highly specific). Return ONLY the number.

Query: {query}"""
    
    try:
        if qa_generator.gemini_client:
            response = qa_generator.gemini_client.get_response(prompt)
        elif hasattr(qa_generator, 'client'):
            response_obj = qa_generator.client.chat.completions.create(
                model=qa_generator.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            response = response_obj.choices[0].message.content.strip()
        else:
            logger.warning("No LLM client available for static semantic completeness scoring")
            return 0.5
        
        # Extract number from response
        response = response.strip()
        # Remove any quotes
        response = response.strip('"').strip("'")
        # Extract first float number
        match = re.search(r'([0-9]*\.?[0-9]+)', response)
        if match:
            score = float(match.group(1))
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            return score
        else:
            logger.warning(f"Could not parse static semantic completeness score from response: {response}")
            return 0.5
    except Exception as e:
        logger.error(f"Error computing static semantic completeness score: {e}")
        return 0.5


def getSQLPrecisionScore(sql: str) -> float:
    """
    Score SQL query precision based on filtering criteria.
    
    Args:
        sql: SQL query string
    
    Returns:
        Float score between 0.0 and 1.0
    """
    if not sql:
        return 0.0
    
    sql_lower = sql.lower()
    score = 0.0
    
    # +0.3 if SQL contains WHERE
    if 'where' in sql_lower:
        score += 0.3
    
    # +0.3 if SQL filters proposal_index
    if 'proposal_index' in sql_lower and 'where' in sql_lower:
        score += 0.3
    
    # +0.2 if SQL filters network (source_network)
    if 'source_network' in sql_lower and 'where' in sql_lower:
        score += 0.2
    
    # +0.1 if SQL filters onchaininfo_status
    if 'onchaininfo_status' in sql_lower and 'where' in sql_lower:
        score += 0.1
    
    # -0.3 if SQL uses SELECT * without WHERE
    if 'select *' in sql_lower and 'where' not in sql_lower:
        score -= 0.3
    
    # -0.3 if SQL uses LIMIT without WHERE
    if 'limit' in sql_lower and 'where' not in sql_lower:
        score -= 0.3
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


async def compute_retrieval_confidence(
    route: str,
    router_confidence: float,
    static_chunks: Optional[List[Dict[str, Any]]] = None,
    sql_result_count: Optional[int] = None,
    sql_success: Optional[bool] = None,
    hybrid_static_available: Optional[bool] = None,
    hybrid_dynamic_available: Optional[bool] = None,
    file_fallback_available: bool = False,
    is_ambiguous_query: bool = False,
    query: Optional[str] = None,
    sql_query: Optional[List[str]] = None,
    qa_generator = None
) -> tuple[float, Optional[float]]:
    """
    Compute retrieval confidence based on multiple factors.
    
    Args:
        route: The route type (static, dynamic, hybrid, generic)
        router_confidence: Confidence from the router LLM (0.0-1.0)
        static_chunks: List of static chunks with similarity scores
        sql_result_count: Number of rows returned from SQL query
        sql_success: Whether SQL query was successful
        hybrid_static_available: Whether static data is available in hybrid route
        hybrid_dynamic_available: Whether dynamic data is available in hybrid route
        file_fallback_available: Whether file fallback API is available
        is_ambiguous_query: Whether the query is ambiguous
        query: The user query (required for dynamic route)
        sql_query: List of SQL query strings (required for dynamic route)
        qa_generator: QA generator instance (required for dynamic route)
    
    Returns:
        Tuple of (retrieval confidence score (0.0-1.0), semantic_completeness for static route or None)
    """
    base_confidence = router_confidence * 0.25
    
    file_fallback_bonus = 0.05 if file_fallback_available else 0.0
    
    if route == "static":
        semantic_completeness = 0.5
        if query and qa_generator:
            semantic_completeness = await getSemanticCompletenessScoreForStatic(query, qa_generator)
        
        static_similarity = 0.0
        if static_chunks and len(static_chunks) > 0:
            similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in static_chunks]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            static_similarity = (avg_similarity * 0.5 + max_similarity * 0.5)
        
        chunk_count_factor = min(len(static_chunks) / 5.0, 1.0) if static_chunks else 0.0
        
        # Check if chunks are from Polkassembly docs (pa_docs)
        is_polkassembly_docs = False
        if static_chunks and len(static_chunks) > 0:
            for chunk in static_chunks:
                metadata = chunk.get('metadata', {})
                source = metadata.get('source', '').lower() if metadata.get('source') else ''
                title = metadata.get('title', '').lower() if metadata.get('title') else ''
                # Check multiple fields to detect Polkassembly docs
                if ('pa_docs' in source or 
                    'pa_docs' in title or 
                    'pa_docs' in str(metadata).lower() or
                    metadata.get('doc_type', '').lower() == 'pa_docs' or
                    metadata.get('source_type', '').lower() == 'pa_docs'):
                    is_polkassembly_docs = True
                    break
        
        # Adjust weights based on chunk quality
        # If we have good chunks, reduce semantic_completeness weight and increase similarity weight
        has_good_chunks = static_chunks and len(static_chunks) > 0 and static_similarity >= 0.45
        
        if has_good_chunks:
            # When chunks are found with decent similarity, trust the chunks more
            static_confidence = (
                0.30 * router_confidence +
                0.25 * semantic_completeness +  # Fixed at 25%
                0.30 * static_similarity +      # Increased from 0.20
                0.15 * chunk_count_factor      # Increased from 0.10
            )
        else:
            # Default weights when no good chunks
            static_confidence = (
                0.30 * router_confidence +
                0.25 * semantic_completeness +  # Fixed at 25%
                0.30 * static_similarity +      # Increased from 0.20
                0.15 * chunk_count_factor       # Increased from 0.10
            )
        
        # Polkassembly docs bonus: if using Polkassembly docs chunks, boost confidence significantly
        polkassembly_docs_bonus = 0.0
        if is_polkassembly_docs:
            # Polkassembly docs chunks are preferred - boost confidence to ensure answer
            polkassembly_docs_bonus = 0.3
        
        # High similarity bonus: if chunks exist with high similarity, boost confidence significantly
        high_similarity_bonus = 0.0
        if static_chunks and len(static_chunks) > 0 and static_similarity >= 0.6 and not is_polkassembly_docs:
            # High similarity chunks found - boost confidence to ensure answer
            # Only apply if not already using Polkassembly docs (which has its own bonus)
            high_similarity_bonus = 0.3
        
        # Medium similarity bonus: if chunks exist with medium similarity (0.5-0.6), give small bonus
        medium_similarity_bonus = 0.0
        if static_chunks and len(static_chunks) > 0 and 0.5 <= static_similarity < 0.6 and not is_polkassembly_docs:
            # Medium similarity chunks found - small bonus to help reach threshold
            medium_similarity_bonus = 0.15
        
        ambiguity_penalty = 0.0
        if semantic_completeness < 0.45:
            # Only apply penalty if no chunks found or chunks have low similarity
            # If we have good chunks, the query might be vague but we still have relevant data
            if not static_chunks or len(static_chunks) == 0:
                ambiguity_penalty = -0.4
            elif is_polkassembly_docs:
                # Polkassembly docs chunks - no penalty (bonus handles it)
                ambiguity_penalty = 0.0
            elif static_similarity < 0.40:
                # Low similarity chunks - moderate penalty
                ambiguity_penalty = -0.2
            elif static_similarity >= 0.6:
                # Good similarity chunks - no penalty (bonus handles it)
                ambiguity_penalty = 0.0
            elif static_similarity >= 0.5:
                # Medium similarity (0.5-0.6) - small penalty since we have medium_similarity_bonus
                ambiguity_penalty = -0.05
            elif static_similarity >= 0.45:
                # Medium similarity (0.45-0.5) - small penalty since we're using adjusted weights
                ambiguity_penalty = -0.05
            else:
                # Should not reach here, but keep for safety
                ambiguity_penalty = -0.15
        
        final_static_confidence = static_confidence + polkassembly_docs_bonus + high_similarity_bonus + medium_similarity_bonus + ambiguity_penalty
        
        return (max(0.0, min(1.0, final_static_confidence)), semantic_completeness)
    
    elif route == "dynamic":
        semantic_completeness = 0.5
        if query and qa_generator:
            semantic_completeness = await getSemanticCompletenessScore(query, qa_generator)
        
        sql_precision = 0.0
        if sql_query and len(sql_query) > 0:
            combined_sql = ' '.join(sql_query)
            sql_precision = getSQLPrecisionScore(combined_sql)
        
        result_specificity = 0.0
        if sql_success and sql_result_count is not None:
            if sql_result_count > 0:
                result_specificity = min(sql_result_count / 10.0, 1.0)
            else:
                result_specificity = 0.1
        
        # If we have successful SQL results, trust the router confidence more
        # Successful results indicate the query was clear enough to execute
        has_results = sql_success and sql_result_count and sql_result_count > 0
        
        ambiguity_penalty = 0.0
        if is_ambiguous_query:
            ambiguity_penalty = -0.3
        
        # Adjust weights based on whether we have results
        # If we have results, trust router_confidence more (it was right about routing)
        if has_results:
            # High router confidence + results = query was clear enough
            final_confidence = (
                0.50 * router_confidence +
                0.20 * semantic_completeness +
                0.15 * sql_precision +
                0.15 * result_specificity +
                file_fallback_bonus +
                ambiguity_penalty
            )
        else:
            # No results - be more cautious, weight semantic completeness more
            final_confidence = (
                0.35 * router_confidence +
                0.35 * semantic_completeness +
                0.20 * sql_precision +
                0.10 * result_specificity +
                file_fallback_bonus +
                ambiguity_penalty
            )
        
        return (max(0.0, min(1.0, final_confidence)), None)
    
    elif route == "hybrid":
        static_confidence = 0.0
        if hybrid_static_available and static_chunks and len(static_chunks) > 0:
            similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in static_chunks]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            static_confidence = avg_similarity * 0.25
        
        dynamic_confidence = 0.0
        if hybrid_dynamic_available:
            if sql_result_count and sql_result_count > 0:
                row_count_factor = min(sql_result_count / 10.0, 1.0)
                dynamic_confidence = 0.25 + (row_count_factor * 0.15)
            else:
                dynamic_confidence = 0.15
        else:
            dynamic_confidence = 0.05
        
        completeness_bonus = 0.0
        if hybrid_static_available and hybrid_dynamic_available:
            completeness_bonus = 0.1
        
        ambiguity_penalty = 0.0
        if is_ambiguous_query:
            ambiguity_penalty = -0.3
        
        final_confidence = base_confidence + static_confidence + dynamic_confidence + completeness_bonus + file_fallback_bonus + ambiguity_penalty
        
        return (max(0.0, min(1.0, final_confidence)), None)
    
    return (base_confidence, None)

