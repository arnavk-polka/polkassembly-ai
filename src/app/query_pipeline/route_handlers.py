"""
Route-specific handlers for processing queries.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ..clarification import generate_clarification_question
from ..internet_fallback import generate_internet_search_response
from .utils import log_step, _get_reranker
from .validation import validate_static_answer_with_llm
from ..greeting import handle_generic_query_llm


async def handle_static_route(
    analyzed_query: str,
    userMessage: str,
    static_embedding_manager,
    qa_generator,
    conversationHistory: Optional[List[Dict[str, Any]]],
    route: str,
    confidence: float,
    max_chunks: int,
    custom_prompt: Optional[str],
    user_id: str,
    pipeline_start: datetime
) -> Dict[str, Any]:
    """Handle static route queries."""
    log_step("static_route_start", {})
    
    initial_chunks_to_retrieve = max(max_chunks * 10, 50)
    static_chunks = static_embedding_manager.search_similar_chunks(
        query=analyzed_query,
        n_results=initial_chunks_to_retrieve
    )
    
    from ...core.reranking.semantic_reranker import get_reranker
    from ...core.reranking.chunks_reranker import keyword_filter, final_rerank
    
    reranker = get_reranker()
    
    static_chunks = keyword_filter(analyzed_query, static_chunks)
    
    if reranker:
        static_chunks = reranker.rerank(analyzed_query, static_chunks)
    
    static_chunks = final_rerank(analyzed_query, static_chunks)
    
    static_chunks = static_chunks[:max_chunks]
    log_step("static_retrieval_complete", {
        "chunks_count": len(static_chunks)
    })
    
    qa_result = None
    if static_chunks:
        qa_result = await qa_generator.generate_answer(
            query=analyzed_query,
            chunks=static_chunks,
            custom_prompt=custom_prompt,
            user_id=user_id,
            conversation_history=conversationHistory,
            route=route,
            route_confidence=confidence
        )
        
        validator_passed = await validate_static_answer_with_llm(
            query=userMessage,
            answer=qa_result.get('answer', ''),
            conversation_history=conversationHistory,
            qa_generator=qa_generator,
            log_step=log_step
        )
        
        if validator_passed:
            log_step("static_route_complete", {
                "chunks_used": qa_result.get('chunks_used', 0),
                "search_method": qa_result.get('search_method', 'unknown'),
                "validator_passed": True
            })
        else:
            log_step("static_validator_rejected", {
                "note": "LLM validator flagged static answer as irrelevant",
                "validator_passed": False,
                "rejected_answer": qa_result.get('answer', '')
            })
            qa_result = None
    
    if not qa_result:
        fallback_reason = "no_chunks" if not static_chunks else "validator_rejected"
        log_step("static_fallback_triggered", {
            "route": "static",
            "fallback_reason": fallback_reason
        })
        
        internet_result = await generate_internet_search_response(
            query=analyzed_query,
            qa_generator=qa_generator,
            log_step=log_step,
            route=route,
            conversation_history=conversationHistory
        )
        
        internet_result['route'] = route
        internet_result['route_confidence'] = confidence
        internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
        
        log_step("pipeline_complete", {
            "route": route,
            "confidence": confidence,
            "processing_time_ms": internet_result['processing_time_ms'],
            "internet_fallback": True,
            "success": True
        })
        
        return internet_result
    
    return qa_result


async def handle_dynamic_route(
    analyzed_query: str,
    userMessage: str,
    qa_generator,
    conversationHistory: Optional[List[Dict[str, Any]]],
    dynamic_embedding_manager,
    route: str,
    confidence: float,
    custom_prompt: Optional[str],
    user_id: str,
    pipeline_start: datetime,
    is_ambiguous_query: bool,
    is_clarification_followup: bool
) -> Dict[str, Any]:
    """Handle dynamic route queries."""
    log_step("dynamic_route_start", {})
    
    selected_table = qa_generator._determine_table_from_query(analyzed_query)
    log_step("table_selection_check", {
        "selected_table": selected_table
    })
    
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
    
    validator_verdict = qa_result.get('validator_verdict')
    validator_reason = qa_result.get('validator_reason')
    requires_clarification = qa_result.get('requires_clarification', False)
    requires_fallback = qa_result.get('requires_fallback', False)
    result_count = qa_result.get('result_count', 0)
    connection_error = qa_result.get('connection_error', False)
    data_fetch_failed = connection_error or not qa_result.get('success', False) or qa_result.get('error') is not None
    
    decision = None
    
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
                log_step=log_step,
                conversation_history=conversationHistory
            )
            
            clarification_result['route'] = route
            clarification_result['route_confidence'] = confidence
            clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
            clarification_result['original_query'] = userMessage
            clarification_result['validator_verdict'] = validator_verdict
            clarification_result['validator_reason'] = validator_reason
            
            log_step("pipeline_complete", {
                "route": route,
                "confidence": confidence,
                "processing_time_ms": clarification_result['processing_time_ms'],
                "requires_clarification": True,
                "validator_verdict": validator_verdict,
                "success": True
            })
            
            return clarification_result
        else:
            decision = "ANSWER"
    
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
        
        sql_query_context = qa_result.get('sql_query') or (qa_result.get('sql_queries', [None])[0] if qa_result.get('sql_queries') else None)
        data_fetch_failed = not qa_result.get('success', False) or qa_result.get('error') is not None
        internet_result = await generate_internet_search_response(
            query=analyzed_query,
            qa_generator=qa_generator,
            log_step=log_step,
            route=route,
            sql_query=sql_query_context,
            validator_reason=validator_reason,
            data_fetch_failed=data_fetch_failed
        )
        
        internet_result['route'] = route
        internet_result['route_confidence'] = confidence
        internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
        internet_result['validator_verdict'] = validator_verdict
        internet_result['validator_reason'] = validator_reason
        
        log_step("pipeline_complete", {
            "route": route,
            "confidence": confidence,
            "processing_time_ms": internet_result['processing_time_ms'],
            "internet_fallback": True,
            "validator_verdict": validator_verdict,
            "success": True
        })
        
        return internet_result
    
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
    
    else:
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
                    log_step=log_step,
                    conversation_history=conversationHistory
                )
                
                clarification_result['route'] = route
                clarification_result['route_confidence'] = confidence
                clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                clarification_result['original_query'] = userMessage
                clarification_result['validator_verdict'] = validator_verdict
                clarification_result['validator_reason'] = validator_reason
                
                log_step("pipeline_complete", {
                    "route": route,
                    "confidence": confidence,
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
            
            sql_query_context = qa_result.get('sql_query') or (qa_result.get('sql_queries', [None])[0] if qa_result.get('sql_queries') else None)
            data_fetch_failed = connection_error or not qa_result.get('success', False) or qa_result.get('error') is not None
            internet_result = await generate_internet_search_response(
                query=analyzed_query,
                qa_generator=qa_generator,
                log_step=log_step,
                route=route,
                conversation_history=conversationHistory,
                sql_query=sql_query_context,
                validator_reason=validator_reason,
                connection_error=connection_error,
                data_fetch_failed=data_fetch_failed
            )
            
            internet_result['route'] = route
            internet_result['route_confidence'] = confidence
            internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
            internet_result['validator_verdict'] = validator_verdict
            internet_result['validator_reason'] = validator_reason
            
            log_step("pipeline_complete", {
                "route": route,
                "confidence": confidence,
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
    
    log_step("dynamic_route_decision", {
        "validator_verdict": validator_verdict,
        "validator_reason": validator_reason,
        "result_count": result_count,
        "requires_clarification": requires_clarification,
        "requires_fallback": requires_fallback,
        "decision": decision
    })
    qa_result['route'] = route
    qa_result['route_confidence'] = confidence
    
    return qa_result


async def handle_hybrid_route(
    analyzed_query: str,
    static_embedding_manager,
    qa_generator,
    conversationHistory: Optional[List[Dict[str, Any]]],
    dynamic_embedding_manager,
    route: str,
    confidence: float,
    max_chunks: int,
    custom_prompt: Optional[str],
    user_id: str,
    pipeline_start: datetime
) -> Dict[str, Any]:
    """Handle hybrid route queries."""
    log_step("hybrid_route_start", {})
    
    initial_chunks_to_retrieve = max(max_chunks * 2, 10)
    static_chunks = static_embedding_manager.search_similar_chunks(
        query=analyzed_query,
        n_results=initial_chunks_to_retrieve
    )
    from ...core.reranking.chunks_reranker import rerank_static_chunks, final_rerank
    static_chunks = rerank_static_chunks(query=analyzed_query, static_chunks=static_chunks)
    
    reranker = _get_reranker()
    if reranker:
        static_chunks = reranker.rerank(analyzed_query, static_chunks, top_k=max_chunks)
    
    static_chunks = final_rerank(analyzed_query, static_chunks)
    
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
    result_count = qa_result.get('result_count', 0)
    
    if not hybrid_static_available and not hybrid_dynamic_available:
        log_step("hybrid_fallback_triggered", {
            "route": "hybrid",
            "fallback_reason": "no_static_or_dynamic_data",
            "static_available": hybrid_static_available,
            "dynamic_available": hybrid_dynamic_available,
            "result_count": result_count
        })
        
        sql_query_context = qa_result.get('sql_query') or (qa_result.get('sql_queries', [None])[0] if qa_result.get('sql_queries') else None)
        data_fetch_failed = not qa_result.get('success', False) or qa_result.get('error') is not None
        internet_result = await generate_internet_search_response(
            query=analyzed_query,
            qa_generator=qa_generator,
            log_step=log_step,
            route=route,
            conversation_history=conversationHistory,
            sql_query=sql_query_context,
            validator_reason=qa_result.get('validator_reason'),
            data_fetch_failed=data_fetch_failed
        )
        
        internet_result['route'] = route
        internet_result['route_confidence'] = confidence
        internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
        
        log_step("pipeline_complete", {
            "route": route,
            "confidence": confidence,
            "processing_time_ms": internet_result['processing_time_ms'],
            "internet_fallback": True,
            "success": True
        })
        
        return internet_result
    
    log_step("hybrid_route_complete", {
        "chunks_used": qa_result.get('chunks_used', 0),
        "search_method": qa_result.get('search_method', 'unknown'),
        "hybrid_static_available": hybrid_static_available,
        "hybrid_dynamic_available": hybrid_dynamic_available
    })
    
    return qa_result


async def handle_generic_route(
    analyzed_query: str,
    conversationHistory: Optional[List[Dict[str, Any]]],
    qa_generator,
    user_id: str,
    log_step
) -> Dict[str, Any]:
    """Handle generic route queries."""
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
    
    return qa_result

