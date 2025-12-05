"""
Main query processing pipeline orchestration.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ..handlers.clarification import generate_clarification_question
from src.core.errors import is_insufficient_quota_error, get_quota_error_message
from .utils import log_step
from ..handlers.ambiguity import is_query_truly_ambiguous
from ..handlers.clarification_handler import detect_and_handle_clarification_response
from src.core.routing import get_router, RouterDecision
from ..handlers.route_handlers import (
    handle_static_route,
    handle_dynamic_route,
    handle_hybrid_route,
    handle_generic_route
)


async def processUserQuery(
    userMessage: str,
    conversationHistory: Optional[List[Dict[str, Any]]],
    static_embedding_manager,
    dynamic_embedding_manager,
    qa_generator,
    max_chunks: int = 5,
    custom_prompt: Optional[str] = None,
    user_id: str = "default_user",
    router_embedding_manager=None
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
        router_embedding_manager: Manager for router example embeddings (for few-shot retrieval)
        
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
        clarification_info = await detect_and_handle_clarification_response(
            userMessage,
            conversationHistory,
            qa_generator,
            log_step
        )
        
        is_clarification_followup = clarification_info is not None
        is_voting_data = False
        is_ambiguous_query = False
        
        if is_clarification_followup:
            log_step("clarification_followup_detected", {
                "original_query": clarification_info['original_query'],
                "original_route": clarification_info.get('original_route', 'unknown'),
                "original_router_confidence": clarification_info.get('original_router_confidence', 'unknown'),
                "clarification_response": clarification_info['clarification_response']
            })
        
        if is_clarification_followup:
            analyzed_query = userMessage
        else:
            analyzed_query = userMessage
        
        if conversationHistory and qa_generator:
            log_step("query_analysis_start", {})
            try:
                analyzed_query = qa_generator.analyze_query_with_context(
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
                pass
        
        if not is_clarification_followup:
            log_step("ambiguity_check_pre_route_start", {
                "query": analyzed_query[:100]
            })
            is_ambiguous_query = await is_query_truly_ambiguous(analyzed_query, qa_generator, None, conversationHistory)
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
                    log_step=log_step,
                    conversation_history=conversationHistory
                )
                
                clarification_result['route'] = 'ambiguous_pre_route'
                clarification_result['route_confidence'] = 0.0
                clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                clarification_result['original_query'] = userMessage
                
                log_step("pipeline_complete", {
                    "route": "ambiguous_pre_route",
                    "confidence": 0.0,
                    "processing_time_ms": clarification_result['processing_time_ms'],
                    "requires_clarification": True,
                    "success": True
                })
                
                return clarification_result
            else:
                is_ambiguous_query = False
        
        router = get_router(qa_generator, log_step, router_embedding_manager=router_embedding_manager)
        
        if is_clarification_followup:
            log_step("routing_clarification_followup", {
                "is_clarification_followup": is_clarification_followup,
                "query_preview": analyzed_query[:100],
                "note": "Re-routing clarification response with conversation history for context"
            })
        else:
            log_step("routing_start", {
                "is_clarification_followup": is_clarification_followup,
                "query_for_routing": analyzed_query[:100]
            })
        
        decision = await router.route(analyzed_query, conversationHistory)
        route = decision.route.value
        confidence = decision.confidence
        
        log_step("routing_complete", {
            "route": route,
            "confidence": confidence,
            "network": decision.network,
            "proposal_index": decision.proposal_index,
            "needs": decision.needs,
            "is_clarification_followup": is_clarification_followup
        })
        
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        
        if route == "static":
            qa_result = await handle_static_route(
                analyzed_query=analyzed_query,
                userMessage=userMessage,
                static_embedding_manager=static_embedding_manager,
                qa_generator=qa_generator,
                conversationHistory=conversationHistory,
                route=route,
                confidence=confidence,
                max_chunks=max_chunks,
                custom_prompt=custom_prompt,
                user_id=user_id,
                pipeline_start=pipeline_start
            )
            
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
        
        elif route == "dynamic":
            qa_result = await handle_dynamic_route(
                analyzed_query=analyzed_query,
                userMessage=userMessage,
                qa_generator=qa_generator,
                conversationHistory=conversationHistory,
                dynamic_embedding_manager=dynamic_embedding_manager,
                route=route,
                confidence=confidence,
                custom_prompt=custom_prompt,
                user_id=user_id,
                pipeline_start=pipeline_start,
                is_ambiguous_query=is_ambiguous_query,
                is_clarification_followup=is_clarification_followup
            )
            
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
        
        elif route == "hybrid":
            qa_result = await handle_hybrid_route(
                analyzed_query=analyzed_query,
                static_embedding_manager=static_embedding_manager,
                qa_generator=qa_generator,
                conversationHistory=conversationHistory,
                dynamic_embedding_manager=dynamic_embedding_manager,
                route=route,
                confidence=confidence,
                max_chunks=max_chunks,
                custom_prompt=custom_prompt,
                user_id=user_id,
                pipeline_start=pipeline_start
            )
            
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
        
        elif route == "generic":
            qa_result = await handle_generic_route(
                analyzed_query=analyzed_query,
                conversationHistory=conversationHistory,
                qa_generator=qa_generator,
                user_id=user_id,
                log_step=log_step
            )
            
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
        
        if is_insufficient_quota_error(e):
            return {
                'answer': get_quota_error_message(),
                'sources': [],
                'chunks_used': 0,
                'search_method': 'error',
                'route': 'generic',
                'route_confidence': 0.0,
                'processing_time_ms': processing_time,
                'error': str(e)
            }
        
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

