"""
LangGraph node implementations for Klara query processing pipeline.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .state import KlaraState
from ..pipeline.utils import log_step
from ..handlers.ambiguity import is_query_truly_ambiguous
from ..handlers.clarification import generate_clarification_question
from ..handlers.clarification_handler import detect_and_handle_clarification_response
from ..handlers.route_handlers import (
    handle_static_route,
    handle_dynamic_route,
    handle_hybrid_route,
    handle_generic_route
)
from ...core.routing import get_router
from ...safety.bedrock_guardrail import check_with_guardrail_async, generate_user_friendly_block_message
from ...core.errors import is_insufficient_quota_error, get_quota_error_message

logger = logging.getLogger(__name__)

def _ensure_dependencies(state: KlaraState) -> tuple:
    """Auto-initialize dependencies if not in state (for Studio mode)"""
    static_mgr = state.get("static_embedding_manager")
    dynamic_mgr = state.get("dynamic_embedding_manager")
    router_mgr = state.get("router_embedding_manager")
    qa_gen = state.get("qa_generator")
    
    if not static_mgr or not dynamic_mgr or not qa_gen or not router_mgr:
        try:
            from .studio import get_dependencies
            deps = get_dependencies()
            static_mgr = static_mgr or deps.get("static_embedding_manager")
            dynamic_mgr = dynamic_mgr or deps.get("dynamic_embedding_manager")
            router_mgr = router_mgr or deps.get("router_embedding_manager")
            qa_gen = qa_gen or deps.get("qa_generator")
        except Exception as e:
            logger.warning(f"Could not auto-init dependencies: {e}")
    
    return static_mgr, dynamic_mgr, router_mgr, qa_gen


async def safety_node(state: KlaraState) -> Dict[str, Any]:
    """Safety check node - guardrail validation"""
    query = state.get("query", "")
    if not query and state.get("messages"):
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("type", ""))
                if role in ("human", "user"):
                    query = msg.get("content", "")
                    break
            elif hasattr(msg, "content"):
                query = msg.content
                break
        if not query and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                query = last_msg.get("content", str(last_msg))
            else:
                query = str(last_msg)
    
    user_id = state.get("user_id", "studio_user")
    
    init_updates = {
        "query": query,
        "user_id": user_id,
        "analyzed_query": query,
        "pipeline_start": state.get("pipeline_start", datetime.now()),
        "max_chunks": state.get("max_chunks", 5),
    }
    
    log_step("langgraph_safety_node_start", {
        "query_preview": query[:100] if query else "",
        "user_id": user_id
    })
    
    try:
        guardrail_result = await check_with_guardrail_async(query)
        
        is_blocked = guardrail_result.get("status") == "blocked"
        block_message = None
        
        if is_blocked:
            violation_details = guardrail_result.get('violation_details', {})
            reason = guardrail_result.get('reason', 'Content policy violation')
            logger.warning(f"Query blocked by guardrail for user {user_id}: {reason}")
            
            try:
                block_message = await generate_user_friendly_block_message(violation_details, query)
            except Exception as e:
                logger.error(f"Failed to generate user-friendly block message: {e}")
                block_message = "Your query was blocked because it violates our content policy. Please revise your query to comply with our terms of service."
        
        log_step("langgraph_safety_node_complete", {
            "is_blocked": is_blocked,
            "status": guardrail_result.get("status")
        })
        
        return {
            **init_updates,
            "guardrail_result": guardrail_result,
            "is_blocked": is_blocked,
            "block_message": block_message
        }
    except Exception as e:
        logger.error(f"Error in safety node: {e}")
        log_step("langgraph_safety_node_error", {"error": str(e)}, "error")
        return {
            **init_updates,
            "guardrail_result": {"status": "error", "reason": str(e)},
            "is_blocked": False,
            "block_message": None
        }


async def ambiguity_vote_advice_node(state: KlaraState) -> Dict[str, Any]:
    """Ambiguity check and vote advice detection node"""
    query = state.get("query", "")
    log_step("langgraph_ambiguity_vote_advice_node_start", {
        "query_preview": query[:100] if query else "",
        "is_clarification_followup": state.get("is_clarification_followup", False)
    })
    
    static_mgr, dynamic_mgr, _, qa_gen = _ensure_dependencies(state)
    
    try:
        analyzed_query = state.get("analyzed_query") or query
        is_ambiguous = False
        is_vote_advice_query = False
        clarification_needed = False
        clarification_result = None
        
        if not state.get("is_clarification_followup", False) and qa_gen:
            is_ambiguous = await is_query_truly_ambiguous(
                analyzed_query,
                qa_gen,
                None,
                state.get("conversation_history")
            )
            
            if is_ambiguous:
                log_step("langgraph_ambiguous_query_detected", {
                    "query": analyzed_query,
                    "note": "Query is ambiguous - will return clarification"
                })
                
                clarification_result = await generate_clarification_question(
                    query=state.get("query", ""),
                    route=None,
                    router_confidence=0.0,
                    qa_generator=qa_gen,
                    log_step=log_step,
                    conversation_history=state.get("conversation_history")
                )
                
                clarification_result['route'] = 'ambiguous_pre_route'
                clarification_result['route_confidence'] = 0.0
                clarification_result['original_query'] = state.get("query", "")
                clarification_needed = True
        
        query_lower = analyzed_query.lower()
        vote_advice_keywords = [
            "should i vote", "how should i vote", "vote yes or no",
            "recommend voting", "voting advice", "should i support"
        ]
        is_vote_advice_query = any(keyword in query_lower for keyword in vote_advice_keywords)
        
        log_step("langgraph_ambiguity_vote_advice_node_complete", {
            "is_ambiguous": is_ambiguous,
            "is_vote_advice_query": is_vote_advice_query,
            "clarification_needed": clarification_needed
        })
        
        return {
            "is_ambiguous": is_ambiguous,
            "is_vote_advice_query": is_vote_advice_query,
            "clarification_needed": clarification_needed,
            "clarification_result": clarification_result
        }
    except Exception as e:
        logger.error(f"Error in ambiguity/vote advice node: {e}")
        log_step("langgraph_ambiguity_vote_advice_node_error", {"error": str(e)}, "error")
        return {
            "is_ambiguous": False,
            "is_vote_advice_query": False,
            "clarification_needed": False,
            "clarification_result": None
        }


async def router_node(state: KlaraState) -> Dict[str, Any]:
    """Router node - determines query route"""
    log_step("langgraph_router_node_start", {
        "query_preview": state.get("analyzed_query") or state.get("query", "")[:100]
    })
    
    static_mgr, dynamic_mgr, router_mgr, qa_gen = _ensure_dependencies(state)
    
    try:
        analyzed_query = state.get("analyzed_query") or state.get("query", "")
        
        router = get_router(qa_gen, log_step, router_embedding_manager=router_mgr)
        decision = await router.route(
            analyzed_query,
            state.get("conversation_history")
        )
        
        route = decision.route.value
        confidence = decision.confidence
        
        log_step("langgraph_router_node_complete", {
            "route": route,
            "confidence": confidence,
            "network": decision.network,
            "proposal_index": decision.proposal_index,
            "needs": decision.needs
        })
        
        return {
            "route": route,
            "route_confidence": confidence,
            "network": decision.network,
            "proposal_index": decision.proposal_index,
            "needs": decision.needs
        }
    except Exception as e:
        logger.error(f"Error in router node: {e}")
        log_step("langgraph_router_node_error", {"error": str(e)}, "error")
        return {
            "route": "static",
            "route_confidence": 0.0
        }


async def planner_node(state: KlaraState) -> Dict[str, Any]:
    """Planner node - creates execution plan"""
    log_step("langgraph_planner_node_start", {
        "route": state.get("route"),
        "query_preview": state.get("analyzed_query") or state.get("query", "")[:100]
    })
    
    static_mgr, dynamic_mgr, _, qa_gen = _ensure_dependencies(state)
    
    try:
        route = state.get("route", "static")
        analyzed_query = state.get("analyzed_query") or state.get("query", "")
        plan = {}
        
        if route in ["dynamic", "hybrid"] and qa_gen:
            selected_table = qa_gen._determine_table_from_query(analyzed_query)
            plan = {
                "table": selected_table,
                "needs_sql": True,
                "needs_static_chunks": route == "hybrid"
            }
        elif route == "static":
            plan = {
                "needs_static_chunks": True,
                "needs_sql": False
            }
        else:
            plan = {
                "needs_static_chunks": False,
                "needs_sql": False
            }
        
        log_step("langgraph_planner_node_complete", {
            "route": route,
            "plan": plan
        })
        
        return {
            "plan": plan
        }
    except Exception as e:
        logger.error(f"Error in planner node: {e}")
        log_step("langgraph_planner_node_error", {"error": str(e)}, "error")
        return {
            "plan": {}
        }


async def static_tools_node(state: KlaraState) -> Dict[str, Any]:
    """Static tools node - embedding search and processing"""
    log_step("langgraph_static_tools_node_start", {
        "route": state.get("route"),
        "query_preview": state.get("analyzed_query") or state.get("query", "")[:100]
    })
    
    static_mgr, dynamic_mgr, _, qa_gen = _ensure_dependencies(state)
    
    logger.info(f"[STATIC] deps: static_mgr={type(static_mgr).__name__}, qa_gen={type(qa_gen).__name__}")
    if not static_mgr or not qa_gen:
        err = f"Missing: static_mgr={static_mgr is not None}, qa_gen={qa_gen is not None}"
        logger.error(f"[STATIC] {err}")
        return {"static_chunks": [], "static_validation_passed": False, "static_qa_result": {"error": err}}
    
    try:
        analyzed_query = state.get("analyzed_query") or state.get("query", "")
        route = state.get("route", "static")
        confidence = state.get("route_confidence", 0.0)
        
        logger.info(f"[STATIC] Calling handle_static_route: query='{analyzed_query[:50]}...'")
        qa_result = await handle_static_route(
            analyzed_query=analyzed_query,
            userMessage=state.get("query", ""),
            static_embedding_manager=static_mgr,
            qa_generator=qa_gen,
            conversationHistory=state.get("conversation_history"),
            route=route,
            confidence=confidence,
            max_chunks=state.get("max_chunks", 5),
            custom_prompt=state.get("custom_prompt"),
            user_id=state.get("user_id", "studio_user"),
            pipeline_start=state.get("pipeline_start", datetime.now())
        )
        
        logger.info(f"[STATIC] Got result: keys={list(qa_result.keys())}")
        static_chunks = qa_result.get("sources", [])
        static_validation_passed = qa_result.get("answer") is not None and len(qa_result.get("answer", "")) > 0
        
        logger.info(f"[STATIC] chunks={len(static_chunks)}, has_answer={static_validation_passed}")
        
        return {
            "static_chunks": static_chunks,
            "static_validation_passed": static_validation_passed,
            "static_qa_result": qa_result
        }
    except Exception as e:
        import traceback
        logger.error(f"[STATIC] Error: {e}\n{traceback.format_exc()}")
        return {
            "static_chunks": [],
            "static_validation_passed": False,
            "static_qa_result": {"error": str(e), "traceback": traceback.format_exc()}
        }


async def dynamic_tools_node(state: KlaraState) -> Dict[str, Any]:
    """Dynamic tools node - SQL execution and data retrieval"""
    log_step("langgraph_dynamic_tools_node_start", {
        "route": state.get("route"),
        "query_preview": state.get("analyzed_query") or state.get("query", "")[:100]
    })
    
    static_mgr, dynamic_mgr, _, qa_gen = _ensure_dependencies(state)
    
    try:
        analyzed_query = state.get("analyzed_query") or state.get("query", "")
        route = state.get("route", "dynamic")
        confidence = state.get("route_confidence", 0.0)
        is_ambiguous_query = state.get("is_ambiguous", False)
        is_clarification_followup = state.get("is_clarification_followup", False)
        
        qa_result = await handle_dynamic_route(
            analyzed_query=analyzed_query,
            userMessage=state.get("query", ""),
            qa_generator=qa_gen,
            conversationHistory=state.get("conversation_history"),
            dynamic_embedding_manager=dynamic_mgr,
            route=route,
            confidence=confidence,
            custom_prompt=state.get("custom_prompt"),
            user_id=state.get("user_id", "studio_user"),
            pipeline_start=state.get("pipeline_start", datetime.now()),
            is_ambiguous_query=is_ambiguous_query,
            is_clarification_followup=is_clarification_followup
        )
        
        validator_verdict = qa_result.get('validator_verdict')
        validator_reason = qa_result.get('validator_reason')
        requires_clarification = qa_result.get('requires_clarification', False)
        requires_fallback = qa_result.get('requires_fallback', False)
        result_count = qa_result.get('result_count', 0)
        connection_error = qa_result.get('connection_error', False)
        data_fetch_failed = connection_error or not qa_result.get('success', False) or qa_result.get('error') is not None
        
        sql_queries = qa_result.get('sql_queries', [])
        if not sql_queries and qa_result.get('sql_query'):
            sql_queries = [qa_result.get('sql_query')]
        
        clarification_result = None
        if validator_verdict == "bad" and requires_clarification and not state.get("is_ambiguous", False):
            try:
                clarification_result = await generate_clarification_question(
                    query=state.get("query", ""),
                    route=route,
                    router_confidence=confidence,
                    qa_generator=qa_gen,
                    log_step=log_step,
                    conversation_history=state.get("conversation_history")
                )
                clarification_result['route'] = route
                clarification_result['route_confidence'] = confidence
                clarification_result['original_query'] = state.get("query", "")
                clarification_result['validator_verdict'] = validator_verdict
                clarification_result['validator_reason'] = validator_reason
            except Exception as e:
                logger.error(f"Error generating clarification in dynamic tools: {e}")
        
        log_step("langgraph_dynamic_tools_node_complete", {
            "validator_verdict": validator_verdict,
            "result_count": result_count,
            "requires_clarification": requires_clarification,
            "requires_fallback": requires_fallback,
            "has_clarification_result": clarification_result is not None
        })
        
        return {
            "sql_queries": sql_queries,
            "sql_results": qa_result,
            "validator_verdict": validator_verdict,
            "validator_reason": validator_reason,
            "result_count": result_count,
            "connection_error": connection_error,
            "data_fetch_failed": data_fetch_failed,
            "requires_clarification": requires_clarification,
            "requires_fallback": requires_fallback,
            "clarification_result": clarification_result
        }
    except Exception as e:
        logger.error(f"Error in dynamic tools node: {e}")
        log_step("langgraph_dynamic_tools_node_error", {"error": str(e)}, "error")
        return {
            "sql_queries": [],
            "sql_results": None,
            "validator_verdict": None,
            "validator_reason": None,
            "result_count": 0,
            "connection_error": False,
            "data_fetch_failed": True,
            "requires_clarification": False,
            "requires_fallback": True
        }


async def hybrid_tools_node(state: KlaraState) -> Dict[str, Any]:
    """Hybrid tools node - combines static and dynamic processing"""
    log_step("langgraph_hybrid_tools_node_start", {
        "query_preview": state.get("analyzed_query") or state.get("query", "")[:100]
    })
    
    static_mgr, dynamic_mgr, _, qa_gen = _ensure_dependencies(state)
    
    try:
        analyzed_query = state.get("analyzed_query") or state.get("query", "")
        route = state.get("route", "hybrid")
        confidence = state.get("route_confidence", 0.0)
        
        qa_result = await handle_hybrid_route(
            analyzed_query=analyzed_query,
            static_embedding_manager=static_mgr,
            qa_generator=qa_gen,
            conversationHistory=state.get("conversation_history"),
            dynamic_embedding_manager=dynamic_mgr,
            route=route,
            confidence=confidence,
            max_chunks=state.get("max_chunks", 5),
            custom_prompt=state.get("custom_prompt"),
            user_id=state.get("user_id", "studio_user"),
            pipeline_start=state.get("pipeline_start", datetime.now())
        )
        
        static_chunks = qa_result.get("sources", [])
        validator_verdict = qa_result.get('validator_verdict')
        result_count = qa_result.get('result_count', 0)
        
        log_step("langgraph_hybrid_tools_node_complete", {
            "chunks_count": len(static_chunks),
            "validator_verdict": validator_verdict,
            "result_count": result_count
        })
        
        return {
            "static_chunks": static_chunks,
            "static_qa_result": qa_result,
            "sql_results": qa_result,
            "validator_verdict": validator_verdict,
            "result_count": result_count
        }
    except Exception as e:
        logger.error(f"Error in hybrid tools node: {e}")
        log_step("langgraph_hybrid_tools_node_error", {"error": str(e)}, "error")
        return {
            "static_chunks": [],
            "static_qa_result": None,
            "sql_results": None,
            "validator_verdict": None,
            "result_count": 0
        }


async def generic_tools_node(state: KlaraState) -> Dict[str, Any]:
    """Generic tools node - handles greetings and casual queries"""
    log_step("langgraph_generic_tools_node_start", {
        "query_preview": state.get("analyzed_query") or state.get("query", "")[:100]
    })
    
    static_mgr, dynamic_mgr, _, qa_gen = _ensure_dependencies(state)
    
    try:
        analyzed_query = state.get("analyzed_query") or state.get("query", "")
        
        from ..handlers.greeting import handle_generic_query_llm
        
        qa_result = await handle_generic_query_llm(
            query=analyzed_query,
            conversation_history=state.get("conversation_history"),
            qa_generator=qa_gen,
            log_step=log_step
        )
        
        log_step("langgraph_generic_tools_node_complete", {
            "has_answer": bool(qa_result.get("answer"))
        })
        
        return {
            "static_qa_result": qa_result
        }
    except Exception as e:
        logger.error(f"Error in generic tools node: {e}")
        log_step("langgraph_generic_tools_node_error", {"error": str(e)}, "error")
        return {
            "static_qa_result": None
        }


async def fallback_node(state: KlaraState) -> Dict[str, Any]:
    """Fallback node - internet search fallback"""
    log_step("langgraph_fallback_node_start", {
        "route": state.get("route"),
        "query_preview": state.get("analyzed_query") or state.get("query", "")[:100]
    })
    
    static_mgr, dynamic_mgr, _, qa_gen = _ensure_dependencies(state)
    
    try:
        analyzed_query = state.get("analyzed_query") or state.get("query", "")
        route = state.get("route", "static")
        
        from ..handlers.internet_fallback import generate_internet_search_response
        
        sql_query_context = None
        if state.get("sql_queries"):
            sql_query_context = state["sql_queries"][0]
        elif state.get("sql_results") and state["sql_results"].get("sql_query"):
            sql_query_context = state["sql_results"].get("sql_query")
        
        internet_result = await generate_internet_search_response(
            query=analyzed_query,
            qa_generator=qa_gen,
            log_step=log_step,
            route=route,
            conversation_history=state.get("conversation_history"),
            sql_query=sql_query_context,
            validator_reason=state.get("validator_reason"),
            connection_error=state.get("connection_error", False),
            data_fetch_failed=state.get("data_fetch_failed", False)
        )
        
        log_step("langgraph_fallback_node_complete", {
            "has_answer": bool(internet_result.get("answer"))
        })
        
        return {
            "fallback_triggered": True,
            "internet_search_result": internet_result
        }
    except Exception as e:
        logger.error(f"Error in fallback node: {e}")
        log_step("langgraph_fallback_node_error", {"error": str(e)}, "error")
        return {
            "fallback_triggered": True,
            "internet_search_result": {
                "answer": "I encountered an error while trying to answer your question. Please try again later.",
                "sources": [],
                "confidence": 0.3
            }
        }


async def answer_node(state: KlaraState) -> Dict[str, Any]:
    """Answer node - final answer generation and formatting"""
    log_step("langgraph_answer_node_start", {
        "route": state.get("route")
    })
    
    try:
        pipeline_start = state.get("pipeline_start", datetime.now())
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        
        answer = None
        sources = []
        follow_up_questions = []
        confidence = 0.0
        model_used = "unknown"
        search_method = "unknown"
        
        route = state.get("route", "static")
        result = None
        
        if route == "dynamic" and state.get("sql_results"):
            result = state["sql_results"]
            answer = result.get("natural_response") or result.get("answer", "")
            search_method = result.get("search_method", "dynamic")
            if not answer and result.get("error"):
                answer = f"I encountered an error: {result.get('error')}"
        elif route in ["static", "hybrid"] and state.get("static_qa_result"):
            result = state["static_qa_result"]
            answer = result.get("answer", "")
            search_method = result.get("search_method", "static")
        elif state.get("internet_search_result"):
            result = state["internet_search_result"]
            answer = result.get("answer", "")
            search_method = result.get("search_method", "internet_fallback")
        elif state.get("sql_results"):
            result = state["sql_results"]
            answer = result.get("natural_response") or result.get("answer", "")
            search_method = result.get("search_method", "dynamic")
        elif state.get("static_qa_result"):
            result = state["static_qa_result"]
            answer = result.get("answer", "")
            search_method = result.get("search_method", "static")
        
        if result:
            sources = result.get("sources", [])
            follow_up_questions = result.get("follow_up_questions", [])
            confidence = result.get("confidence", 0.0)
            model_used = result.get("model_used", "unknown")
        else:
            answer = "I'm having trouble processing your query. Please try again or rephrase your question."
            follow_up_questions = [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ]
            confidence = 0.0
            model_used = "fallback"
            search_method = "error"
        
        if not answer:
            answer = "I couldn't generate a response. Please try rephrasing your question."
        
        if not follow_up_questions:
            follow_up_questions = [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ]
        
        log_step("langgraph_answer_node_complete", {
            "answer_length": len(answer) if answer else 0,
            "sources_count": len(sources),
            "confidence": confidence,
            "search_method": search_method
        })
        
        return {
            "answer": answer,
            "sources": sources,
            "follow_up_questions": follow_up_questions,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "model_used": model_used,
            "search_method": search_method
        }
    except Exception as e:
        logger.error(f"Error in answer node: {e}")
        log_step("langgraph_answer_node_error", {"error": str(e)}, "error")
        pipeline_start = state.get("pipeline_start", datetime.now())
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        return {
            "answer": "I encountered an error processing your query. Please try again.",
            "sources": [],
            "follow_up_questions": [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ],
            "confidence": 0.0,
            "processing_time_ms": processing_time,
            "model_used": "error",
            "search_method": "error",
            "error": str(e)
        }

