"""
LangGraph graph definition and runner for Klara query processing pipeline.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from langgraph.graph import StateGraph, END

from .state import KlaraState
from .nodes import (
    safety_node,
    ambiguity_vote_advice_node,
    router_node,
    planner_node,
    static_tools_node,
    dynamic_tools_node,
    hybrid_tools_node,
    generic_tools_node,
    fallback_node,
    answer_node
)
from ..pipeline.utils import log_step
from ..handlers.clarification_handler import detect_and_handle_clarification_response

logger = logging.getLogger(__name__)

_graph = None


def _should_block(state: KlaraState) -> str:
    """Conditional: Check if query should be blocked"""
    if state.get("is_blocked", False):
        return "blocked"
    return "continue"


def _create_blocked_node(state: KlaraState) -> Dict[str, Any]:
    """Node that creates blocked response"""
    result = _create_block_response(state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "follow_up_questions": result["follow_up_questions"],
        "confidence": result["confidence"],
        "processing_time_ms": result["processing_time_ms"],
        "model_used": result["model_used"],
        "search_method": result["search_method"],
        "route": result["route"],
        "route_confidence": result["route_confidence"]
    }


def _should_clarify(state: KlaraState) -> str:
    """Conditional: Check if clarification is needed"""
    if state.get("is_ambiguous", False) and state.get("clarification_result"):
        return "clarify"
    return "route"


def _create_clarification_node(state: KlaraState) -> Dict[str, Any]:
    """Node that creates clarification response"""
    result = _create_clarification_response(state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "follow_up_questions": result["follow_up_questions"],
        "confidence": result["confidence"],
        "processing_time_ms": result["processing_time_ms"],
        "model_used": result["model_used"],
        "search_method": result["search_method"],
        "route": result["route"],
        "route_confidence": result["route_confidence"],
        "clarification_result": state.get("clarification_result")
    }


def _route_to_tools(state: KlaraState) -> str:
    """Conditional: Route to appropriate tools based on route type"""
    route = state.get("route", "static")
    
    if route == "static":
        return "static_tools"
    elif route == "dynamic":
        return "dynamic_tools"
    elif route == "hybrid":
        return "hybrid_tools"
    elif route == "generic":
        return "generic_tools"
    else:
        return "static_tools"


def _static_tools_decision(state: KlaraState) -> str:
    """Conditional: Decide next step after static tools"""
    if state.get("static_validation_passed", False) and state.get("static_qa_result"):
        return "answer"
    return "fallback"


def _dynamic_tools_decision(state: KlaraState) -> str:
    """Conditional: Decide next step after dynamic tools"""
    validator_verdict = state.get("validator_verdict")
    requires_clarification = state.get("requires_clarification", False)
    requires_fallback = state.get("requires_fallback", False)
    result_count = state.get("result_count", 0)
    clarification_result = state.get("clarification_result")
    
    if clarification_result:
        return "clarify"
    elif validator_verdict in ["good", "partial"]:
        return "answer"
    elif validator_verdict == "empty" or result_count == 0 or requires_fallback:
        return "fallback"
    else:
        return "answer"


def _hybrid_tools_decision(state: KlaraState) -> str:
    """Conditional: Decide next step after hybrid tools"""
    static_available = len(state.get("static_chunks", [])) > 0
    dynamic_available = state.get("result_count", 0) > 0
    
    if static_available or dynamic_available:
        return "answer"
    return "fallback"


def _generic_tools_decision(state: KlaraState) -> str:
    """Conditional: Decide next step after generic tools"""
    if state.get("static_qa_result") and state.get("static_qa_result").get("answer"):
        return "answer"
    return "fallback"


def _create_block_response(state: KlaraState) -> Dict[str, Any]:
    """Create response for blocked queries"""
    processing_time = (datetime.now() - state["pipeline_start"]).total_seconds() * 1000
    
    return {
        "answer": state.get("block_message", "Your query was blocked because it violates our content policy."),
        "sources": [],
        "follow_up_questions": [
            "How does Polkadot's governance system work?",
            "What are the benefits of staking DOT tokens?",
            "How do parachains connect to Polkadot?"
        ],
        "confidence": 0.0,
        "context_used": False,
        "model_used": "model_armor",
        "chunks_used": 0,
        "processing_time_ms": processing_time,
        "timestamp": datetime.now().isoformat(),
        "search_method": "model_armor_blocked",
        "route": "blocked",
        "route_confidence": 0.0
    }


def _create_clarification_response(state: KlaraState) -> Dict[str, Any]:
    """Create response for clarification queries"""
    clarification_result = state.get("clarification_result", {})
    processing_time = (datetime.now() - state["pipeline_start"]).total_seconds() * 1000
    
    result = {
        "answer": clarification_result.get("answer", ""),
        "sources": clarification_result.get("sources", []),
        "follow_up_questions": clarification_result.get("follow_up_questions", []),
        "confidence": clarification_result.get("confidence", 0.0),
        "context_used": clarification_result.get("context_used", False),
        "model_used": clarification_result.get("model_used", "unknown"),
        "chunks_used": clarification_result.get("chunks_used", 0),
        "processing_time_ms": processing_time,
        "timestamp": datetime.now().isoformat(),
        "search_method": clarification_result.get("search_method", "clarification"),
        "route": clarification_result.get("route", "ambiguous"),
        "route_confidence": clarification_result.get("route_confidence", 0.0),
        "requires_clarification": True,
        "original_query": state.get("query", "")
    }
    
    return result


def _build_graph() -> StateGraph:
    """Build and compile the LangGraph graph"""
    workflow = StateGraph(KlaraState)
    
    workflow.add_node("safety", safety_node)
    workflow.add_node("blocked_response", _create_blocked_node)
    workflow.add_node("ambiguity_vote_advice", ambiguity_vote_advice_node)
    workflow.add_node("clarification_response", _create_clarification_node)
    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("static_tools", static_tools_node)
    workflow.add_node("dynamic_tools", dynamic_tools_node)
    workflow.add_node("hybrid_tools", hybrid_tools_node)
    workflow.add_node("generic_tools", generic_tools_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("answer_generator", answer_node)
    
    workflow.set_entry_point("safety")
    
    workflow.add_conditional_edges(
        "safety",
        _should_block,
        {
            "blocked": "blocked_response",
            "continue": "ambiguity_vote_advice"
        }
    )
    
    workflow.add_edge("blocked_response", END)
    
    workflow.add_conditional_edges(
        "ambiguity_vote_advice",
        _should_clarify,
        {
            "clarify": "clarification_response",
            "route": "router"
        }
    )
    
    workflow.add_edge("clarification_response", END)
    
    workflow.add_edge("router", "planner")
    
    workflow.add_conditional_edges(
        "planner",
        _route_to_tools,
        {
            "static_tools": "static_tools",
            "dynamic_tools": "dynamic_tools",
            "hybrid_tools": "hybrid_tools",
            "generic_tools": "generic_tools"
        }
    )
    
    workflow.add_conditional_edges(
        "static_tools",
        _static_tools_decision,
        {
            "answer": "answer_generator",
            "fallback": "fallback"
        }
    )
    
    workflow.add_conditional_edges(
        "dynamic_tools",
        _dynamic_tools_decision,
        {
            "answer": "answer_generator",
            "clarify": "clarification_response",
            "fallback": "fallback"
        }
    )
    
    workflow.add_conditional_edges(
        "hybrid_tools",
        _hybrid_tools_decision,
        {
            "answer": "answer_generator",
            "fallback": "fallback"
        }
    )
    
    workflow.add_conditional_edges(
        "generic_tools",
        _generic_tools_decision,
        {
            "answer": "answer_generator",
            "fallback": "fallback"
        }
    )
    
    workflow.add_edge("fallback", "answer_generator")
    workflow.add_edge("answer_generator", END)
    
    return workflow.compile()


def _get_graph():
    """Get or create the compiled graph"""
    global _graph
    if _graph is None:
        try:
            _graph = _build_graph()
        except Exception as e:
            logger.error(f"Failed to build LangGraph: {e}", exc_info=True)
            raise
    return _graph


async def run_langgraph_query(
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
    Run query through LangGraph pipeline.
    Returns same format as processUserQuery for compatibility.
    """
    pipeline_start = datetime.now()
    
    log_step("langgraph_pipeline_start", {
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
        analyzed_query = userMessage
        
        if is_clarification_followup:
            log_step("langgraph_clarification_followup_detected", {
                "original_query": clarification_info.get('original_query', '')[:100],
                "clarification_response": userMessage[:100]
            })
        
        if conversationHistory and qa_generator:
            try:
                analyzed_query = qa_generator.analyze_query_with_context(
                    analyzed_query,
                    conversationHistory
                )
                log_step("langgraph_query_analysis_complete", {
                    "original": userMessage[:100],
                    "analyzed": analyzed_query[:100]
                })
            except Exception as e:
                log_step("langgraph_query_analysis_error", {"error": str(e)}, "error")
        
        initial_state: KlaraState = {
            "query": userMessage,
            "user_id": user_id,
            "conversation_history": conversationHistory,
            "max_chunks": max_chunks,
            "custom_prompt": custom_prompt,
            "static_embedding_manager": static_embedding_manager,
            "dynamic_embedding_manager": dynamic_embedding_manager,
            "router_embedding_manager": router_embedding_manager,
            "qa_generator": qa_generator,
            "guardrail_result": None,
            "is_blocked": False,
            "block_message": None,
            "clarification_info": clarification_info,
            "is_clarification_followup": is_clarification_followup,
            "analyzed_query": analyzed_query,
            "is_ambiguous": False,
            "is_vote_advice_query": False,
            "clarification_needed": False,
            "clarification_result": None,
            "route": None,
            "route_confidence": 0.0,
            "plan": None,
            "static_chunks": [],
            "static_validation_passed": False,
            "static_qa_result": None,
            "sql_queries": [],
            "sql_results": None,
            "validator_verdict": None,
            "validator_reason": None,
            "result_count": 0,
            "connection_error": False,
            "data_fetch_failed": False,
            "requires_clarification": False,
            "requires_fallback": False,
            "fallback_triggered": False,
            "internet_search_result": None,
            "answer": None,
            "sources": [],
            "follow_up_questions": [],
            "confidence": 0.0,
            "processing_time_ms": 0.0,
            "error": None,
            "pipeline_start": pipeline_start,
            "model_used": "unknown",
            "search_method": "unknown",
            "route_metadata": None
        }
        
        graph = _get_graph()
        
        final_node_state = {}
        try:
            async for state in graph.astream(initial_state):
                for node_name, node_state in state.items():
                    log_step("langgraph_node_complete", {
                        "node": node_name,
                        "has_answer": "answer" in node_state if isinstance(node_state, dict) else False
                    }, "debug")
                    if isinstance(node_state, dict):
                        final_node_state.update(node_state)
        except Exception as graph_error:
            logger.error(f"Error during graph execution: {graph_error}", exc_info=True)
            raise Exception(f"Graph execution failed: {graph_error}") from graph_error
        
        if not final_node_state:
            raise Exception("Graph execution did not produce final state")
        
        if "answer" in final_node_state and final_node_state.get("answer"):
            result = {
                "answer": final_node_state.get("answer", ""),
                "sources": final_node_state.get("sources", []),
                "follow_up_questions": final_node_state.get("follow_up_questions", []),
                "confidence": final_node_state.get("confidence", 0.0),
                "context_used": bool(final_node_state.get("sources")),
                "model_used": final_node_state.get("model_used", "unknown"),
                "chunks_used": len(final_node_state.get("static_chunks", [])),
                "processing_time_ms": final_node_state.get("processing_time_ms", 0.0),
                "timestamp": datetime.now().isoformat(),
                "search_method": final_node_state.get("search_method", "unknown"),
                "route": final_node_state.get("route", "unknown"),
                "route_confidence": final_node_state.get("route_confidence", 0.0)
            }
        else:
            answer_result = await answer_node(final_node_state)
            final_node_state.update(answer_result)
            
            result = {
                "answer": final_node_state.get("answer", ""),
                "sources": final_node_state.get("sources", []),
                "follow_up_questions": final_node_state.get("follow_up_questions", []),
                "confidence": final_node_state.get("confidence", 0.0),
                "context_used": bool(final_node_state.get("sources")),
                "model_used": final_node_state.get("model_used", "unknown"),
                "chunks_used": len(final_node_state.get("static_chunks", [])),
                "processing_time_ms": final_node_state.get("processing_time_ms", 0.0),
                "timestamp": datetime.now().isoformat(),
                "search_method": final_node_state.get("search_method", "unknown"),
                "route": final_node_state.get("route", "unknown"),
                "route_confidence": final_node_state.get("route_confidence", 0.0)
            }
        
        if final_node_state.get("is_clarification_followup") and clarification_info:
            result["is_clarification_followup"] = True
            result["original_query"] = clarification_info.get("original_query", "")
        
        log_step("langgraph_pipeline_complete", {
            "route": result.get("route", "unknown"),
            "processing_time_ms": result.get("processing_time_ms", 0.0),
            "success": True
        })
        
        return result
        
    except Exception as e:
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        log_step("langgraph_pipeline_error", {
            "error": str(e),
            "processing_time_ms": processing_time
        }, "error")
        
        from ...core.errors import is_insufficient_quota_error, get_quota_error_message
        
        if is_insufficient_quota_error(e):
            return {
                "answer": get_quota_error_message(),
                "sources": [],
                "follow_up_questions": [],
                "confidence": 0.0,
                "context_used": False,
                "model_used": "error",
                "chunks_used": 0,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat(),
                "search_method": "quota_error",
                "route": "error",
                "route_confidence": 0.0,
                "error": str(e)
            }
        
        return {
            "answer": "I encountered an error processing your query. Please try again.",
            "sources": [],
            "follow_up_questions": [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ],
            "confidence": 0.0,
            "context_used": False,
            "model_used": "error",
            "chunks_used": 0,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat(),
            "search_method": "error",
            "route": "error",
            "route_confidence": 0.0,
            "error": str(e)
        }

