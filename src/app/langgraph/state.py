"""
LangGraph state schema for Klara query processing pipeline.
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class KlaraState(TypedDict, total=False):
    query: str
    messages: List[Dict[str, Any]]
    
    user_id: str
    conversation_history: Optional[List[Dict[str, Any]]]
    max_chunks: int
    custom_prompt: Optional[str]
    
    static_embedding_manager: Any
    dynamic_embedding_manager: Any
    qa_generator: Any
    
    guardrail_result: Optional[Dict[str, Any]]
    is_blocked: bool
    block_message: Optional[str]
    
    clarification_info: Optional[Dict[str, Any]]
    is_clarification_followup: bool
    analyzed_query: str
    
    is_ambiguous: bool
    is_vote_advice_query: bool
    clarification_needed: bool
    clarification_result: Optional[Dict[str, Any]]
    
    route: Optional[str]
    route_confidence: float
    
    plan: Optional[Dict[str, Any]]
    
    static_chunks: List[Dict[str, Any]]
    static_validation_passed: bool
    static_qa_result: Optional[Dict[str, Any]]
    
    sql_queries: List[str]
    sql_results: Optional[Dict[str, Any]]
    validator_verdict: Optional[str]
    validator_reason: Optional[str]
    result_count: int
    connection_error: bool
    data_fetch_failed: bool
    requires_clarification: bool
    requires_fallback: bool
    
    fallback_triggered: bool
    internet_search_result: Optional[Dict[str, Any]]
    
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    follow_up_questions: List[str]
    confidence: float
    processing_time_ms: float
    error: Optional[str]
    
    pipeline_start: datetime
    model_used: str
    search_method: str
    route_metadata: Optional[Dict[str, Any]]
