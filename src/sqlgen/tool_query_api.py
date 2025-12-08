import json
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from .tools.query_processor import ToolBasedQueryProcessor, create_tool_processor
from .governance.response_generator import generate_natural_response
from .governance.result_processor import format_amount_by_asset_id, add_proposal_links
from .utils.formatting import format_number_for_prompt
from .governance.query2sql import Query2SQL

try:
    from ..integrations.slack_bot import SlackBot
except ImportError:
    SlackBot = None

logger = logging.getLogger(__name__)


def send_error_to_slack(query: str, error: str, error_source: str = "ToolQuery") -> None:
    if not SlackBot:
        return
    
    try:
        slack_bot = SlackBot()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        error_context = {
            "query": query,
            "timestamp": timestamp,
            "error_source": error_source,
            "error_details": str(error)
        }
        
        slack_bot.post_error_to_slack(
            "Tool-based query processing failed", 
            context=error_context
        )
    except Exception as slack_error:
        logger.warning(f"Failed to send error notification to Slack: {slack_error}")


def ask_question_with_tools(
    question: str, 
    conversation_history: Optional[List[Dict[str, Any]]] = None, 
    table: Optional[str] = None, 
    embedding_manager=None,
    use_fallback: bool = True
) -> dict:
    """
    Process a natural language question using tool-based SQL generation.
    Falls back to LLM-based SQL generation if tools fail and use_fallback is True.
    
    Args:
        question: Natural language question
        conversation_history: Previous conversation context
        table: Target table (governance_data or voting_data)
        embedding_manager: Optional embedding manager
        use_fallback: Whether to fallback to LLM SQL generation if tools fail
        
    Returns:
        dict: Response containing results and natural language answer
    """
    
    if table == 'voting_data':
        from .voting.vote_query2sql import VoteQuery2SQL
        logger.info("Voting data table - using VoteQuery2SQL directly")
        processor = VoteQuery2SQL()
        return processor.process_query(question, conversation_history)
    
    try:
        tool_processor = create_tool_processor(embedding_manager)
        tool_result = tool_processor.process_query(question, conversation_history)
        
        if tool_result.get("success") and tool_result.get("result_count", 0) > 0:
            logger.info(f"Tool-based query successful: {tool_result.get('tool_used')}")
            
            results = tool_result.get("results", [])
            results = format_amount_by_asset_id(results)
            results = add_proposal_links(results)
            tool_result["results"] = results
            
            try:
                from openai import OpenAI
                from src.core.gemini_client import GeminiClient
                
                openai_client = None
                gemini_client = None
                
                openai_key = os.getenv('OPENAI_API_KEY')
                if openai_key:
                    openai_client = OpenAI(api_key=openai_key)
                
                try:
                    gemini_client = GeminiClient()
                except:
                    pass
                
                sql_query = tool_result.get("sql_queries", [""])[0]
                columns = tool_result.get("columns", [])
                
                natural_response = generate_natural_response(
                    question, sql_query, results, columns, conversation_history,
                    gemini_client, openai_client, format_number_for_prompt
                )
                tool_result["natural_response"] = natural_response
                
            except Exception as e:
                logger.warning(f"Failed to generate natural response for tool result: {e}")
                tool_result["natural_response"] = _generate_simple_response(question, results)
            
            return {
                "original_query": question,
                "sql_query": tool_result.get("sql_queries", [""])[0],
                "sql_queries": tool_result.get("sql_queries", []),
                "result_count": tool_result.get("result_count", 0),
                "results": tool_result.get("results", []),
                "columns": tool_result.get("columns", []),
                "natural_response": tool_result.get("natural_response", ""),
                "success": True,
                "tool_used": tool_result.get("tool_used"),
                "search_method": "tool_based",
                "requires_fallback": False,
                "requires_clarification": False,
                "validator_verdict": None,
                "validator_reason": None
            }
        
        if tool_result.get("tool_fallback_needed") and use_fallback:
            logger.info("Tool-based query returned no results or failed, falling back to LLM SQL generation")
            return _fallback_to_llm_sql(question, conversation_history, embedding_manager)
        
        if not tool_result.get("success"):
            error = tool_result.get("error", "Unknown error")
            error_type = tool_result.get("error_type", "unknown")
            
            if error_type == "no_tool_match" and use_fallback:
                logger.info("No matching tool found, falling back to LLM SQL generation")
                return _fallback_to_llm_sql(question, conversation_history, embedding_manager)
            
            return {
                "original_query": question,
                "sql_query": None,
                "sql_queries": [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": "",
                "success": False,
                "error": error,
                "error_type": error_type,
                "tool_used": tool_result.get("tool_used"),
                "search_method": "tool_based",
                "requires_fallback": True,
                "validator_verdict": None,
                "validator_reason": None
            }
        
        return {
            "original_query": question,
            "sql_query": tool_result.get("sql_queries", [""])[0] if tool_result.get("sql_queries") else None,
            "sql_queries": tool_result.get("sql_queries", []),
            "result_count": 0,
            "results": [],
            "columns": [],
            "natural_response": "",
            "success": False,
            "error": "no_results",
            "tool_used": tool_result.get("tool_used"),
            "search_method": "tool_based",
            "requires_fallback": True,
            "validator_verdict": None,
            "validator_reason": None
        }
        
    except Exception as e:
        logger.error(f"Tool-based query processing failed: {e}")
        send_error_to_slack(question, str(e), "ToolQuery")
        
        if use_fallback:
            logger.info("Tool processing exception, falling back to LLM SQL generation")
            return _fallback_to_llm_sql(question, conversation_history, embedding_manager)
        
        return {
            "original_query": question,
            "sql_query": None,
            "sql_queries": [],
            "result_count": 0,
            "results": [],
            "columns": [],
            "natural_response": f"I'm having trouble processing your query. Please try again.",
            "success": False,
            "error": str(e),
            "search_method": "tool_based_error",
            "validator_verdict": None,
            "validator_reason": None
        }


def _fallback_to_llm_sql(
    question: str, 
    conversation_history: Optional[List[Dict[str, Any]]], 
    embedding_manager
) -> dict:
    """Fallback to the original LLM-based SQL generation"""
    try:
        processor = Query2SQL(embedding_manager=embedding_manager)
        result = processor.process_query(question, conversation_history)
        result["search_method"] = "llm_sql_fallback"
        return result
    except Exception as e:
        logger.error(f"LLM SQL fallback also failed: {e}")
        return {
            "original_query": question,
            "sql_query": None,
            "sql_queries": [],
            "result_count": 0,
            "results": [],
            "columns": [],
            "natural_response": f"I'm having trouble processing your query. Please try again or rephrase your question.",
            "success": False,
            "error": str(e),
            "search_method": "fallback_error",
            "validator_verdict": None,
            "validator_reason": None
        }


def _generate_simple_response(question: str, results: List[Dict[str, Any]]) -> str:
    """Generate a simple response when LLM response generation fails"""
    if not results:
        return f"No results found for: {question}"
    
    count = len(results)
    sample = results[0] if results else {}
    
    key_fields = []
    for key in ['title', 'index', 'onchaininfo_status', 'source_network']:
        if key in sample and sample[key]:
            key_fields.append(f"{key}: {sample[key]}")
    
    if key_fields:
        return f"Found {count} result(s). First result: {', '.join(key_fields)}"
    
    return f"Found {count} result(s) for your query."

