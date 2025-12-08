import json
from logging import Logger
import logging
import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from .governance.query2sql import Query2SQL
from .voting.vote_query2sql import VoteQuery2SQL

logger = logging.getLogger(__name__)

USE_TOOL_BASED_QUERIES = os.getenv('USE_TOOL_BASED_QUERIES', 'true').lower() == 'true'

try:
    from ..integrations.slack_bot import SlackBot
except ImportError:
    SlackBot = None

def send_error_to_slack(query: str, error: str, error_source: str = "Query2SQL") -> None:
    """Send error notification to Slack channel"""
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
            "Query processing failed", 
            context=error_context
        )
        print("Error notification sent to Slack")
    except Exception as slack_error:
        print(f"Failed to send error notification to Slack: {slack_error}")


def ask_question_tool_based(question: str, conversation_history: Optional[List[Dict[str, Any]]] = None, table: Optional[str] = None, embedding_manager=None) -> dict:
    """
    Process query using tool-based SQL generation (new approach).
    Falls back to LLM SQL if tools fail.
    """
    from .tool_query_api import ask_question_with_tools
    return ask_question_with_tools(question, conversation_history, table, embedding_manager, use_fallback=True)


def ask_question(question: str, conversation_history: Optional[List[Dict[str, Any]]] = None, table: Optional[str] = None, embedding_manager=None) -> dict:
    """
    Process a natural language question and get results.
    Uses tool-based SQL generation by default (controlled by USE_TOOL_BASED_QUERIES env var).
    Falls back to LLM-based SQL generation if tools fail.
    
    Args:
        question (str): Natural language question about governance or voting data
        conversation_history: Previous conversation context
        table (str): Target table - 'governance_data' or 'voting_data'
        embedding_manager: Optional EmbeddingManager for dynamic Chroma collection (governance only)
        
    Returns:
        dict: Response containing SQL query, results, and natural language answer
    """
    if USE_TOOL_BASED_QUERIES and table != 'voting_data':
        logger.info("Using tool-based query processing")
        try:
            result = ask_question_tool_based(question, conversation_history, table, embedding_manager)
            if result.get("success") or not result.get("requires_fallback", True):
                return result
            logger.info("Tool-based query needs fallback, using LLM SQL generation")
        except Exception as e:
            logger.warning(f"Tool-based query failed, falling back to LLM: {e}")
    
    try:
        if table == 'voting_data':
            print("Extracting from voting table")
            processor = VoteQuery2SQL()
        else:
            processor = Query2SQL(embedding_manager=embedding_manager)
        
        result = processor.process_query(question, conversation_history)
        
        return result
        
    except Exception as e:
        send_error_to_slack(question, str(e), f"Query2SQL-{table or 'governance'}")
        
        return {
            "original_query": question,
            "sql_query": None,
            "result_count": 0,
            "results": [],
            "columns": [],
            "natural_response": f"I'm having trouble processing your query. Please try again or rephrase your question in the next prompt",
            "success": False,
            "error": str(e),
            "table": table or "governance_data",
            "validator_verdict": None,
            "validator_reason": None
        }

def main():
   
    question = "How much they ask in clarys resubmission proposal"
    result = ask_question(question)
    print(f"\nQuery: {question}")
    print(f"SQL Queries: {result.get('sql_queries', [])}")
    print(f"Result Count: {result.get('result_count', 0)}")
    print(f"Natural Response: {result.get('natural_response', 'No response')}")
    print(f"Success: {result.get('success', False)}")
    
if __name__ == "__main__":
    main()