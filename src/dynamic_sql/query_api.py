import json
from logging import Logger
import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from .query2sql import Query2SQL
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from query2sql import Query2SQL


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

def ask_question(question: str, conversation_history: Optional[List[Dict[str, Any]]] = None, table: Optional[str] = None, embedding_manager=None) -> dict:
    """
    Simple function to ask a natural language question and get results
    
    Args:
        question (str): Natural language question about governance or voting data
        conversation_history: Previous conversation context
        table (str): Target table - 'governance_data' or 'voting_data'
        embedding_manager: Optional EmbeddingManager for dynamic Chroma collection (governance only)
        
    Returns:
        dict: Response containing SQL query, results, and natural language answer
    """
    try:
        if table == 'voting_data':
            from .query2sql import VoteQuery2SQL
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