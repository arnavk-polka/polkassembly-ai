"""
VoteQuery2SQL class for voting data - refactored version
This is a simplified version that maintains the same interface as the original
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..base.base_query2sql import BaseQuery2SQL
from ..base.schema_loader import load_schema_info, get_table_schema_fallback
from ..base.client_factory import initialize_clients
from ..execution.executor import execute_sql_query_voting, execute_sql_queries_voting

logger = logging.getLogger(__name__)

class VoteQuery2SQL(BaseQuery2SQL):
    def __init__(self):
        """Initialize the VoteQuery2SQL converter specifically for voting data"""
        
        db_config = {
            'host': os.getenv('POSTGRES_HOST_PA'),
            'port': int(os.getenv('POSTGRES_PORT_PA', '5432')),
            'database': os.getenv('POSTGRES_DATABASE_PA'),
            'user': os.getenv('POSTGRES_USER_PA'),
            'password': os.getenv('POSTGRES_PASSWORD_PA')
        }
        
        required_vars = ['POSTGRES_HOST_PA', 'POSTGRES_PORT_PA', 'POSTGRES_USER_PA', 'POSTGRES_PASSWORD_PA']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.api_timeout = float(os.getenv('API_TIMEOUT', '30'))
        
        super().__init__(db_config, 'flattened_conviction_votes', self.api_timeout)
        self.db_config = db_config
        
        self.openai_client, self.gemini_client = initialize_clients(
            self.openai_api_key, self.api_timeout
        )
        
        self.schema_info = load_schema_info('POSTGRES_SCHEMA_VOTE_PATH')
        self.table_schema = get_table_schema_fallback(self.schema_info, self.table_name)
        
        logger.info(f"Initialized VoteQuery2SQL for table: {self.table_name}")
        logger.info(f"Loaded schema for {len(self.schema_info)} columns")
    
    def execute_sql_query(self, sql_query: str) -> Tuple[List[List[Any]], List[str], Optional[str]]:
        """Execute SQL query and return results with column names and error type"""
        return execute_sql_query_voting(sql_query, self.db_config, self.api_timeout)
    
    def execute_sql_queries(self, sql_queries: List[str]) -> Tuple[List[Tuple[List[List[Any]], List[str]]], Optional[str]]:
        """Execute multiple SQL queries and return all results with error type"""
        return execute_sql_queries_voting(sql_queries, self.db_config, self.api_timeout)
    
    def process_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Main method to process a natural language query for voting data"""
        try:
            logger.info(f"Processing voting query: {natural_query}")
            
            from .sql_generator import generate_sql_queries_only_voting
            sql_queries = generate_sql_queries_only_voting(
                natural_query, conversation_history, self.table_schema, self.table_name,
                self.openai_client, self.gemini_client, self.trim_prompt_to_fit_tokens
            )
            
            all_results, connection_error = self.execute_sql_queries(sql_queries)
            
            total_result_count = sum(len(results) for results, _ in all_results)
            if connection_error:
                logger.error("Database connection failed, triggering fallback flow")
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0] if sql_queries else None,
                    "sql_queries": sql_queries,
                    "result_count": 0,
                    "results": [],
                    "columns": [],
                    "natural_response": "",
                    "success": False,
                    "error": "connection_error",
                    "requires_fallback": True,
                    "connection_error": True
                }
            if total_result_count == 0:
                logger.info("No results found, triggering fallback flow")
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0] if sql_queries else None,
                    "sql_queries": sql_queries,
                    "result_count": 0,
                    "results": [],
                    "columns": [],
                    "natural_response": "",
                    "success": False,
                    "error": "no_results",
                    "requires_fallback": True
                }
            
            if len(sql_queries) == 1:
                results, columns = all_results[0]
                
                from .response_generator import generate_natural_response
                natural_response = generate_natural_response(
                    natural_query, sql_queries[0], results, columns, conversation_history,
                    self.gemini_client, self.openai_client
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0],
                    "sql_queries": sql_queries,
                    "result_count": len(results),
                    "results": results,
                    "columns": columns,
                    "natural_response": natural_response,
                    "success": True,
                    "table": "voting_data"
                }
            else:
                combined_results = []
                combined_columns = []
                total_result_count = 0
                
                for results, columns in all_results:
                    combined_results.extend(results)
                    if not combined_columns:
                        combined_columns = columns
                    total_result_count += len(results)
                
                from .response_generator import generate_natural_response
                natural_response = generate_natural_response(
                    natural_query, "; ".join(sql_queries), combined_results, combined_columns, conversation_history,
                    self.gemini_client, self.openai_client
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": "; ".join(sql_queries),
                    "sql_queries": sql_queries,
                    "result_count": total_result_count,
                    "results": combined_results,
                    "columns": combined_columns,
                    "natural_response": natural_response,
                    "success": True,
                    "table": "voting_data"
                }
                
        except Exception as e:
            logger.error(f"Error processing voting query: {e}")
            return {
                "original_query": natural_query,
                "sql_query": None,
                "sql_queries": [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": "I'm sorry, I encountered an error processing your voting query. Please try rephrasing your question or try again later.",
                "success": False,
                "error": str(e),
                "table": "voting_data"
            }

