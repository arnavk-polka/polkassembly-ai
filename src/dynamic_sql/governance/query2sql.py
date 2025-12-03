"""
Main Query2SQL class for governance data
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..base.base_query2sql import BaseQuery2SQL
from ..base.schema_loader import load_schema_info, get_table_schema
from ..base.client_factory import initialize_clients
from ..base.database import get_connection
from ..execution.executor import execute_sql_queries_governance
from ..utils.formatting import format_number_for_prompt
from ..utils.model_usage import GEMINI_MODEL_NAME, GEMINI_TIMEOUT

from .result_processor import format_amount_by_asset_id, add_proposal_links
from .intent_extractor import extract_sql_intent
from .sql_generator import generate_sql_queries_only, generate_sql_with_model_deterministic

logger = logging.getLogger(__name__)

class Query2SQL(BaseQuery2SQL):
    def __init__(self, embedding_manager=None):
        """Initialize the Query2SQL converter with database and OpenAI connections"""
        
        self.embedding_manager = embedding_manager
        
        db_config = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DATABASE'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        required_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.api_timeout = float(os.getenv('API_TIMEOUT', '30'))
        
        super().__init__(db_config, os.getenv('POSTGRES_TABLE_NAME', 'governance_data'), self.api_timeout)
        self.db_config = db_config
        
        self.openai_client, self.gemini_client = initialize_clients(
            self.openai_api_key, self.api_timeout
        )
        
        self.schema_info = load_schema_info('POSTGRES_SCHEMA_PATH')
        self.table_schema = get_table_schema(self.schema_info, self.table_name, 'POSTGRES_SCHEMA_PATH')
        
        logger.info(f"Initialized Query2SQL for table: {self.table_name}")
        logger.info(f"Loaded schema for {len(self.schema_info)} columns")
    
    def format_amount_by_asset_id(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return format_amount_by_asset_id(results)
    
    def add_proposal_links(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return add_proposal_links(results)
    
    def _format_number_for_prompt(self, value: Any) -> str:
        return format_number_for_prompt(value)
    
    def _extract_sql_intent(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return extract_sql_intent(natural_query, conversation_history, self.openai_client, self.gemini_client)
    
    def _generate_sql_queries_only(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3) -> List[str]:
        intent = self._extract_sql_intent(natural_query, conversation_history)
        logger.info(f"Intent extracted - entity_type: {intent['entity_type']}, network: {intent['network']}, metric: {intent['metric']}")
        
        return generate_sql_queries_only(
            natural_query, conversation_history, intent, self.embedding_manager,
            self.table_schema, self.table_name, self.openai_client,
            self.gemini_client, self.trim_prompt_to_fit_tokens, max_retries
        )
    
    def execute_sql_queries(self, sql_queries: List[str]) -> List[Tuple[List[Dict[str, Any]], List[str]]]:
        """Execute multiple SQL queries against PostgreSQL database"""
        return execute_sql_queries_governance(sql_queries, self.db_config, self.api_timeout)
    
    def execute_sql_query(self, sql_query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute single SQL query - wrapper for backwards compatibility"""
        results = self.execute_sql_queries([sql_query])
        return results[0]
    
    def generate_sql_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Convert natural language query to SQL"""
        try:
            sql_queries = self._generate_sql_queries_only(natural_query, conversation_history)
            return sql_queries
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            raise
    
    def process_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, table: Optional[str] = None) -> Dict[str, Any]:
        """Main method to process a natural language query end-to-end"""
        try:
            logger.info(f"Processing query: {natural_query}")
            
            sql_queries = self._generate_sql_queries_only(natural_query, conversation_history)
            all_results = self.execute_sql_queries(sql_queries)
            
            total_result_count = sum(len(results) for results, _ in all_results)
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
                    "requires_fallback": True,
                    "validator_verdict": None,
                    "validator_reason": None
                }
            
            if len(sql_queries) == 1:
                results, columns = all_results[0]
                results = self.format_amount_by_asset_id(results)
                results = self.add_proposal_links(results)
                
                from .response_generator import generate_natural_response
                natural_response = generate_natural_response(
                    natural_query, sql_queries[0], results, columns, conversation_history,
                    self.gemini_client, self.openai_client, self._format_number_for_prompt
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
                    "requires_fallback": False,
                    "requires_clarification": False,
                    "search_method": "sql_query",
                    "validator_verdict": None,
                    "validator_reason": None
                }
            else:
                formatted_all_results = []
                for results, columns in all_results:
                    formatted_results = self.format_amount_by_asset_id(results)
                    enhanced_results = self.add_proposal_links(formatted_results)
                    formatted_all_results.append((enhanced_results, columns))
                all_results = formatted_all_results
                
                combined_results = []
                combined_columns = []
                total_result_count = 0
                
                for results, columns in all_results:
                    combined_results.extend(results)
                    if columns not in combined_columns:
                        combined_columns.extend(columns)
                    total_result_count += len(results)
                
                from .response_generator import generate_natural_response_multiple
                natural_response = generate_natural_response_multiple(
                    natural_query, sql_queries, all_results, conversation_history,
                    self.gemini_client, self.openai_client
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": "; ".join(sql_queries),
                    "sql_queries": sql_queries,
                    "result_count": total_result_count,
                    "results": combined_results,
                    "columns": list(set(combined_columns)),
                    "all_results": all_results,
                    "natural_response": natural_response,
                    "success": True,
                    "requires_fallback": False,
                    "requires_clarification": False,
                    "search_method": "sql_query",
                    "validator_verdict": None,
                    "validator_reason": None
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "original_query": natural_query,
                "sql_query": None,
                "sql_queries": [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": "I'm sorry, I encountered an error processing your query. Please try rephrasing your question or try again later.",
                "success": False,
                "error": str(e),
                "validator_verdict": None,
                "validator_reason": None
            }

