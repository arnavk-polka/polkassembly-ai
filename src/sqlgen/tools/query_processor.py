import logging
import os
from typing import Any, Dict, List, Optional

from .base import ToolResult
from .registry import get_tool_registry, reset_registry
from .selector import ToolSelector

logger = logging.getLogger(__name__)


class ToolBasedQueryProcessor:
    def __init__(self, embedding_manager=None, openai_client=None, gemini_client=None, table='governance_data'):
        self.embedding_manager = embedding_manager
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        
        if table == 'voting_data':
            self.db_config = {
                'host': os.getenv('POSTGRES_HOST_PA'),
                'port': int(os.getenv('POSTGRES_PORT_PA', '5432')),
                'database': os.getenv('POSTGRES_DATABASE_PA'),
                'user': os.getenv('POSTGRES_USER_PA'),
                'password': os.getenv('POSTGRES_PASSWORD_PA')
            }
            required_vars = ['POSTGRES_HOST_PA', 'POSTGRES_DATABASE_PA', 'POSTGRES_USER_PA', 'POSTGRES_PASSWORD_PA']
            self.table_name = 'flattened_conviction_votes'
        else:
            self.db_config = {
                'host': os.getenv('POSTGRES_HOST'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DATABASE'),
                'user': os.getenv('POSTGRES_USER'),
                'password': os.getenv('POSTGRES_PASSWORD')
            }
            required_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
            self.table_name = os.getenv('POSTGRES_TABLE_NAME', 'governance_data')
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Missing required database environment variables: {', '.join(missing_vars)}")
        
        self.timeout = float(os.getenv('API_TIMEOUT', 30.0))
        
        reset_registry()
        self.registry = get_tool_registry(self.db_config, self.table_name, self.timeout, table_type=table)
        self.selector = ToolSelector(self.registry, openai_client, gemini_client)
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        logger.info(f"[ToolBasedProcessor] Processing query: {query[:100]}")
        
        tool_name, result = self.selector.process_query(query, conversation_history)
        
        if not result.success:
            logger.warning(f"[ToolBasedProcessor] Tool execution failed: {result.error}")
            return {
                "original_query": query,
                "sql_queries": [result.sql_query] if result.sql_query else [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": None,
                "success": False,
                "error": result.error,
                "error_type": result.error_type,
                "tool_used": tool_name,
                "tool_fallback_needed": True,
                "metadata": result.metadata
            }
        
        logger.info(f"[ToolBasedProcessor] Tool '{tool_name}' returned {result.total_count} results")
        
        return {
            "original_query": query,
            "sql_queries": [result.sql_query],
            "result_count": result.total_count,
            "results": result.data,
            "columns": result.columns,
            "natural_response": None,
            "success": True,
            "error": None,
            "tool_used": tool_name,
            "tool_fallback_needed": False,
            "metadata": result.metadata
        }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        return self.registry.get_all_schemas()
    
    def get_tools_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.registry.get_schemas_by_category()


def create_tool_processor(embedding_manager=None, table='governance_data') -> ToolBasedQueryProcessor:
    from openai import OpenAI
    
    openai_client = None
    gemini_client = None
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            openai_client = OpenAI(api_key=openai_key)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    try:
        from src.core.gemini_client import GeminiClient
        gemini_client = GeminiClient()
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")
    
    return ToolBasedQueryProcessor(
        embedding_manager=embedding_manager,
        openai_client=openai_client,
        gemini_client=gemini_client,
        table=table
    )

