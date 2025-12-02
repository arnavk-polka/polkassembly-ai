import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from .database import get_connection
from ..utils.token_utils import count_tokens, trim_prompt_to_fit_tokens

logger = logging.getLogger(__name__)

class BaseQuery2SQL(ABC):
    """Base class for Query2SQL implementations with shared functionality"""
    
    def __init__(self, db_config: Dict, table_name: str, api_timeout: float):
        self.db_config = db_config
        self.table_name = table_name
        self.api_timeout = api_timeout
        self.schema_info = {}
        self.table_schema = ""
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text using tiktoken or approximate counting"""
        return count_tokens(text, model)
    
    def trim_prompt_to_fit_tokens(self, system_prompt: str, max_tokens: int = 20000, 
                                   completion_tokens: int = 1000, buffer_tokens: int = 500) -> str:
        """Trim the system prompt to fit within token limits"""
        return trim_prompt_to_fit_tokens(system_prompt, max_tokens, completion_tokens, buffer_tokens)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with timeout"""
        yield from get_connection(self.db_config, self.api_timeout)
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    count = cur.fetchone()[0]
                    logger.info(f"Connection test successful. Table {self.table_name} has {count:,} rows")
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def debug_schema(self):
        """Debug method to inspect loaded schema"""
        print(f"Schema info type: {type(self.schema_info)}")
        print(f"Schema info keys: {list(self.schema_info.keys())[:10]}")
        
        if self.schema_info:
            first_key = list(self.schema_info.keys())[0]
            first_value = self.schema_info[first_key]
            print(f"First item: {first_key} -> {first_value} (type: {type(first_value)})")
        
        print(f"\nTable schema preview:\n{self.table_schema[:500]}...")

