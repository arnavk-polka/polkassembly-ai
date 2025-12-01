# DEPRECATED - Use src.core.* and src.integrations.* instead
# This module is kept for backward compatibility only.

from src.core.qa_generator import QAGenerator
from src.core.embeddings import EmbeddingManager
from src.core.gemini_client import GeminiClient
from src.core.text_chunker import TextChunker
from src.core.web_search import search_tavily
from src.core.memory import get_memory_manager, add_user_query, add_assistant_response
from src.core.errors import is_insufficient_quota_error, get_quota_error_message
from src.core.rate_limiter import check_rate_limit, get_client_stats
from src.integrations.slack_bot import SlackBot

__all__ = [
    'QAGenerator',
    'EmbeddingManager',
    'GeminiClient',
    'TextChunker',
    'search_tavily',
    'get_memory_manager',
    'add_user_query',
    'add_assistant_response',
    'is_insufficient_quota_error',
    'get_quota_error_message',
    'check_rate_limit',
    'get_client_stats',
    'SlackBot',
]
