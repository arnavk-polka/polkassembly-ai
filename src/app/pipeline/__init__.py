"""
Query processing pipeline for the Polkadot AI Chatbot.
Implements the new routing-first architecture with structured logging.
"""

from .main import processUserQuery
from .utils import set_reranker, log_step

__all__ = [
    'processUserQuery',
    'set_reranker',
    'log_step'
]

