"""
Shared utilities for the query pipeline.
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

_reranker = None


def set_reranker(reranker):
    """Set the global reranker instance (called at startup)"""
    global _reranker
    _reranker = reranker


def _get_reranker():
    """Get the global reranker instance"""
    return _reranker


def log_step(step_name: str, data: Dict[str, Any], level: str = "info"):
    """Log a pipeline step with structured data"""
    log_data = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        **data
    }
    if level == "info":
        logger.info(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "warning":
        logger.warning(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "error":
        logger.error(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "debug":
        logger.debug(f"[{step_name}] {json.dumps(log_data, default=str)}")


from typing import Optional, List


def _format_conversation_history_for_validation(conversation_history: Optional[List[Dict[str, Any]]]) -> str:
    """Convert conversation history into a compact string for validator prompts."""
    if not conversation_history:
        return ""
    
    formatted_messages = []
    recent_history = conversation_history[-10:]
    for msg in recent_history:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content') or msg.get('response') or msg.get('query') or msg.get('answer')
        else:
            role = getattr(msg, 'role', 'user')
            content = getattr(msg, 'content', None) or getattr(msg, 'response', None) or getattr(msg, 'query', None)
        if content:
            formatted_messages.append(f"{role.upper()}: {str(content).strip()}")
    
    return "\n".join(formatted_messages)

