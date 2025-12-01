"""
Utility functions for handling API errors, specifically OpenAI quota errors.
"""

import openai
from typing import Optional


def is_insufficient_quota_error(error: Exception) -> bool:
    """
    Check if an exception is an OpenAI insufficient quota error (429 with insufficient_quota type).
    
    Args:
        error: The exception to check
        
    Returns:
        True if it's an insufficient quota error, False otherwise
    """
    # Check if it's an OpenAI APIError
    if isinstance(error, openai.APIError):
        error_code = getattr(error, 'status_code', None)
        error_body = getattr(error, 'body', None) or getattr(error, 'response', None)
        
        if error_code == 429:
            if error_body:
                if isinstance(error_body, dict):
                    error_info = error_body.get('error', {})
                    if isinstance(error_info, dict):
                        error_type = error_info.get('type', '')
                        error_code_str = error_info.get('code', '')
                        if error_type == 'insufficient_quota' or error_code_str == 'insufficient_quota':
                            return True
                elif isinstance(error_body, str):
                    if 'insufficient_quota' in error_body.lower() or 'exceeded your current quota' in error_body.lower():
                        return True
    
    # Check error string representation (handles cases where error is formatted as string)
    error_str = str(error).lower()
    
    # Check for insufficient_quota in error message
    if 'insufficient_quota' in error_str:
        return True
    
    # Check for quota exceeded message
    if 'exceeded your current quota' in error_str:
        return True
    
    # Check for error code 429 with quota-related messages
    if 'error code: 429' in error_str and ('quota' in error_str or 'insufficient' in error_str):
        return True
    
    return False


def get_quota_error_message() -> str:
    """
    Get the user-friendly message for insufficient quota errors.
    
    Returns:
        The error message to show to users
    """
    return "Sorry, We're facing some technical issues right now. Please try again later"

