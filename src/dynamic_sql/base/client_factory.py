import os
import logging
from openai import OpenAI
from typing import Optional

from ..utils.model_usage import GeminiClient, GEMINI_MODEL_SQL, GEMINI_SQL_TIMEOUT

logger = logging.getLogger(__name__)

def initialize_clients(sql_model: str, openai_api_key: Optional[str], api_timeout: float, 
                      force_gemini: bool = False) -> tuple[Optional[OpenAI], Optional]:
    """
    Initialize OpenAI and Gemini clients based on configuration
    
    Returns:
        Tuple of (openai_client, gemini_client)
    """
    openai_client = None
    gemini_client = None
    
    if force_gemini:
        sql_model = 'gemini'
    
    if sql_model == 'chatgpt':
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when SQL_MODEL=chatgpt")
        openai_client = OpenAI(api_key=openai_api_key, timeout=api_timeout)
        logger.info("OpenAI client initialized as primary SQL model")
        
        if GeminiClient is not None:
            try:
                gemini_client = GeminiClient(model_name=GEMINI_MODEL_SQL, timeout=GEMINI_SQL_TIMEOUT)
                logger.info(f"Gemini {GEMINI_MODEL_SQL} initialized as fallback")
            except Exception as e:
                logger.warning(f"Gemini fallback initialization failed: {e}")
    else:
        if GeminiClient is not None:
            try:
                gemini_client = GeminiClient(model_name=GEMINI_MODEL_SQL, timeout=GEMINI_SQL_TIMEOUT)
                logger.info(f"Gemini {GEMINI_MODEL_SQL} initialized as primary SQL model")
            except Exception as e:
                logger.error(f"Gemini 2.5 Pro initialization failed: {e}")
                raise ValueError("Failed to initialize Gemini 2.5 Pro. Please check GEMINI_API_KEY.")
        else:
            raise ValueError("Gemini client not available. Please install required dependencies.")
        
        if openai_api_key:
            openai_client = OpenAI(api_key=openai_api_key, timeout=api_timeout)
            logger.info("OpenAI client initialized as fallback")
        else:
            logger.warning("OpenAI API key not provided, no fallback available")
    
    return openai_client, gemini_client

