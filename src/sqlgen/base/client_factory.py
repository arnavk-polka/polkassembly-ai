import os
import logging
from openai import OpenAI
from typing import Optional

from ..utils.model_usage import GeminiClient, GEMINI_MODEL_SQL, GEMINI_SQL_TIMEOUT

logger = logging.getLogger(__name__)

def initialize_clients(openai_api_key: Optional[str], api_timeout: float) -> tuple[Optional[OpenAI], Optional]:
    """
    Initialize Gemini as primary SQL model and OpenAI as fallback
    
    Returns:
        Tuple of (openai_client, gemini_client)
    """
    openai_client = None
    gemini_client = None
    
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

