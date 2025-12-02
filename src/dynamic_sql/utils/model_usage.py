import os

def print_model_usage(model_name: str, purpose: str):
    """Print model usage information in green color"""
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    END = '\033[0m'
    print(f"{GREEN}{BOLD}🤖 Using {model_name} for {purpose}{END}")

GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-pro')
GEMINI_MODEL_SQL = os.getenv('GEMINI_MODEL_SQL', 'gemini-2.5-pro')
GEMINI_TIMEOUT = float(os.getenv('GEMINI_TIMEOUT', '30'))
GEMINI_SQL_TIMEOUT = float(os.getenv('GEMINI_SQL_TIMEOUT', '120'))

try:
    from src.core.gemini_client import GeminiClient
except ImportError:
    GeminiClient = None

