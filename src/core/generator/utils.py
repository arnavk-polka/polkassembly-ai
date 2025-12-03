import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_TIMEOUT = float(os.getenv('GEMINI_TIMEOUT', '30'))

def print_model_usage(model_name: str, purpose: str):
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    END = '\033[0m'
    print(f"{GREEN}{BOLD}🤖 Using {model_name} for {purpose}{END}")

