#!/usr/bin/env python3
"""
Entry point script for running the API server.
"""

import os
import sys
import signal
import traceback
from datetime import datetime

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler for unhandled exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_msg = f"Unhandled exception: {exc_type.__name__}: {str(exc_value)}"
    traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    print(f"CRITICAL: {error_msg}", file=sys.stderr)
    print(traceback_str, file=sys.stderr)
    
    try:
        from src.utils.slack_bot import SlackBot
        slack_bot = SlackBot()
        slack_bot.post_error_to_slack(
            f"API Server crashed: {error_msg}",
            context={
                "error_type": exc_type.__name__,
                "error_message": str(exc_value),
                "traceback": traceback_str,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as slack_error:
        print(f"Failed to send crash notification to Slack: {slack_error}", file=sys.stderr)

sys.excepthook = handle_unhandled_exception

if __name__ == "__main__":
    import uvicorn
    from src.rag.config import Config
    
    print("Starting Polkadot AI Chatbot API server...")
    try:
        uvicorn.run(
            "src.rag.api_server:app",
            host=Config.API_HOST,
            port=Config.API_PORT,
            log_level="info",
            reload=False
        )
    except Exception as e:
        error_msg = f"Server startup/runtime error: {type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()
        print(f"CRITICAL: {error_msg}", file=sys.stderr)
        print(traceback_str, file=sys.stderr)
        
        try:
            from src.utils.slack_bot import SlackBot
            slack_bot = SlackBot()
            slack_bot.post_error_to_slack(
                f"API Server failed: {error_msg}",
                context={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback_str,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as slack_error:
            print(f"Failed to send error notification to Slack: {slack_error}", file=sys.stderr)
        
        sys.exit(1) 