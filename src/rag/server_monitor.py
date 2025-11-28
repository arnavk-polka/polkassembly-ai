#!/usr/bin/env python3
"""
Server monitoring and Slack notification module.
Handles startup, shutdown, and error notifications for the API server.
"""

import os
import sys
import fcntl
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config
from ..utils.slack_bot import SlackBot

logger = None

def get_logger():
    """Get or create logger"""
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger

# Startup notification tracking
STARTUP_LOCK_FILE = Path("/tmp/polkassembly_ai_startup_notification.lock")
startup_notification_sent = False

# Global Slack bot instance
slack_bot: Optional[SlackBot] = None

# Shutdown tracking
shutdown_reason: Optional[str] = None
shutdown_exception: Optional[Exception] = None


def initialize_slack_bot() -> Optional[SlackBot]:
    """Initialize Slack bot for notifications"""
    if not Config.ENABLE_SLACK_NOTIFICATIONS:
        get_logger().info("Slack notifications disabled (ENABLE_SLACK_NOTIFICATIONS=false)")
        return None
    
    try:
        get_logger().info("Initializing Slack bot for error reporting...")
        bot = SlackBot()
        get_logger().info("Slack bot initialized successfully")
        return bot
    except Exception as e:
        get_logger().warning(f"Failed to initialize Slack bot: {e}. Error reporting to Slack will be disabled.")
        return None


def send_startup_notification(bot: Optional[SlackBot]) -> None:
    """Send startup notification to Slack, ensuring only one notification is sent even with multiple workers"""
    if not bot or not Config.ENABLE_SLACK_NOTIFICATIONS:
        return
    
    global startup_notification_sent
    if startup_notification_sent:
        return
    
    import time
    should_send = False
    lock_fd = None
    
    try:
        # Check if notification was sent recently (within last 30 seconds)
        if STARTUP_LOCK_FILE.exists():
            try:
                mtime = STARTUP_LOCK_FILE.stat().st_mtime
                if time.time() - mtime < 30:
                    get_logger().info("Startup notification was sent recently, skipping to avoid duplicates")
                    should_send = False
                else:
                    should_send = True
            except:
                should_send = True
        else:
            should_send = True
        
        if should_send:
            # Try to acquire exclusive lock
            lock_fd = os.open(STARTUP_LOCK_FILE, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Double-check timestamp after acquiring lock
                if STARTUP_LOCK_FILE.exists():
                    try:
                        mtime = STARTUP_LOCK_FILE.stat().st_mtime
                        if time.time() - mtime < 30:
                            get_logger().info("Another process sent notification while we were waiting, skipping")
                            should_send = False
                    except:
                        pass
                
                if should_send:
                    # Send notification
                    bot.post_to_slack({
                        "event": "API Server Started",
                        "status": "RUNNING 🚀",
                        "timestamp": datetime.now().isoformat(),
                        "process_id": os.getpid(),
                    })
                    startup_notification_sent = True
                    get_logger().info("Startup notification sent to Slack")
                    
                    # Update lock file timestamp
                    os.write(lock_fd, str(time.time()).encode())
                    os.fsync(lock_fd)
                    
            except BlockingIOError:
                get_logger().info("Startup notification already being sent by another process, skipping")
            finally:
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
                    except:
                        pass
    except Exception as lock_error:
        get_logger().warning(f"Failed to handle startup notification lock: {lock_error}")


def set_shutdown_reason(reason: str, exception: Optional[Exception] = None) -> None:
    """Set the shutdown reason and exception"""
    global shutdown_reason, shutdown_exception
    shutdown_reason = reason
    shutdown_exception = exception


def send_shutdown_notification(bot: Optional[SlackBot]) -> None:
    """Send shutdown notification to Slack"""
    if not bot or not Config.ENABLE_SLACK_NOTIFICATIONS:
        return
    
    global shutdown_reason, shutdown_exception
    
    if not shutdown_reason:
        shutdown_reason = "Graceful shutdown"
    
    get_logger().info(f"Shutting down Polkadot AI Chatbot API... Reason: {shutdown_reason}")
    
    try:
        shutdown_context = {
            "reason": shutdown_reason,
            "timestamp": datetime.now().isoformat(),
        }
        
        if shutdown_exception:
            shutdown_context.update({
                "error_type": type(shutdown_exception).__name__,
                "error_message": str(shutdown_exception),
                "traceback": traceback.format_exc(),
            })
        
        bot.post_to_slack({
            "event": "API Server Shutdown",
            "status": "SHUTDOWN 🛑",
            **shutdown_context
        })
    except Exception as e:
        get_logger().error(f"Failed to send shutdown notification to Slack: {e}")


def send_startup_error_notification(bot: Optional[SlackBot], error: Exception) -> None:
    """Send startup error notification to Slack"""
    if not bot or not Config.ENABLE_SLACK_NOTIFICATIONS:
        return
    
    try:
        bot.post_error_to_slack(
            f"API Server startup failed: {str(error)}",
            context={
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as slack_error:
        get_logger().error(f"Failed to send startup error to Slack: {slack_error}")


def send_runtime_error_notification(bot: Optional[SlackBot], error: Exception) -> None:
    """Send runtime error notification to Slack"""
    if not bot or not Config.ENABLE_SLACK_NOTIFICATIONS:
        return
    
    try:
        bot.post_error_to_slack(
            f"API Server runtime error: {str(error)}",
            context={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as slack_error:
        get_logger().error(f"Failed to send runtime error to Slack: {slack_error}")


def send_query_error_notification(bot: Optional[SlackBot], query: str, user_id: str, error: Exception) -> None:
    """Send query processing error notification to Slack"""
    if not bot or not Config.ENABLE_SLACK_NOTIFICATIONS:
        return
    
    try:
        error_context = {
            "query": query,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": "query_processor_error"
        }
        bot.post_error_to_slack(
            error_message=f"Query Processor Error: {str(error)}",
            context=error_context
        )
    except Exception as slack_error:
        get_logger().error(f"Failed to send error to Slack: {slack_error}")


def send_crash_notification(error_type: type, error_value: Exception, traceback_str: str) -> None:
    """Send crash notification for unhandled exceptions"""
    if not Config.ENABLE_SLACK_NOTIFICATIONS:
        return
    
    try:
        bot = SlackBot()
        bot.post_error_to_slack(
            f"API Server crashed: {error_type.__name__}: {str(error_value)}",
            context={
                "error_type": error_type.__name__,
                "error_message": str(error_value),
                "traceback": traceback_str,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as slack_error:
        get_logger().error(f"Failed to send crash notification to Slack: {slack_error}")

