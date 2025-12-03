import logging
import psycopg2
from contextlib import contextmanager
from typing import Dict

logger = logging.getLogger(__name__)

@contextmanager
def get_connection(db_config: Dict, api_timeout: float):
    """Context manager for database connections with timeout"""
    conn = None
    try:
        db_config_with_timeout = db_config.copy()
        db_config_with_timeout['connect_timeout'] = int(api_timeout)
        
        conn = psycopg2.connect(**db_config_with_timeout)
        yield conn
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        if "timed out" in str(e) or "Operation timed out" in str(e):
            logger.error(f"Connection timeout after {api_timeout} seconds. Database may be unreachable or network issues.")
        if conn:
            conn.rollback()
        raise
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

