import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from contextlib import contextmanager

from ..base.database import get_connection

logger = logging.getLogger(__name__)

def execute_sql_queries_governance(sql_queries: List[str], db_config: Dict, api_timeout: float) -> List[Tuple[List[Dict[str, Any]], List[str]]]:
    """Execute multiple SQL queries against PostgreSQL database (governance data)"""
    all_results = []
    
    try:
        with get_connection(db_config, api_timeout) as conn:
            for i, sql_query in enumerate(sql_queries):
                logger.info(f"Executing query {i+1}/{len(sql_queries)}: {sql_query}")
                
                df = pd.read_sql_query(sql_query, conn)
                results = df.to_dict('records')
                columns = df.columns.tolist()
                
                all_results.append((results, columns))
                logger.info(f"Query {i+1} executed successfully. Retrieved {len(results)} rows")
            
            return all_results
            
    except Exception as e:
        logger.error(f"Error executing SQL queries: {e}")
        logger.error(f"Queries: {sql_queries}")
        raise

def execute_sql_query_voting(sql_query: str, db_config: Dict, api_timeout: float) -> Tuple[List[List[Any]], List[str], Optional[str]]:
    """Execute SQL query and return results with column names and error type (voting data)"""
    try:
        with get_connection(db_config, api_timeout) as conn:
            with conn.cursor() as cur:
                logger.info(f"Executing SQL: {sql_query}")
                cur.execute(sql_query)
                
                columns = [desc[0] for desc in cur.description] if cur.description else []
                results = cur.fetchall()
                
                logger.info(f"Query executed successfully: {len(results)} rows returned")
                return results, columns, None
                
    except Exception as e:
        error_str = str(e).lower()
        if "timed out" in error_str or "operation timed out" in error_str or "connection" in error_str:
            logger.error(f"Database connection error: {e}")
            return [], [], "connection_error"
        logger.error(f"Database error: {e}")
        return [], [], "database_error"

def execute_sql_queries_voting(sql_queries: List[str], db_config: Dict, api_timeout: float) -> Tuple[List[Tuple[List[List[Any]], List[str]]], Optional[str]]:
    """Execute multiple SQL queries and return all results with error type (voting data)"""
    all_results = []
    connection_error = None
    for i, query in enumerate(sql_queries):
        logger.info(f"Executing query {i+1}/{len(sql_queries)}")
        results, columns, error_type = execute_sql_query_voting(query, db_config, api_timeout)
        all_results.append((results, columns))
        if error_type == "connection_error" and connection_error is None:
            connection_error = error_type
    return all_results, connection_error

