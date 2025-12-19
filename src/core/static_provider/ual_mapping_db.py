"""
Database operations for UAL (Universal Asset Locator) mappings.

Stores mappings: (chunk_id, chunk_hash) -> {ual, asset_version, published_at}
in PostgreSQL database table 'ual_mapping'.
"""

import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


def get_db_config() -> Dict[str, Any]:
    """Get database configuration from environment variables."""
    return {
        'host': os.getenv('POSTGRES_HOST'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'database': os.getenv('POSTGRES_DATABASE'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = None
    try:
        db_config = get_db_config()
        required_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        conn = psycopg2.connect(**db_config)
        conn.autocommit = False
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def create_ual_mapping_table() -> bool:
    """
    Create the ual_mapping table if it doesn't exist.
    
    Table structure:
    - chunk_id: VARCHAR (deterministic chunk identifier)
    - chunk_hash: VARCHAR (SHA256 hash of chunk content)
    - ual: VARCHAR (DKG Universal Asset Locator)
    - asset_version: INTEGER (version of the asset)
    - published_at: TIMESTAMP (when published to DKG)
    - created_at: TIMESTAMP (when record was created)
    - updated_at: TIMESTAMP (when record was last updated)
    
    Primary key: (chunk_id, chunk_hash)
    Indexes: chunk_id, chunk_hash, ual
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ual_mapping (
                        chunk_id VARCHAR(500) NOT NULL,
                        chunk_hash VARCHAR(64) NOT NULL,
                        ual VARCHAR(500) NOT NULL,
                        asset_version INTEGER DEFAULT 1,
                        published_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (chunk_id, chunk_hash)
                    );
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ual_mapping_chunk_id 
                    ON ual_mapping(chunk_id);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ual_mapping_chunk_hash 
                    ON ual_mapping(chunk_hash);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ual_mapping_ual 
                    ON ual_mapping(ual);
                """)
                
                conn.commit()
                logger.info("Created ual_mapping table and indexes")
                return True
    except Exception as e:
        logger.error(f"Failed to create ual_mapping table: {e}")
        return False


def upsert_ual_mapping(
    chunk_id: str,
    chunk_hash: str,
    ual: str,
    asset_version: int = 1,
    published_at: Optional[datetime] = None
) -> bool:
    """
    Insert or update a UAL mapping.
    
    Args:
        chunk_id: Deterministic chunk identifier
        chunk_hash: SHA256 hash of chunk content
        ual: DKG Universal Asset Locator
        asset_version: Version of the asset
        published_at: When published to DKG (defaults to now)
        
    Returns:
        True if successful, False otherwise
    """
    if not published_at:
        published_at = datetime.now()
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ual_mapping 
                        (chunk_id, chunk_hash, ual, asset_version, published_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (chunk_id, chunk_hash)
                    DO UPDATE SET
                        ual = EXCLUDED.ual,
                        asset_version = EXCLUDED.asset_version,
                        published_at = EXCLUDED.published_at,
                        updated_at = CURRENT_TIMESTAMP;
                """, (chunk_id, chunk_hash, ual, asset_version, published_at))
                
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to upsert UAL mapping: {e}")
        return False


def batch_upsert_ual_mappings(
    mappings: list[Dict[str, Any]]
) -> int:
    """
    Batch insert/update UAL mappings.
    
    Args:
        mappings: List of dicts with keys: chunk_id, chunk_hash, ual, asset_version, published_at
        
    Returns:
        Number of mappings successfully inserted/updated
    """
    if not mappings:
        return 0
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                now = datetime.now()
                values = []
                for mapping in mappings:
                    chunk_id = mapping.get('chunk_id', '')
                    chunk_hash = mapping.get('chunk_hash', '')
                    ual = mapping.get('ual', '')
                    asset_version = mapping.get('asset_version', 1)
                    published_at = mapping.get('published_at', now)
                    
                    if not chunk_id or not chunk_hash or not ual:
                        continue
                    
                    if isinstance(published_at, str):
                        try:
                            published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        except:
                            published_at = now
                    
                    values.append((chunk_id, chunk_hash, ual, asset_version, published_at))
                
                if not values:
                    return 0
                
                execute_values(
                    cur,
                    """
                    INSERT INTO ual_mapping 
                        (chunk_id, chunk_hash, ual, asset_version, published_at)
                    VALUES %s
                    ON CONFLICT (chunk_id, chunk_hash)
                    DO UPDATE SET
                        ual = EXCLUDED.ual,
                        asset_version = EXCLUDED.asset_version,
                        published_at = EXCLUDED.published_at,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    values,
                    template=None,
                    page_size=100
                )
                
                conn.commit()
                count = len(values)
                logger.info(f"Batch upserted {count} UAL mappings")
                return count
    except Exception as e:
        logger.error(f"Failed to batch upsert UAL mappings: {e}")
        return 0


def get_ual_mapping(chunk_id: str, chunk_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get UAL mapping for a chunk.
    
    Args:
        chunk_id: Chunk identifier
        chunk_hash: Chunk hash
        
    Returns:
        Dict with ual, asset_version, published_at, or None if not found
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ual, asset_version, published_at, created_at, updated_at
                    FROM ual_mapping
                    WHERE chunk_id = %s AND chunk_hash = %s;
                """, (chunk_id, chunk_hash))
                
                row = cur.fetchone()
                if row:
                    return {
                        'ual': row[0],
                        'asset_version': row[1],
                        'published_at': row[2],
                        'created_at': row[3],
                        'updated_at': row[4],
                        'chunk_hash': chunk_hash,
                    }
                return None
    except Exception as e:
        logger.error(f"Failed to get UAL mapping: {e}")
        return None


def load_all_ual_mappings() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load all UAL mappings from database.
    
    Returns:
        Dict mapping (chunk_id, chunk_hash) tuple to UAL metadata
    """
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, chunk_hash, ual, asset_version, published_at
                    FROM ual_mapping;
                """)
                
                for row in cur.fetchall():
                    chunk_id, chunk_hash, ual, asset_version, published_at = row
                    key = (chunk_id, chunk_hash)
                    index[key] = {
                        'ual': ual,
                        'asset_version': asset_version,
                        'chunk_hash': chunk_hash,
                        'published_at': published_at.isoformat() if published_at else '',
                    }
                
                logger.info(f"Loaded {len(index)} UAL mappings from database")
                return index
    except Exception as e:
        logger.error(f"Failed to load UAL mappings from database: {e}")
        return index


def get_uals_by_source(source_prefix: str) -> list[str]:
    """
    Get all unique UALs for a given source prefix.
    
    Args:
        source_prefix: Source prefix (e.g., "wiki", "aag", "pa_md")
        
    Returns:
        List of unique UALs
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ual
                    FROM ual_mapping
                    WHERE chunk_id LIKE %s;
                """, (f"{source_prefix}:%",))
                
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get UALs by source: {e}")
        return []


def delete_ual_mappings_by_source(source_prefix: str) -> int:
    """
    Delete all UAL mappings for a given source prefix.
    
    Use with caution! This removes all mappings for a source.
    
    Args:
        source_prefix: Source prefix (e.g., "wiki", "aag")
        
    Returns:
        Number of mappings deleted
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM ual_mapping
                    WHERE chunk_id LIKE %s;
                """, (f"{source_prefix}:%",))
                
                count = cur.rowcount
                conn.commit()
                logger.info(f"Deleted {count} UAL mappings for source '{source_prefix}'")
                return count
    except Exception as e:
        logger.error(f"Failed to delete UAL mappings by source: {e}")
        return 0

