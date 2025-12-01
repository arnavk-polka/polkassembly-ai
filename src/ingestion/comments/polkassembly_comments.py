import os
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2.extras import execute_values, Json

# Imports use src.* paths, no sys.path manipulation needed

from src.ingestion.onchain.onchain_data import PolkassemblyDataFetcher, _resolve_data_dir

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _get_db_config() -> Dict[str, Any]:
    config = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DATABASE"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    missing = [k for k, v in config.items() if not v]
    if missing:
        raise ValueError(f"Missing database env vars: {', '.join(missing)}")
    return config


TABLE_NAME = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")


def _prepare_records(comments: List[Dict[str, Any]], source_file: str, fetched_at: datetime):
    for comment in comments:
        yield (
            comment.get("id"),
            comment.get("network"),
            comment.get("proposalType"),
            comment.get("indexOrHash"),
            comment.get("parentCommentId"),
            comment.get("userId"),
            comment.get("content"),
            comment.get("createdAt"),
            comment.get("updatedAt"),
            comment.get("isDeleted"),
            comment.get("dataSource"),
            comment.get("authorAddress"),
            comment.get("aiSentiment"),
            Json(comment.get("history") or []),
            Json(comment.get("publicUser") or {}),
            Json(comment.get("children") or []),
            Json(comment.get("reactions") or []),
            source_file,
            fetched_at,
            Json(comment),
        )


def _insert_comments(conn, records: List[tuple]):
    if not records:
        return

    insert_sql = f"""
    INSERT INTO {TABLE_NAME} (
        id, network, proposal_type, index_or_hash, parent_comment_id,
        user_id, content, created_at, updated_at, is_deleted, data_source,
        author_address, ai_sentiment, history, public_user, children, reactions,
        source_file, fetched_at, raw
    )
    VALUES %s
    ON CONFLICT (id) DO UPDATE SET
        network = EXCLUDED.network,
        proposal_type = EXCLUDED.proposal_type,
        index_or_hash = EXCLUDED.index_or_hash,
        parent_comment_id = EXCLUDED.parent_comment_id,
        user_id = EXCLUDED.user_id,
        content = EXCLUDED.content,
        created_at = EXCLUDED.created_at,
        updated_at = EXCLUDED.updated_at,
        is_deleted = EXCLUDED.is_deleted,
        data_source = EXCLUDED.data_source,
        author_address = EXCLUDED.author_address,
        ai_sentiment = EXCLUDED.ai_sentiment,
        history = EXCLUDED.history,
        public_user = EXCLUDED.public_user,
        children = EXCLUDED.children,
        reactions = EXCLUDED.reactions,
        source_file = EXCLUDED.source_file,
        fetched_at = EXCLUDED.fetched_at,
        raw = EXCLUDED.raw;
    """

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, records, page_size=1000)
    conn.commit()


def fetch_comments_data(
    network: str = "polkadot",
    data_dir: Optional[str] = None,
    max_items: int = 1000,
):
    """Fetch Polkassembly comments for a specific network, store locally, and load into DB."""
    data_dir = _resolve_data_dir(data_dir)
    logger.info(f"Storing comments data in: {data_dir}")

    try:
        logger.info(f"Starting comments fetch for {network}...")
        fetcher = PolkassemblyDataFetcher(network=network, data_dir=data_dir)

        comments_data = fetcher.fetch_all_comments(max_items=max_items)

        if not comments_data:
            logger.warning(f"No comments data fetched for {network}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{network}_comments_{timestamp}.json"
        fetcher.save_to_file(comments_data, filename)
        logger.info(
            f"Completed comments fetch for {network}. Total comments: {len(comments_data)}"
        )

        db_config = _get_db_config()
        fetched_at = datetime.now(timezone.utc)
        source_file = os.path.join(data_dir, filename)

        with psycopg2.connect(**db_config) as conn:
            batch = []
            for record in _prepare_records(comments_data, source_file, fetched_at):
                batch.append(record)
                if len(batch) >= 2000:
                    _insert_comments(conn, batch)
                    batch.clear()

            if batch:
                _insert_comments(conn, batch)

        logger.info(f"Comments successfully inserted into the '{TABLE_NAME}' table.")

    except Exception as e:
        logger.error(f"Error processing comments for network {network}: {e}")
        raise


if __name__ == "__main__":
    target_network = os.getenv("POLKASSEMBLY_COMMENTS_NETWORK", "polkadot")
    max_items = int(os.getenv("POLKASSEMBLY_COMMENTS_MAX_ITEMS", "1000"))
    fetch_comments_data(network=target_network, max_items=max_items)

