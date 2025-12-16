#!/usr/bin/env python3
"""
Publish dkg_asset_doc.json to DKG in batches.
Splits large asset into smaller ones and builds chunk index.
"""
import os
import sys
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
asset_file = project_root / "dkg_asset_doc.json"
index_file = project_root / "dkg_chunk_index.json"

BATCH_SIZE = 50


def load_asset_json() -> dict:
    if not asset_file.exists():
        logger.error(f"Asset file not found: {asset_file}")
        sys.exit(1)
    
    with open(asset_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_into_batches(asset_data: dict) -> list:
    """Split hasPart into batches, each becoming a separate asset."""
    has_part = asset_data.get("hasPart", [])
    batches = []
    
    for i in range(0, len(has_part), BATCH_SIZE):
        batch_parts = has_part[i:i + BATCH_SIZE]
        batch_asset = {
            "@context": asset_data.get("@context", "https://schema.org"),
            "@type": asset_data.get("@type", "CreativeWork"),
            "name": f"{asset_data.get('name', 'Asset')} - Part {i // BATCH_SIZE + 1}",
            "description": asset_data.get("description", ""),
            "identifier": f"{asset_data.get('identifier', 'asset')}_{i // BATCH_SIZE + 1}",
            "version": asset_data.get("version", 1),
            "hasPart": batch_parts,
        }
        batches.append(batch_asset)
    
    return batches


def get_chunk_ids_from_batch(batch: dict) -> list:
    """Extract chunk IDs and hashes from a batch."""
    chunks = []
    for part in batch.get("hasPart", []):
        chunk_id = part.get("identifier", "")
        chunk_hash = ""
        for prop in part.get("additionalProperty", []):
            if prop.get("name") == "chunkHash":
                chunk_hash = prop.get("value", "")
                break
        if chunk_id:
            chunks.append({"chunk_id": chunk_id, "chunk_hash": chunk_hash})
    return chunks


def publish_batch(dkg, batch: dict, batch_num: int, publish_options: dict, max_retries: int = 3) -> str:
    """Publish a single batch and return its UAL. Retries on timeout."""
    logger.info(f"\n--- Publishing batch {batch_num} ({len(batch.get('hasPart', []))} chunks) ---")
    
    content = {"public": batch}
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.info(f"Retry attempt {attempt}/{max_retries} for batch {batch_num}...")
                time.sleep(10)
            
            result = dkg.asset.create(content, publish_options)
            ual = result.get("UAL", "")
            if ual:
                logger.info(f"✓ Batch {batch_num} published: {ual}")
                return ual
            else:
                logger.error(f"✗ Batch {batch_num} failed - no UAL returned")
                if attempt < max_retries:
                    continue
                return ""
        except Exception as e:
            error_str = str(e)
            if "not in the chain" in error_str or "timeout" in error_str.lower():
                if attempt < max_retries:
                    logger.warning(f"Batch {batch_num} timeout (attempt {attempt}/{max_retries}), retrying...")
                    continue
                else:
                    logger.error(f"✗ Batch {batch_num} failed after {max_retries} attempts: {e}")
                    return ""
            else:
                logger.error(f"✗ Batch {batch_num} failed: {e}")
                return ""
    
    return ""


def main():
    node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
    blockchain = os.getenv("DKG_BLOCKCHAIN")
    private_key = os.getenv("PRIVATE_KEY")
    
    if not blockchain or not private_key:
        logger.error("DKG_BLOCKCHAIN and PRIVATE_KEY required")
        sys.exit(1)
    
    asset_data = load_asset_json()
    total_chunks = len(asset_data.get("hasPart", []))
    logger.info(f"Loaded asset with {total_chunks} chunks")
    
    batches = split_into_batches(asset_data)
    logger.info(f"Split into {len(batches)} batches of ~{BATCH_SIZE} chunks each")
    
    node_provider = NodeHTTPProvider(endpoint_uri=node_endpoint, api_version="v1")
    blockchain_provider = BlockchainProvider(blockchain)
    
    try:
        blockchain_provider.set_account(private_key)
    except Exception as e:
        if "same un-named instance twice" in str(e):
            blockchain_provider.account = blockchain_provider.w3.eth.account.from_key(private_key)
            blockchain_provider.w3.eth.default_account = blockchain_provider.account.address
        else:
            raise
    
    dkg = DKG(node_provider, blockchain_provider, config={"max_number_of_retries": 100, "frequency": 3})
    
    try:
        node_info = dkg.node.info
        logger.info(f"Node accessible: v{node_info.get('version')}")
    except Exception as e:
        logger.error(f"Node not accessible: {e}")
        sys.exit(1)
    
    publish_options = {
        "epochs_num": 2,
        "minimum_number_of_finalization_confirmations": 1,
        "minimum_number_of_node_replications": 0,
    }
    
    chunk_index = {}
    if index_file.exists():
        try:
            with open(index_file, 'r') as f:
                chunk_index = json.load(f)
            logger.info(f"Loaded existing index with {len(chunk_index)} chunks - will resume from here")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}, starting fresh")
    
    def save_index():
        """Save index to file."""
        with open(index_file, 'w') as f:
            json.dump(chunk_index, f, indent=2)
        logger.info(f"Index saved ({len(chunk_index)} chunks)")
    
    for i, batch in enumerate(batches, 1):
        ual = publish_batch(dkg, batch, i, publish_options)
        
        if ual:
            chunks = get_chunk_ids_from_batch(batch)
            for chunk in chunks:
                chunk_index[chunk["chunk_id"]] = {
                    "ual": ual,
                    "chunk_hash": chunk["chunk_hash"]
                }
            logger.info(f"Added {len(chunks)} chunks to index")
            save_index()
        else:
            logger.warning(f"Batch {i} failed - chunks not indexed")
        
        if i < len(batches):
            wait_time = 10 if ual else 5
            logger.info(f"Waiting {wait_time}s before next batch...")
            time.sleep(wait_time)
    
    save_index()
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total chunks indexed: {len(chunk_index)}")
    logger.info(f"Index saved to: {index_file}")
    
    uals = set(v["ual"] for v in chunk_index.values())
    logger.info(f"Unique UALs: {len(uals)}")
    for ual in uals:
        logger.info(f"  - {ual}")


if __name__ == "__main__":
    main()

