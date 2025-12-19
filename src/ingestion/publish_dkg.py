#!/usr/bin/env python3
"""
Publish dkg_asset_doc.json to DKG in batches.
Splits large asset into smaller ones and stores UAL mappings in database.
"""
import os
import sys
import json
import logging
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent
asset_file = project_root / "dkg_asset_doc.json"
export_jsonl_file = project_root / "dkg_polkawiki_export.jsonl"

BATCH_SIZE = 50

try:
    from src.core.static_provider.ual_mapping_db import (
        create_ual_mapping_table,
        batch_upsert_ual_mappings,
    )
    USE_DATABASE = True
except ImportError:
    logger.warning("Could not import database module, falling back to JSON file")
    USE_DATABASE = False
    index_file = project_root / "dkg_chunk_index.json"


def load_asset_json() -> dict:
    """Load asset data from JSON file (legacy format with hasPart array)."""
    if not asset_file.exists():
        logger.error(f"Asset file not found: {asset_file}")
        sys.exit(1)
    
    with open(asset_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_assets_from_jsonl(jsonl_path: Optional[Path] = None, source_filter: Optional[str] = None) -> List[dict]:
    """
    Load assets from JSONL file (one asset per line, per-doc format).
    
    Args:
        jsonl_path: Path to JSONL file (defaults to export_jsonl_file)
        source_filter: Optional source type to filter by (e.g., "polka_wiki")
    
    Returns:
        List of asset dictionaries
    """
    file_path = jsonl_path or export_jsonl_file
    if not file_path.exists():
        logger.error(f"Export JSONL file not found: {file_path}")
        sys.exit(1)
    
    assets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                asset = json.loads(line)
                
                if source_filter:
                    asset_source = None
                    for prop in asset.get("additionalProperty", []):
                        if prop.get("name") == "source":
                            asset_source = prop.get("value", "")
                            break
                    
                    if asset_source != source_filter:
                        continue
                
                assets.append(asset)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    
    if source_filter:
        logger.info(f"Loaded {len(assets)} assets from JSONL (filtered by source: {source_filter})")
    else:
        logger.info(f"Loaded {len(assets)} assets from JSONL")
    return assets


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


def create_knowledge_collection(dkg, collection_name: str, description: str, publish_options: dict) -> Optional[str]:
    """
    Create a Knowledge Collection in DKG.
    
    Args:
        dkg: DKG client instance
        collection_name: Name of the collection (e.g., "PolkaWiki Documentation")
        description: Description of the collection
        publish_options: Publishing options
        
    Returns:
        Collection UAL if successful, None otherwise
    """
    try:
        collection_asset = {
            "@context": "https://schema.org",
            "@type": "Collection",
            "name": collection_name,
            "description": description,
            "keywords": ["polkadot", "wiki", "documentation"],
        }
        
        content = {"public": collection_asset}
        result = dkg.asset.create(content, publish_options)
        collection_ual = result.get("UAL", "")
        
        if collection_ual:
            logger.info(f"✓ Knowledge Collection created: {collection_ual}")
            return collection_ual
        else:
            logger.warning("Knowledge Collection creation returned no UAL")
            return None
    except Exception as e:
        logger.warning(f"Failed to create Knowledge Collection (may not be supported by DKG SDK): {e}")
        logger.info("Continuing with individual asset publishing...")
        return None


def publish_batch(dkg, batch: dict, batch_num: int, publish_options: dict, collection_ual: Optional[str] = None, max_retries: int = 3) -> str:
    """
    Publish a single batch and return its UAL. Retries on timeout.
    
    IMPORTANT: Returns the INDIVIDUAL ASSET UAL, not the collection UAL.
    The collection UAL is only added as metadata for organizational purposes.
    Each chunk will be mapped to this individual asset UAL in the database.
    
    Args:
        dkg: DKG client instance
        batch: Batch asset data
        batch_num: Batch number
        publish_options: Publishing options
        collection_ual: Optional Knowledge Collection UAL to associate with (metadata only)
        max_retries: Maximum retry attempts
        
    Returns:
        Individual asset UAL (NOT collection UAL) - this is what gets stored in mappings
    """
    logger.info(f"\n--- Publishing batch {batch_num} ({len(batch.get('hasPart', []))} chunks) ---")
    
    if collection_ual:
        additional_props = batch.get("additionalProperty", [])
        additional_props.append({
            "@type": "PropertyValue",
            "name": "knowledgeCollection",
            "value": collection_ual
        })
        batch["additionalProperty"] = additional_props
    
    content = {"public": batch}
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.info(f"Retry attempt {attempt}/{max_retries} for batch {batch_num}...")
                time.sleep(10)
            
            logger.info(f"Publishing batch {batch_num} to DKG (attempt {attempt}/{max_retries})...")
            logger.info(f"  This may take 30-60 seconds for blockchain confirmation...")
            try:
                result = dkg.asset.create(content, publish_options)
                logger.info(f"✓ Received response from DKG for batch {batch_num}")
            except Exception as api_error:
                logger.error(f"✗ DKG API error for batch {batch_num}: {api_error}")
                if attempt < max_retries:
                    continue
                raise
            
            asset_ual = result.get("UAL", "")
            if asset_ual:
                logger.info(f"✓ Batch {batch_num} published: {asset_ual}")
                if collection_ual:
                    logger.info(f"  Associated with collection: {collection_ual} (collection UAL is metadata only)")
                logger.info(f"  → Storing individual asset UAL in mappings: {asset_ual}")
                return asset_ual
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
    parser = argparse.ArgumentParser(description="Publish DKG assets to testnet/mainnet")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of assets to publish (for testing). If not set, publishes all assets.",
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="Filter by source type (e.g., 'polka_wiki'). Defaults to DKG_SOURCE_FILTER env var or 'polka_wiki'.",
    )
    args = parser.parse_args()
    
    node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
    blockchain = os.getenv("DKG_BLOCKCHAIN")
    private_key = os.getenv("PRIVATE_KEY")
    source_filter = args.source_filter or os.getenv("DKG_SOURCE_FILTER", "polka_wiki")
    limit = args.limit
    
    if not blockchain or not private_key:
        logger.error("DKG_BLOCKCHAIN and PRIVATE_KEY required")
        sys.exit(1)
    
    if limit:
        logger.info(f"⚠️  LIMIT MODE: Will only publish first {limit} assets (for testing)")
    
    use_jsonl = export_jsonl_file.exists()
    if use_jsonl and asset_file.exists():
        logger.info(f"Both files exist. Using JSONL format: {export_jsonl_file.name}")
        logger.info(f"Ignoring legacy JSON file: {asset_file.name}")
    
    if use_jsonl:
        assets = load_assets_from_jsonl(source_filter=source_filter)
        logger.info(f"Loaded {len(assets)} assets from JSONL (per-doc format, filtered by source: {source_filter})")
        total_chunks = sum(len(asset.get("hasPart", [])) for asset in assets)
        logger.info(f"Total chunks across all assets: {total_chunks}")
        
        batches = assets
        if limit:
            batches = batches[:limit]
            logger.info(f"Limited to first {len(batches)} assets (requested limit: {limit})")
        logger.info(f"Will publish {len(batches)} assets (one per document, no splitting)")
    else:
        asset_data = load_asset_json()
        total_chunks = len(asset_data.get("hasPart", []))
        logger.info(f"Loaded asset with {total_chunks} chunks")
        
        batches = split_into_batches(asset_data)
        if limit:
            batches = batches[:limit]
            logger.info(f"Limited to first {len(batches)} batches (requested limit: {limit})")
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
    
    epochs_num = int(os.getenv("DKG_EPOCHS_NUM", "1"))
    confirmations = int(os.getenv("DKG_CONFIRMATIONS", "1"))
    
    publish_options = {
        "epochs_num": epochs_num,
        "minimum_number_of_finalization_confirmations": confirmations,
        "minimum_number_of_node_replications": 0,
    }
    
    logger.info(f"Publish options: epochs={epochs_num}, confirmations={confirmations}, replications=0")
    
    if USE_DATABASE:
        create_ual_mapping_table()
        logger.info("Using database for UAL mappings")
    
    collection_ual = None
    if use_jsonl and assets:
        first_asset = assets[0]
        source_value = None
        for prop in first_asset.get("additionalProperty", []):
            if prop.get("name") == "source":
                source_value = prop.get("value", "")
                break
        
        if source_value == "polka_wiki" or "wiki" in str(first_asset.get("name", "")).lower():
            logger.info("Creating Knowledge Collection for PolkaWiki...")
            collection_ual = create_knowledge_collection(
                dkg,
                collection_name="PolkaWiki Documentation",
                description="Comprehensive documentation from the Polkadot Wiki, including guides, tutorials, and technical references",
                publish_options=publish_options
            )
            if collection_ual:
                logger.info(f"Knowledge Collection UAL: {collection_ual}")
            else:
                logger.info("Note: Knowledge Collection creation not available, publishing individual assets")
    elif not use_jsonl:
        source_type = asset_data.get("additionalProperty", [])
        source_value = None
        for prop in source_type:
            if prop.get("name") == "source":
                source_value = prop.get("value", "")
                break
        
        if source_value == "polka_wiki" or "wiki" in str(asset_data.get("name", "")).lower():
            logger.info("Creating Knowledge Collection for PolkaWiki...")
            collection_ual = create_knowledge_collection(
                dkg,
                collection_name="PolkaWiki Documentation",
                description="Comprehensive documentation from the Polkadot Wiki, including guides, tutorials, and technical references",
                publish_options=publish_options
            )
            if collection_ual:
                logger.info(f"Knowledge Collection UAL: {collection_ual}")
            else:
                logger.info("Note: Knowledge Collection creation not available, publishing individual assets")
    else:
        chunk_index = {}
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    raw_index = json.load(f)
                    for key, value in raw_index.items():
                        if isinstance(key, list) and len(key) == 2:
                            chunk_index[tuple(key)] = value
                        elif isinstance(value, dict) and "chunk_hash" in value:
                            chunk_id = key
                            chunk_hash = value["chunk_hash"]
                            chunk_index[(chunk_id, chunk_hash)] = value
                        else:
                            logger.warning(f"Skipping invalid index entry: {key}")
                logger.info(f"Loaded existing index with {len(chunk_index)} chunks - will resume from here")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}, starting fresh")
        
        def save_index():
            """Save index to file using compound key format."""
            serializable_index = {}
            for (chunk_id, chunk_hash), data in chunk_index.items():
                key_str = f"{chunk_id}||{chunk_hash}"
                serializable_index[key_str] = data
            with open(index_file, 'w') as f:
                json.dump(serializable_index, f, indent=2)
            logger.info(f"Index saved ({len(chunk_index)} chunks)")
    
    for i, batch in enumerate(batches, 1):
        ual = publish_batch(dkg, batch, i, publish_options, collection_ual=collection_ual)
        
        if ual:
            chunks = get_chunk_ids_from_batch(batch)
            mappings = []
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                chunk_hash = chunk["chunk_hash"]
                mappings.append({
                    "chunk_id": chunk_id,
                    "chunk_hash": chunk_hash,
                    "ual": ual,
                    "asset_version": 1,
                    "published_at": datetime.now(),
                })
            
            if USE_DATABASE:
                count = batch_upsert_ual_mappings(mappings)
                logger.info(f"Saved {count} chunks to database")
            else:
                for mapping in mappings:
                    key = (mapping["chunk_id"], mapping["chunk_hash"])
                    chunk_index[key] = {
                        "ual": mapping["ual"],
                        "chunk_hash": mapping["chunk_hash"],
                        "asset_version": mapping["asset_version"],
                        "published_at": mapping["published_at"].isoformat(),
                    }
                logger.info(f"Added {len(chunks)} chunks to index")
                save_index()
        else:
            logger.warning(f"Batch {i} failed - chunks not indexed")
        
        if i < len(batches):
            wait_time = 10 if ual else 5
            logger.info(f"Waiting {wait_time}s before next batch...")
            time.sleep(wait_time)
    
    if not USE_DATABASE:
        save_index()
    
    logger.info(f"\n=== Summary ===")
    if USE_DATABASE:
        from src.core.static_provider.ual_mapping_db import load_all_ual_mappings
        index = load_all_ual_mappings()
        logger.info(f"Total chunks indexed in database: {len(index)}")
        uals = set(v.get("ual", "") for v in index.values() if v.get("ual"))
    else:
        logger.info(f"Total chunks indexed: {len(chunk_index)}")
        logger.info(f"Index saved to: {index_file}")
        uals = set(v.get("ual", "") for v in chunk_index.values() if v.get("ual"))
    
    logger.info(f"Unique UALs: {len(uals)}")
    for ual in uals:
        logger.info(f"  - {ual}")


if __name__ == "__main__":
    main()

