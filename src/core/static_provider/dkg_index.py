import json
import logging
import os
from functools import lru_cache
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path

from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider
from dkg.constants import BlockchainIds

from ..config import Config

logger = logging.getLogger(__name__)


def _load_json_file(path: str) -> Optional[Any]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load DKG index file '{path}': {e}")
        return None


def _load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    assets: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    assets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Failed to load DKG JSONL file '{path}': {e}")
    return assets


def _build_index_from_asset(asset: Dict[str, Any], asset_ual: Optional[str] = None) -> Dict[str, Tuple[str, str]]:
    index: Dict[str, Tuple[str, str]] = {}
    if not asset:
        return index
    
    ual = asset_ual or asset.get("asset_ual") or asset.get("ual") or asset.get("@id") or ""
    
    has_part = asset.get("hasPart") or []
    for part in has_part:
        identifier = part.get("identifier") or part.get("@id") or ""
        if not identifier:
            continue
        chunk_hash = ""
        for prop in part.get("additionalProperty", []):
            if prop.get("name") in ("chunkHash", "chunk_hash"):
                chunk_hash = str(prop.get("value") or "")
                break
        if not chunk_hash:
            continue
        index[identifier] = (ual, chunk_hash)
    return index


def _parse_assertion_graph(assertion: List[Dict[str, Any]], asset_ual: str) -> Dict[str, Tuple[str, str]]:
    """
    Parse DKG assertion graph (JSON-LD format) to extract chunk_id -> (asset_ual, chunk_hash) mappings.
    
    Looks for Schema.org hasPart relationships and chunkHash properties in the assertion.
    Handles both expanded JSON-LD (full IRIs) and compact form.
    """
    index: Dict[str, Tuple[str, str]] = {}
    
    if not assertion or not isinstance(assertion, list):
        return index
    
    for node in assertion:
        if not isinstance(node, dict):
            continue
        
        node_id = node.get("@id", "")
        if not node_id:
            continue
        
        has_part_refs = (
            node.get("http://schema.org/hasPart", []) or
            node.get("schema:hasPart", []) or
            node.get("hasPart", [])
        )
        
        if not has_part_refs:
            continue
        
        if not isinstance(has_part_refs, list):
            has_part_refs = [has_part_refs]
        
        for part_ref in has_part_refs:
            if isinstance(part_ref, dict):
                part_id = part_ref.get("@id") or part_ref.get("identifier") or ""
            else:
                part_id = str(part_ref)
            
            if not part_id:
                continue
            
            chunk_hash = ""
            for prop_node in assertion:
                if not isinstance(prop_node, dict):
                    continue
                prop_id = prop_node.get("@id", "")
                if prop_id != part_id:
                    continue
                
                chunk_hash_prop = (
                    prop_node.get("https://ontology.origintrail.io/dkg/1.0#chunkHash") or
                    prop_node.get("dkg:chunkHash") or
                    prop_node.get("chunkHash") or
                    prop_node.get("chunk_hash")
                )
                
                if chunk_hash_prop:
                    if isinstance(chunk_hash_prop, list) and len(chunk_hash_prop) > 0:
                        chunk_hash_value = chunk_hash_prop[0].get("@value") if isinstance(chunk_hash_prop[0], dict) else chunk_hash_prop[0]
                    elif isinstance(chunk_hash_prop, dict):
                        chunk_hash_value = chunk_hash_prop.get("@value") or chunk_hash_prop.get("value")
                    else:
                        chunk_hash_value = chunk_hash_prop
                    
                    if chunk_hash_value:
                        chunk_hash = str(chunk_hash_value)
                        break
            
            if chunk_hash and part_id:
                index[part_id] = (asset_ual, chunk_hash)
    
    return index


def trigger_asset_sync(ual: str, node_endpoint: str, blockchain: str) -> bool:
    """
    Manually trigger a sync/fetch of a Knowledge Asset from the DKG network.
    
    This function initiates a 'get' operation which tells the node to fetch
    the asset from the network. The actual sync happens asynchronously on the node.
    
    Parameters
    ----------
    ual : str
        Knowledge Asset UAL to sync
    node_endpoint : str
        DKG node endpoint URL
    blockchain : str
        Blockchain ID (e.g., 'otp:20430')
        
    Returns
    -------
    bool
        True if operation was initiated successfully, False otherwise
    """
    try:
        node_provider = NodeHTTPProvider(endpoint_uri=node_endpoint, api_version="v1")
        blockchain_provider = BlockchainProvider(blockchain)
        dkg = DKG(
            node_provider,
            blockchain_provider,
            config={"max_number_of_retries": 10, "frequency": 3},
        )
        
        logger.info(f"Triggering sync for asset UAL: {ual}")
        response = dkg.asset.get(ual)
        
        if response:
            logger.info(f"Sync operation initiated successfully for {ual}")
            return True
        else:
            logger.warning(f"Sync operation returned empty response for {ual}")
            return False
    except Exception as e:
        logger.error(f"Failed to trigger sync for {ual}: {e}")
        return False


def _load_index_from_remote() -> Dict[str, Tuple[str, str]]:
    """
    Load chunk index from a live DKG knowledge asset using the SDK.

    Config.DKG_POLKAWIKI_ASSET_UAL must be set to a Knowledge Asset UAL.
    Uses DKG SDK to fetch the asset and parse its assertion graph.
    """
    ual = getattr(Config, "DKG_POLKAWIKI_ASSET_UAL", "") or os.getenv("DKG_POLKAWIKI_ASSET_UAL", "")
    if not ual:
        return {}
    
    node_endpoint = os.getenv("DKG_NODE_ENDPOINT")
    blockchain = os.getenv("DKG_BLOCKCHAIN")
    environment = os.getenv("DKG_ENVIRONMENT", "TESTNET")
    
    if not node_endpoint or not blockchain:
        logger.warning("DKG_NODE_ENDPOINT and DKG_BLOCKCHAIN must be set to load index from remote")
        return {}
    
    try:
        logger.info(f"Fetching DKG knowledge asset via SDK: UAL={ual}, endpoint={node_endpoint}, blockchain={blockchain}")
        
        node_provider = NodeHTTPProvider(endpoint_uri=node_endpoint, api_version="v1")
        blockchain_provider = BlockchainProvider(blockchain)
        dkg = DKG(
            node_provider,
            blockchain_provider,
            config={"max_number_of_retries": 5, "frequency": 2},
        )
        logger.info(f"DKG client initialized successfully")
        
        try:
            node_info = dkg.node.info
            logger.info(f"DKG node is accessible. Node info: {node_info}")
        except Exception as node_check_error:
            logger.error(f"Failed to connect to DKG node at {node_endpoint}. Is the node running? Error: {node_check_error}")
            return {}
        
        logger.info(f"Attempting to fetch asset via asset.get('{ual}')...")
        try:
            response = dkg.asset.get(ual)
            logger.info(f"Asset.get succeeded. Response type: {type(response)}, keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
            
            if not response or not isinstance(response, dict):
                logger.warning(f"DKG SDK returned invalid response for UAL '{ual}': {response}")
                return {}
            
            operation_status = response.get("operation", {})
            if operation_status:
                logger.info(f"Operation status: {operation_status}")
            
            assertion = response.get("assertion", [])
            if not assertion:
                logger.warning(f"DKG asset '{ual}' has no assertion data. Full response: {response}")
                return {}
            
            logger.info(f"Found assertion with {len(assertion)} nodes")
            index = _parse_assertion_graph(assertion, ual)
        except Exception as asset_get_error:
            logger.warning(f"asset.get() failed (asset may not be synced to node yet): {asset_get_error}")
            logger.info(f"Attempting alternative: querying graph directly via SPARQL...")
            
            try:
                logger.info("Trying SPARQL query to find hasPart relationships...")
                sparql_query = f"""
                PREFIX dkg: <https://ontology.origintrail.io/dkg/1.0#>
                PREFIX schema: <http://schema.org/>
                SELECT ?partId ?chunkHash
                WHERE {{
                    GRAPH ?g {{
                        ?kc dkg:hasKnowledgeAsset <{ual}> .
                        ?kc dkg:hasNamedGraph ?g .
                        ?asset schema:hasPart ?part .
                        ?part schema:identifier ?partId .
                        ?part dkg:chunkHash ?chunkHash .
                    }}
                }}
                LIMIT 1000
                """
                
                query_response = dkg.graph.query(sparql_query)
                logger.info(f"SPARQL query response type: {type(query_response)}")
                
                if isinstance(query_response, list):
                    data = query_response
                elif isinstance(query_response, dict):
                    data = query_response.get("data", [])
                else:
                    logger.warning(f"Unexpected SPARQL response type: {type(query_response)}")
                    return {}
                
                logger.info(f"SPARQL returned {len(data)} results")
                
                if len(data) > 0:
                    logger.info("Building index from SPARQL results...")
                    index = {}
                    for binding in data:
                        if not isinstance(binding, dict):
                            continue
                        part_id = binding.get("partId") or binding.get("partId", {}).get("@value") or binding.get("partId", {}).get("value")
                        chunk_hash = binding.get("chunkHash") or binding.get("chunkHash", {}).get("@value") or binding.get("chunkHash", {}).get("value")
                        
                        if part_id and chunk_hash:
                            if isinstance(part_id, dict):
                                part_id = part_id.get("@value") or part_id.get("value") or str(part_id)
                            if isinstance(chunk_hash, dict):
                                chunk_hash = chunk_hash.get("@value") or chunk_hash.get("value") or str(chunk_hash)
                            index[str(part_id)] = (ual, str(chunk_hash))
                    logger.info(f"Built index with {len(index)} entries from SPARQL query")
                else:
                    logger.warning("SPARQL query returned no hasPart data - asset may not be synced to node yet")
                    logger.info("Trying simpler query to check if asset exists in graph...")
                    
                    simple_query = f"""
                    PREFIX dkg: <https://ontology.origintrail.io/dkg/1.0#>
                    SELECT ?kc ?g
                    WHERE {{
                        ?kc dkg:hasKnowledgeAsset <{ual}> .
                        ?kc dkg:hasNamedGraph ?g .
                    }}
                    LIMIT 1
                    """
                    simple_response = dkg.graph.query(simple_query)
                    simple_data = simple_response if isinstance(simple_response, list) else (simple_response.get("data", []) if isinstance(simple_response, dict) else [])
                    if len(simple_data) > 0:
                        logger.info(f"Asset UAL found in graph metadata, but hasPart data not available yet")
                    else:
                        logger.warning(
                            f"Asset UAL '{ual}' not found in local node graph. "
                            f"The node needs to sync this asset from the blockchain network first. "
                            f"Falling back to local index files. Once synced, DKG verification will work automatically."
                        )
                    return {}
            except Exception as sparql_error:
                import traceback
                logger.error(f"SPARQL query also failed: {sparql_error}")
                logger.error(f"SPARQL traceback: {traceback.format_exc()}")
                return {}
        
        if index:
            logger.info(f"Loaded DKG chunk index from SDK with {len(index)} entries for UAL '{ual}'")
        else:
            logger.warning(f"DKG asset '{ual}' assertion graph contains no hasPart entries with chunkHash. Parsed {len(assertion)} assertion nodes but found no chunk mappings.")
        
        return index
    except Exception as e:
        import traceback
        logger.error(f"Failed to load DKG index from SDK for UAL '{ual}': {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {}


def _load_published_chunk_index(index_path: str = "dkg_chunk_index.json") -> Dict[str, Tuple[str, str]]:
    """
    Load the chunk index created by publish_dkg_batched.py.
    This contains actual UALs from published assets.
    Format: {chunk_id: {ual: str, chunk_hash: str}}
    """
    if not os.path.exists(index_path):
        return {}
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            raw_index = json.load(f)
        
        index: Dict[str, Tuple[str, str]] = {}
        for chunk_id, data in raw_index.items():
            if isinstance(data, dict):
                ual = data.get("ual", "")
                chunk_hash = data.get("chunk_hash", "")
                if ual and chunk_hash:
                    index[chunk_id] = (ual, chunk_hash)
        
        if index:
            logger.info(f"Loaded published chunk index from '{index_path}' with {len(index)} entries")
        return index
    except Exception as e:
        logger.warning(f"Failed to load published chunk index '{index_path}': {e}")
        return {}


_cache_key = None
_cached_index = None

def load_dkg_chunk_index(
    asset_doc_path: str = "dkg_asset_doc.json",
    docs_export_path: str = "dkg_docs_export.jsonl",
    published_index_path: str = "dkg_chunk_index.json",
) -> Dict[str, Tuple[str, str]]:
    """
    Build an in-memory index:
        chunk_id -> (asset_ual, chunk_hash)

    Priority:
    1. Published chunk index (dkg_chunk_index.json) - created by publish_dkg_batched.py with actual UALs
    2. Live DKG knowledge asset via SDK (Config.DKG_POLKAWIKI_ASSET_UAL)
    3. dkg_asset_doc.json (single asset with hasPart and a known asset UAL, if present)
    4. dkg_docs_export.jsonl (multiple assets; caller is expected to know UALs separately)
    
    Uses file modification time for cache invalidation.
    """
    global _cache_key, _cached_index
    
    published_path = Path(published_index_path)
    current_key = published_path.stat().st_mtime if published_path.exists() else 0
    
    if _cache_key == current_key and _cached_index is not None:
        return _cached_index
    
    index = _load_published_chunk_index(published_index_path)
    
    if not index:
        index = _load_index_from_remote()

    if not index:
        asset_doc = _load_json_file(asset_doc_path)
        if asset_doc:
            index.update(_build_index_from_asset(asset_doc))

    if not index:
        assets = _load_jsonl_file(docs_export_path)
        for asset in assets:
            asset_ual = asset.get("asset_ual") or ""
            has_part = asset.get("hasPart") or []
            for part in has_part:
                identifier = part.get("identifier")
                if not identifier:
                    continue
                chunk_hash = ""
                for prop in part.get("additionalProperty", []):
                    if prop.get("name") in ("chunkHash", "chunk_hash"):
                        chunk_hash = str(prop.get("value") or "")
                        break
                if not chunk_hash:
                    continue
                index.setdefault(identifier, (asset_ual, chunk_hash))

    if not index:
        logger.info("DKG chunk index is empty; dkg_match will remain None.")
    else:
        logger.info(f"Loaded DKG chunk index with {len(index)} entries.")
    
    _cache_key = current_key
    _cached_index = index

    return index


