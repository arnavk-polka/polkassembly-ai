"""
Create embeddings for router training examples.
This enables dynamic few-shot retrieval at inference time.
"""

import os
import sys
import csv
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.core.config import Config
from src.core.embeddings import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_router_examples(csv_path: str) -> list:
    examples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get('question', '').strip()
            route = row.get('route_expected', '').strip().lower()
            
            if not question or not route:
                continue
            
            if route not in ['static', 'dynamic', 'hybrid', 'generic']:
                continue
            
            network = row.get('network', '').strip().lower()
            proposal_index = row.get('proposal_index', '').strip()
            needs = row.get('needs', '').strip()
            
            examples.append({
                'content': question,
                'metadata': {
                    'route': route,
                    'network': network if network in ['polkadot', 'kusama'] else '',
                    'proposal_index': proposal_index,
                    'needs': needs,
                    'source': 'router_training'
                }
            })
    
    return examples


def create_router_embeddings(csv_path: str = None, collection_name: str = None) -> bool:
    try:
        if not csv_path:
            csv_path = os.getenv("ROUTER_TRAINING_DATA", "src/core/routing/data/route.csv")
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(project_root, csv_path)
        
        collection_name = collection_name or Config.CHROMA_ROUTER_COLLECTION_NAME
        
        logger.info("=" * 60)
        logger.info("ROUTER EMBEDDINGS CREATION STARTED")
        logger.info("=" * 60)
        logger.info(f"CSV path: {csv_path}")
        logger.info(f"Collection name: {collection_name}")
        logger.info(f"Embedding model: {Config.ROUTER_EMBEDDING_MODEL}")
        logger.info("-" * 60)
        
        embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.ROUTER_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=collection_name
        )
        
        logger.info("Clearing existing collection...")
        embedding_manager.clear_collection()
        
        examples = load_router_examples(csv_path)
        if not examples:
            logger.warning("No router examples found")
            return False
        
        logger.info(f"Loaded {len(examples)} router examples")
        
        route_counts = {}
        for ex in examples:
            route = ex['metadata']['route']
            route_counts[route] = route_counts.get(route, 0) + 1
        logger.info(f"Route distribution: {route_counts}")
        
        logger.info("Generating embeddings and adding to collection...")
        success = embedding_manager.add_chunks_to_collection(examples)
        
        if success:
            stats = embedding_manager.get_collection_stats()
            logger.info("=" * 60)
            logger.info("ROUTER EMBEDDINGS CREATION COMPLETE")
            logger.info(f"Total examples indexed: {stats.get('total_chunks', 0)}")
            logger.info("=" * 60)
        
        return success
        
    except Exception as e:
        logger.error(f"Error creating router embeddings: {e}", exc_info=True)
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create router training embeddings")
    parser.add_argument("--csv", type=str, help="Path to route.csv file")
    parser.add_argument("--collection", type=str, help="ChromaDB collection name")
    args = parser.parse_args()
    
    success = create_router_embeddings(
        csv_path=args.csv,
        collection_name=args.collection
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

