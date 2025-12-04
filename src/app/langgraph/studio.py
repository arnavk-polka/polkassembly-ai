"""
LangGraph Studio entry point for visualization and debugging.
This file exposes the graph for LangGraph Studio with auto-initialized dependencies.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.app.langgraph.graph import _build_graph

_cached_static_manager = None
_cached_dynamic_manager = None
_cached_qa_generator = None

def _get_static_embedding_manager():
    global _cached_static_manager
    if _cached_static_manager is None:
        try:
            from src.core.embeddings import EmbeddingManager
            from src.core.config import Config
            _cached_static_manager = EmbeddingManager(
                openai_api_key=Config.OPENAI_API_KEY,
                embedding_model=Config.OPENAI_EMBEDDING_MODEL,
                chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                collection_name=Config.CHROMA_COLLECTION_NAME
            )
        except Exception as e:
            print(f"Warning: Could not init static embedding manager: {e}")
            _cached_static_manager = None
    return _cached_static_manager

def _get_dynamic_embedding_manager():
    global _cached_dynamic_manager
    if _cached_dynamic_manager is None:
        try:
            from src.core.embeddings import EmbeddingManager
            from src.core.config import Config
            _cached_dynamic_manager = EmbeddingManager(
                openai_api_key=Config.OPENAI_API_KEY,
                embedding_model=Config.OPENAI_EMBEDDING_MODEL,
                chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                collection_name=Config.CHROMA_DYNAMIC_COLLECTION_NAME
            )
        except Exception as e:
            print(f"Warning: Could not init dynamic embedding manager: {e}")
            _cached_dynamic_manager = None
    return _cached_dynamic_manager

def _get_qa_generator():
    global _cached_qa_generator
    if _cached_qa_generator is None:
        try:
            from src.core.generator import QAGenerator
            from src.core.config import Config
            _cached_qa_generator = QAGenerator(
                openai_api_key=Config.OPENAI_API_KEY,
                model=Config.OPENAI_MODEL,
                temperature=0.1,
            )
        except Exception as e:
            print(f"Warning: Could not init QA generator: {e}")
            _cached_qa_generator = None
    return _cached_qa_generator

def get_dependencies():
    return {
        "static_embedding_manager": _get_static_embedding_manager(),
        "dynamic_embedding_manager": _get_dynamic_embedding_manager(),
        "qa_generator": _get_qa_generator(),
    }

graph = _build_graph()

def get_graph():
    return graph

if __name__ == "__main__":
    print("Graph compiled successfully!")
    print(f"Graph type: {type(graph)}")
    deps = get_dependencies()
    print(f"Static manager: {'✅' if deps['static_embedding_manager'] else '❌'}")
    print(f"Dynamic manager: {'✅' if deps['dynamic_embedding_manager'] else '❌'}")
    print(f"QA generator: {'✅' if deps['qa_generator'] else '❌'}")

