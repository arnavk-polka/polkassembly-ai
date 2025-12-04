import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "polkadot_embeddings")
    CHROMA_DYNAMIC_COLLECTION_NAME = os.getenv("CHROMA_DYNAMIC_COLLECTION_NAME", "polkadot_embeddings_dynamic")
    
    SEARCH_STATIC_DATA = os.getenv("SEARCH_STATIC_DATA", "true").lower() == "true"
    SEARCH_DYNAMIC_DATA = os.getenv("SEARCH_DYNAMIC_DATA", "true").lower() == "true"
    
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", 20))
    RATE_LIMIT_EXPIRE_SECONDS = int(os.getenv("RATE_LIMIT_EXPIRE_SECONDS", 3600))
    
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    
    STATIC_DATA_PATH = os.getenv("STATIC_DATA_PATH", "data/joined_data/static")
    
    ENABLE_CONTENT_FILTERING = os.getenv("ENABLE_CONTENT_FILTERING", "true").lower() == "true"
    BLOCKED_DOMAINS = os.getenv("BLOCKED_DOMAINS", "subsquare.io,subsquare.com,subsquare.network").split(",")
    PREFERRED_DOMAINS = os.getenv("PREFERRED_DOMAINS", "polkadot.io,polkadot.network,polkassembly.io").split(",")
    
    
    
    POLKASSEMBLY_AI_TOKEN = os.getenv("POLKASSEMBLY_AI_TOKEN", "")
    ENABLE_AUTHENTICATION = os.getenv("ENABLE_AUTHENTICATION", "true").lower() == "true"
    
    ENABLE_SLACK_NOTIFICATIONS = os.getenv("ENABLE_SLACK_NOTIFICATIONS", "false").lower() in ("true", "1", "yes", "on")
    
    USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        
        if cls.ENABLE_AUTHENTICATION and not cls.POLKASSEMBLY_AI_TOKEN:
            raise ValueError("POLKASSEMBLY_AI_TOKEN must be set when authentication is enabled")

        return True 