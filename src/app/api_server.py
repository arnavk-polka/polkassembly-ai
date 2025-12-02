"""
FastAPI server for the Polkadot AI Chatbot system.
Provides endpoints for querying the knowledge base and getting AI-generated answers.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.config import Config
from .auth import authenticate_request, get_auth_status
from ..core.embeddings import EmbeddingManager
from ..core.qa_generator import QAGenerator
from ..safety.bedrock_guardrail import check_with_guardrail_async, generate_user_friendly_block_message
from ..core.rate_limiter import check_rate_limit, get_client_stats
from ..core.reranking.chunks_reranker import rerank_static_chunks
from ..ops.monitoring import (
    initialize_slack_bot,
    send_startup_notification,
    send_shutdown_notification,
    send_startup_error_notification,
    send_runtime_error_notification,
    send_query_error_notification,
    send_crash_notification,
    set_shutdown_reason,
)
from .query_pipeline import processUserQuery, set_reranker
from ..core.errors import is_insufficient_quota_error, get_quota_error_message

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

static_embedding_manager: Optional[EmbeddingManager] = None
dynamic_embedding_manager: Optional[EmbeddingManager] = None
qa_generator: Optional[QAGenerator] = None
slack_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global static_embedding_manager, dynamic_embedding_manager, qa_generator, slack_bot
    
    try:
        Config.validate_config()
        
        static_embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.OPENAI_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        
        dynamic_embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.OPENAI_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=Config.CHROMA_DYNAMIC_COLLECTION_NAME
        )
        
        # Check if collections have data
        if not static_embedding_manager.collection_exists():
            logger.warning("Static ChromaDB collection is empty. Please run create_embeddings.py first.")
        
        if not dynamic_embedding_manager.collection_exists():
            logger.warning("Dynamic ChromaDB collection is empty. Please run create_dynamic_embeddings.py first.")
        
        qa_generator = QAGenerator(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_MODEL,
            temperature=0.1,
            enable_web_search=Config.ENABLE_WEB_SEARCH,
            web_search_context_size=Config.WEB_SEARCH_CONTEXT_SIZE,
            enable_memory=Config.USE_MEM0 and bool(Config.MEM0_API_KEY)
        )
        
        slack_bot = initialize_slack_bot()
        send_startup_notification(slack_bot)
        
        try:
            from ..core.reranking.semantic_reranker import SemanticReranker
            reranker = SemanticReranker()
            set_reranker(reranker)
        except ImportError as e:
            logger.warning(f"Semantic reranker not available: {e}. Reranking will be skipped.")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic reranker: {e}. Reranking will be skipped.")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        logger.error(traceback.format_exc())
        set_shutdown_reason(f"Startup failed: {type(e).__name__}: {str(e)}", e)
        send_startup_error_notification(slack_bot, e)
        raise e
    
    try:
        yield
    except Exception as e:
        logger.error(f"Unhandled exception during runtime: {e}")
        logger.error(traceback.format_exc())
        set_shutdown_reason(f"Runtime error: {type(e).__name__}: {str(e)}", e)
    
    send_shutdown_notification(slack_bot)

app = FastAPI(
    title="Polkadot AI Chatbot API",
    description="AI-powered chatbot for Polkadot ecosystem questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConversationMessage(BaseModel):
    query: str = Field(..., description="Previous user query")
    response: str = Field(..., description="Previous AI response")
    timestamp: str = Field(..., description="ISO timestamp of the message")
    original_answer: Optional[str] = Field(default=None, description="Deprecated - no longer used")

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask", min_length=1, max_length=500)
    user_id: str = Field(..., description="Unique user identifier", min_length=1, max_length=100)
    client_ip: str = Field(..., description="Client IP address or client identifier")
    max_chunks: int = Field(default=5, description="Maximum number of chunks to retrieve", ge=1, le=10)
    include_sources: bool = Field(default=True, description="Whether to include source information")
    custom_prompt: Optional[str] = Field(default=None, description="Custom system prompt for the AI")
    conversation_history: Optional[List[ConversationMessage]] = Field(default=[], description="Previous conversation history")

class Source(BaseModel):
    title: str
    url: str
    source_type: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    follow_up_questions: List[str]
    remaining_requests: int = Field(..., description="Number of remaining requests for this user")
    confidence: float = Field(..., description="Confidence score of the answer")
    context_used: bool = Field(..., description="Whether context was used")
    model_used: str = Field(..., description="AI model used")
    chunks_used: int = Field(..., description="Number of chunks used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")
    search_method: str = Field(..., description="Method used for search")
    original_answer: Optional[str] = Field(default=None, description="Original answer with markers for conversation history")

class HealthResponse(BaseModel):
    status: str
    collection_stats: Dict[str, Any]
    timestamp: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=200)
    n_results: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    source_filter: Optional[str] = Field(default=None, description="Filter by source type")

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float
    timestamp: str



@app.get("/health", response_model=HealthResponse)
async def health_check(authenticated: bool = Depends(authenticate_request)):
    """Health check endpoint"""
    try:
        if not static_embedding_manager or not dynamic_embedding_manager:
            raise HTTPException(status_code=503, detail="Embedding managers not initialized")
        
        stats = static_embedding_manager.get_collection_stats()
        
        return HealthResponse(
            status="healthy" if stats.get('total_chunks', 0) > 0 else "no_data",
            collection_stats=stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest, authenticated: bool = Depends(authenticate_request)):
    """Main chatbot query endpoint with enhanced guardrails and rate limiting"""
    start_time = datetime.now()
    
    try:
        if not static_embedding_manager or not dynamic_embedding_manager or not qa_generator:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not static_embedding_manager.collection_exists() and not dynamic_embedding_manager.collection_exists():
            raise HTTPException(status_code=503, detail="No data available. Please create embeddings first.")
        
        # Check rate limit using user_id as primary identifier
        is_allowed, remaining_requests = check_rate_limit(request.user_id)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for user {request.user_id} from IP {request.client_ip}")
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        guardrail_result = await check_with_guardrail_async(request.question)
        
        if guardrail_result["status"] == "blocked":
            violation_details = guardrail_result.get('violation_details', {})
            reason = guardrail_result.get('reason', 'Content policy violation')
            logger.warning(f"Query blocked by guardrail for user {request.user_id} from IP {request.client_ip}: {reason}")
            
            try:
                answer = await generate_user_friendly_block_message(violation_details, request.question)
            except Exception as e:
                logger.error(f"Failed to generate user-friendly block message: {e}")
                answer = "Your query was blocked because it violates our content policy. Please revise your query to comply with our terms of service. Continued violations may result in your IP being blocked."
            
            return QueryResponse(
                answer=answer,
                sources=[],
                follow_up_questions=[
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains communicate with each other?"
                ],
                remaining_requests=remaining_requests,
                confidence=0.0,
                context_used=False,
                model_used="guardrail",
                chunks_used=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat(),
                search_method="guardrail_blocked"
            )
        elif guardrail_result["status"] == "error":
            logger.error(f"Guardrail error for user {request.user_id}: {guardrail_result['reason']}")
        
        conversation_history_dicts = None
        if request.conversation_history:
            conversation_history_dicts = []
            for i, msg in enumerate(request.conversation_history):
                if msg.query:
                    conversation_history_dicts.append({
                        'role': 'user',
                        'content': msg.query,
                        'timestamp': msg.timestamp
                    })
                if msg.response:
                    conversation_history_dicts.append({
                        'role': 'assistant',
                        'content': msg.response,
                        'timestamp': msg.timestamp
                    })
        
        try:
            qa_result = await processUserQuery(
                userMessage=request.question,
                conversationHistory=conversation_history_dicts,
                static_embedding_manager=static_embedding_manager,
                dynamic_embedding_manager=dynamic_embedding_manager,
                qa_generator=qa_generator,
                max_chunks=request.max_chunks,
                custom_prompt=request.custom_prompt,
                user_id=request.user_id
            )
        except Exception as qa_error:
            if is_insufficient_quota_error(qa_error):
                logger.error(f"Insufficient quota error in processUserQuery: {qa_error}")
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                return QueryResponse(
                    answer=get_quota_error_message(),
                    sources=[],
                    follow_up_questions=[],
                    remaining_requests=remaining_requests,
                    confidence=0.0,
                    context_used=False,
                    model_used=Config.OPENAI_MODEL,
                    chunks_used=0,
                    processing_time_ms=processing_time,
                    timestamp=datetime.now().isoformat(),
                    search_method="quota_error"
                )
            
            logger.error(f"Error in processUserQuery: {qa_error}")
            send_query_error_notification(slack_bot, request.question, request.user_id, qa_error)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return QueryResponse(
                answer="I am currently having problems processing your prompt. Try it again in your next prompt.",
                sources=[],
                follow_up_questions=[
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains communicate with each other?"
                ],
                remaining_requests=remaining_requests,
                confidence=0.0,
                context_used=False,
                model_used=Config.OPENAI_MODEL,
                chunks_used=0,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                    search_method="query_processor_error"
            )
        
        sources = []
        if request.include_sources:
            sources = [
                Source(
                    title=src['title'],
                    url=src['url'],
                    source_type=src['source_type'],
                    similarity_score=src['similarity_score']
                )
                for src in qa_result.get('sources', [])
            ]
        
        processing_time = qa_result.get('processing_time_ms', (datetime.now() - start_time).total_seconds() * 1000)
        
        answer = qa_result['answer']
        if answer:
            answer = answer.strip()

        return QueryResponse(
            answer=answer,
            sources=sources,
            follow_up_questions=qa_result.get('follow_up_questions', []),
            remaining_requests=remaining_requests,
            confidence=qa_result.get('confidence', 0.0),
            context_used=qa_result.get('context_used', False),
            model_used=qa_result.get('model_used', Config.OPENAI_MODEL),
            chunks_used=qa_result.get('chunks_used', 0),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            search_method=qa_result.get('search_method', 'unknown'),
            original_answer=None
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        
        if is_insufficient_quota_error(e):
            try:
                _, remaining_requests = check_rate_limit(request.user_id)
            except:
                remaining_requests = 0
            
            return QueryResponse(
                answer=get_quota_error_message(),
                sources=[],
                follow_up_questions=[],
                remaining_requests=remaining_requests,
                confidence=0.0,
                context_used=False,
                model_used=Config.OPENAI_MODEL,
                chunks_used=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat(),
                search_method="quota_error"
            )
        
        try:
            _, remaining_requests = check_rate_limit(request.user_id)
        except:
            remaining_requests = 0
        
        return QueryResponse(
            answer="I'd be happy to help you with Polkadot questions! What would you like to know about governance, staking, or parachains?",
            sources=[],
            follow_up_questions=[
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains communicate with each other?"
            ],
            remaining_requests=remaining_requests,
            confidence=0.0,
            context_used=False,
            model_used=Config.OPENAI_MODEL,
            chunks_used=0,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            timestamp=datetime.now().isoformat(),
            search_method="error"
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, authenticated: bool = Depends(authenticate_request)):
    """Search for relevant document chunks"""
    start_time = datetime.now()
    
    try:
        if not static_embedding_manager or not dynamic_embedding_manager:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not static_embedding_manager.collection_exists() and not dynamic_embedding_manager.collection_exists():
            raise HTTPException(status_code=503, detail="No data available. Please create embeddings first.")
        
        filter_metadata = None
        if request.source_filter:
            filter_metadata = {"source": request.source_filter}
        
        static_chunks = static_embedding_manager.search_similar_chunks(
            query=request.query,
            n_results=request.n_results,
            filter_metadata=filter_metadata
        )
        
        dynamic_chunks = dynamic_embedding_manager.search_similar_chunks(
            query=request.query,
            n_results=request.n_results,
            filter_metadata=filter_metadata
        )

        all_chunks = static_chunks + dynamic_chunks
        all_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        results = [
            SearchResult(
                content=chunk['content'],
                metadata=chunk['metadata'],
                similarity_score=chunk['similarity_score']
            )
            for chunk in all_chunks[:request.n_results]
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_collection_stats(authenticated: bool = Depends(authenticate_request)):
    """Get collection statistics"""
    try:
        if not static_embedding_manager or not dynamic_embedding_manager:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        stats = static_embedding_manager.get_collection_stats()
        return {
            "collection_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rate-limit/{user_id}")
async def get_rate_limit_status(user_id: str, authenticated: bool = Depends(authenticate_request)):
    """Get rate limit status for a user"""
    try:
        stats = get_client_stats(user_id)
        return {
            "user_id": user_id,
            "rate_limit_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limit stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth-status")
async def get_authentication_status():
    """Get authentication configuration status (public endpoint)"""
    return get_auth_status()

@app.get("/")
async def root():
    """Root endpoint with API information (public endpoint)"""
    auth_info = get_auth_status()
    
    base_info = {
        "message": "Polkadot AI Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "authentication": auth_info,
        "endpoints": {
            "query": "POST /query - Ask questions about Polkadot",
            "search": "POST /search - Search document chunks", 
            "stats": "GET /stats - Get collection statistics",
            "health": "GET /health - Health check",
            "rate-limit": "GET /rate-limit/{user_id} - Get rate limit status",
            "auth-status": "GET /auth-status - Get authentication status"
        }
    }
    
    if auth_info["authentication_enabled"]:
        base_info["authentication_required"] = True
        base_info["auth_header_example"] = auth_info["auth_header_format"]
        base_info["note"] = "All endpoints except / and /auth-status require authentication"
    
    return base_info

if __name__ == "__main__":
    import uvicorn
    
    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_msg = f"Unhandled exception: {exc_type.__name__}: {str(exc_value)}"
        traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        logger.error(f"CRITICAL: {error_msg}")
        logger.error(traceback_str)
        
        send_crash_notification(exc_type, exc_value, traceback_str)
    
    sys.excepthook = handle_unhandled_exception
    
    logger.info("Starting Klara API server...")
    try:
        uvicorn.run(
            app,
            host=Config.API_HOST,
            port=Config.API_PORT,
            log_level="info",
            reload=False
        )
    except Exception as e:
        error_msg = f"Server startup/runtime error: {type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()
        logger.error(f"CRITICAL: {error_msg}")
        logger.error(traceback_str)
        
        send_crash_notification(type(e), e, traceback_str)
        
        sys.exit(1)