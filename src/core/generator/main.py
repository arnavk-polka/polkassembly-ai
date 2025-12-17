import openai
import logging
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

from ..gemini_client import GeminiClient
from ...integrations.slack_bot import SlackBot
from .utils import GEMINI_TIMEOUT
from .context_processing import create_context_from_chunks, remove_double_asterisks, clean_example_urls
from .query_analysis import analyze_query_with_context, format_conversation_history, parse_gemini_response
from .sql_handlers import determine_table_from_query, handle_dynamic_route, handle_hybrid_route
from .llm_response import get_default_system_prompt, create_user_prompt
from .response_processing import extract_sources, estimate_confidence
from .follow_up_questions import generate_follow_up_questions, get_fallback_follow_ups
from ..errors import is_insufficient_quota_error, get_quota_error_message

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAGenerator:
    def __init__(self, 
                 openai_api_key: str,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        
        self.openai_api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.api_timeout = float(os.getenv('API_TIMEOUT', '10'))
        
        self.client = openai.OpenAI(api_key=self.openai_api_key, timeout=self.api_timeout)
        
        try:
            self.gemini_client = GeminiClient(timeout=GEMINI_TIMEOUT)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.warning(f"Gemini client initialization failed: {e}")
            logger.info("Continuing without Gemini client (OpenAI only mode)")
            self.gemini_client = None
        
        logger.info("Content moderation will be handled by Model Armor")
        
        try:
            self.slack_bot = SlackBot()
            logger.info("Slack bot initialized for error notifications")
        except Exception as e:
            logger.warning(f"Slack bot initialization failed: {e}")
            logger.info("Continuing without Slack notifications")
            self.slack_bot = None
    
    def send_error_to_slack(self, query: str, error: str, error_source: str = "Klara") -> None:
        if not self.slack_bot:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            error_context = {
                "query": query,
                "timestamp": timestamp,
                "error_source": error_source,
                "error_details": str(error)
            }
            
            self.slack_bot.post_error_to_slack(
                "Query processing failed", 
                context=error_context
            )
            logger.info("Error notification sent to Slack")
        except Exception as slack_error:
            logger.error(f"Failed to send error notification to Slack: {slack_error}")
    
    def create_context_from_chunks(self, chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
        return create_context_from_chunks(chunks, max_context_length)
    
    def remove_double_asterisks(self, text):
        return remove_double_asterisks(text)
    
    def clean_example_urls(self, text):
        return clean_example_urls(text)
    
    def _fallback_to_static_response(self, query: str, user_id: str) -> Dict[str, Any]:
        return {
            'answer': "I encountered an issue fetching the specific proposal data you requested. Please try rephrasing your question or check if the proposal ID is correct. You can also try asking about general Polkadot governance topics.",
            'sources': [
                {
                    'title': 'Polkadot Governance',
                    'url': 'https://polkadot.polkassembly.io',
                    'source_type': 'platform',
                    'similarity_score': 0.8
                }
            ],
            'confidence': 0.3,
            'context_used': False,
            'model_used': self.model,
            'chunks_used': 0,
            'search_method': 'dynamic_fallback',
            'error': 'Failed to fetch proposal data'
        }

    def analyze_query_with_context(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        return analyze_query_with_context(self, query, conversation_history)
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        return format_conversation_history(history)
    
    def _parse_gemini_response(self, response: str, fallback_query: str) -> str:
        return parse_gemini_response(response, fallback_query)
    
    def _determine_table_from_query(self, query: str) -> Optional[str]:
        return determine_table_from_query(self, query)
    
    def _get_default_system_prompt(self) -> str:
        return get_default_system_prompt()
    
    def _create_user_prompt(self, query: str, context: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        return create_user_prompt(self, query, context, conversation_history)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return extract_sources(self, chunks)
    
    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        return estimate_confidence(self, chunks)
    
    
    def _generate_follow_up_questions(self, query: str, chunks: List[Dict[str, Any]], answer: str) -> List[str]:
        return generate_follow_up_questions(self, query, chunks, answer)
    
    def _get_fallback_follow_ups(self, query: str) -> List[str]:
        return get_fallback_follow_ups(query)

    async def generate_answer(self, 
                       query: str, 
                       chunks: List[Dict[str, Any]], 
                       custom_prompt: Optional[str] = None,
                       user_id: str = "default_user",
                       conversation_history: Optional[List[Dict[str, Any]]] = None,
                       route: Optional[str] = None,
                       route_confidence: Optional[float] = None,
                       dynamic_embedding_manager=None) -> Dict[str, Any]:
        try:
            print(f"\033[92m📝 User Query: {query}\033[0m")
            
            analyzed_query = query
            
            if route == 'dynamic':
                route_result_data_source = 'ONCHAIN'
                route_result_table = self._determine_table_from_query(query)
            elif route == 'hybrid':
                route_result_data_source = 'HYBRID'
                route_result_table = self._determine_table_from_query(query)
                logger.info(f"Hybrid route detected: will execute SQL query and combine with static chunks")
            elif route == 'generic':
                route_result_data_source = 'STATIC'
                route_result_table = None
            else:
                route_result_data_source = 'STATIC'
                route_result_table = None
            
            logger.info(f"Using route: {route} -> {route_result_data_source}, table: {route_result_table}")
            
            if route_result_data_source == 'ONCHAIN':
                return handle_dynamic_route(self, analyzed_query, conversation_history, route_result_table, dynamic_embedding_manager)
            
            if route_result_data_source == 'HYBRID':
                logger.info("Hybrid route: Entering hybrid processing block")
                analyzed_query = handle_hybrid_route(self, analyzed_query, conversation_history, route_result_table, dynamic_embedding_manager)
            
            try:
                context = self.create_context_from_chunks(chunks, max_context_length=8000)
                print("context from chunks", context)
                print("context after strip", context.strip())
            except Exception as context_error:
                logger.error(f"Error creating context from chunks: {context_error}")
                return {
                    'answer': "I'm sorry, I encountered an error processing your request. Please try again.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': 'error',
                    'chunks_used': len(chunks),
                    'follow_up_questions': get_fallback_follow_ups(query),
                    'search_method': 'context_error'
                }
            
            has_sufficient_context = (
                context.strip() and 
                len(chunks) > 0 and 
                any(chunk.get('similarity_score', 0) > float(os.getenv("SIMILARITY_THRESHOLD", "0.7")) for chunk in chunks)
            )

            if not has_sufficient_context:
                max_score = max([chunk.get('similarity_score', 0) for chunk in chunks]) if chunks else 0
                has_good_similarity = any(chunk.get('similarity_score', 0) > 0.6 for chunk in chunks)
                has_chunks = len(chunks) > 0
                has_content = bool(context.strip())
                
                if not has_chunks:
                    reason = "No relevant chunks found"
                elif not has_good_similarity:
                    reason = f"Low similarity scores (max: {max_score:.3f} < 0.6)"
                elif not has_content:
                    reason = f"Retrieved chunks contain no usable content (similarity: {max_score:.3f})"
                else:
                    reason = "Unknown context issue"
                
                logger.info(f"Insufficient context: {reason}. Query is outside knowledge boundary.")
                
                follow_up_questions = [
                    "How does Polkadot's governance system work?",
                    "What are parachains and how do they connect to Polkadot?",
                    "How can I stake DOT tokens on Polkadot?",
                    "What is the difference between Polkadot and Kusama?"
                ]
                
                return {
                    'answer': "I couldn't find sufficient information to answer your question accurately. This query appears to be outside my knowledge boundary for Polkadot-related topics. Please try rephrasing your question or ask about a more specific Polkadot topic.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': self.model,
                    'chunks_used': len(chunks),
                    'follow_up_questions': follow_up_questions,
                    'search_method': 'insufficient_context',
                    'max_similarity_score': max_score,
                    'similarity_threshold': 0.6,
                    'failure_reason': reason
                }
            
            if not context.strip():
                return {
                    'answer': "I could not find sufficient information about the query. Please try rephrasing your query or ask about a different topic related to Polkaseembly.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'search_method': 'local_only'
                }
            
            try:
                system_prompt = custom_prompt or self._get_default_system_prompt()
            except Exception as system_prompt_error:
                logger.warning(f"Error creating system prompt: {system_prompt_error}")
                system_prompt = "You are a helpful AI assistant for Polkadot-related questions."
            
            print("context before going to openAI prompt", context)
            try:
                user_prompt = self._create_user_prompt(analyzed_query, context, conversation_history)
            except Exception as user_prompt_error:
                logger.error(f"Error creating user prompt: {user_prompt_error}")
                return {
                    'answer': "I'm sorry, I encountered an error preparing your request. Please try again.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': 'error',
                    'chunks_used': len(chunks),
                    'follow_up_questions': get_fallback_follow_ups(query),
                    'search_method': 'prompt_error'
                }
            
            answer = None
            openai_enabled = os.getenv("ENABLE_OPENAI", "").lower() == "true"
            gemini_enabled = os.getenv("ENABLE_GEMINI", "").lower() == "true"
            system_prompt = self._get_default_system_prompt()
            
            try:
                if openai_enabled:
                    from .utils import print_model_usage
                    print_model_usage("GPT-3.5-turbo", "response generation (static data)")
                    logger.info("Using OpenAI for response generation")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    answer = response.choices[0].message.content
                    logger.info("OpenAI response received successfully")
                
                elif gemini_enabled and self.gemini_client:
                    from .utils import print_model_usage
                    model_name = getattr(self.gemini_client, 'model_name', 'Gemini')
                    print_model_usage(f"{model_name}", "response generation (static data)")
                    logger.info("Using Gemini for response generation")
                    try:
                        answer = self.gemini_client.get_response(system_prompt + "\n\n" + user_prompt)
                        logger.info("Gemini response received successfully")
                    except Exception as gemini_error:
                        logger.warning(f"Gemini response failed: {gemini_error}. Falling back to OpenAI.")
                        if self.client:
                            print_model_usage(self.model, "response generation fallback after Gemini error")
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=self.temperature,
                                max_tokens=self.max_tokens
                            )
                            answer = response.choices[0].message.content
                            logger.info("OpenAI fallback response received successfully after Gemini error")
                        else:
                            raise gemini_error
                    
                else:
                    from .utils import print_model_usage
                    print_model_usage("GPT-3.5-turbo", "response generation fallback (static data)")
                    logger.warning("No AI service explicitly enabled, falling back to OpenAI")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    answer = response.choices[0].message.content
                    logger.info("OpenAI fallback response received successfully")
            except Exception as llm_error:
                if is_insufficient_quota_error(llm_error):
                    logger.error(f"Insufficient quota error in LLM response generation: {llm_error}")
                    return {
                        'answer': get_quota_error_message(),
                        'sources': [],
                        'confidence': 0.0,
                        'context_used': False,
                        'model_used': 'error',
                        'chunks_used': len(chunks),
                        'follow_up_questions': [],
                        'search_method': 'quota_error'
                    }
                logger.error(f"Error in LLM response generation: {llm_error}")
                return {
                    'answer': "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': 'error',
                    'chunks_used': len(chunks),
                    'follow_up_questions': get_fallback_follow_ups(query),
                    'search_method': 'error_fallback'
                }
            
            print("--------answer without strip-------\n", answer)
            
            try:
                answer = self.clean_example_urls(answer)
                print("--------answer after cleaning example.com URLs-------\n", answer)
            except Exception as clean_error:
                logger.warning(f"Error cleaning URLs from response: {clean_error}")
            
            try:
                sources = self._extract_sources(chunks)
            except Exception as source_error:
                logger.warning(f"Error extracting sources: {source_error}")
                sources = []
            
            try:
                confidence = self._estimate_confidence(chunks)
            except Exception as confidence_error:
                logger.warning(f"Error estimating confidence: {confidence_error}")
                confidence = 0.5
            
            try:
                follow_up_questions = self._generate_follow_up_questions(query, chunks, answer)
            except Exception as followup_error:
                logger.warning(f"Error generating follow-up questions: {followup_error}")
                follow_up_questions = get_fallback_follow_ups(query)
            
            search_method = 'local_knowledge'
            if route_result_data_source == 'HYBRID':
                search_method = 'hybrid_static_and_dynamic'
            elif route_result_data_source == 'STATIC':
                search_method = 'static_embeddings'
            
            result = {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'follow_up_questions': follow_up_questions,
                'context_used': True,
                'model_used': self.model,
                'chunks_used': len(chunks),
                'search_method': search_method
            }
            
            logger.info(f"Generated answer for query: '{query[:50]}...' using {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            self.send_error_to_slack(query, str(e))
            
            return {
                'answer': "I'm having trouble processing your query. Please try again or rephrase your question in the next prompt.",
                'sources': [],
                'confidence': 0.0,
                'context_used': False,
                'model_used': self.model,
                'chunks_used': 0,
                'search_method': 'error_fallback',
                'error': True,
                'follow_up_questions': get_fallback_follow_ups(query)
            }

