import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def determine_table_from_query(self, query: str) -> Optional[str]:
    import json
    
    from ...prompts.table_classifier_prompt import PROMPT_TEMPLATE
    prompt = PROMPT_TEMPLATE.format(query=query)

    try:
        if self.gemini_client:
            response = self.gemini_client.get_response(prompt)
            response = response.strip()
            
            try:
                result = json.loads(response)
                table = result.get('table', 'governance_data')
                if table in ['governance_data', 'voting_data']:
                    logger.info(f"Table selected by Gemini: {table}")
                    return table
            except json.JSONDecodeError:
                if 'voting_data' in response.lower():
                    logger.info("Table selected by Gemini (text parsing): voting_data")
                    return "voting_data"
        
        if self.client:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            response_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(response_text)
                table = result.get('table', 'governance_data')
                if table in ['governance_data', 'voting_data']:
                    logger.info(f"Table selected by OpenAI: {table}")
                    return table
            except json.JSONDecodeError:
                if 'voting_data' in response_text.lower():
                    logger.info("Table selected by OpenAI (text parsing): voting_data")
                    return "voting_data"
    
    except Exception as e:
        logger.warning(f"Error in LLM table selection: {e}")
    
    logger.info("Table selected (fallback): governance_data")
    return "governance_data"

def handle_dynamic_route(self, analyzed_query: str, conversation_history: Optional[List[Dict[str, Any]]], route_result_table: Optional[str], dynamic_embedding_manager) -> Dict[str, Any]:
    from ...dynamic_sql.query_api import ask_question
    from .follow_up_questions import generate_follow_up_questions, get_fallback_follow_ups
    
    try:
        sql_result = ask_question(analyzed_query, conversation_history, route_result_table, dynamic_embedding_manager)
        
        if sql_result.get('requires_clarification', False):
            return {
                'answer': '',
                'sources': [],
                'confidence': 0.0,
                'follow_up_questions': [],
                'context_used': False,
                'model_used': 'sql_query',
                'chunks_used': 0,
                'search_method': 'sql_precision_too_low',
                'sql_query': sql_result.get('sql_queries', []),
                'result_count': 0,
                'success': False,
                'requires_clarification': True,
                'sql_precision': sql_result.get('sql_precision', 0.0)
            }
        
        if sql_result.get('requires_fallback', False):
            return {
                'answer': '',
                'sources': [],
                'confidence': 0.0,
                'follow_up_questions': [],
                'context_used': False,
                'model_used': 'sql_query',
                'chunks_used': 0,
                'search_method': 'no_results',
                'sql_query': sql_result.get('sql_queries', []),
                'result_count': 0,
                'success': False,
                'requires_fallback': True
            }
        
        try:
            follow_up_questions = generate_follow_up_questions(self, analyzed_query, [], sql_result.get('natural_response', ''))
        except Exception:
            follow_up_questions = get_fallback_follow_ups(analyzed_query)
        
        return {
            'answer': sql_result.get('natural_response', 'No response available'),
            'sources': [],
            'confidence': 0.9 if sql_result.get('success', False) else 0.5,
            'follow_up_questions': follow_up_questions,
            'context_used': False,
            'model_used': 'sql_query',
            'chunks_used': 0,
            'search_method': 'sql_query',
            'sql_query': sql_result.get('sql_queries', []),
            'result_count': sql_result.get('result_count', 0),
            'success': sql_result.get('success', False)
        }
    except Exception as e:
        logger.error(f"Error in SQL query processing: {e}")
        from .follow_up_questions import get_fallback_follow_ups
        return {
            'answer': "I'm sorry, I encountered an error processing your database query. Please try rephrasing your question or try again later.",
            'sources': [],
            'confidence': 0.0,
            'follow_up_questions': get_fallback_follow_ups(analyzed_query),
            'context_used': False,
            'model_used': 'sql_query_error',
            'chunks_used': 0,
            'search_method': 'sql_error_fallback',
            'sql_query': [],
            'result_count': 0,
            'success': False
        }

def handle_hybrid_route(self, analyzed_query: str, conversation_history: Optional[List[Dict[str, Any]]], route_result_table: Optional[str], dynamic_embedding_manager) -> str:
    from ...dynamic_sql.query_api import ask_question
    
    dynamic_answer = None
    dynamic_data_available = False
    try:
        logger.info(f"Hybrid route: Executing SQL query for dynamic data. Query: {analyzed_query[:100]}, Table: {route_result_table}")
        sql_result = ask_question(analyzed_query, conversation_history, route_result_table, dynamic_embedding_manager)
        dynamic_answer = sql_result.get('natural_response', '')
        dynamic_data_available = sql_result.get('success', False) and bool(dynamic_answer)
        logger.info(f"Hybrid route: SQL query completed. Success: {dynamic_data_available}, Response length: {len(dynamic_answer) if dynamic_answer else 0}, Result count: {sql_result.get('result_count', 0)}")
    except Exception as e:
        logger.error(f"Error in hybrid SQL query processing: {e}", exc_info=True)
        dynamic_data_available = False
    
    if dynamic_answer and dynamic_data_available:
        updated_query = f"{analyzed_query}\n\nIMPORTANT: The user also requested specific data. Here is the dynamic data from the database:\n{dynamic_answer}\n\nPlease incorporate this data into your response along with the static context."
        logger.info(f"Hybrid route: Added dynamic data to query context. Dynamic answer preview: {dynamic_answer[:200]}")
        return updated_query
    elif not dynamic_data_available:
        logger.warning("Hybrid route: Dynamic data not available, proceeding with static only")
        return analyzed_query
    
    return analyzed_query

