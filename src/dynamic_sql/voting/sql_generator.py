import json
import logging
from typing import Dict, List, Any, Optional

from ..utils.model_usage import print_model_usage, GeminiClient, GEMINI_MODEL_NAME, GEMINI_MODEL_SQL, GEMINI_SQL_TIMEOUT, GEMINI_TIMEOUT
from ...prompts.voting_sql_system_prompt import PROMPT as voting_sql_system_prompt
from ...prompts.voting_sql_generation_prompt import PROMPT_TEMPLATE as voting_sql_generation_template

logger = logging.getLogger(__name__)

def _gemini_response_has_error(response_text: Optional[str]) -> bool:
    """Gemini client returns human-readable error strings instead of raising. Detect those."""
    if not response_text:
        return True
    normalized = response_text.strip().lower()
    error_markers = [
        "error generating response",
        "request timed out",
        "operation timed out",
        "model is overloaded",
        "503",
        "service unavailable",
        "unavailable"
    ]
    return any(marker in normalized for marker in error_markers)

def generate_sql_with_model(system_prompt: str, sql_model: str, openai_client, gemini_client) -> str:
    """Generate SQL using Gemini as primary and OpenAI as fallback for voting data"""
    try:
        if gemini_client:
            print_model_usage(f"{GEMINI_MODEL_SQL}", "SQL generation (voting data)")
            logger.debug("Using Gemini for voting SQL generation")
            
            full_prompt = f"""{voting_sql_system_prompt}

{system_prompt}"""
            
            try:
                response = gemini_client.get_response(full_prompt)
                if _gemini_response_has_error(response):
                    raise RuntimeError(response)
                return response.strip()
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                    logger.warning(f"Gemini SQL model overloaded (503 error), falling back to general Gemini model for voting: {e}")
                    try:
                        print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (voting data)")
                        fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                        response = fallback_client.get_response(full_prompt)
                        if _gemini_response_has_error(response):
                            raise RuntimeError(response)
                        logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for voting SQL generation")
                        return response.strip()
                    except Exception as fallback_error:
                        logger.error(f"Fallback Gemini model also failed for voting: {fallback_error}")
                else:
                    raise e
        
        if openai_client:
            print_model_usage("GPT-4", "SQL generation fallback (voting data)")
            logger.debug("Using ChatGPT as fallback for voting SQL generation")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": voting_sql_system_prompt},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        else:
            raise ValueError("No SQL generation model available for voting data")
            
    except Exception as e:
        logger.error(f"Error in primary SQL model for voting, trying fallback: {e}")
        
        if sql_model != 'chatgpt' and openai_client:
            logger.info("Falling back to ChatGPT for voting SQL generation")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": voting_sql_system_prompt},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        elif sql_model == 'chatgpt' and gemini_client:
            logger.info("Falling back to Gemini 2.5 Pro for voting SQL generation")
            full_prompt = f"""{voting_sql_system_prompt}

{system_prompt}"""
            response = gemini_client.get_response(full_prompt)
            return response.strip()
        else:
            raise e

def generate_sql_queries_only_voting(natural_query: str, conversation_history: Optional[List[Dict[str, Any]]], 
                                    table_schema: str, table_name: str, sql_model: str,
                                    openai_client, gemini_client, trim_prompt_func, max_retries: int = 3) -> List[str]:
    """Generate SQL queries for voting data without executing them"""
    base_system_prompt = voting_sql_generation_template.format(
        table_name=table_name,
        table_schema=table_schema,
        natural_query=natural_query
    )
    
    for attempt in range(max_retries):
        try:
            system_prompt = base_system_prompt
            system_prompt = trim_prompt_func(system_prompt)
            
            response_content = generate_sql_with_model(system_prompt, sql_model, openai_client, gemini_client)
            response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
            
            try:
                sql_queries = json.loads(response_content)
                
                if isinstance(sql_queries, list) and len(sql_queries) > 0 and isinstance(sql_queries[0], dict):
                    sql_queries = [item.get('query', str(item)) for item in sql_queries]
                elif isinstance(sql_queries, str):
                    sql_queries = [sql_queries]
                elif not isinstance(sql_queries, list):
                    sql_queries = [str(sql_queries)]
                
                normalized_queries = []
                for q in sql_queries:
                    if isinstance(q, dict):
                        if 'query' in q:
                            normalized_queries.append(q['query'])
                        else:
                            normalized_queries.append(str(q))
                    else:
                        normalized_queries.append(str(q))
                sql_queries = normalized_queries
                
                logger.info(f"Generated {len(sql_queries)} SQL queries for voting (attempt {attempt + 1}): {sql_queries}")
                return sql_queries
                
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed to parse JSON.")
                    return [response_content.strip()]
                else:
                    continue
                    
        except Exception as e:
            logger.error(f"Error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise e
            continue
    
    return []

