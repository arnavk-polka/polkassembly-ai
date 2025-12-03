import logging
from typing import Dict, List, Any, Optional

from ..utils.model_usage import print_model_usage, GeminiClient, GEMINI_MODEL_NAME, GEMINI_TIMEOUT
from ...prompts.voting_natural_response_prompt import PROMPT as voting_natural_response_system_prompt, PROMPT_TEMPLATE as voting_natural_response_template

logger = logging.getLogger(__name__)

def generate_natural_response(natural_query: str, sql_query: str, results: List[List[Any]], 
                             columns: List[str], conversation_history: Optional[List[Dict[str, Any]]],
                             gemini_client, openai_client) -> str:
    """Generate natural language response from SQL results for voting data"""
    try:
        if not results:
            return f"I didn't find any voting records matching your query '{natural_query}'. This could mean there are no votes matching your criteria, or the voting data might not contain the specific information you're looking for."
        
        is_count_query = 'COUNT(*)' in sql_query.upper() or (len(columns) == 1 and 'count' in columns[0].lower())
        count_value = None
        if is_count_query and results and len(results) > 0 and len(results[0]) > 0:
            count_value = results[0][0] if isinstance(results[0][0], (int, float)) else None
        
        total_count_from_window = None
        if results and len(results) > 0 and len(results[0]) > 0:
            if 'total_count' in str(results[0]):
                for i, col in enumerate(columns):
                    if 'total_count' in col.lower():
                        total_count_from_window = results[0][i]
                        break
        
        if count_value is not None:
            actual_total_count = int(count_value)
        elif total_count_from_window is not None:
            actual_total_count = total_count_from_window
        else:
            actual_total_count = len(results)
        
        displayed_count = min(10, len(results))
        if actual_total_count <= displayed_count:
            results_summary = f"Found {actual_total_count} voting records"
            sample_data = results[:displayed_count]
        else:
            results_summary = f"Found {actual_total_count} voting records (showing few due to length)"
            sample_data = results[:displayed_count]
        
        has_context = conversation_history and len(conversation_history) > 0
        context_info = ""
        if has_context:
            recent_topics = []
            for msg in conversation_history[-6:]:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if content and len(content) > 10:
                        recent_topics.append(content[:100])
            
            if recent_topics:
                context_info = f"Previous conversation topics: {'; '.join(recent_topics)}"
        
        context_prompt = voting_natural_response_template.format(
            natural_query=natural_query,
            sql_query=sql_query,
            results_summary=results_summary,
            columns=columns,
            sample_data=sample_data,
            context_info=context_info
        )
        
        if gemini_client is not None:
            try:
                print_model_usage(f"{GEMINI_MODEL_NAME}", "natural response generation (voting data)")
                logger.info("Using Gemini as primary LLM for voting natural response generation")
                system_prompt = voting_natural_response_system_prompt
                full_prompt = system_prompt + "\n\n" + context_prompt
                natural_response_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                natural_response = natural_response_client.get_response(full_prompt)
                logger.info("Generated voting natural language response using Gemini")
                disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                return natural_response + disclaimer
            except Exception as gemini_error:
                error_str = str(gemini_error).lower()
                if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                    logger.warning(f"Gemini model overloaded (503 error) for voting natural response, falling back to general Gemini model: {gemini_error}")
                    try:
                        print_model_usage(f"{GEMINI_MODEL_NAME}", "natural response generation fallback (voting data)")
                        fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                        system_prompt = voting_natural_response_system_prompt
                        full_prompt = system_prompt + "\n\n" + context_prompt
                        natural_response = fallback_client.get_response(full_prompt)
                        logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for voting natural response generation")
                        disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                        return natural_response + disclaimer
                    except Exception as fallback_error:
                        logger.error(f"Fallback Gemini model also failed for voting natural response: {fallback_error}")
                        logger.warning(f"Gemini failed for voting natural response, falling back to OpenAI: {gemini_error}")
                else:
                    logger.warning(f"Gemini failed for voting natural response, falling back to OpenAI: {gemini_error}")
        
        print_model_usage("GPT-4", "natural response generation fallback (voting data)")
        logger.info("Using OpenAI for voting natural response generation (fallback)")
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": voting_natural_response_system_prompt},
                {"role": "user", "content": context_prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        natural_response = response.choices[0].message.content.strip()
        disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
        return natural_response + disclaimer
        
    except Exception as e:
        logger.error(f"Error generating natural response: {e}")
        return f"I found {len(results)} voting records for your query '{natural_query}', but I'm having trouble formatting the response. Here's a summary: The query returned {len(results)} rows from the voting database."

