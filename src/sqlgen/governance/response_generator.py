import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable

from ..utils.model_usage import print_model_usage, GeminiClient, GEMINI_MODEL_NAME, GEMINI_TIMEOUT
from ...prompts.natural_response_prompt import PROMPT_TEMPLATE as natural_response_prompt_template
from ...prompts.natural_response_multiple_prompt import PROMPT_TEMPLATE as natural_response_multiple_prompt_template

logger = logging.getLogger(__name__)

def generate_natural_response(natural_query: str, sql_query: str, results: List[Dict[str, Any]], 
                             columns: List[str], conversation_history: Optional[List[Dict[str, Any]]],
                             gemini_client, openai_client, format_number_func: Callable) -> str:
    """Generate natural language response from query results - uses GPT-4o as primary, Gemini as fallback"""
    try:
        result_count = len(results)
        
        if result_count == 0:
            return f"I couldn't find any data matching your query: '{natural_query}'. The database might not contain the specific information you're looking for."
        
        sample_results = results[:10]
        trimmed_results = []
        for result in sample_results:
            trimmed_result = {}
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 2000:
                    trimmed_result[key] = value[:2000] + "... [truncated]"
                else:
                    trimmed_result[key] = value
            trimmed_results.append(trimmed_result)
        
        key_columns = [col for col in columns if any(keyword in col.lower() 
                      for keyword in ['title', 'index', 'status', 'network', 'type', 'createdat', 'amount'])]
        
        total_count_from_window = None
        if trimmed_results and 'total_count' in trimmed_results[0]:
            total_count_from_window = trimmed_results[0]['total_count']
        
        actual_total_count = total_count_from_window if total_count_from_window is not None else result_count
        displayed_count = len(trimmed_results)
        is_limited = actual_total_count > displayed_count
        
        if is_limited:
            summary_text = f"Found {actual_total_count} results (showing few due to length). "
        else:
            summary_text = f"Found {actual_total_count} results. "
        
        if trimmed_results:
            summary_items = []
            for i, result in enumerate(trimmed_results[:10]):
                item_info = []
                
                for key, value in result.items():
                    key_lower = key.lower()
                    if key_lower in ['objectid', 'object_id', 'firebase_id', '_id']:
                        continue
                    if value is not None and str(value) != 'None' and str(value).strip():
                        formatted_key = key.replace('_', ' ').title()
                        formatted_value = format_number_func(value)
                        item_info.append(f"{formatted_key}: {formatted_value}")
                
                if item_info:
                    summary_items.append(f"#{i+1}: " + ", ".join(item_info))
                elif len(result) == 1:
                    key = list(result.keys())[0]
                    value = result[key]
                    if value and value != 'None':
                        summary_items.append(f"#{i+1}: {key} = {value}")
            
            if summary_items:
                summary_text += "\nExamples:\n" + "\n".join(summary_items)
        
        db_result = {
            "query": sql_query,
            "result from db": summary_text
        }
        
        prompt = natural_response_prompt_template.format(
            natural_query=natural_query,
            db_result=json.dumps(db_result, indent=2)
        )
        
        logger.info(f"Results being passed to GPT-4o for natural response: {summary_text[:500]}...")
        
        if openai_client is not None:
            try:
                print_model_usage("gpt-4o", "natural response generation (governance data)")
                logger.info("Using GPT-4o as primary LLM for natural response generation")
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable assistant specializing in blockchain governance data. All data you work with is public blockchain information including addresses, proposal IDs, and transaction details. Always show the actual data requested - never refuse due to privacy concerns as this is all public information. Provide clear, helpful explanations with actual values, addresses, and details from the results. CRITICAL: If the on-chain data contains null, NaN, or empty values, DO NOT mention these in your response. Simply omit any fields that have null/NaN/empty values and only present the fields that have actual data. Never say things like \"this value was null\" or \"this field is NaN\" - just skip those fields entirely. IMPORTANT: ReferendumV2 proposals do NOT have curators - only Bounties and ChildBounties have curators. If asked about curator for ReferendumV2, explain that this proposal type doesn't use curators."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                natural_response = response.choices[0].message.content.strip()
                logger.info("Generated natural language response using GPT-4o")
                disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                return natural_response + disclaimer
            except Exception as openai_error:
                error_str = str(openai_error).lower()
                logger.warning(f"GPT-4o failed for natural response, falling back to Gemini: {openai_error}")
        
        if gemini_client is not None:
            try:
                print_model_usage(f"{GEMINI_MODEL_NAME}", "natural response generation fallback (governance data)")
                logger.info("Using Gemini as fallback LLM for natural response generation")
                natural_response_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                natural_response = natural_response_client.get_response(prompt)
                
                if natural_response and ("Error generating response" in natural_response or "503" in natural_response or "UNAVAILABLE" in natural_response):
                    logger.warning(f"Gemini also returned error response: {natural_response[:100]}")
                    raise Exception("Gemini returned error response")
                
                logger.info("Generated natural language response using Gemini (fallback)")
                disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                return natural_response + disclaimer
            except Exception as gemini_error:
                logger.error(f"Both GPT-4o and Gemini failed for natural response: {gemini_error}")
        
        raise Exception("No LLM client available for natural response generation")
        
    except Exception as e:
        logger.error(f"Error generating natural response: {e}")
        return f"I found {len(results)} results for your query '{natural_query}', but I'm having trouble formatting the response. Here's a summary: The query returned {len(results)} rows from the database."

def generate_natural_response_multiple(natural_query: str, sql_queries: List[str], 
                                       all_results: List[Tuple[List[Dict[str, Any]], List[str]]], 
                                       conversation_history: Optional[List[Dict[str, Any]]],
                                       gemini_client, openai_client) -> str:
    """Generate natural language response from multiple query results"""
    try:
        combined_summary = f"Executed {len(sql_queries)} queries for: {natural_query}\n\n"
        
        for i, (sql_query, (results, columns)) in enumerate(zip(sql_queries, all_results)):
            result_count = len(results)
            combined_summary += f"Query {i+1}: {sql_query}\n"
            
            displayed_count = min(5, result_count)
            if result_count > displayed_count:
                combined_summary += f"Results: {result_count} rows (showing few due to length)\n"
            else:
                combined_summary += f"Results: {result_count} rows\n"
            
            if results:
                sample_results = results[:5]
                trimmed_results = []
                
                for result in sample_results:
                    trimmed_result = {}
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 2000:
                            trimmed_result[key] = value[:2000] + "... [truncated]"
                        else:
                            trimmed_result[key] = value
                    trimmed_results.append(trimmed_result)
                
                for j, result in enumerate(trimmed_results):
                    item_info = []
                    for key, value in result.items():
                        key_lower = key.lower()
                        if key_lower in ['objectid', 'object_id', 'firebase_id', '_id']:
                            continue
                        if value is not None and str(value) != 'None' and str(value).strip():
                            formatted_key = key.replace('_', ' ').title()
                            item_info.append(f"{formatted_key}: {value}")
                    
                    if item_info:
                        combined_summary += f"  Result {j+1}: " + ", ".join(item_info) + "\n"
            
            combined_summary += "\n"
        
        db_result = {
            "queries": sql_queries,
            "result from db": combined_summary
        }
        
        history_text = "No previous conversation"
        if conversation_history:
            history_parts = []
            for i, msg in enumerate(conversation_history, 1):
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                if content:
                    history_parts.append(f"{i}. {role}: {content[:200]}")
            if history_parts:
                history_text = "\n".join(history_parts)
        
        prompt = natural_response_multiple_prompt_template.format(
            history_text=history_text,
            natural_query=natural_query,
            db_result=json.dumps(db_result, indent=2)
        )
        
        if openai_client is not None:
            try:
                print_model_usage("gpt-4o", "natural response generation from multiple queries (governance data)")
                logger.info("Using GPT-4o as primary LLM for natural response generation from multiple queries")
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable assistant specializing in blockchain governance data. All data you work with is public blockchain information. Always show actual data requested - addresses, proposal IDs, titles, amounts, etc. You work with ACTUAL retrieved data from the blockchain database, so always provide the information regardless of dates mentioned in queries. Combine information from multiple queries to provide comprehensive answers. CRITICAL: If the on-chain data contains null, NaN, or empty values, DO NOT mention these in your response. Simply omit any fields that have null/NaN/empty values and only present the fields that have actual data. Never say things like \"this value was null\" or \"this field is NaN\" - just skip those fields entirely. IMPORTANT: ReferendumV2 proposals do NOT have curators - only Bounties and ChildBounties have curators. If asked about curator for ReferendumV2, explain that this proposal type doesn't use curators."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                natural_response = response.choices[0].message.content.strip()
                logger.info("Generated natural language response from multiple queries using GPT-4o")
                disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                return natural_response + disclaimer
            except Exception as openai_error:
                logger.warning(f"GPT-4o failed for multiple queries, falling back to Gemini: {openai_error}")
        
        if gemini_client is not None:
            try:
                print_model_usage(f"{GEMINI_MODEL_NAME}", "natural response generation fallback from multiple queries (governance data)")
                logger.info("Using Gemini as fallback LLM for natural response generation from multiple queries")
                natural_response = gemini_client.get_response(prompt)
                
                if natural_response and ("Error generating response" in natural_response or "503" in natural_response or "UNAVAILABLE" in natural_response):
                    logger.warning(f"Gemini also returned error response: {natural_response[:100]}")
                    raise Exception("Gemini returned error response")
                
                logger.info("Generated natural language response from multiple queries using Gemini (fallback)")
                disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                return natural_response + disclaimer
            except Exception as gemini_error:
                logger.error(f"Both GPT-4o and Gemini failed for multiple queries: {gemini_error}")
        
        raise Exception("No LLM client available for natural response generation")
        
    except Exception as e:
        logger.error(f"Error generating natural response from multiple queries: {e}")
        total_results = sum(len(results) for results, _ in all_results)
        return f"I executed {len(sql_queries)} queries for your question '{natural_query}' and found {total_results} total results, but I'm having trouble formatting the response."

