import json
import logging
from typing import Dict, List, Any, Optional

from ..utils.model_usage import print_model_usage, GeminiClient, GEMINI_MODEL_NAME, GEMINI_MODEL_SQL, GEMINI_SQL_TIMEOUT, GEMINI_TIMEOUT
from ..utils.token_utils import trim_prompt_to_fit_tokens
from ...prompts.sql_generation_with_intent_prompt import PROMPT as sql_generation_system_prompt, PROMPT_TEMPLATE as sql_generation_with_intent_template
from ...core.generator.query_analysis import format_conversation_history

logger = logging.getLogger(__name__)

def generate_sql_with_model_deterministic(system_prompt: str, openai_client, gemini_client) -> str:
    """Generate SQL using deterministic settings (temperature=0, top_p=1, seed=42) - always uses Gemini first"""
    full_prompt = f"""{sql_generation_system_prompt}

{system_prompt}"""
    
    if gemini_client:
        try:
            print_model_usage(f"{GEMINI_MODEL_SQL}", "SQL generation (governance data, deterministic)")
            logger.debug("Using Gemini for deterministic SQL generation")
                response = gemini_client.get_response(full_prompt)
                return response.strip()
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                    logger.warning(f"Gemini SQL model overloaded (503 error), falling back to general Gemini model: {e}")
                    try:
                        print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (governance data)")
                        fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_SQL_TIMEOUT)
                        response = fallback_client.get_response(full_prompt)
                        logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for SQL generation")
                        return response.strip()
                    except Exception as fallback_error:
                        logger.error(f"Fallback Gemini model also failed: {fallback_error}")
                    if openai_client:
                        logger.info("Falling back to ChatGPT for deterministic SQL generation")
            print_model_usage("GPT-4", "SQL generation fallback (governance data, deterministic)")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sql_generation_system_prompt},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.0,
                top_p=1.0,
                seed=42,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        else:
                        raise e
            else:
                if openai_client:
                    logger.warning(f"Gemini failed, falling back to ChatGPT: {e}")
                    print_model_usage("GPT-4", "SQL generation fallback (governance data, deterministic)")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sql_generation_system_prompt},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.0,
                top_p=1.0,
                seed=42,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        else:
            raise e

    elif openai_client:
        print_model_usage("GPT-4", "SQL generation (governance data, deterministic)")
        logger.debug("Using ChatGPT for deterministic SQL generation (Gemini not available)")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sql_generation_system_prompt},
                    {"role": "user", "content": system_prompt}
                ],
            temperature=0.0,
            top_p=1.0,
            seed=42,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
    else:
        raise ValueError("No SQL generation model available")

def generate_sql_with_model(system_prompt: str, openai_client, gemini_client, user_message: str = None) -> str:
    """Generate SQL using Gemini as primary, OpenAI as fallback"""
            full_prompt = f"""{sql_generation_system_prompt}

{system_prompt}"""
            
    if gemini_client:
            try:
            print_model_usage(f"{GEMINI_MODEL_SQL}", "SQL generation (governance data)")
            logger.debug("Using Gemini for SQL generation")
                response = gemini_client.get_response(full_prompt)
                return response.strip()
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                    logger.warning(f"Gemini SQL model overloaded (503 error), falling back to general Gemini model: {e}")
                    try:
                        print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (governance data)")
                        fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_SQL_TIMEOUT)
                        response = fallback_client.get_response(full_prompt)
                        logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for SQL generation")
                        return response.strip()
                    except Exception as fallback_error:
                        logger.error(f"Fallback Gemini model also failed: {fallback_error}")
                    if openai_client:
                        logger.info("Falling back to ChatGPT for SQL generation")
                        print_model_usage("GPT-4", "SQL generation fallback (governance data)")
                        response = openai_client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": sql_generation_system_prompt},
                                {"role": "user", "content": system_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=800
                        )
                        return response.choices[0].message.content.strip()
                    else:
                        raise e
                else:
                if openai_client:
                    logger.warning(f"Gemini failed, falling back to ChatGPT: {e}")
            print_model_usage("GPT-4", "SQL generation fallback (governance data)")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sql_generation_system_prompt},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        else:
                    raise e
    
    elif openai_client:
        print_model_usage("GPT-4", "SQL generation (governance data)")
        logger.debug("Using ChatGPT for SQL generation (Gemini not available)")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sql_generation_system_prompt},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        else:
        raise ValueError("No SQL generation model available")

def get_governance_context(embedding_manager, natural_query: str) -> str:
    """Retrieve relevant governance proposals from Chroma as contextual examples"""
    governance_context = ""
    if embedding_manager:
        logger.info("📊 Embedding manager available - will retrieve governance examples")
        try:
            logger.info("=" * 70)
            logger.info("🔍 SEMANTIC SEARCH FOR SQL CONTEXT")
            logger.info("=" * 70)
            logger.info(f"Query sent to Chroma: '{natural_query}'")
            logger.info(f"Collection: polkadot_embeddings_dynamic")
            logger.info(f"Filter: doc_type='governance'")
            logger.info(f"Requesting: 3 results")
            logger.info("=" * 70)
            
            results = embedding_manager.search_similar_chunks(
                query=natural_query,
                n_results=10,
                filter_metadata={"doc_type": "governance"}
            )
            
            if results and len(results) > 0:
                logger.info(f"✅ Found {len(results)} results from Chroma")
                logger.info("-" * 70)
                
                def sort_key(chunk):
                    metadata = chunk.get('metadata', {})
                    network = metadata.get('network', 'unknown')
                    proposal_idx = metadata.get('proposal_index', 'unknown')
                    try:
                        proposal_idx_int = int(proposal_idx) if proposal_idx != 'unknown' else 0
                    except (ValueError, TypeError):
                        proposal_idx_int = 0
                    created_at = metadata.get('created_at', metadata.get('createdat', ''))
                    return (network, proposal_idx_int, created_at)
                
                sorted_results = sorted(results, key=sort_key)
                selected_results = sorted_results[:3]
                
                context_parts = []
                for i, chunk in enumerate(selected_results, 1):
                    content = chunk.get('content', '')
                    metadata = chunk.get('metadata', {})
                    network = metadata.get('network', 'unknown')
                    proposal_idx = metadata.get('proposal_index', 'unknown')
                    proposal_type = metadata.get('proposal_type', 'unknown')
                    
                    logger.info(f"Result {i} (deterministic order):")
                    logger.info(f"  Network: {network}")
                    logger.info(f"  Proposal Index: {proposal_idx}")
                    logger.info(f"  Proposal Type: {proposal_type}")
                    logger.info(f"  Content Preview: {content[:150]}...")
                    logger.info("-" * 70)
                    
                    context_parts.append(f"Example {i} (Proposal {network}#{proposal_idx}):\n{content[:500]}")
                
                governance_context = "\n\nRELEVANT GOVERNANCE PROPOSALS (for reference):\n" + "\n\n".join(context_parts) + "\n\nUse these examples to understand the data structure and write better SQL queries.\n"
                logger.info(f"✅ Added {len(selected_results)} governance proposals as context for SQL generation (deterministic order)")
                logger.info("=" * 70)
            else:
                logger.info("❌ No relevant governance proposals found in Chroma")
                logger.info("=" * 70)
        except Exception as e:
            logger.error("=" * 70)
            logger.error("❌ SEMANTIC SEARCH FAILED")
            logger.error(f"Error: {e}")
            logger.error("=" * 70)
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.info("⚠️  No embedding manager - SQL generation without governance examples")
    
    return governance_context

def generate_sql_queries_only(natural_query: str, conversation_history: Optional[List[Dict[str, Any]]], 
                             intent: Dict[str, Any], embedding_manager, table_schema: str, 
                             table_name: str, openai_client, gemini_client,
                             trim_prompt_func, max_retries: int = 3) -> List[str]:
    """Generate SQL queries without executing them - uses intent extraction for deterministic generation"""
    
    governance_context = get_governance_context(embedding_manager, natural_query)
    history_text = format_conversation_history(conversation_history)
    
    intent_json_str = json.dumps(intent, indent=2)
    
    network_filter_instruction = ""
    if intent["network"] in ["polkadot", "kusama"]:
        network_filter_instruction = f'\n- CRITICAL: Add WHERE filter: "source_network" = \'{intent["network"]}\' AND "source_network" IS NOT NULL'
    elif intent["network"] == "both":
        network_filter_instruction = '\n- CRITICAL: Do NOT filter by network - include both Polkadot and Kusama'
    else:
        network_filter_instruction = '\n- CRITICAL: Do NOT filter by network unless explicitly mentioned in filters'
    
    time_filter_instruction = ""
    if intent["time_range"] == "last_30_days":
        time_filter_instruction = '\n- Add date filter: "createdat" >= CURRENT_DATE - INTERVAL \'30 days\' AND "createdat" IS NOT NULL'
    elif intent["time_range"] == "last_90_days":
        time_filter_instruction = '\n- Add date filter: "createdat" >= CURRENT_DATE - INTERVAL \'90 days\' AND "createdat" IS NOT NULL'
    elif intent["time_range"] == "all_time":
        time_filter_instruction = '\n- No time filter needed - include all time periods'
    else:
        time_filter_instruction = '\n- Use time filter only if explicitly mentioned in the query or filters field'
    
    metric_instruction = ""
    if intent["metric"] == "count":
        metric_instruction = '\n- Use COUNT(*) aggregation'
    elif intent["metric"] == "sum":
        metric_instruction = '\n- Use SUM() aggregation on appropriate amount fields'
    elif intent["metric"] == "avg":
        metric_instruction = '\n- Use AVG() aggregation on appropriate numeric fields'
    elif intent["metric"] == "details":
        metric_instruction = '\n- Return full details (SELECT multiple columns) for the specific item'
    else:
        metric_instruction = '\n- Return list of items (SELECT with LIMIT)'
    
    id_filter_instruction = ""
    if intent["id"] is not None:
        id_filter_instruction = f'\n- CRITICAL: Filter by ID: "index" = {intent["id"]} AND "index" IS NOT NULL'
    
    base_system_prompt = sql_generation_with_intent_template.format(
        intent_json_str=intent_json_str,
        network_filter_instruction=network_filter_instruction,
        time_filter_instruction=time_filter_instruction,
        metric_instruction=metric_instruction,
        id_filter_instruction=id_filter_instruction,
        history_text=history_text,
        table_schema=table_schema,
        governance_context=governance_context,
        table_name=table_name,
        natural_query=natural_query
    )
    
    for attempt in range(max_retries):
        try:
            system_prompt = base_system_prompt
            system_prompt = trim_prompt_func(system_prompt)
            
            response_content = generate_sql_with_model_deterministic(system_prompt, openai_client, gemini_client)
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
                
                logger.info(f"Generated {len(sql_queries)} SQL queries (attempt {attempt + 1})")
                logger.info(f"SQL query preview: {sql_queries[0][:200]}..." if sql_queries else "No queries generated")
                logger.info(f"Intent used - network: {intent['network']}, entity_type: {intent['entity_type']}, metric: {intent['metric']}")
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

