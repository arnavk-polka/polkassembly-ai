import json
import logging
from typing import Dict, List, Any, Optional

from ...prompts.intent_extraction_prompt import PROMPT as intent_extractor_system_prompt, PROMPT_TEMPLATE as intent_prompt_template

logger = logging.getLogger(__name__)

def extract_sql_intent(natural_query: str, conversation_history: Optional[List[Dict[str, Any]]], 
                       openai_client, gemini_client) -> Dict[str, Any]:
    """
    Extract structured intent from natural language query.
    Returns a deterministic intent object that will be used for SQL generation.
    """
    default_intent = {
        "entity_type": "unknown",
        "network": "unspecified",
        "id": None,
        "time_range": "unspecified",
        "metric": "list",
        "filters": ""
    }
    
    history_text = "No previous conversation"
    if conversation_history:
        history_parts = []
        for i, msg in enumerate(conversation_history, 1):
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if content:
                history_parts.append(f"{i}. {role}: {content[:150]}")
        if history_parts:
            history_text = "\n".join(history_parts)
    
    intent_prompt = intent_prompt_template.format(
        natural_query=natural_query,
        history_text=history_text
    )

    try:
        if gemini_client:
            full_prompt = f"""{intent_extractor_system_prompt}

{intent_prompt}"""
            try:
                response_text = gemini_client.get_response(full_prompt).strip()
            except Exception as e:
                logger.warning(f"Gemini intent extraction failed, falling back to OpenAI: {e}")
                if openai_client:
                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": intent_extractor_system_prompt},
                            {"role": "user", "content": intent_prompt}
                        ],
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=300
                    )
                    response_text = response.choices[0].message.content.strip()
                else:
                    logger.warning("No LLM client available for intent extraction, using default intent")
                    return default_intent
        elif openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": intent_extractor_system_prompt},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=300
            )
            response_text = response.choices[0].message.content.strip()
        else:
            logger.warning("No LLM client available for intent extraction, using default intent")
            return default_intent
        
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                intent = json.loads(json_str)
                
                validated_intent = {
                    "entity_type": intent.get("entity_type", "unknown"),
                    "network": intent.get("network", "unspecified"),
                    "id": intent.get("id"),
                    "time_range": intent.get("time_range", "unspecified"),
                    "metric": intent.get("metric", "list"),
                    "filters": intent.get("filters", "")
                }
                
                valid_entity_types = ["referenda", "treasury_proposal", "bounty", "discussion", "voter", "delegate", "unknown"]
                valid_networks = ["polkadot", "kusama", "both", "unspecified"]
                valid_time_ranges = ["last_30_days", "last_90_days", "all_time", "unspecified"]
                valid_metrics = ["count", "list", "sum", "avg", "details"]
                
                if validated_intent["entity_type"] not in valid_entity_types:
                    validated_intent["entity_type"] = "unknown"
                if validated_intent["network"] not in valid_networks:
                    validated_intent["network"] = "unspecified"
                if validated_intent["time_range"] not in valid_time_ranges:
                    validated_intent["time_range"] = "unspecified"
                if validated_intent["metric"] not in valid_metrics:
                    validated_intent["metric"] = "list"
                
                logger.info(f"Extracted intent: {json.dumps(validated_intent)}")
                return validated_intent
            else:
                logger.warning(f"Could not find JSON in intent extraction response: {response_text[:200]}")
                return default_intent
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse intent JSON: {e}, response: {response_text[:200]}")
            return default_intent
            
    except Exception as e:
        logger.error(f"Error extracting intent: {e}")
        return default_intent

