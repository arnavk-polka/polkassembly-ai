import logging
import json
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def analyze_query_with_context(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    try:
        if not conversation_history or not hasattr(self, 'client') or not self.client:
            return query
        
        max_history_length = 5
        recent_history = conversation_history[-max_history_length:] if len(conversation_history) > max_history_length else conversation_history
        
        serializable_history = []
        for msg in recent_history:
            content = None
            role = None
            
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                role = msg.role
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", str(msg))
            else:
                role = "user"
                content = str(msg)
            
            if role == "assistant" and content:
                content_str = str(content).strip()
                if content_str.endswith('?'):
                    continue
            
            serializable_history.append({
                "role": role,
                "content": content
            })
        
        current_date = datetime.utcnow()
        current_date_str = current_date.strftime("%Y-%m-%d")
        current_month_str = current_date.strftime("%B %Y")
        last_month = (current_date.replace(day=1) - timedelta(days=1))
        last_month_str = last_month.strftime("%B %Y")
        yesterday_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        last_year_str = (current_date.replace(month=1, day=1) - timedelta(days=1)).strftime("%Y")
        
        from ...prompts.query_analysis_prompt import PROMPT_TEMPLATE
        analysis_prompt = PROMPT_TEMPLATE.format(
            current_date_str=current_date_str,
            current_month_str=current_month_str,
            last_month_str=last_month_str,
            yesterday_str=yesterday_str,
            last_year_str=last_year_str,
            conversation_history=format_conversation_history(serializable_history),
            query=query
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query context analyzer. Return only valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            openai_response = response.choices[0].message.content
            if not openai_response:
                return query
            
            analyzed_query = parse_gemini_response(openai_response, query)
        except Exception as e:
            logger.warning(f"OpenAI query analysis failed: {e}, returning original query")
            return query
        
        if not analyzed_query or analyzed_query.strip() == "":
            logger.warning("Analyzed query is empty, returning original")
            return query
        
        analyzed_lower = analyzed_query.lower().strip()
        clarification_patterns = [
            'are you referring to',
            'are you looking for',
            'are you asking about',
            'can you clarify',
            'which network',
            'polkadot or kusama',
            'which proposal',
            'which referendum'
        ]
        
        starts_with_clarification = any(analyzed_lower.startswith(pattern) for pattern in clarification_patterns)
        
        contains_clarification_prefix = False
        for pattern in clarification_patterns:
            if pattern in analyzed_lower:
                original_in_analyzed = query.lower().strip() in analyzed_lower
                if original_in_analyzed and analyzed_lower.index(pattern) < analyzed_lower.index(query.lower().strip()):
                    contains_clarification_prefix = True
                    break
        
        if starts_with_clarification or contains_clarification_prefix:
            logger.warning(f"Analyzed query contains clarification question pattern, returning original. Analyzed: '{analyzed_query[:100]}'")
            return query
        
        if analyzed_query.lower().strip() != query.lower().strip():
            logger.info(f"Query modified: '{query}' → '{analyzed_query}'")
        
        return analyzed_query
            
    except Exception as e:
        logger.error(f"Error in query analysis: {e}", exc_info=True)
        return query

def format_conversation_history(history: Optional[List[Dict[str, Any]]]) -> str:
    if not history:
        return "No previous conversation"
    
    formatted = []
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        formatted.append(f"{i}. {role}: {content}")
    
    return "\n".join(formatted)

def get_gemini_response_with_retry(self, prompt: str, max_retries: int = 2) -> str:
    for attempt in range(max_retries):
        try:
            response = self.gemini_client.get_response(prompt)
            if response and response.strip():
                return response
        except Exception as e:
            logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    
    raise Exception("All Gemini API attempts failed")

def parse_gemini_response(response: str, fallback_query: str) -> str:
    try:
        response_json = json.loads(response.strip())
        return response_json.get("analyzed_query", fallback_query)
    
    except json.JSONDecodeError:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                response_json = json.loads(json_match.group(1))
                return response_json.get("analyzed_query", fallback_query)
            except json.JSONDecodeError:
                pass
        
        json_match = re.search(r'\{[^}]*"analyzed_query"[^}]*\}', response, re.DOTALL)
        if json_match:
            try:
                response_json = json.loads(json_match.group(0))
                return response_json.get("analyzed_query", fallback_query)
            except json.JSONDecodeError:
                pass
        
        quote_match = re.search(r'analyzed_query["\s:]+(["\'])(.*?)\1', response, re.DOTALL)
        if quote_match:
            return quote_match.group(2).strip()
        
        logger.warning(f"Could not parse LLM response: {response[:200]}")
        return fallback_query

