import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def get_default_system_prompt() -> str:
    from ...prompts.default_system_prompt import PROMPT
    return PROMPT

def create_user_prompt(self, query: str, context: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    from .query_analysis import format_conversation_history
    
    prompt_parts = []
    
    if conversation_history:
        formatted_history = format_conversation_history(conversation_history)
        prompt_parts.append(f"Conversation History:\n{formatted_history}")
    
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    prompt_parts.append(f"Current Date (UTC): {current_date}")

    if context:
        prompt_parts.append(f"Context Information:\n{context}")
    
    if "IMPORTANT: The user also requested specific data" in query or "Dynamic Data Context:" in query:
        prompt_parts.append("NOTE: This query requires both explaining concepts AND providing specific data. Make sure to include both the explanation from the context AND the specific data requested by the user.")
    
    prompt_parts.append(f"Current Question: {query}")
    
    prompt_parts.append("CRITICAL INSTRUCTIONS:\n- Provide a comprehensive and detailed answer to the user's question using ALL relevant information from the retrieved chunks.\n- For questions asking \"what is\", \"explain\", \"how does\", or similar conceptual questions, provide a thorough explanation covering the key aspects, how it works, and relevant details from the context.\n- For specific data questions (counts, lists, specific values), provide the exact information requested.\n- Use ALL relevant chunks that address the question - do not limit yourself to just one chunk if multiple chunks contain relevant information.\n- Do NOT include information about unrelated topics, but DO include all relevant details about the topic being asked about.\n- Answer the question directly without mentioning the context, sources, documentation, or previous conversations. Do not start with phrases like \"Based on the provided context\", \"According to the documentation\", \"From the Polkadot Wiki\", \"From our previous conversation\", etc. Simply provide the answer as if you have direct knowledge of the topic.\n\nCRITICAL FORMATTING REQUIREMENTS:\n- NEVER start with headers (##, ###)\n- Start directly with answer content\n- ALWAYS add line breaks between numbered steps (1. step one [LINE BREAK] 2. step two [LINE BREAK])\n- ALWAYS add line breaks between bullet points\n- Use professional markdown formatting throughout\n- IMPORTANT: Include all images from the context using exact markdown format: ![Step Image](url)")
    
    return "\n\n".join(prompt_parts)

def generate_llm_response(self, system_prompt: str, user_prompt: str) -> str:
    from .utils import print_model_usage
    from ..errors import is_insufficient_quota_error, get_quota_error_message
    
    answer = None
    openai_enabled = os.getenv("ENABLE_OPENAI", "").lower() == "true"
    gemini_enabled = os.getenv("ENABLE_GEMINI", "").lower() == "true"
    
    try:
        if openai_enabled:
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
            raise ValueError(get_quota_error_message())
        logger.error(f"Error in LLM response generation: {llm_error}")
        raise
    
    return answer

