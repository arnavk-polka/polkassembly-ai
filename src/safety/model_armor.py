import os
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import httpx
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
import logging

load_dotenv()

logger = logging.getLogger(__name__)

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL_ARMOR_TEMPLATE_ID = os.environ.get("MODEL_ARMOR_TEMPLATE_ID")
GCP_SERVICE_ACCOUNT_KEY = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")


def get_base_url() -> str:
    """Get the base URL for Model Armor API"""
    if not GCP_PROJECT_ID or not MODEL_ARMOR_TEMPLATE_ID:
        raise ValueError("GCP_PROJECT_ID and MODEL_ARMOR_TEMPLATE_ID must be set")
    
    template_id = MODEL_ARMOR_TEMPLATE_ID.strip()
    
    if template_id.startswith("projects/"):
        base_url = f"https://modelarmor.googleapis.com/v1/{template_id}"
    elif "/templates/" in template_id:
        base_url = f"https://modelarmor.googleapis.com/v1/{template_id}"
    else:
        base_url = f"https://modelarmor.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/templates/{template_id}"
    
    return base_url

async def get_access_token() -> str:
    """Get Google Cloud access token for authentication"""
    try:
        loop = asyncio.get_event_loop()
        
        def _get_token():
            from google.oauth2 import service_account
            import json as json_lib
            
            if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
                credentials = service_account.Credentials.from_service_account_file(
                    GOOGLE_APPLICATION_CREDENTIALS,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                credentials.refresh(Request())
                return credentials.token
            elif GCP_SERVICE_ACCOUNT_KEY:
                credentials_info = json_lib.loads(GCP_SERVICE_ACCOUNT_KEY)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                credentials.refresh(Request())
                return credentials.token
            else:
                credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                credentials.refresh(Request())
                return credentials.token
        
        token = await loop.run_in_executor(None, _get_token)
        return token
    except Exception as e:
        logger.error(f"Failed to get access token: {str(e)}", exc_info=True)
        raise Exception(f"Failed to get access token: {str(e)}")

async def verify_model_armor_api_enabled() -> bool:
    """Check if Model Armor API is enabled for the project"""
    return True

async def check_with_guardrail_async(query: str) -> Dict[str, Any]:
    """
    Async version of Model Armor check to avoid blocking FastAPI event loop
    Returns:
      {
        "status": "blocked" | "not_blocked" | "error",
        "action": "NONE" | "BLOCKED",
        "reason": str | None,
        "violation_details": dict | None
      }
    """
    if not GCP_PROJECT_ID or not MODEL_ARMOR_TEMPLATE_ID:
        return {
            "status": "error",
            "action": "ERROR",
            "reason": "Model Armor not configured: GCP_PROJECT_ID and MODEL_ARMOR_TEMPLATE_ID must be set"
        }
    
    try:
        api_enabled = await verify_model_armor_api_enabled()
        if not api_enabled:
            return {
                "status": "error",
                "action": "ERROR",
                "reason": "Model Armor API is not enabled. Enable it at: https://console.cloud.google.com/apis/library/modelarmor.googleapis.com"
            }
    except:
        pass
    
    try:
        loop = asyncio.get_event_loop()
        
        def _make_request():
            from google.cloud import modelarmor_v1
            from google.api_core.client_options import ClientOptions
            
            if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
            elif GCP_SERVICE_ACCOUNT_KEY:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(GCP_SERVICE_ACCOUNT_KEY)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
            
            client_options = ClientOptions(
                api_endpoint=f"modelarmor.{GCP_LOCATION}.rep.googleapis.com"
            )
            client = modelarmor_v1.ModelArmorClient(client_options=client_options)
            
            template_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/templates/{MODEL_ARMOR_TEMPLATE_ID}"
            
            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=template_name,
                user_prompt_data=modelarmor_v1.DataItem(text=query)
            )
            
            response = client.sanitize_user_prompt(request=request, timeout=5.0)
            return response
        
        result_obj = await loop.run_in_executor(None, _make_request)
        
        sanitization_result = result_obj.sanitization_result
        
        filter_match_state_enum = sanitization_result.filter_match_state
        filter_match_state = filter_match_state_enum.name if filter_match_state_enum else "NO_MATCH"
        blocked = filter_match_state == "MATCH_FOUND"
        
        sanitized_text = query
        was_sanitized = False
        
        for field_name in ['sanitized_user_prompt_data', 'sanitized_data', 'sanitizedData', 'sanitizedUserPromptData']:
            if hasattr(sanitization_result, field_name):
                sanitized_data = getattr(sanitization_result, field_name)
                if sanitized_data:
                    if hasattr(sanitized_data, 'text') and sanitized_data.text:
                        sanitized_text = sanitized_data.text
                        was_sanitized = sanitized_text != query
                        break
        
        violation_details = {}
        filter_matches = []
        
        if blocked or was_sanitized:
            if hasattr(sanitization_result, 'filter_matches') and sanitization_result.filter_matches:
                for match in sanitization_result.filter_matches:
                    filter_type = "Unknown"
                    if hasattr(match, 'filter_type'):
                        filter_type_enum = match.filter_type
                        if hasattr(filter_type_enum, 'name'):
                            filter_type = filter_type_enum.name
                        elif hasattr(filter_type_enum, 'value'):
                            filter_type = str(filter_type_enum.value)
                        else:
                            filter_type = str(filter_type_enum)
                    elif hasattr(match, 'filterType'):
                        filter_type = str(getattr(match, 'filterType'))
                    
                    confidence = "Unknown"
                    if hasattr(match, 'confidence'):
                        confidence_enum = match.confidence
                        if hasattr(confidence_enum, 'name'):
                            confidence = confidence_enum.name
                        elif hasattr(confidence_enum, 'value'):
                            confidence = str(confidence_enum.value)
                        else:
                            confidence = str(confidence_enum)
                    
                    filter_matches.append({
                        "filterType": filter_type,
                        "confidence": confidence
                    })
            
            violation_details = {
                "filter_matches": filter_matches,
                "filter_match_state": filter_match_state,
                "sanitized_text": sanitized_text if was_sanitized else None,
                "was_sanitized": was_sanitized
            }
        
        reason = None
        if blocked:
            filter_types = [f.get("filterType", "Unknown") for f in filter_matches if f.get("filterType") != "Unknown"]
            if filter_types:
                reason = f"Content blocked by Model Armor filters: {', '.join(filter_types)}"
            else:
                reason = "Content blocked by Model Armor"
        elif was_sanitized:
            reason = "Content was sanitized by Model Armor"
        
        return {
            "status": "blocked" if blocked else ("not_blocked" if not was_sanitized else "sanitized"),
            "action": "BLOCKED" if blocked else "NONE",
            "reason": reason,
            "violation_details": violation_details if (blocked or was_sanitized) else None,
            "raw_response": None
        }
    except Exception as e:
        logger.error(f"Model Armor API error: {str(e)}")
        return {
            "status": "error",
            "action": "ERROR",
            "reason": f"Error calling Model Armor: {str(e)}"
        }

async def generate_user_friendly_block_message(violation_details: Dict[str, Any], user_query: str) -> str:
    """
    Generate a natural language explanation for why a query was blocked using GPT-3.5-turbo.
    
    Args:
        violation_details: Dictionary containing violation details from Model Armor
        user_query: The original user query that was blocked
    
    Returns:
        A user-friendly natural language explanation
    """
    try:
        import openai
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "Your query was blocked because it violates our content policy. Please revise your query to comply with our terms of service."
        
        client = openai.OpenAI(api_key=openai_api_key, timeout=5.0)
        
        violation_summary = []
        
        filter_matches = violation_details.get("filter_matches", [])
        if filter_matches:
            filter_types = [f.get("filterType", "Unknown") for f in filter_matches if f.get("filterType") != "Unknown"]
            if filter_types:
                violation_summary.append(f"Filters triggered: {', '.join(filter_types)}")
        
        was_sanitized = violation_details.get("was_sanitized", False)
        if was_sanitized:
            violation_summary.append("Content was automatically sanitized")
        
        violation_text = "\n".join(violation_summary) if violation_summary else "Content policy violation detected"
        
        prompt = f"""You are a helpful assistant explaining why a user's query was blocked by content moderation.

The user's query was: "{user_query}"

The content moderation system detected the following violations:
{violation_text}

Generate a clear, helpful, and professional message explaining to the user why their query was blocked. The message should:
1. Be polite and respectful
2. Clearly explain what type of content policy was violated (without being too technical)
3. Suggest how they can revise their query
4. Be concise (2-3 sentences maximum)
5. Not include the exact violation details verbatim - translate them into user-friendly language

Example good responses:
- "I'm unable to process queries that contain inappropriate language. Please rephrase your question using respectful language, and I'll be happy to help."
- "Your query was blocked because it contains content that violates our safety policies. Could you please rephrase your question in a way that complies with our community guidelines?"
- "I can't process requests that attempt to manipulate system prompts or access restricted information. Please ask a question about Polkadot governance, and I'll be happy to help."

Generate the response now:"""
        
        from src.core.config import Config
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains content moderation decisions in a clear, user-friendly way."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
        )
        
        message = response.choices[0].message.content.strip()
        return message
        
    except Exception as e:
        logger.error(f"Error generating user-friendly block message: {e}")
        return "Your query was blocked because it violates our content policy. Please revise your query to comply with our terms of service. Continued violations may result in your IP being blocked."

async def sanitize_model_response(response: str) -> Dict[str, Any]:
    """
    Sanitize a model response using Model Armor
    
    Args:
        response: The model response text to sanitize
    
    Returns:
        Dict with sanitized text and blocking status
    """
    if not GCP_PROJECT_ID or not MODEL_ARMOR_TEMPLATE_ID:
        return {
            "status": "error",
            "sanitized_text": response,
            "blocked": False,
            "reason": "Model Armor not configured"
        }
    
    try:
        access_token = await get_access_token()
        base_url = get_base_url()
        sanitize_url = f"{base_url}:sanitizeModelResponse"
        
        payload = {
            "modelResponseData": {
                "text": response
            }
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(sanitize_url, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()
        
        sanitization_result = result.get("sanitizationResult", {})
        filter_match_state = sanitization_result.get("filterMatchState", "")
        blocked = filter_match_state == "MATCH_FOUND"
        
        sanitized_data = sanitization_result.get("sanitizedData", {})
        sanitized_text = sanitized_data.get("text", response)
        
        return {
            "status": "blocked" if blocked else "not_blocked",
            "sanitized_text": sanitized_text,
            "blocked": blocked,
            "raw_response": result
        }
    except Exception as e:
        return {
            "status": "error",
            "sanitized_text": response,
            "blocked": False,
            "reason": f"Error sanitizing response: {str(e)}"
        }

