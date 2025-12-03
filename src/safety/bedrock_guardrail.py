import os
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import boto3
from botocore.config import Config
import openai

load_dotenv()

REGION = os.environ["AWS_REGION"]
GUARDRAIL_ID = os.environ["BEDROCK_GUARDRAIL_ID"]

bedrock_rt = boto3.client("bedrock-runtime", region_name=REGION, config=Config(retries={"max_attempts": 3}))

async def check_with_guardrail_async(query: str) -> Dict[str, Any]:
    """
    Async version of guardrail check to avoid blocking FastAPI event loop
    Returns:
      {
        "status": "blocked" | "not_blocked",
        "action": "NONE" | "GUARDRAIL_INTERVENED",
        "reason": str | None
      }
    """
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: bedrock_rt.apply_guardrail(
                guardrailIdentifier=GUARDRAIL_ID,
                guardrailVersion=GUARDRAIL_VERSION,
                source="INPUT",
                content=[
                    {
                        "text": {
                            "text": query
                        }
                    }
                ]
            )
        )
        
        action = resp.get("action", "NONE")
        blocked = (action == "GUARDRAIL_INTERVENED")
        
        violation_details = {}
        if blocked:
            outputs = resp.get("outputs", [])
            content_filters = resp.get("contentFilters", [])
            topics = resp.get("topics", [])
            word_policy = resp.get("wordPolicy", {})
            
            violation_details = {
                "content_filters": content_filters,
                "topics": topics,
                "word_policy": word_policy,
                "custom_message": outputs[0].get("text") if outputs and len(outputs) > 0 else None
            }
        
        reason = "Content policy violation" if blocked else None
        
        return {
            "status": "blocked" if blocked else "not_blocked",
            "action": action,
            "reason": reason,
            "violation_details": violation_details if blocked else None,
            "raw_response": resp if blocked else None
        }
    except Exception as e:
        print(f"Error calling guardrail: {e}")
        return {
            "status": "error",
            "action": "ERROR",
            "reason": str(e)
        }

def debug_environment():
    """Debug function to check environment variables"""
    print(f"AWS_REGION: {os.environ.get('AWS_REGION', 'NOT SET')}")
    print(f"BEDROCK_GUARDRAIL_ID: {os.environ.get('BEDROCK_GUARDRAIL_ID', 'NOT SET')}")

async def generate_user_friendly_block_message(violation_details: Dict[str, Any], user_query: str) -> str:
    """
    Generate a natural language explanation for why a query was blocked using GPT-3.5-turbo.
    
    Args:
        violation_details: Dictionary containing violation details from guardrail
        user_query: The original user query that was blocked
    
    Returns:
        A user-friendly natural language explanation
    """
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "Your query was blocked because it violates our content policy. Please revise your query to comply with our terms of service."
        
        client = openai.OpenAI(api_key=openai_api_key, timeout=5.0)
        
        violation_summary = []
        
        content_filters = violation_details.get("content_filters", [])
        if content_filters:
            filter_info = []
            for cf in content_filters:
                filter_type = cf.get("type", "")
                confidence = cf.get("confidence", "")
                if filter_type:
                    filter_info.append(f"{filter_type}" + (f" ({confidence} confidence)" if confidence else ""))
            if filter_info:
                violation_summary.append(f"Content filter violations: {', '.join(filter_info)}")
        
        topics = violation_details.get("topics", [])
        if topics:
            topic_info = []
            for topic in topics:
                topic_name = topic.get("name", "")
                if topic_name:
                    topic_info.append(topic_name)
            if topic_info:
                violation_summary.append(f"Topic policy violations: {', '.join(topic_info)}")
        
        word_policy = violation_details.get("word_policy", {})
        if word_policy:
            managed_word_lists = word_policy.get("managedWordLists", [])
            if managed_word_lists:
                list_names = [mw.get("name", "") for mw in managed_word_lists if mw.get("name")]
                if list_names:
                    violation_summary.append(f"Word policy violations: {', '.join(list_names)}")
        
        custom_message = violation_details.get("custom_message")
        if custom_message and custom_message.strip().lower() != "blocked":
            violation_summary.append(f"Custom policy: {custom_message}")
        
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
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
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
        print(f"Error generating user-friendly block message: {e}")
        return "Your query was blocked because it violates our content policy. Please revise your query to comply with our terms of service. Continued violations may result in your IP being blocked."

def test_guardrail_exists():
    """Test if the guardrail exists and is accessible"""
    try:
        bedrock = boto3.client("bedrock", region_name=REGION)
        response = bedrock.get_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION
        )
        print(f"✅ Guardrail found: {response['name']}")
        print(f"   Status: {response['status']}")
        return True
    except Exception as e:
        print(f"❌ Guardrail not accessible: {e}")
        return False