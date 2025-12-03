"""
Greeting handler utilities for generic route queries.
"""

from typing import Dict, Any, Optional, List
import logging
from src.core.errors import is_insufficient_quota_error, get_quota_error_message

logger = logging.getLogger(__name__)

_DEFAULT_FOLLOW_UP_QUESTIONS = [
    "How does Polkadot's governance system work?",
    "What are the benefits of staking DOT tokens?",
    "How do parachains connect to Polkadot?"
]

_DEFAULT_SOURCES = [
    {
        'title': 'Polkassembly Main Platform',
        'url': 'https://polkassembly.io',
        'source_type': 'platform',
        'similarity_score': 1.0
    },
    {
        'title': 'Polkadot Governance on Polkassembly',
        'url': 'https://polkadot.polkassembly.io',
        'source_type': 'platform',
        'similarity_score': 1.0
    }
]

_FALLBACK_ANSWER = "We're unable to answer your question at this time. Please try again later or rephrase your question."


def is_greeting_query(query: str) -> bool:
    """Check if query is a greeting"""
    greeting_keywords = [
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
        'good evening', 'what\'s up', 'whats up', 'wassup', 'howdy',
        'intro', 'introduction', 'who are you', 'what are you',
        'what do you do', 'what is this', 'help', 'start'
    ]
    
    query_lower = query.lower().strip()
    
    for keyword in greeting_keywords:
        if query_lower == keyword:
            return True
    
    if len(query_lower.split()) <= 3:
        for keyword in greeting_keywords:
            if keyword in query_lower:
                data_indicators = ['number', 'count', 'total', 'sum', 'average', 'highest', 'lowest', 
                                   'vote', 'votes', 'voter', 'proposal', 'referendum', 'treasury',
                                   'month', 'year', 'date', 'time', 'day', 'week']
                has_data_indicator = any(indicator in query_lower for indicator in data_indicators)
                if not has_data_indicator:
                    return True
    
    return False


def get_polkassembly_introduction() -> Dict[str, Any]:
    """Get the hardcoded greeting response"""
    introduction = """Hello! I'm **Klara** 👋 – your AI-powered governance assistant for **Polkadot** and **Kusama**!

I'm here to help you explore the governance ecosystem through **Polkassembly**, making it easy to query on-chain data, analyze proposals, and understand the voting process—all in natural language.

**What I can help you with:**

🗳 **Governance Data** - Query proposals, referenda, bounties, and treasury activities

📊 **Voting Analysis** - Track voter behavior, delegation, and voting power

💰 **Treasury Insights** - Explore funding proposals and beneficiary data

🧭 **Platform Guidance** - Learn how to use Polkassembly features and OpenGov

**Who I'm built for:**

- Community members exploring proposals
- Delegates analyzing voting patterns  
- Builders tracking treasury activities
- Researchers studying governance trends

**Try asking me things like:**

- "Show all active referenda on Polkadot"
- "Who voted on referendum 472?"
- "List treasury proposals above 100k DOT"
- "How does conviction voting work?"

**Useful Links:**

- **Klara Guide**: [klara.polkassembly.io/guide](https://klara.polkassembly.io/guide) - If you want detailed guidance on how to use Klara, follow this doc
- **Polkadot Governance**: [polkadot.polkassembly.io](https://polkadot.polkassembly.io)
- **Kusama Governance**: [kusama.polkassembly.io](https://kusama.polkassembly.io)
- **Documentation**: [docs.polkassembly.io](https://docs.polkassembly.io)"""

    sources = [
        {
            'title': 'Polkassembly Main Platform',
            'url': 'https://polkassembly.io',
            'source_type': 'platform',
            'similarity_score': 1.0
        },
        {
            'title': 'Polkadot Governance on Polkassembly',
            'url': 'https://polkadot.polkassembly.io',
            'source_type': 'platform',
            'similarity_score': 1.0
        },
        {
            'title': 'Kusama Governance on Polkassembly',
            'url': 'https://kusama.polkassembly.io',
            'source_type': 'platform',
            'similarity_score': 1.0
        },
        {
            'title': 'Polkassembly Documentation',
            'url': 'https://docs.polkassembly.io',
            'source_type': 'documentation',
            'similarity_score': 1.0
        }
    ]

    greeting_follow_ups = [
        "How does Polkadot governance work?",
        "What are parachains and how do they work?",
        "How can I start staking DOT tokens?"
    ]
    
    return {
        'answer': introduction,
        'sources': sources,
        'confidence': 1.0,
        'follow_up_questions': greeting_follow_ups,
        'context_used': True,
        'model_used': 'polkassembly_intro',
        'chunks_used': 0,
        'search_method': 'greeting_response'
    }


async def handle_generic_query_llm(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    qa_generator,
    log_step
) -> Dict[str, Any]:
    """
    Handle generic queries (greetings, non-Polkadot queries).
    For greetings, returns the exact same hardcoded response as before.
    For other generic queries, uses LLM.
    
    Returns:
        Dictionary with answer, sources, and metadata for generic responses
    """
    log_step("generic_handler_start", {"query_preview": query[:100]})
    
    if is_greeting_query(query):
        log_step("generic_handler_greeting_detected", {})
        return get_polkassembly_introduction()
    
    log_step("generic_handler_non_greeting", {"note": "Using LLM for non-greeting generic query"})
    try:

        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_messages = conversation_history[-10:]  
            context_parts = []
            for msg in recent_messages:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '')
                    if content and len(content) > 5:
                        role_display = role if role else 'user'
                        context_parts.append(f"{role_display}: {content[:300]}")
            if context_parts:
                conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand what the user is referring to in their query."
        
        from ..prompts.generic_query_prompt import PROMPT as system_prompt, PROMPT_TEMPLATE as user_prompt_template
        user_prompt = user_prompt_template.format(
            query=query,
            conversation_context=conversation_context
        )
        
        model_name = "gpt-4o-mini"
        
        if hasattr(qa_generator, 'client'):
            try:
                log_step("generic_handler_llm_call", {"model": model_name})
                
                response = qa_generator.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                answer = response.choices[0].message.content
                
                log_step("generic_handler_complete", {
                    "response_length": len(answer)
                })
                
                return {
                    'answer': answer.strip(),
                    'sources': _DEFAULT_SOURCES,
                    'confidence': 0.8,
                    'follow_up_questions': _DEFAULT_FOLLOW_UP_QUESTIONS,
                    'context_used': False,
                    'model_used': model_name,
                    'chunks_used': 0,
                    'search_method': 'generic_llm_response'
                }
            except Exception as e:
                if is_insufficient_quota_error(e):
                    log_step("generic_handler_error", {"error": str(e), "quota_error": True}, "error")
                    return {
                        'answer': get_quota_error_message(),
                        'sources': [],
                        'confidence': 0.0,
                        'follow_up_questions': [],
                        'context_used': False,
                        'model_used': 'error',
                        'chunks_used': 0,
                        'search_method': 'quota_error'
                    }
                log_step("generic_handler_error", {"error": str(e)}, "error")
        
        if qa_generator.gemini_client:
            log_step("generic_handler_fallback", {"reason": "openai_failed_using_gemini"}, "warning")
            try:
                fallback_prompt = f"""{system_prompt}

{user_prompt}"""
                model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
                response = qa_generator.gemini_client.get_response(fallback_prompt)
                
                return {
                    'answer': response.strip(),
                    'sources': _DEFAULT_SOURCES,
                    'confidence': 0.7,
                    'follow_up_questions': _DEFAULT_FOLLOW_UP_QUESTIONS,
                    'context_used': False,
                    'model_used': model_name,
                    'chunks_used': 0,
                    'search_method': 'generic_llm_fallback'
                }
            except Exception as e:
                log_step("generic_handler_error", {"error": str(e)}, "error")
            
            return {
                'answer': _FALLBACK_ANSWER,
                'sources': [],
                'confidence': 0.5,
                'follow_up_questions': _DEFAULT_FOLLOW_UP_QUESTIONS,
                'context_used': False,
                'model_used': 'fallback',
                'chunks_used': 0,
                'search_method': 'generic_fallback'
            }
            
    except Exception as e:
        log_step("generic_handler_error", {"error": str(e)}, "error")
        return {
            'answer': _FALLBACK_ANSWER,
            'sources': [],
            'confidence': 0.5,
            'follow_up_questions': _DEFAULT_FOLLOW_UP_QUESTIONS,
            'context_used': False,
            'model_used': 'error_fallback',
            'chunks_used': 0,
            'search_method': 'generic_error'
        }

