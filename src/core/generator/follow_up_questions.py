import logging
from typing import List, Dict, Any
import random

logger = logging.getLogger(__name__)

def generate_follow_up_questions(self, query: str, chunks: List[Dict[str, Any]], answer: str) -> List[str]:
    try:
        topics = set()
        for chunk in chunks[:3]:
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', '')
            content = chunk.get('content', '')
            
            if title:
                topics.add(title.split(' - ')[0])
            
            polkadot_terms = [
                'parachain', 'relay chain', 'governance', 'staking', 'validator',
                'nominator', 'treasury', 'referendum', 'proposal', 'council',
                'DOT', 'KSM', 'kusama', 'democracy', 'xcm', 'bridge',
                'consensus', 'runtime', 'substrate', 'crowdloan', 'auction'
            ]
            
            content_lower = content.lower()
            for term in polkadot_terms:
                if term.lower() in content_lower and term.lower() not in query.lower():
                    topics.add(term)
        
        topics_list = list(topics)[:5]
        topics_str = ', '.join(topics_list) if topics_list else 'Polkadot ecosystem'
        
        from ...prompts.follow_up_questions_prompt import PROMPT_TEMPLATE as follow_up_questions_template
        follow_up_prompt = follow_up_questions_template.format(
            query=query,
            topics_str=topics_str
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": follow_up_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        follow_up_text = response.choices[0].message.content.strip()
        
        from .context_processing import clean_example_urls
        follow_up_text = clean_example_urls(follow_up_text)
        
        follow_up_questions = [q.strip() for q in follow_up_text.split('\n') if q.strip()]
        
        if len(follow_up_questions) < 2:
            follow_up_questions = get_fallback_follow_ups(query)
        elif len(follow_up_questions) > 3:
            follow_up_questions = follow_up_questions[:3]
        
        return follow_up_questions
        
    except Exception as e:
        logger.warning(f"Error generating follow-up questions: {e}")
        return get_fallback_follow_ups(query)

def get_helpful_follow_ups() -> List[str]:
    helpful_questions = [
        "How does Polkadot's governance system work?",
        "What are the benefits of staking DOT tokens?",
        "How do parachains communicate with each other?",
        "What makes Polkadot different from other blockchains?",
        "How can I participate in Polkadot governance?",
        "What are the risks and rewards of DOT staking?",
    ]
    
    return random.sample(helpful_questions, 3)

def get_fallback_follow_ups(query: str) -> List[str]:
    query_lower = query.lower()
    
    if any(term in query_lower for term in ['governance', 'vote', 'proposal', 'referendum']):
        return [
            "How do I participate in Polkadot governance?",
            "What are the different governance tracks?",
            "How does voting power work in OpenGov?"
        ]
    elif any(term in query_lower for term in ['staking', 'validator', 'nominator']):
        return [
            "What are the risks of staking DOT?",
            "How do I choose good validators?",
            "What is the minimum amount needed to stake?"
        ]
    elif any(term in query_lower for term in ['parachain', 'slot', 'auction']):
        return [
            "How do parachain auctions work?",
            "What is the difference between parachains and parathreads?",
            "How do parachains communicate with each other?"
        ]
    elif any(term in query_lower for term in ['dot', 'token', 'price', 'economics']):
        return [
            "What are the main uses of DOT token?",
            "How does DOT inflation work?",
            "Where can I buy and store DOT?"
        ]
    else:
        return [
            "How does Polkadot differ from other blockchains?",
            "What are the main benefits of using Polkadot?",
            "How can I get started with Polkadot?"
        ]

