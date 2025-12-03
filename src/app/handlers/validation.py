"""
Answer validation logic.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from src.core.errors import is_insufficient_quota_error
from ..pipeline.utils import log_step, _format_conversation_history_for_validation

logger = logging.getLogger(__name__)


async def validate_static_answer_with_llm(
    query: str,
    answer: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    qa_generator,
    log_step
) -> bool:
    """
    Validate that the generated static answer truly addresses the query,
    considering the entire conversation history.
    """
    if not answer:
        return False
    
    if not qa_generator or not getattr(qa_generator, "client", None):
        log_step("static_answer_validator_skipped", {
            "reason": "no_llm_client_available"
        }, "warning")
        return True
    
    history_text = _format_conversation_history_for_validation(conversation_history) or "No prior messages."
    
    from ...prompts.static_answer_validation_prompt import PROMPT_TEMPLATE as validation_prompt_template
    validation_prompt = validation_prompt_template.format(
        history_text=history_text,
        query=query,
        answer=answer
    )
    
    try:
        response = qa_generator.client.chat.completions.create(
            model=os.getenv("STATIC_VALIDATION_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a validator that only replies 'yes' or 'no'. Reject answers that say they don't have information. Accept answers that provide actual information, even if incomplete."},
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0.0,
            max_tokens=3
        )
        decision = (response.choices[0].message.content or "").strip().lower()
        is_valid = decision.startswith("y")
        log_step("static_answer_validation_complete", {
            "decision": decision,
            "is_valid": is_valid
        })
        return is_valid
    except Exception as e:
        if is_insufficient_quota_error(e):
            log_step("static_answer_validation_error", {
                "error": str(e),
                "quota_error": True
            }, "warning")
            raise
        log_step("static_answer_validation_error", {
            "error": str(e)
        }, "warning")
        return True

