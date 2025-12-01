"""Static answer validation prompt for checking if generated answer addresses the query"""

PROMPT_TEMPLATE = """
Conversation History:
{history_text}

Current User Query:
{query}

Candidate Answer:
{answer}

Does the candidate answer directly and accurately address the user's query while respecting the conversation context?

CRITICAL: Reject answers that indicate no information was found, such as:
- "I don't have information about"
- Any answer that explicitly states it cannot answer the question

If the answer reasonably addresses the user's query, even if it's not perfect, respond "yes".
If the answer says it doesn't have information or cannot answer, respond "no".

Respond with exactly one word: "yes" or "no".
"""

