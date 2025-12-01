"""Query combination prompt for merging original query with clarification response"""

PROMPT_TEMPLATE = """
You are helping to combine a user's original query with their clarification response.

Original query: "{original_query}"

Clarification question that was asked: "{clarification_question}"

User's clarification response: "{clarification_response}"{conversation_context}

Your task:
- Preserve ALL keywords, topics, and specific terms from the original query (e.g., "vitro connect", proposal names, track names, etc.)
- The clarification response provides ADDITIONAL context (like network name, specific ID, etc.), NOT replacing the original topic
- Use the conversation history to understand the full context
- Create a single, clear, and coherent query that combines both the original intent AND the clarification
- Output ONLY the combined query, no explanations

Now create the combined query:
"""

