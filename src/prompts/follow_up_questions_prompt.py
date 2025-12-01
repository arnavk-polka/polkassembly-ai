"""Prompt template for generating follow-up questions"""

PROMPT_TEMPLATE = """Based on this query: "{query}"
And these related topics: {topics_str}

Generate exactly 3 short, relevant follow-up questions that a user might want to ask next. Each question should:
- Be directly related to the original query or mentioned topics
- Be concise (under 15 words)
- Explore different aspects or dive deeper
- Be practical and useful

Format: Return only the 3 questions, one per line, without numbers or bullets."""

