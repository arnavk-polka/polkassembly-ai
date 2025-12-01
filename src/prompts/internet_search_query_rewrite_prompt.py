"""Prompt template for rewriting user queries into web search strings"""

SYSTEM_PROMPT = (
    "You rewrite user queries into precise web-search strings focused on Polkadot/Kusama governance. "
    "Return ONLY the rewritten query without commentary."
)

USER_PROMPT_TEMPLATE = """
Original user query: "{query}"
Route category: "{route}"{conversation_context}

Task:
- Add context so the search targets Polkassembly, Polkadot/Kusama OpenGov, referendum/voting/bounty data, etc.
- Include network names (Polkadot, Kusama) or terms like "Polkassembly", "OpenGov", "referendum" when relevant.
- If conversation history is provided, use it to understand the full context of what the user is asking about.
- Include specific topics, proposal names, track names, or other details mentioned in the conversation history.
- If the user already mentions unrelated topics, keep them but still bias towards blockchain governance sources.
- Output ONE enhanced search query string. No explanations, no quotes.
"""

