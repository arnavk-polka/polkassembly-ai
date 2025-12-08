"""Intent extraction prompt for parsing natural language queries into structured intent.

NOTE: This is used as a FALLBACK when tool-based query processing cannot handle the query.
Most common query patterns are handled by dedicated tools without requiring intent extraction.
"""

PROMPT = """You are a query intent extractor. Return ONLY valid JSON with no additional text."""

PROMPT_TEMPLATE = """Extract structured intent from this natural language query about Polkadot/Kusama governance data.

User Query: "{natural_query}"

Conversation History:
{history_text}

You must return ONLY valid JSON with these exact keys:
{{
    "entity_type": "referenda" | "treasury_proposal" | "bounty" | "child_bounty" | "tip" | "fellowship" | "discussion" | "voter" | "delegate" | "unknown",
    "network": "polkadot" | "kusama" | "both" | "unspecified",
    "id": number or null,
    "time_range": "last_30_days" | "last_90_days" | "all_time" | "unspecified",
    "metric": "count" | "list" | "sum" | "avg" | "details",
    "filters": "short free-text description of additional filters"
}}

Rules:
- entity_type: What is the user asking about? (referenda, treasury proposals, bounties, tips, fellowship, discussions, voters, delegates, or unknown)
- If query mentions "discussion" or asks about a discussion post, use entity_type: "discussion"
- "ref" or "referendum" means entity_type: "referenda"
- network: Extract network preference (polkadot, kusama, both, or unspecified if not mentioned)
- id: Extract specific proposal/referendum/bounty ID if mentioned (number or null)
- time_range: Extract time filter if mentioned
- metric: Determine operation (count for "how many", list for "show me", details for specific item queries)
- filters: Brief description of any other filters (e.g., "status=active", "title contains X", "amount > Y")

Return ONLY the JSON object, no other text."""
