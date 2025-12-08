"""Natural language response prompt for multiple SQL query results."""

PROMPT_TEMPLATE = """
Convert these multiple query results into a unified natural response.

Conversation History:
{history_text}

Current Query: {natural_query}

Database Results:
{db_result}

RESPONSE GUIDELINES:

1. COMBINE RESULTS: Synthesize information from all queries into a coherent answer
2. BE CONCISE: 50-300 words ideal
3. ANSWER DIRECTLY: Start with the main answer to the user's question

DATA PRESENTATION:
- For COUNT + EXAMPLES queries: State the count, then show examples
- Use 'proposal_link' or 'proposal_link_display' for URLs
- Use 'amount_formatted' or 'amount_display' for currency amounts
- NEVER use Firebase IDs - only use 'index' field

HANDLING NULL/EMPTY:
- Simply OMIT fields with null/NaN/empty values
- Do NOT mention missing or null fields

EXAMPLE FORMAT:
"There are 45 active proposals. Here are some examples:
1. [Proposal Title] (Index: 1234) - Status: Deciding - [Link]
2. [Proposal Title] (Index: 1235) - Status: ConfirmStarted - [Link]
..."

FOLLOW-UP (optional):
End with a brief relevant follow-up suggestion if appropriate.

Response:
"""
