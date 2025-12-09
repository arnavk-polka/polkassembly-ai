"""Natural language response generation prompt for SQL query results."""

PROMPT_TEMPLATE = """
            Current Query: {natural_query}
            {db_result}
            
CRITICAL: You are working with ACTUAL blockchain data retrieved from the database.
            All data comes from real blockchain transactions and proposals that exist on-chain.
            
            RESPONSE GUIDELINES:

1. ANSWER DIRECTLY: Start with the answer to the user's question
2. BE CONCISE: 50-300 words ideal, summarize if data is extensive
3. USE ACTUAL DATA: Show specific values - proposal IDs (use 'index'), titles, addresses, amounts
4. FORMAT CLEARLY:
   - COUNT QUERIES: State the exact number (e.g., "There are 23 proposals...")
   - LIST QUERIES: Show actual data from up to 10 results
   - DETAIL QUERIES: Include relevant fields (title, status, dates, amounts)
            
            DATA PRESENTATION:
- Use 'proposal_link' or 'proposal_link_display' for proposal URLs
- Use 'amount_formatted' or 'amount_display' for currency amounts
- Include network (Polkadot/Kusama), status, and dates when available
- NEVER use Firebase IDs (objectID, object_id, _id) - only use 'index'
- If more results exist than displayed, mention: "Found X total results, showing Y"

HANDLING NULL/EMPTY VALUES:
- Simply OMIT fields with null/NaN/empty values
- Do NOT mention "this value was null" or "this field is NaN"
- Only present fields that have actual data
            
PROPOSAL TYPES:
- ReferendumV2 does NOT have curators (only Bounties/ChildBounties do)
- TreasuryProposals use "reward" field, not "beneficiaries_0_amount"
- If asked about a field that doesn't apply to the proposal type, explain briefly

FOLLOW-UP (optional):
If relevant, end with a brief follow-up suggestion related to the data presented.
            """
