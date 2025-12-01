"""Table classifier prompt for determining which database table to query"""

PROMPT_TEMPLATE = """You are a query classifier for a blockchain governance database. Determine which table to query.

Query: "{query}"

Tables:

1. governance_data - Contains proposal information:
   - Proposal details (title, content, description, status, type)
   - Proposal metadata (dates, network, proposer)
   - Financial data (amounts, beneficiaries, asset IDs)
   - Proposal metrics (likes, comments)
   - Examples:
     * "Show me recent treasury proposals"
     * "What's the status of proposal 123?"
     * "Find proposals requesting more than 10000 DOT"
     * "List all executed referendums"
     * "Who proposed referendum 456?"
     * "Show me proposals about topic X"

2. voting_data - Contains voter activity and behavior:
   - Voter accounts and addresses
   - Vote decisions (Aye/Nay/Abstain)
   - Voting power and locked amounts
   - Conviction multipliers and lock periods
   - Vote delegation
   - Voting timestamps
   - Examples:
     * "How many people voted on proposal 123?"
     * "Show me votes with 6x conviction"
     * "Who voted Aye on referendum 456?"
     * "List voters with >1000 DOT voting power"
     * "Show delegated votes for proposal X"
     * "What was voter Y's decision?"
     * "Count unique voters in the last 30 days"

Decision Rules:
- If query asks about WHO voted, HOW people voted, VOTER behavior → voting_data
- If query asks about WHAT proposals exist, proposal STATUS, proposal DETAILS → governance_data
- If query mentions both, prioritize the main focus:
  * "Show me voters who participated in treasury proposals" → voting_data (focus: voters)
  * "Show me treasury proposals and their vote counts" → governance_data (focus: proposals)

Respond with ONLY valid JSON:
{{"table": "governance_data"}} or {{"table": "voting_data"}}"""

