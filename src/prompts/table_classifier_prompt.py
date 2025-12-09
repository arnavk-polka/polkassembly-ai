"""Table classifier prompt for determining which database table to query.

NOTE: This is used as a FALLBACK when tool-based query processing cannot handle the query.
The tool system automatically routes to the correct table based on the selected tool.
"""

PROMPT_TEMPLATE = """You are a query classifier for a blockchain governance database. Determine which table to query.

Query: "{query}"

Tables:

1. governance_data - Contains proposal/referendum information:
   - Proposal details (title, content, description, status, type)
   - Proposal metadata (dates, network, proposer, curator)
   - Financial data (amounts, beneficiaries, rewards)
   - Proposal types: ReferendumV2, TreasuryProposal, Bounty, ChildBounty, Tip, FellowshipReferendum, Discussion
   - Pre-aggregated vote metrics (aye/nay counts stored with proposals - less accurate)
   - USE FOR: Proposal details, status, content, proposer info, spending amounts ONLY

2. voting_data - Contains individual voter activity:
   - Voter accounts and addresses
   - Vote decisions (Aye/Nay/Abstain)
   - Voting power and conviction/lock periods
   - Vote delegation information
   - Voting timestamps
   - One record per vote (source of truth for vote counts)
   - USE FOR: Vote counts, who voted, how they voted, voting power, delegation patterns

CRITICAL RULE - VOTE COUNT QUERIES:
ANY query asking about "how many votes", "vote count", "number of votes", "total votes", "votes received" MUST use voting_data.
This includes queries like:
- "How many votes did referenda X receive?" → voting_data
- "How many votes did proposal X get?" → voting_data
- "What's the vote count for referendum X?" → voting_data
- "Number of votes for proposal X" → voting_data

Decision Rules:
- Questions about VOTE COUNTS / NUMBER OF VOTES → ALWAYS voting_data (count individual vote records)
- Questions about VOTERS, WHO voted, individual voting behavior → voting_data
- Questions about PROPOSALS, status, content, amounts (NOT vote counts) → governance_data
- If both are mentioned, prioritize the main focus:
  * "Show voters for proposal X" → voting_data
  * "How many votes did proposal X receive?" → voting_data (MUST use voting_data)
  * "Who voted for proposal X?" → voting_data
  * "What are the vote counts for proposal X?" → voting_data
  * "Proposal status of X" → governance_data (no vote count mentioned)

EXAMPLES:
- "How many votes did referenda 1781 receive?" → {{"table": "voting_data"}}
- "Who voted for proposal 100?" → {{"table": "voting_data"}}
- "What is the status of referendum 50?" → {{"table": "governance_data"}}
- "Show me proposal 200 details" → {{"table": "governance_data"}}

Respond with ONLY valid JSON:
{{"table": "governance_data"}} or {{"table": "voting_data"}}"""
