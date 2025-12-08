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
   - Vote metrics (aye/nay counts and values from proposals)
   - USE FOR: Proposal details, status, content, proposer info, spending amounts

2. voting_data - Contains individual voter activity:
   - Voter accounts and addresses
   - Vote decisions (Aye/Nay/Abstain)
   - Voting power and conviction/lock periods
   - Vote delegation information
   - Voting timestamps
   - USE FOR: Who voted, how they voted, voting power, delegation patterns

Decision Rules:
- Questions about VOTERS, WHO voted, voting behavior → voting_data
- Questions about PROPOSALS, status, content, amounts → governance_data
- If both are mentioned, prioritize the main focus:
  * "Show voters for proposal X" → voting_data (focus: voters)
  * "Show proposal vote counts" → governance_data (focus: proposal metrics)

Respond with ONLY valid JSON:
{{"table": "governance_data"}} or {{"table": "voting_data"}}"""
