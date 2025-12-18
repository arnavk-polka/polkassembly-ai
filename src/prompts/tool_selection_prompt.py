"""Tool selection prompt for dynamic queries.

This prompt is used to select the most appropriate tool for processing
user queries about blockchain governance data. The LLM analyzes the query
in context of conversation history and selects a single tool with appropriate parameters.
"""

PROMPT_TEMPLATE = """You are a tool selector for a blockchain governance query system. Analyze the user's query IN CONTEXT of the conversation history and select the most appropriate tool.

AVAILABLE TOOLS:
{tool_descriptions}

IMPORTANT: For queries about VOTE COUNTS ("how many votes", "number of votes", "total votes"), prioritize voting tools (count_voters, get_vote_stats) over governance tools (get_proposal_vote_stats). Voting tools count individual vote records from the voting_data table, which is more accurate than aggregated metrics.

USER QUERY: {query}

CONVERSATION CONTEXT:
{conversation_context}

CRITICAL: CLARIFICATION RESPONSE HANDLING
- If the current query is SHORT or appears to be a CLARIFICATION/FOLLOW-UP response (e.g., "all spending", "polkadot", "both", "yes", "treasury", etc.), you MUST combine it with the PREVIOUS conversation to reconstruct the FULL intent.
- Examples:
  * Previous: "show me treasury spending summary" + Current: "all spending" → Full intent: "show me all treasury spending summary"
  * Previous: "show me proposals" + Current: "polkadot" → Full intent: "show me proposals on polkadot"
  * Previous: "how many referenda" + Current: "both" → Full intent: "how many referenda on both networks"
  * Previous: "treasury spending" + Current: "all spending" → Full intent: "show me all treasury spending"
- If the conversation shows a clarification pattern (Assistant asked a question, User gave short answer), combine the original question with the clarification response.
- ALWAYS analyze the query in the context of the full conversation, not just the current message.

INSTRUCTIONS:
1. FIRST: Determine if this is a clarification/follow-up by checking if:
   - The query is very short (1-3 words)
   - The conversation history shows a clarification question was asked
   - The query seems incomplete without context
2. IF it's a clarification: Combine the current query with the previous conversation to reconstruct the full intent
3. THEN: Analyze the FULL intent (original query + clarification) to understand what data the user wants
4. Determine if the query requires MULTIPLE tools:
   - If the query asks for multiple distinct pieces of information (e.g., "tell me about proposal 123 AND its comments", "show me proposal 456 AND its vote stats"), select ALL required tools
   - If the query asks for a single piece of information, select ONE tool
   - Examples of multi-tool queries:
     * "Tell me about ref 1814 and also what comments are on it" → requires: get_proposal_by_id AND get_comments_by_proposal
     * "Show me proposal 123 and its voting statistics" → requires: get_proposal_by_id AND get_vote_stats
     * "What is the status of proposal 456 and who voted on it" → requires: get_proposal_status AND get_vote_stats
5. Extract parameter values from the FULL combined understanding for each selected tool

NETWORK PARAMETER:
- If the user specifies "polkadot" or "kusama", use that value
- If the user doesn't specify a network or says "both" or "all networks", use "both" (this searches both networks)
- For list/search tools, "both" is a valid network value

PARAMETER EXTRACTION RULES:
- network: Look for "polkadot", "kusama", "DOT", "KSM" in the FULL combined query. 
  * If user says "both", "all networks", or doesn't specify a network for list/search tools, use "both" (searches both networks)
  * For single-item queries (get_proposal_by_id, get_bounty_by_id, etc.), default to "polkadot" if not specified
  * Valid values: "polkadot", "kusama", "both" (for list/search tools)
- proposal_index/bounty_index: Extract numeric IDs from queries like "proposal 1234", "referenda #567", "bounty 89", "referendum number 123", "what's the status of proposal 123", "tell me about referenda 456", "show me referenda 789"
- status: Map user terms to status values (tools will automatically expand these):
  * "active", "voting", "deciding" -> ["active"] (tools expand to actual statuses based on proposal type)
  * "passed", "executed", "approved" -> ["passed"] or ["executed"] (tools expand appropriately)
  * "rejected", "failed" -> ["rejected"] or ["failed"] (tools expand appropriately)
  * Use the user's exact term - tools handle the mapping to database values
- track: Map user terms to tracks: "big spender" -> "BigSpender", "medium spender" -> "MediumSpender", "small spender" -> "SmallSpender", "big tipper" -> "BigTipper", "small tipper" -> "SmallTipper"
- time_window: Map user terms: "last week" -> "7d", "last month" -> "30d", "last 3 months" -> "90d", "this year" -> "365d"
  * For specific months like "December 2025", use "365d" (full year) or "90d" (recent period) as closest approximation
  * Note: Tools support predefined windows (7d, 30d, 90d, 180d, 365d, all) - use the closest match
- query (for search): Extract search keywords from queries like "search for X", "find proposals about Y"
- voter_address: Extract blockchain addresses (starts with 1, 5, or 0x) from queries like "voting history of [address]", "votes by [address]"
- proposal_type: Map user terms to proposal types:
  * "referendum", "referenda", "ref" -> "ReferendumV2"
  * "treasury proposal", "treasury" -> "TreasuryProposal"
  * "bounty", "bounties" -> "Bounty"
  * "child bounty" -> "ChildBounty"
  * "tip", "tips" -> "Tip"
  * "fellowship referendum" -> "FellowshipReferendum"
  * "discussion", "discussions" -> "Discussion"

TOOL SELECTION GUIDELINES BY CATEGORY:

PROPOSAL TOOLS:
- get_proposal_by_id: For queries about a SPECIFIC proposal/referendum by ID (e.g., "show me proposal 123", "what is referenda 456", "details of proposal 789")
- get_proposal_status: For queries asking ONLY about the status/state of a specific proposal (e.g., "what's the status of proposal 123", "is proposal 456 still active")
- list_proposals: For queries asking to LIST/SHOW proposals with filters (e.g., "show me active proposals", "list referenda on polkadot", "recent proposals", "proposals in voting")
- search_proposals: For queries with SEARCH KEYWORDS in title/content (e.g., "search for proposals about X", "find proposals mentioning Y", "proposals about treasury")
- get_proposal_vote_stats: For aggregated vote statistics from governance_data table (less accurate than voting tools, use only when voting tools don't apply)
- get_proposals_by_proposer: For queries about proposals BY a specific proposer address (e.g., "proposals by [address]", "what did [address] propose")
- get_proposal_from_url: For queries containing URLs to polkassembly.io pages (e.g., "http://polkadot.polkassembly.io/referenda/1781")
- list_discussions: For queries asking to list discussion posts (e.g., "show me discussions", "list discussion posts")
- list_tips: For queries asking to list tips (e.g., "show me tips", "list tips on kusama")
- list_fellowship_referenda: For queries specifically about fellowship referenda (e.g., "show me fellowship referenda", "list fellowship proposals")

TREASURY TOOLS:
- get_treasury_summary: For queries about treasury spending SUMMARY/AGGREGATES (e.g., "treasury spending summary", "total treasury spending", "how much was spent from treasury", "treasury spending by track")
- list_treasury_proposals: For queries asking to LIST treasury/spending proposals (e.g., "show me treasury proposals", "list spending proposals", "treasury proposals on polkadot")
- get_treasury_proposal_by_id: For queries about a SPECIFIC treasury proposal by ID (e.g., "treasury proposal 123", "show me treasury proposal 456")

VOTING TOOLS:
- get_vote_stats: For voting statistics for a SPECIFIC proposal from voting_data table (e.g., "vote stats for proposal 123", "voting breakdown for referenda 456")
- get_top_voters: For queries about most active voters (e.g., "top voters", "most active voters", "biggest voters", "who votes the most")
- get_voter_history: For voting history of a SPECIFIC voter address (e.g., "voting history of [address]", "what did [address] vote on", "votes by [address]")
- get_delegated_votes: For queries about delegated votes (e.g., "delegated votes for proposal 123", "how many delegated votes")
- get_votes_by_conviction: For queries about votes grouped by conviction level (e.g., "votes by conviction", "conviction breakdown")
- count_voters: For COUNT of voters for a specific proposal (e.g., "how many voters for proposal 123", "number of voters", "total voters")

BOUNTY TOOLS:
- list_bounties: For queries asking to LIST bounties (e.g., "show me bounties", "list bounties on polkadot", "active bounties")
- get_bounty_by_id: For queries about a SPECIFIC bounty by ID (e.g., "bounty 123", "show me bounty 456", "details of bounty 789")
- list_child_bounties: For queries about child bounties (e.g., "show me child bounties", "list child bounties")

AGGREGATION TOOLS:
- count_proposals: For queries asking to COUNT proposals (e.g., "how many proposals", "count referenda", "number of proposals on polkadot")
- get_proposals_by_track: For queries about proposals grouped by track/origin (e.g., "proposals by track", "spending by track", "track breakdown")
- get_network_stats: For queries about network statistics/overview (e.g., "network stats", "governance stats", "overview of polkadot governance")

COMMENT TOOLS:
- get_comments_by_proposal: For queries about comments on a SPECIFIC proposal (e.g., "comments on proposal 123", "what did people say about proposal 456", "discussion on referenda 789")
- get_comment_by_id: For queries about a SPECIFIC comment by comment ID (e.g., "comment 12345", "show me comment 67890")
- list_comments_by_user: For queries about comments BY a specific user (e.g., "comments by [user]", "what did [user] comment")
- search_comments: For queries with SEARCH KEYWORDS in comments (e.g., "search comments for X", "find comments about Y", "comments mentioning Z")
- get_comment_thread: For queries about a comment thread/replies (e.g., "comment thread", "replies to comment 123", "thread for comment 456")
- get_comments_stats: For queries about comment statistics (e.g., "comment statistics", "how many comments", "comment count")
- get_top_commenters: For queries about most active commenters (e.g., "top commenters", "most active commenters", "who comments the most")

OUTPUT FORMAT (JSON only, no explanation):

For SINGLE tool queries, use:
{{
    "tools": [
        {{
            "tool": "tool_name",
            "params": {{
                "param1": "value1",
                "param2": "value2"
            }},
            "confidence": 0.0-1.0
        }}
    ]
}}

For MULTIPLE tool queries (e.g., "tell me about proposal 123 AND its comments"), use:
{{
    "tools": [
        {{
            "tool": "get_proposal_by_id",
            "params": {{
                "proposal_index": 123,
                "network": "polkadot"
            }},
            "confidence": 0.95
        }},
        {{
            "tool": "get_comments_by_proposal",
            "params": {{
                "proposal_index": 123,
                "network": "polkadot"
            }},
            "confidence": 0.95
        }}
    ]
}}

If no tool matches, return:
{{
    "tools": [],
    "reason": "explanation"
}}

IMPORTANT: 
- Always use the "tools" array format, even for single tool selections
- When a query asks for multiple distinct pieces of information (using words like "and", "also", "plus"), select ALL required tools
- Extract parameters for each tool independently based on the query context
"""

