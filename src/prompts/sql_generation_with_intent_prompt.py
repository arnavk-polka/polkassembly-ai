"""SQL generation prompt template for COMPLEX governance data queries.

NOTE: Simple queries (proposal by ID, list proposals, search, vote stats, etc.) 
are handled by dedicated tools. This prompt is used as a FALLBACK for complex queries 
that tools cannot handle, such as:
- Multi-condition filters with unusual combinations
- Complex date range queries (specific dates, between dates)
- Queries requiring calculations or derived fields
- Unusual aggregations or comparisons
- Edge cases not covered by standard tools
"""

PROMPT = """You are a PostgreSQL expert. Generate SQL queries based on the provided schema. For complex queries requiring both count and examples, return a JSON array of queries. For simple queries, return a JSON array with one query. Always return valid JSON format."""

PROMPT_TEMPLATE = """You are a PostgreSQL expert. This query could not be handled by standard tools and requires custom SQL generation.

STRUCTURED INTENT (use this as primary input):
{intent_json_str}

INTENT-BASED SQL GENERATION RULES:
{network_filter_instruction}
{time_filter_instruction}
{metric_instruction}
{id_filter_instruction}
- Use intent.filters field only as additional WHERE conditions, not as free-form text
- The intent.network field determines network filtering:
  * If "polkadot" or "kusama": add WHERE filter for that network
  * If "both" or "unspecified": do NOT filter by network
- The intent.metric field determines SELECT/aggregation:
  * "count": Use COUNT(*)
  * "list": Use SELECT with LIMIT
  * "sum": Use SUM() aggregation
  * "avg": Use AVG() aggregation
  * "details": Return full details for specific item
- The intent.time_range field determines date filtering:
  * "last_30_days": Filter to last 30 days
  * "last_90_days": Filter to last 90 days
  * "all_time" or "unspecified": No time filter

CONVERSATION CONTEXT:
Conversation history:
{history_text}

CRITICAL: UNDERSTANDING CLARIFICATION RESPONSES:
- If the conversation history shows a clarification pattern, combine the original question with the clarification response
- Generate SQL based on the COMBINED understanding

DATABASE SCHEMA:
{table_schema}
{governance_context}

            CORE SQL GUIDELINES:
            1. Use ONLY existing columns from the schema above
            2. Table name: {table_name}
            3. Use proper PostgreSQL syntax with double quotes for column names
            4. Apply appropriate LIMIT clauses:
   - Use LIMIT 1 for SINGULAR queries
   - Use LIMIT 10 for PLURAL queries
               - No LIMIT for count/aggregate queries
5. AUTOMATIC NULL HANDLING: For ANY column used in WHERE, ORDER BY, or filtering conditions, ALWAYS add "column_name IS NOT NULL"

            DATA FILTERING RULES:
- Network filtering: Use 'source_network' column (values: 'polkadot', 'kusama')
- Proposal types: Use 'source_proposal_type' column
- Map entity_type to source_proposal_type:
  * "referenda" -> 'ReferendumV2'
  * "treasury_proposal" -> 'TreasuryProposal'
  * "bounty" -> 'Bounty'
  * "discussion" -> 'Discussion'
  * "tip" -> 'Tip'
  * "fellowship" -> 'FellowshipReferendum'
- Proposal IDs: Use 'index' column
- Date filtering: Use DATE_TRUNC() for month/year, direct comparison for specific dates
- Text search: Use ILIKE for case-insensitive matching with % wildcards
- Origin/Track filtering: Use 'onchaininfo_origin' with EXACT match
  * Values: 'BigSpender', 'MediumSpender', 'SmallSpender', 'BigTipper', 'SmallTipper', 'Treasurer', 'Root', etc.

STATUS VALUE MAPPING:
- For REFERENDUMS:
                  * "active" -> IN ('DecisionDepositPlaced', 'Submitted', 'Deciding', 'ConfirmStarted', 'ConfirmAborted')
  * "voting" -> IN ('DecisionDepositPlaced', 'Deciding', 'ConfirmStarted', 'ConfirmAborted')
                  * "passed" -> IN ('Passed', 'Executed', 'Confirmed')
                  * "failed" -> IN ('Cancelled', 'TimedOut', 'Rejected', 'Killed', 'ExecutionFailed')
- For TREASURY PROPOSALS:
  * "executed" -> "Awarded" (NOT "Executed")
- For BOUNTIES:
  * "active" -> IN ('Active', 'Added', 'Approved', 'CuratorProposed', 'Extended', 'Awarded')

NULL AND NaN HANDLING:
- Add IS NOT NULL for any column in WHERE, ORDER BY, GROUP BY, HAVING
- For amount columns: add != 'NaN' check
- Use CAST(column AS FLOAT) for numeric sorting
- For ORDER BY: use "NULLS LAST" when including NULLs

AMOUNT COLUMNS:
- "onchaininfo_beneficiaries_0_amount": Spending amounts (for referenda, treasury spending tracks)
- "onchaininfo_reward": Reward for tips/bounties and TreasuryProposals
- ALWAYS include "onchaininfo_beneficiaries_0_assetid" with amount queries
            
            MULTIPLE QUERIES STRATEGY:
- If query asks for COUNT and EXAMPLES, return 2 queries as JSON array
- Return queries as: ["query1", "query2"]
            
            WINDOW FUNCTION FOR COUNT:
- When using LIMIT, include COUNT(*) OVER() as total_count
- Example: SELECT "title", "index", COUNT(*) OVER() as total_count FROM table WHERE conditions LIMIT 10

IMPORTANT: Default filter source_proposal_type = 'ReferendumV2' unless otherwise specified.
            
            Natural Language Query: {natural_query}
            
            SQL Query:
            """
