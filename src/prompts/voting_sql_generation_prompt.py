"""SQL generation prompt template for COMPLEX voting data queries.

NOTE: Simple voting queries (vote stats, top voters, voter history, delegated votes, 
conviction-based queries) are handled by dedicated tools. This prompt is used as a 
FALLBACK for complex queries that tools cannot handle, such as:
- Complex voting power calculations with joins
- Multi-proposal voting pattern analysis
- Time-series voting behavior
- Complex aggregations across multiple dimensions
- Edge cases not covered by standard tools
"""

PROMPT = """You are a PostgreSQL expert specializing in voting data. Generate SQL queries based on the provided schema. For complex queries requiring both count and examples, return a JSON array of queries. For simple queries, return a JSON array with one query. Always return valid JSON format."""

PROMPT_TEMPLATE = """You are a PostgreSQL expert specializing in voting data analysis. This query could not be handled by standard tools and requires custom SQL generation.

DATABASE SCHEMA:
Main Table: {table_name}
{table_schema}

Related Table: conviction_vote
- Contains "self_voting_power" (voting power/balance for each vote)
- Joined via "parent_vote_id" (foreign key in {table_name}) → "id" (primary key in conviction_vote)

CORE SQL GUIDELINES:
1. Use ONLY existing columns from the schema above.
2. Main table name: {table_name}
3. Use proper PostgreSQL syntax with double quotes for column names.
4. Apply appropriate LIMIT clauses (typically 10 for lists; no LIMIT for counts/aggregates).
5. Always order explicitly when returning recent items (e.g., ORDER BY main."created_at" DESC).
6. AUTOMATIC NULL HANDLING: For ANY column used in WHERE, ORDER BY, or filtering conditions, ALWAYS add "column_name IS NOT NULL".

JOIN REQUIREMENTS (for voting power queries):
- When querying voting power/balance, JOIN with conviction_vote table:
   FROM {table_name} AS main
   LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
- Use "cv.self_voting_power" for all voting power queries (replaces "balance").
- Always use table aliases (main, cv) to avoid ambiguity.

VOTING DATA COLUMNS:
- Voter: "main.voter"
- Proposal ID: "main.proposal_index" or "main.proposal_id"
- Decision: "main.decision" (values: 'aye', 'nay', 'abstain' - use ILIKE for case-insensitive)
- Voting power: "cv.self_voting_power" (FLOAT) - requires JOIN
- Delegation: "main.is_delegated" (BOOLEAN), "main.delegated_to" (target account)
- Vote date: "main.created_at"
- Revoked votes: "main.removed_at" (NULL for active votes)
- Proposal type: "main.type" (e.g., 'ReferendumV2', 'Treasury', 'Fellowship')
- Conviction/Lock: "main.lock_period"

NULL HANDLING:
- Add IS NOT NULL for columns in WHERE, ORDER BY, GROUP BY
- For voting power: "cv.self_voting_power IS NOT NULL" and include JOIN
- For dates: "main.created_at IS NOT NULL"
- Do NOT add IS NOT NULL for columns only in SELECT

MULTIPLE QUERIES STRATEGY:
- If query asks for COUNT and EXAMPLES, return 2 queries:
  * Query 1: COUNT query
  * Query 2: SELECT query with details
- Return as JSON array: ["query1", "query2"]

Natural Language Query: {natural_query}

SQL Query:
"""
