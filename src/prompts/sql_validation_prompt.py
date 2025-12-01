"""SQL result validation prompt for checking if SQL query results match the user's question"""

PROMPT_TEMPLATE = """Validate if the SQL query results match the user's question.

User Question: "{natural_query}"

Generated SQL Query:
{sql_query}

SQL Results:
- Total rows returned: {results_count}
- Sample results (first {sample_count} rows):
{sample_results}

You must return ONLY valid JSON with these exact keys:
{{
    "verdict": "good" | "partial" | "bad" | "empty",
    "reason": "short explanation (1-2 sentences)"
}}

Verdict Definitions:
- "empty": The result set is empty or meaningless for answering the question (no relevant data found).
- "good": Rows clearly match the question (right entity type, IDs, network, time range, metric as requested).
- "partial": Rows are related but clearly missing important constraints (e.g., wrong time range, wrong network, missing filters that were requested).
- "bad": Rows are clearly off-topic or conflicting with the question (wrong entity type, wrong IDs, completely unrelated data).

Return ONLY the JSON object, no other text."""

