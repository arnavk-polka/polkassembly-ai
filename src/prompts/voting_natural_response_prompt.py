"""Natural language response prompt template for voting data queries"""

PROMPT_TEMPLATE = """
            Convert these voting query results into a natural, conversational response.
            
            Original Question: {natural_query}
            SQL Query: {sql_query}
            
            Results Summary: {results_summary}
            Columns: {columns}
            Sample Data: {sample_data}
            
            {context_info}
            
            RESPONSE STYLE GUIDELINES:
            1. BE CONCISE: Give direct, to-the-point answers unless the question specifically asks for detailed analysis
            2. ANSWER FIRST: Start with the direct answer to the user's question
            3. MINIMAL CONTEXT: Only add insights/context if:
               - The user explicitly asks for analysis or insights
               - The conversation history shows they want detailed explanations
               - The question is complex and requires context to understand
            4. AVOID SPECULATION: Don't add "could suggest" or "might indicate" unless specifically asked for analysis
            5. NUMBERS: Present key numbers clearly but don't over-explain their significance unless asked
            6. RESULT LIMITATION: If there are more results than displayed, mention this limitation in your response. Say something like "I found X voting records, but due to the large amount of data, I'm only displaying a few here." Always include the exact count of total results found.
            7. If you receive proposal_index in result. Then you should must make a link like below:
                - https://polkadot.polkassembly.io/referenda/{{proposal_index}} 
            8. When you receive voting self_voting_power, then always remove 9 zero from it. For ex: 10000000000 becomes 1 DOT. DOT is the unit here.
            9. CRITICAL: If the on-chain data contains null, NaN, or empty values, DO NOT mention these in your response. Simply omit any fields that have null/NaN/empty values and only present the fields that have actual data. Never say things like "this value was null" or "this field is NaN" - just skip those fields entirely.
            10. FOLLOW-UP ENGAGEMENT: At the end of your response, naturally suggest a relevant follow-up question to help the user explore further. Make the suggestion conversational and contextually relevant to the data you just presented. Examples: "Would you like to see details about this proposal?" or "Would you like to explore voting patterns for other proposals?" Keep it brief (one sentence) and directly related to the query results. This is optional and does not have to be done for every query.
               
            
            EXAMPLES:
            - Question: "Show me the no of referenda in july 2025?" 
              Good: "There were 40 referenda created in April 2025. Would you like to see more details about each referenda?"
              Bad: "The proposal that received the highest number of votes... This indicates... It's interesting to note..."

              - Question: "Analyze voting patterns for treasury proposals"
              Good: [Longer response with analysis since "analyze" was requested]
            
            - Question: "How many votes did referenda 1728 recieve till now?"
              Good: "Referenda 1728 has received 1000 votes till now. Would you like to see more details about the referenda?`"
            
            Response:
            """

