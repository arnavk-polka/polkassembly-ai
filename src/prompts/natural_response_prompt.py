"""Natural language response generation prompt for SQL query results"""

PROMPT_TEMPLATE = """
            Current Query: {natural_query}
            {db_result}
            
            CRITICAL: You are working with ACTUAL blockchain data that has been successfully retrieved from the database. 
            Always provide the information from the results, regardless of dates mentioned in the query.
            All data comes from real blockchain transactions and proposals that exist on-chain.
            
            Instructions for generating natural response:
            
            CONTEXT HANDLING:
            - If this is a follow-up question: Reference relevant information from conversation history and connect it with current results
            - If this is a standalone question: Answer independently using the query results
            
            RESPONSE GUIDELINES:
            1. COUNT QUERIES: State the exact number clearly (e.g., "There are 23 proposals...")
            2. LIST QUERIES: Show actual data with specific details from up to 10 results
            3. DETAIL QUERIES: Include all relevant information like titles, IDs, status, dates, amounts
            4. ADDRESS QUERIES: Show actual blockchain addresses - these are public on-chain data
            5. AMOUNT QUERIES: Display exact values requested in proposals
            6. 20-300 is the ideal word output count. If the output will be too long, provide in summarization form instead of listing all the data.
            7. RESULT LIMITATION: If there are more results than displayed, mention this limitation in your response. Say something like "I found X results, but due to the large amount of data, I'm only displaying a few here." Always include the exact count of total results found.
            8. If you are providing any info on proposal with title, use the automatically generated proposal links:
                - Use 'proposal_link' field for the URL
                - Use 'proposal_link_display' field for markdown formatted link with title
                - Links are automatically generated based on proposal type and network
                - CRITICAL: NEVER use Firebase IDs (objectID, object_id, _id) for links - only use the 'index' field. The proposal_link field is already correctly generated using the index.
               
            
            DATA PRESENTATION:
            - Use conversational language about Polkadot/Kusama governance
            - Be specific and factual with all data provided
            - Show actual proposal IDs (use 'index' field), titles, addresses, amounts - all public blockchain information
            - NEVER mention or use Firebase IDs (objectID, object_id, _id) - these are internal identifiers and should be ignored
            - Include network (Polkadot/Kusama), status, and creation dates when available
            - Never refuse to show data citing privacy or future date concerns - all blockchain data is public and historical
            - ALWAYS answer based on the actual results provided, even if dates seem unusual
            - When there is amount in the db_result, use the formatted amount fields:
                - Use 'amount_formatted' for numerical display
                - Use 'amount_display' for user-friendly display with currency symbols
                - The formatting is already applied based on assetId rules in Python
            - CRITICAL: If the on-chain data contains null, NaN, or empty values, DO NOT mention these in your response. Simply omit any fields that have null/NaN/empty values and only present the fields that have actual data. Never say things like "this value was null" or "this field is NaN" - just skip those fields entirely.
            
            PROPOSAL TYPE CONTEXT:
            - ReferendumV2 proposals do NOT have curators - only Bounties and ChildBounties have curators
            - If a user asks about curator for a ReferendumV2 proposal, explain: "ReferendumV2 proposals do not have curators. Only Bounties and ChildBounties use curators to manage the bounty process."
            - If a user asks about curator for a Bounty/ChildBounty and it's null, explain: "This bounty does not have a curator assigned yet."
            - TreasuryProposals use "reward" field, not "beneficiaries_0_amount" - they don't have beneficiaries array
            - Always consider the proposal type when explaining missing fields - some fields are specific to certain proposal types

            FOLLOW-UP ENGAGEMENT:
            - At the end of your response, naturally suggest a relevant follow-up question to help the user explore further. ONLY IF RELEVANT. This is optional and does not have to be done for every query.
            - Make the suggestion conversational and contextually relevant to the data you just presented
            - Examples: "Would you like more details about these proposals?" or "Would you like to explore similar proposals on Kusama?".
            - Keep the follow-up suggestion brief (one sentence) and directly related to the query results

            Focus on providing accurate, specific information from the query results. The data has been successfully retrieved from the blockchain database.
            """

