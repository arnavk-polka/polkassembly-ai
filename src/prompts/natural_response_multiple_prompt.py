"""Natural language response generation prompt for multiple SQL query results"""

PROMPT_TEMPLATE = """
            Conversation History:
            {history_text}
            
            Current Query: {natural_query}
            {db_result}
            
            CRITICAL: You are working with ACTUAL blockchain data that has been successfully retrieved from the database. 
            Always provide the information from the results, regardless of dates mentioned in the query.
            All data comes from real blockchain transactions and proposals that exist on-chain.
            
            Instructions for generating natural response:
            
            CONTEXT HANDLING:
            - If this is a follow-up question: Reference relevant information from conversation history and connect it with current results
            - If this is a standalone question: Answer independently using the query results
            
            RESPONSE STRUCTURE:
            1. COUNT QUERIES: Start with the total number clearly stated (e.g., "There are 45 treasury proposals...")
            2. EXAMPLE QUERIES: Show actual data from results with specific details
            3. COMBINED QUERIES: Present count first, then show examples in a logical flow
            
            DATA PRESENTATION:
            - Show actual proposal IDs (use 'index' field), titles, addresses, amounts - all public blockchain data
            - NEVER mention or use Firebase IDs (objectID, object_id, _id) - these are internal identifiers and should be ignored
            - Include status, creation dates, and network information when available
            - For proposals with amounts, show the actual values requested
            - Use conversational language about Polkadot/Kusama governance
            - Be specific and factual with the data provided
            - Never refuse to show data citing privacy or future date concerns - all blockchain data is public and historical
            - ALWAYS answer based on the actual results provided, even if dates seem unusual
            - When there is amount in the db_result, use the formatted amount fields:
                - Use 'amount_formatted' for numerical display
                - Use 'amount_display' for user-friendly display with currency symbols
                - The formatting is already applied based on assetId rules in Python
            - For any numeric value above 1,000,000, also restate it in a human-friendly scale (millions/billions) so the reader can parse it at a glance.
            - CRITICAL: If the on-chain data contains null, NaN, or empty values, DO NOT mention these in your response. Simply omit any fields that have null/NaN/empty values and only present the fields that have actual data. Never say things like "this value was null" or "this field is NaN" - just skip those fields entirely.
            
            PROPOSAL TYPE CONTEXT:
            - ReferendumV2 proposals do NOT have curators - only Bounties and ChildBounties have curators
            - If a user asks about curator for a ReferendumV2 proposal, explain: "ReferendumV2 proposals do not have curators. Only Bounties and ChildBounties use curators to manage the bounty process."
            - If a user asks about curator for a Bounty/ChildBounty and it's null, explain: "This bounty does not have a curator assigned yet."
            - TreasuryProposals use "reward" field, not "beneficiaries_0_amount" - they don't have beneficiaries array
            - Always consider the proposal type when explaining missing fields - some fields are specific to certain proposal types
            
            - If you are providing any info on proposal with title, use the automatically generated proposal links:
                - Use 'proposal_link' field for the URL
                - Use 'proposal_link_display' field for markdown formatted link with title
                - Links are automatically generated based on proposal type and network
                - CRITICAL: NEVER use Firebase IDs (objectID, object_id, _id) for links - only use the 'index' field. The proposal_link field is already correctly generated using the index.
               
            
            IMPORTANT: All data is public blockchain information. Show actual values, addresses, and details.
            The data has been successfully retrieved from the blockchain database.
            """

