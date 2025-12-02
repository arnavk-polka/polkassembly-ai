"""Summary generation prompt for chunk summarization"""

PROMPT_TEMPLATE = """Please provide a brief summary of the following Polkadot-related information using proper markdown formatting.

IMPORTANT FORMATTING RULES:
- DO NOT start with headers (##, ###)
- Start directly with the summary content
- Use **bold** for key terms, *italics* for technical concepts
- Add line breaks between bullet points if used
- Keep it concise and professional

{context}

Summary:"""

