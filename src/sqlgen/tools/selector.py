import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseTool, ToolResult
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

TOOL_SELECTION_PROMPT = '''You are a tool selector for a blockchain governance query system. Analyze the user's query IN CONTEXT of the conversation history and select the most appropriate tool.

AVAILABLE TOOLS:
{tool_descriptions}

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
4. Select the SINGLE most appropriate tool from the available tools based on the FULL intent
5. Extract parameter values from the FULL combined understanding

NETWORK PARAMETER:
- If the user specifies "polkadot" or "kusama", use that value
- If the user doesn't specify a network or says "both" or "all networks", use "both" (this searches both networks)
- For list/search tools, "both" is a valid network value

PARAMETER EXTRACTION RULES:
- network: Look for "polkadot", "kusama", "DOT", "KSM" in the FULL combined query. 
  * If user says "both", "all networks", or doesn't specify a network for list/search tools, use "both" (searches both networks)
  * For single-item queries (get_proposal_by_id, etc.), default to "polkadot" if not specified
  * Valid values: "polkadot", "kusama", "both" (for list/search tools)
- proposal_index/bounty_index: Extract numeric IDs from queries like "proposal 1234", "referenda #567", "bounty 89"
- status: Map user terms to status values (tools will automatically expand these):
  * "active", "voting", "deciding" -> ["active"] (tools expand to actual statuses based on proposal type)
  * "passed", "executed", "approved" -> ["passed"] or ["executed"] (tools expand appropriately)
  * "rejected", "failed" -> ["rejected"] or ["failed"] (tools expand appropriately)
  * Use the user's exact term - tools handle the mapping to database values
- track: Map user terms to tracks: "big spender" -> "BigSpender", "medium spender" -> "MediumSpender", etc.
- time_window: Map user terms: "last week" -> "7d", "last month" -> "30d", "last 3 months" -> "90d", "this year" -> "365d"
  * For specific months like "December 2025", use "365d" (full year) or "90d" (recent period) as closest approximation
  * Note: Tools support predefined windows (7d, 30d, 90d, 180d, 365d, all) - use the closest match
- query (for search): Extract search keywords
- voter_address: Extract blockchain addresses (starts with 1, 5, or 0x)

TREASURY SPENDING QUERIES:
- Queries about "treasury spending", "total treasury", "treasury spend", "spending summary" → use "get_treasury_summary"
- For queries with specific dates/months that tools don't support, still select the tool with closest time_window - the system will fall back to LLM SQL if needed

OUTPUT FORMAT (JSON only, no explanation):
{{
    "tool": "tool_name",
    "params": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "confidence": 0.0-1.0
}}

If no tool matches, return:
{{
    "tool": null,
    "params": {{}},
    "confidence": 0.0,
    "reason": "explanation"
}}
'''


class ToolSelector:
    def __init__(self, registry: ToolRegistry, openai_client=None, gemini_client=None):
        self.registry = registry
        self.openai_client = openai_client
        self.gemini_client = gemini_client
    
    def _format_conversation_context(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        if not conversation_history:
            return "No previous conversation"
        
        parts = []
        for i, msg in enumerate(conversation_history[-5:], 1):
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"{i}. {role}: {content}")
        
        context = "\n".join(parts)
        
        if len(conversation_history) >= 2:
            last_assistant = None
            last_user = None
            for msg in reversed(conversation_history[-5:]):
                role = msg.get("role", "")
                if role == "assistant" and last_assistant is None:
                    last_assistant = msg.get("content", "")
                elif role == "user" and last_user is None:
                    last_user = msg.get("content", "")
                    break
            
            if last_assistant and last_user:
                assistant_lower = last_assistant.lower()
                if any(q in assistant_lower for q in ["?", "which", "are you", "would you", "do you want"]):
                    if len(last_user.split()) <= 5:
                        context += "\n\n⚠️ CLARIFICATION PATTERN DETECTED: The assistant asked a clarification question, and the user gave a short response. You MUST combine the user's short response with the original question from the conversation to understand the full intent."
        
        return context
    
    def _call_llm(self, prompt: str) -> str:
        if self.gemini_client:
            try:
                return self.gemini_client.get_response(prompt).strip()
            except Exception as e:
                logger.warning(f"Gemini tool selection failed, trying OpenAI: {e}")
        
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI tool selection failed: {e}")
                raise
        
        raise ValueError("No LLM client available for tool selection")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        response = response.replace('```json', '').replace('```', '').strip()
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {"tool": None, "params": {}, "confidence": 0.0, "reason": "Failed to parse LLM response"}
    
    def _apply_heuristics(self, query: str) -> Optional[Dict[str, Any]]:
        query_lower = query.lower()
        
        proposal_match = re.search(r'(?:referend(?:a|um)|proposal|ref)\s*#?\s*(\d+)', query_lower)
        if proposal_match:
            proposal_index = int(proposal_match.group(1))
            network = "kusama" if "kusama" in query_lower or "ksm" in query_lower else "polkadot"
            
            if any(word in query_lower for word in ["status", "state", "current"]):
                return {
                    "tool": "get_proposal_status",
                    "params": {"proposal_index": proposal_index, "network": network},
                    "confidence": 0.95
                }
            elif any(word in query_lower for word in ["vote", "voting", "aye", "nay"]):
                return {
                    "tool": "get_proposal_vote_stats",
                    "params": {"proposal_index": proposal_index, "network": network},
                    "confidence": 0.95
                }
            else:
                return {
                    "tool": "get_proposal_by_id",
                    "params": {"proposal_index": proposal_index, "network": network},
                    "confidence": 0.95
                }
        
        bounty_match = re.search(r'bounty\s*#?\s*(\d+)', query_lower)
        if bounty_match:
            bounty_index = int(bounty_match.group(1))
            network = "kusama" if "kusama" in query_lower or "ksm" in query_lower else "polkadot"
            return {
                "tool": "get_bounty_by_id",
                "params": {"bounty_index": bounty_index, "network": network},
                "confidence": 0.95
            }
        
        if any(word in query_lower for word in ["how many", "count", "total number"]):
            network = "kusama" if "kusama" in query_lower or "ksm" in query_lower else "polkadot"
            return {
                "tool": "count_proposals",
                "params": {"network": network},
                "confidence": 0.8
            }
        
        if any(phrase in query_lower for phrase in ["treasury summary", "spending summary", "total spent", "treasury spending", "total treasury", "treasury spend"]):
            network = "kusama" if "kusama" in query_lower or "ksm" in query_lower else ("polkadot" if "polkadot" in query_lower or "dot" in query_lower else "both")
            return {
                "tool": "get_treasury_summary",
                "params": {"network": network},
                "confidence": 0.9
            }
        
        if any(word in query_lower for word in ["top voters", "most active voters", "biggest voters"]):
            return {
                "tool": "get_top_voters",
                "params": {},
                "confidence": 0.9
            }
        
        if any(word in query_lower for word in ["network stats", "governance stats", "overview"]):
            network = "kusama" if "kusama" in query_lower or "ksm" in query_lower else "polkadot"
            return {
                "tool": "get_network_stats",
                "params": {"network": network},
                "confidence": 0.85
            }
        
        return None
    
    def _is_clarification_response(self, query: str, conversation_history: Optional[List[Dict[str, Any]]]) -> bool:
        if not conversation_history or len(conversation_history) < 2:
            return False
        
        query_lower = query.lower().strip()
        query_words = len(query_lower.split())
        
        if query_words > 5:
            return False
        
        last_assistant = None
        for msg in reversed(conversation_history[-3:]):
            role = msg.get("role", "")
            if role == "assistant":
                last_assistant = msg.get("content", "")
                break
        
        if last_assistant:
            assistant_lower = last_assistant.lower()
            clarification_indicators = ["?", "which", "are you", "would you", "do you want", "are you looking", "are you interested"]
            if any(indicator in assistant_lower for indicator in clarification_indicators):
                short_response_indicators = ["all", "both", "yes", "no", "polkadot", "kusama", "treasury", "spending", "proposals", "referenda"]
                if any(indicator in query_lower for indicator in short_response_indicators) or query_words <= 3:
                    return True
        
        return False
    
    def select_tool(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        is_clarification = self._is_clarification_response(query, conversation_history)
        heuristic_result = None
        
        if not is_clarification:
            heuristic_result = self._apply_heuristics(query)
            if heuristic_result and heuristic_result.get("confidence", 0) >= 0.9:
                logger.info(f"Tool selected by heuristics: {heuristic_result['tool']}")
                return heuristic_result
        else:
            logger.info(f"Detected clarification response, skipping heuristics and using LLM for context-aware selection")
        
        tool_descriptions = self.registry.get_tool_descriptions_prompt()
        conversation_context = self._format_conversation_context(conversation_history)
        
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=tool_descriptions,
            query=query,
            conversation_context=conversation_context
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_llm_response(response)
            
            if result.get("tool") and self.registry.get(result["tool"]):
                logger.info(f"Tool selected by LLM: {result['tool']} with confidence {result.get('confidence', 0)}")
                return result
            else:
                logger.warning(f"LLM selected unknown tool: {result.get('tool')}")
                if heuristic_result:
                    return heuristic_result
                return {"tool": None, "params": {}, "confidence": 0.0, "reason": "No matching tool found"}
                
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            if heuristic_result:
                return heuristic_result
            return {"tool": None, "params": {}, "confidence": 0.0, "reason": str(e)}
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        tool = self.registry.get(tool_name)
        if not tool:
            from .base import ToolResult
            return ToolResult(
                success=False,
                data=[],
                total_count=0,
                sql_query="",
                columns=[],
                error=f"Tool '{tool_name}' not found",
                error_type="tool_not_found"
            )
        
        return tool.execute(**params)
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[str], ToolResult]:
        selection = self.select_tool(query, conversation_history)
        
        tool_name = selection.get("tool")
        if not tool_name:
            from .base import ToolResult
            return None, ToolResult(
                success=False,
                data=[],
                total_count=0,
                sql_query="",
                columns=[],
                error=selection.get("reason", "No tool selected"),
                error_type="no_tool_match",
                metadata={"selection": selection}
            )
        
        params = selection.get("params", {})
        result = self.execute_tool(tool_name, params)
        result.metadata["selection"] = selection
        
        return tool_name, result

