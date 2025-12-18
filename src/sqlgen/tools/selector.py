import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseTool, ToolResult
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


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
                    model="gpt-4o-mini",
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
                parsed = json.loads(json_match.group())
                
                if "tools" in parsed:
                    return parsed
                elif "tool" in parsed:
                    return {"tools": [parsed]}
                else:
                    return {"tools": [], "reason": "Invalid response format"}
            except json.JSONDecodeError:
                pass
        
        return {"tools": [], "reason": "Failed to parse LLM response"}
    
    
    def select_tools(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        from ...prompts.tool_selection_prompt import PROMPT_TEMPLATE
        
        tool_descriptions = self.registry.get_tool_descriptions_prompt()
        conversation_context = self._format_conversation_context(conversation_history)
        
        prompt = PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
            query=query,
            conversation_context=conversation_context
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_llm_response(response)
            
            tools = result.get("tools", [])
            if not tools:
                logger.warning(f"LLM selected no tools: {result.get('reason', 'Unknown reason')}")
                return result
            
            validated_tools = []
            for tool_selection in tools:
                tool_name = tool_selection.get("tool")
                if tool_name and self.registry.get(tool_name):
                    validated_tools.append(tool_selection)
                    logger.info(f"Tool selected by LLM: {tool_name} with confidence {tool_selection.get('confidence', 0)}")
                else:
                    logger.warning(f"LLM selected unknown tool: {tool_name}")
            
            if validated_tools:
                result["tools"] = validated_tools
                return result
            else:
                return {"tools": [], "reason": "No valid tools found"}
                
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return {"tools": [], "reason": str(e)}
    
    def select_tool(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        result = self.select_tools(query, conversation_history)
        tools = result.get("tools", [])
        if tools:
            return tools[0]
        return {"tool": None, "params": {}, "confidence": 0.0, "reason": result.get("reason", "No tools selected")}
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        tool = self.registry.get(tool_name)
        if not tool:
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
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[List[str]], ToolResult]:
        selection = self.select_tools(query, conversation_history)
        
        tools = selection.get("tools", [])
        if not tools:
            return None, ToolResult(
                success=False,
                data=[],
                total_count=0,
                sql_query="",
                columns=[],
                error=selection.get("reason", "No tools selected"),
                error_type="no_tool_match",
                metadata={"selection": selection}
            )
        
        if len(tools) == 1:
            tool_selection = tools[0]
            tool_name = tool_selection.get("tool")
            params = tool_selection.get("params", {})
            result = self.execute_tool(tool_name, params)
            result.metadata["selection"] = selection
            return [tool_name], result
        
        all_results = []
        all_sql_queries = []
        all_columns = set()
        tool_names = []
        has_error = False
        error_messages = []
        
        for tool_selection in tools:
            tool_name = tool_selection.get("tool")
            params = tool_selection.get("params", {})
            tool_names.append(tool_name)
            
            try:
                tool_result = self.execute_tool(tool_name, params)
                if tool_result.success:
                    all_results.extend(tool_result.data)
                    if tool_result.sql_query:
                        all_sql_queries.append(tool_result.sql_query)
                    all_columns.update(tool_result.columns)
                else:
                    has_error = True
                    error_messages.append(f"{tool_name}: {tool_result.error}")
            except Exception as e:
                has_error = True
                error_messages.append(f"{tool_name}: {str(e)}")
        
        combined_result = ToolResult(
            success=not has_error or len(all_results) > 0,
            data=all_results,
            total_count=len(all_results),
            sql_query=all_sql_queries[0] if all_sql_queries else "",
            columns=list(all_columns),
            error="; ".join(error_messages) if error_messages else None,
            error_type="multi_tool_error" if has_error and len(all_results) == 0 else None,
            metadata={
                "selection": selection,
                "tools_used": tool_names,
                "individual_results": len(tools),
                "sql_queries": all_sql_queries
            }
        )
        
        return tool_names, combined_result

