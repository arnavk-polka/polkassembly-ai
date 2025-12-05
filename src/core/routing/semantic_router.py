import logging
import os
import json
import asyncio
from typing import Optional, List, Dict, Any

from .base import Router
from .types import Route, RouterDecision

logger = logging.getLogger(__name__)


def format_conversation_history(conversation_history: Optional[List[Dict[str, Any]]]) -> str:
    if not conversation_history or len(conversation_history) == 0:
        return ""
    
    recent_messages = conversation_history[-6:]
    context_parts = []
    for msg in recent_messages:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '') or msg.get('response', '') or msg.get('answer', '')
            if content and len(content) > 5:
                role_display = role if role else 'user'
                context_parts.append(f"{role_display}: {content[:100]}")
    if context_parts:
        return f"\n\nCONVERSATION HISTORY:\n" + "\n".join(context_parts) + "\n\nUse the conversation history to understand what the user is referring to in their query."
    return ""


class SemanticRouter(Router):
    def __init__(
        self, 
        qa_generator, 
        log_step=None, 
        router_embedding_manager=None
    ):
        self.qa_generator = qa_generator
        self.log_step = log_step or (lambda step, data, level="info": None)
        self.router_embedding_manager = router_embedding_manager
        self.metrics = {
            "total_queries": 0,
            "correct_routes": 0,
            "correct_networks": 0,
            "correct_proposal_indices": 0,
            "correct_needs": 0
        }
        
        self.openai_client = None
        if hasattr(qa_generator, 'client'):
            self.openai_client = qa_generator.client
        elif hasattr(qa_generator, 'openai_client'):
            self.openai_client = qa_generator.openai_client
        
        if not self.openai_client:
            try:
                import openai
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
        
        self.router_model = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
        logger.info("Initialized SemanticRouter with dynamic few-shot retrieval")
    
    def _get_routing_prompt(self, query: str, conversation_context: str) -> str:
        from ...prompts.routing_prompt import PROMPT_TEMPLATE as routing_prompt_template
        
        prompt = routing_prompt_template.format(
            query=query,
            conversation_context=conversation_context
        )
        
        return prompt
    
    def _retrieve_similar_examples(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.router_embedding_manager:
            return []
        
        try:
            similar_chunks = self.router_embedding_manager.search_similar_chunks(
                query=query,
                n_results=n_results
            )
            
            examples = []
            for chunk in similar_chunks:
                metadata = chunk.get('metadata', {})
                question = chunk.get('content', '')
                
                if not question:
                    continue
                
                examples.append({
                    'question': question,
                    'route': metadata.get('route', 'generic'),
                    'network': metadata.get('network', ''),
                    'proposal_index': metadata.get('proposal_index', ''),
                    'needs': metadata.get('needs', '')
                })
            
            return examples
            
        except Exception as e:
            logger.warning(f"Failed to retrieve similar examples: {e}")
            return []
    
    def _format_few_shot_examples(self, examples: List[Dict[str, Any]]) -> str:
        if not examples:
            return ""
        
        formatted = "\n\nHere are similar example queries and their routing decisions:\n\n"
        for i, ex in enumerate(examples, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Query: {ex['question']}\n"
            formatted += f"Route: {ex['route']}\n"
            if ex.get('network'):
                formatted += f"Network: {ex['network']}\n"
            if ex.get('proposal_index'):
                formatted += f"Proposal Index: {ex['proposal_index']}\n"
            if ex.get('needs'):
                formatted += f"Needs: {ex['needs']}\n"
            formatted += "\n"
        
        return formatted
    
    def _build_llm_prompt(self, routing_instruction: str, few_shot_examples: str) -> str:
        return f"""{routing_instruction}{few_shot_examples}

Now analyze the query and respond with a JSON object containing:
- "route": one of "static", "dynamic", "hybrid", "generic"
- "network": "polkadot" or "kusama" if mentioned, otherwise empty string
- "proposal_index": proposal/referenda ID if mentioned, otherwise empty string
- "needs": comma-separated list of needs (e.g., "docs", "proposal_details", "analytics")

Respond with ONLY valid JSON, no other text."""
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
        
        result_text = ""
        try:
            def _make_call():
                return self.openai_client.chat.completions.create(
                    model=self.router_model,
                    messages=[
                        {"role": "system", "content": "You are a query router. Analyze queries and return routing decisions as JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
            
            response = await asyncio.to_thread(_make_call)
            result_text = response.choices[0].message.content.strip()
            return json.loads(result_text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}, response: {result_text[:200] if result_text else 'empty'}")
            return {"route": "generic", "network": "", "proposal_index": "", "needs": ""}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _calculate_confidence(self, result: Dict[str, Any], route: Route) -> float:
        confidence = 0.5
        
        route_str = (result.get('route') or "").strip().lower()
        if route_str == route.value:
            confidence += 0.3
        
        if result.get('network') and result.get('network', '').strip():
            confidence += 0.1
        
        if result.get('proposal_index') and result.get('proposal_index', '').strip():
            confidence += 0.05
        
        if result.get('needs') and result.get('needs', '').strip():
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    async def route(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> RouterDecision:
        self.log_step("semantic_router_start", {"query_preview": query[:100]})
        
        try:
            conversation_context = format_conversation_history(conversation_history)
            
            routing_prompt = self._get_routing_prompt(query, conversation_context)
            
            similar_examples = self._retrieve_similar_examples(query, n_results=5)
            
            if similar_examples:
                self.log_step("semantic_router_demos", {
                    "num_demos": len(similar_examples),
                    "demo_routes": [ex['route'] for ex in similar_examples]
                })
            
            few_shot_text = self._format_few_shot_examples(similar_examples)
            full_prompt = self._build_llm_prompt(routing_prompt, few_shot_text)
            
            result = await self._call_llm(full_prompt)
            
            route_str = (result.get('route') or "").strip().lower()
            if route_str.endswith('.'):
                route_str = route_str[:-1]
            
            try:
                route = Route(route_str)
            except ValueError:
                route = Route.GENERIC
            
            network_str = (result.get('network') or "").strip().lower()
            network = network_str if network_str in ['polkadot', 'kusama'] else None
            
            proposal_index = None
            proposal_str = (result.get('proposal_index') or "").strip()
            if proposal_str:
                try:
                    proposal_index = int(proposal_str)
                except ValueError:
                    pass
            
            needs_str = (result.get('needs') or "").strip()
            needs = [n.strip() for n in needs_str.split(",") if n.strip()] if needs_str else []
            
            if not needs:
                if route == Route.STATIC:
                    needs = ["docs"]
                elif route == Route.DYNAMIC:
                    needs = ["proposal_details"]
                elif route == Route.HYBRID:
                    needs = ["docs", "proposal_details"]
            
            confidence = self._calculate_confidence(result, route)
            
            self.metrics["total_queries"] += 1
            
            self.log_step("semantic_router_complete", {
                "route": route.value,
                "network": network,
                "proposal_index": proposal_index,
                "needs": needs,
                "confidence": confidence
            })
            
            return RouterDecision(
                route=route,
                network=network,
                proposal_index=proposal_index,
                needs=needs,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Semantic router error: {e}", exc_info=True)
            self.log_step("semantic_router_error", {"error": str(e)}, "error")
            
            return RouterDecision(
                route=Route.GENERIC,
                confidence=0.3
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        from .evaluation import RouterMetrics
        
        metrics = RouterMetrics()
        metrics.total_queries = self.metrics["total_queries"]
        metrics.correct_routes = self.metrics["correct_routes"]
        metrics.correct_networks = self.metrics["correct_networks"]
        metrics.correct_proposal_indices = self.metrics["correct_proposal_indices"]
        metrics.correct_needs = self.metrics["correct_needs"]
        
        if metrics.total_queries > 0:
            metrics.route_accuracy = metrics.correct_routes / metrics.total_queries
            metrics.network_accuracy = metrics.correct_networks / metrics.total_queries
            metrics.proposal_index_accuracy = metrics.correct_proposal_indices / metrics.total_queries
            metrics.needs_accuracy = metrics.correct_needs / metrics.total_queries
            metrics.overall_accuracy = (
                metrics.route_accuracy * 0.5 +
                metrics.network_accuracy * 0.2 +
                metrics.proposal_index_accuracy * 0.2 +
                metrics.needs_accuracy * 0.1
            )
        return metrics
    
    def update_metrics(self, predicted: RouterDecision, expected_route: str, 
                       expected_network: Optional[str] = None,
                       expected_proposal_index: Optional[int] = None,
                       expected_needs: Optional[List[str]] = None):
        from .evaluation import evaluate_router_decision
        
        results = evaluate_router_decision(
            predicted, expected_route, expected_network, 
            expected_proposal_index, expected_needs
        )
        
        self.metrics["total_queries"] += 1
        if results["route_correct"]:
            self.metrics["correct_routes"] += 1
        if results["network_correct"]:
            self.metrics["correct_networks"] += 1
        if results["proposal_index_correct"]:
            self.metrics["correct_proposal_indices"] += 1
        if results["needs_correct"]:
            self.metrics["correct_needs"] += 1

