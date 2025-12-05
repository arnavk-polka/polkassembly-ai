from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from .types import RouterDecision


class Router(ABC):
    @abstractmethod
    async def route(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> RouterDecision:
        pass

