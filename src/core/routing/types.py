from enum import Enum
from dataclasses import dataclass
from typing import Optional, List


class Route(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"
    GENERIC = "generic"


@dataclass
class RouterDecision:
    route: Route
    network: Optional[str] = None
    proposal_index: Optional[int] = None
    needs: List[str] = None
    confidence: float = 0.0

    def __post_init__(self):
        if self.needs is None:
            self.needs = []

