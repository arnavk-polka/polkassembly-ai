import logging
from typing import Dict, List, Optional, Type, Any
from .base import BaseTool

logger = logging.getLogger(__name__)

_registry_instance: Optional['ToolRegistry'] = None


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)
        
        logger.debug(f"Registered tool: {tool.name} in category: {tool.category}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)
    
    def get_all(self) -> Dict[str, BaseTool]:
        return self._tools.copy()
    
    def get_by_category(self, category: str) -> List[BaseTool]:
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_categories(self) -> List[str]:
        return list(self._categories.keys())
    
    def get_all_schemas(self) -> List[Dict]:
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_schemas_by_category(self) -> Dict[str, List[Dict]]:
        result = {}
        for category, tool_names in self._categories.items():
            result[category] = [
                self._tools[name].get_schema() 
                for name in tool_names 
                if name in self._tools
            ]
        return result
    
    def get_tool_descriptions_prompt(self) -> str:
        lines = ["Available governance query tools:\n"]
        
        for category in sorted(self._categories.keys()):
            lines.append(f"\n## {category.upper()}")
            for tool_name in self._categories[category]:
                tool = self._tools.get(tool_name)
                if tool:
                    lines.append(f"\n### {tool.name}")
                    lines.append(f"Description: {tool.description}")
                    lines.append("Parameters:")
                    for param in tool.get_params():
                        req = "(required)" if param.required else "(optional)"
                        default = f", default: {param.default}" if param.default is not None else ""
                        enum_info = f", values: {param.enum_values}" if param.enum_values else ""
                        lines.append(f"  - {param.name} [{param.param_type.value}] {req}: {param.description}{default}{enum_info}")
        
        return "\n".join(lines)


def get_tool_registry(db_config: Dict = None, table_name: str = "governance_data", timeout: float = 30.0, table_type: str = "governance_data") -> ToolRegistry:
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
        
        if db_config:
            from .proposals import (
                GetProposalById,
                GetProposalStatus,
                ListProposals,
                SearchProposals,
                GetProposalVoteStats,
                GetProposalsByProposer,
                GetProposalFromUrl,
                ListDiscussions,
                ListTips,
                ListFellowshipReferenda
            )
            from .treasury import (
                GetTreasurySummary,
                ListTreasuryProposals,
                GetTreasuryProposalById
            )
            from .voting import (
                GetVoteStats,
                GetTopVoters,
                GetVoterHistory,
                GetDelegatedVotes,
                GetVotesByConviction,
                CountVoters
            )
            from .bounties import (
                ListBounties,
                GetBountyById,
                ListChildBounties
            )
            from .aggregations import (
                CountProposals,
                GetProposalsByTrack,
                GetNetworkStats
            )
            
            tools = [
                GetProposalById(db_config, table_name, timeout),
                GetProposalStatus(db_config, table_name, timeout),
                ListProposals(db_config, table_name, timeout),
                SearchProposals(db_config, table_name, timeout),
                GetProposalVoteStats(db_config, table_name, timeout),
                GetProposalsByProposer(db_config, table_name, timeout),
                GetProposalFromUrl(db_config, table_name, timeout),
                ListDiscussions(db_config, table_name, timeout),
                ListTips(db_config, table_name, timeout),
                ListFellowshipReferenda(db_config, table_name, timeout),
                GetTreasurySummary(db_config, table_name, timeout),
                ListTreasuryProposals(db_config, table_name, timeout),
                GetTreasuryProposalById(db_config, table_name, timeout),
                GetVoteStats(db_config, table_name, timeout),
                GetTopVoters(db_config, table_name, timeout),
                GetVoterHistory(db_config, table_name, timeout),
                GetDelegatedVotes(db_config, table_name, timeout),
                GetVotesByConviction(db_config, table_name, timeout),
                CountVoters(db_config, table_name, timeout),
                ListBounties(db_config, table_name, timeout),
                GetBountyById(db_config, table_name, timeout),
                ListChildBounties(db_config, table_name, timeout),
                CountProposals(db_config, table_name, timeout),
                GetProposalsByTrack(db_config, table_name, timeout),
                GetNetworkStats(db_config, table_name, timeout),
            ]
            
            for tool in tools:
                _registry_instance.register(tool)
    
    return _registry_instance


def reset_registry():
    global _registry_instance
    _registry_instance = None

