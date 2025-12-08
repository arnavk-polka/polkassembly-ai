from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolParam, ParamType


class ListBounties(BaseTool):
    name = "list_bounties"
    description = "List bounties with optional filters for status, network, and time period"
    category = "bounties"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query (use 'both' to search both networks)", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
            ToolParam("order_by", ParamType.ENUM, "Order results by", 
                     enum_values=["newest", "oldest", "highest_reward"], default="newest"),
        ]
    
    def build_sql(self, network: str = "polkadot", status: Optional[List[str]] = None,
                  time_window: str = "all", limit: int = 10, order_by: str = "newest", **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'Bounty\'',
            self._build_time_filter(time_window),
            self._build_status_filter(status, "Bounty"),
            '"createdat" IS NOT NULL'
        ]
        
        where_clause = self._combine_filters(*filters)
        
        order_map = {
            "newest": '"createdat" DESC',
            "oldest": '"createdat" ASC',
            "highest_reward": 'CAST("onchaininfo_reward" AS FLOAT) DESC NULLS LAST'
        }
        order = order_map.get(order_by, '"createdat" DESC')
        
        return f'''
            SELECT 
                "index", "title", "source_network",
                "onchaininfo_status", "onchaininfo_curator",
                "onchaininfo_reward", "onchaininfo_description",
                "onchaininfo_childbountiescount",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {order}
            LIMIT {limit}
        '''


class GetBountyById(BaseTool):
    name = "get_bounty_by_id"
    description = "Get detailed information about a specific bounty"
    category = "bounties"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("bounty_index", ParamType.INTEGER, "The bounty index number", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
        ]
    
    def build_sql(self, bounty_index: int, network: str = "polkadot", **kwargs) -> str:
        return f'''
            SELECT 
                "index", "title", "content", "source_network",
                "onchaininfo_status", "onchaininfo_curator",
                "onchaininfo_reward", "onchaininfo_description",
                "onchaininfo_childbountiescount",
                "onchaininfo_proposer",
                "createdat", "updatedat",
                "metrics_comments", "metrics_reactions_like", "metrics_reactions_dislike"
            FROM {self.table_name}
            WHERE "index" = {bounty_index}
            AND "source_network" = '{network}'
            AND "source_proposal_type" = 'Bounty'
            LIMIT 1
        '''


class ListChildBounties(BaseTool):
    name = "list_child_bounties"
    description = "List child bounties, optionally filtered by parent bounty"
    category = "bounties"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query (use 'both' to search both networks)", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("parent_bounty_index", ParamType.INTEGER, "Filter by parent bounty index", default=None),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
        ]
    
    def build_sql(self, network: str = "polkadot", parent_bounty_index: Optional[int] = None,
                  status: Optional[List[str]] = None, limit: int = 10, **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'ChildBounty\'',
            self._build_status_filter(status, "ChildBounty"),
            '"createdat" IS NOT NULL'
        ]
        
        if parent_bounty_index is not None:
            filters.append(f'"linkedpost_indexorhash" = \'{parent_bounty_index}\'')
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                "index", "title", "source_network",
                "onchaininfo_status", "onchaininfo_curator",
                "onchaininfo_reward", "onchaininfo_description",
                "linkedpost_indexorhash" as parent_bounty,
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "createdat" DESC
            LIMIT {limit}
        '''

