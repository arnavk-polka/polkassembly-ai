from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolParam, ParamType


class GetTreasurySummary(BaseTool):
    name = "get_treasury_summary"
    description = "Get treasury spending summary with aggregations (total spent, count, averages)"
    category = "treasury"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query (use 'both' to search both networks)", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("time_window", ParamType.ENUM, "Time period for summary", 
                     enum_values=self.VALID_TIME_WINDOWS, default="30d"),
            ToolParam("track", ParamType.ARRAY, "Filter by spender track(s)", default=None),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
        ]
    
    def build_sql(self, network: str = "polkadot", time_window: str = "30d",
                  track: Optional[List[str]] = None, status: Optional[List[str]] = None, **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'ReferendumV2\'',
            '"onchaininfo_origin" IN (\'SmallSpender\', \'MediumSpender\', \'BigSpender\', \'SmallTipper\', \'BigTipper\', \'Treasurer\')',
            self._build_time_filter(time_window),
            self._build_track_filter(track),
            self._build_status_filter(status, "ReferendumV2"),
            '"onchaininfo_beneficiaries_0_amount" IS NOT NULL',
            '"onchaininfo_beneficiaries_0_amount"::text != \'NaN\''
        ]
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                "onchaininfo_origin" as track,
                COUNT(*) as proposal_count,
                SUM(CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT)) as total_amount,
                AVG(CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT)) as avg_amount,
                MIN(CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT)) as min_amount,
                MAX(CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT)) as max_amount
            FROM {self.table_name}
            WHERE {where_clause}
            GROUP BY "onchaininfo_origin"
            ORDER BY total_amount DESC
        '''


class ListTreasuryProposals(BaseTool):
    name = "list_treasury_proposals"
    description = "List treasury/spending proposals with optional filters"
    category = "treasury"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query (use 'both' to search both networks)", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("track", ParamType.ARRAY, "Filter by spender track(s)", default=None),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("min_amount", ParamType.FLOAT, "Minimum amount filter", default=None),
            ToolParam("max_amount", ParamType.FLOAT, "Maximum amount filter", default=None),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
            ToolParam("order_by", ParamType.ENUM, "Order results by", 
                     enum_values=["newest", "oldest", "highest_amount", "lowest_amount"], default="newest"),
        ]
    
    def build_sql(self, network: str = "polkadot", track: Optional[List[str]] = None,
                  status: Optional[List[str]] = None, time_window: str = "all",
                  min_amount: Optional[float] = None, max_amount: Optional[float] = None,
                  limit: int = 10, order_by: str = "newest", **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'ReferendumV2\'',
            '"onchaininfo_origin" IN (\'SmallSpender\', \'MediumSpender\', \'BigSpender\', \'SmallTipper\', \'BigTipper\', \'Treasurer\')',
            self._build_time_filter(time_window),
            self._build_track_filter(track),
            self._build_status_filter(status, "ReferendumV2"),
            '"createdat" IS NOT NULL'
        ]
        
        if min_amount is not None or max_amount is not None:
            filters.append('"onchaininfo_beneficiaries_0_amount" IS NOT NULL')
            filters.append('"onchaininfo_beneficiaries_0_amount"::text != \'NaN\'')
            if min_amount is not None:
                filters.append(f'CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT) >= {min_amount}')
            if max_amount is not None:
                filters.append(f'CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT) <= {max_amount}')
        
        where_clause = self._combine_filters(*filters)
        
        order_map = {
            "newest": '"createdat" DESC',
            "oldest": '"createdat" ASC',
            "highest_amount": 'CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT) DESC NULLS LAST',
            "lowest_amount": 'CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT) ASC NULLS LAST'
        }
        order = order_map.get(order_by, '"createdat" DESC')
        
        return f'''
            SELECT 
                "index", "title", "source_network",
                "onchaininfo_status", "onchaininfo_origin",
                "onchaininfo_proposer",
                "onchaininfo_beneficiaries_0_address",
                "onchaininfo_beneficiaries_0_amount",
                "onchaininfo_beneficiaries_0_assetid",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {order}
            LIMIT {limit}
        '''


class GetTreasuryProposalById(BaseTool):
    name = "get_treasury_proposal_by_id"
    description = "Get details of a specific treasury proposal (TreasuryProposal type)"
    category = "treasury"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The treasury proposal index number", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
        ]
    
    def build_sql(self, proposal_index: int, network: str = "polkadot", **kwargs) -> str:
        return f'''
            SELECT 
                "index", "title", "content", "source_network",
                "onchaininfo_status", "onchaininfo_proposer",
                "onchaininfo_beneficiaries_0_address",
                "onchaininfo_reward",
                "createdat", "updatedat",
                "metrics_comments", "metrics_reactions_like", "metrics_reactions_dislike"
            FROM {self.table_name}
            WHERE "index" = {proposal_index}
            AND "source_network" = '{network}'
            AND "source_proposal_type" = 'TreasuryProposal'
            LIMIT 1
        '''

