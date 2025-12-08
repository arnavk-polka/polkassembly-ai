from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolParam, ParamType


class CountProposals(BaseTool):
    name = "count_proposals"
    description = "Count proposals with optional filters for type, status, network, and time period"
    category = "aggregations"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal", 
                     enum_values=self.VALID_PROPOSAL_TYPES, default="ReferendumV2"),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
            ToolParam("track", ParamType.ARRAY, "Filter by track(s)", default=None),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("group_by", ParamType.ENUM, "Group results by", 
                     enum_values=["none", "status", "track", "network", "month"], default="none"),
        ]
    
    def build_sql(self, network: str = "polkadot", proposal_type: str = "ReferendumV2",
                  status: Optional[List[str]] = None, track: Optional[List[str]] = None,
                  time_window: str = "all", group_by: str = "none", **kwargs) -> str:
        
        filters = [
            self._build_proposal_type_filter(proposal_type),
            self._build_status_filter(status, proposal_type),
            self._build_track_filter(track),
            self._build_time_filter(time_window),
        ]
        
        if network != "both":
            filters.append(self._build_network_filter(network))
        
        where_clause = self._combine_filters(*filters)
        
        if group_by == "none":
            return f'''
                SELECT COUNT(*) as total_count
                FROM {self.table_name}
                WHERE {where_clause}
            '''
        
        group_map = {
            "status": '"onchaininfo_status"',
            "track": '"onchaininfo_origin"',
            "network": '"source_network"',
            "month": 'DATE_TRUNC(\'month\', "createdat")'
        }
        group_col = group_map.get(group_by, '"onchaininfo_status"')
        
        select_col = group_col
        if group_by == "month":
            select_col = f'{group_col}::date as month'
        else:
            select_col = f'{group_col} as {group_by}'
        
        return f'''
            SELECT 
                {select_col},
                COUNT(*) as count
            FROM {self.table_name}
            WHERE {where_clause}
            AND {group_col} IS NOT NULL
            GROUP BY {group_col}
            ORDER BY count DESC
        '''


class GetProposalsByTrack(BaseTool):
    name = "get_proposals_by_track"
    description = "Get proposal counts and summaries grouped by track/origin"
    category = "aggregations"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="30d"),
            ToolParam("include_amounts", ParamType.BOOLEAN, "Include spending amounts in summary", default=True),
        ]
    
    def build_sql(self, network: str = "polkadot", time_window: str = "30d",
                  include_amounts: bool = True, **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'ReferendumV2\'',
            self._build_time_filter(time_window),
            '"onchaininfo_origin" IS NOT NULL'
        ]
        
        where_clause = self._combine_filters(*filters)
        
        amount_cols = ""
        if include_amounts:
            amount_cols = ''',
                SUM(CASE 
                    WHEN "onchaininfo_beneficiaries_0_amount" IS NOT NULL 
                    AND "onchaininfo_beneficiaries_0_amount"::text != 'NaN'
                    THEN CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT) 
                    ELSE 0 
                END) as total_amount,
                AVG(CASE 
                    WHEN "onchaininfo_beneficiaries_0_amount" IS NOT NULL 
                    AND "onchaininfo_beneficiaries_0_amount"::text != 'NaN'
                    THEN CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT) 
                    ELSE NULL 
                END) as avg_amount'''
        
        return f'''
            SELECT 
                "onchaininfo_origin" as track,
                COUNT(*) as proposal_count,
                SUM(CASE WHEN "onchaininfo_status" IN ('Executed', 'Confirmed', 'Approved') THEN 1 ELSE 0 END) as passed_count,
                SUM(CASE WHEN "onchaininfo_status" IN ('Rejected', 'TimedOut', 'Cancelled') THEN 1 ELSE 0 END) as failed_count,
                SUM(CASE WHEN "onchaininfo_status" IN ('Deciding', 'ConfirmStarted', 'DecisionDepositPlaced') THEN 1 ELSE 0 END) as active_count
                {amount_cols}
            FROM {self.table_name}
            WHERE {where_clause}
            GROUP BY "onchaininfo_origin"
            ORDER BY proposal_count DESC
        '''


class GetNetworkStats(BaseTool):
    name = "get_network_stats"
    description = "Get high-level governance statistics for a network"
    category = "aggregations"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="30d"),
        ]
    
    def build_sql(self, network: str = "polkadot", time_window: str = "30d", **kwargs) -> str:
        
        time_filter = self._build_time_filter(time_window)
        network_filter = self._build_network_filter(network)
        
        base_where = self._combine_filters(network_filter, '"source_proposal_type" = \'ReferendumV2\'')
        time_where = self._combine_filters(base_where, time_filter) if time_filter else base_where
        
        bounty_where = self._combine_filters(network_filter, '"source_proposal_type" = \'Bounty\'')
        treasury_where = self._combine_filters(network_filter, '"source_proposal_type" = \'TreasuryProposal\'')
        
        time_suffix = f" AND {time_filter}" if time_filter else ""
        
        return f'''
            SELECT 
                '{network}' as network,
                '{time_window}' as time_window,
                (SELECT COUNT(*) FROM {self.table_name} WHERE {time_where}) as total_referenda,
                (SELECT COUNT(*) FROM {self.table_name} WHERE {time_where} AND "onchaininfo_status" IN ('Executed', 'Confirmed', 'Approved')) as passed_referenda,
                (SELECT COUNT(*) FROM {self.table_name} WHERE {time_where} AND "onchaininfo_status" IN ('Rejected', 'TimedOut', 'Cancelled', 'Killed')) as failed_referenda,
                (SELECT COUNT(*) FROM {self.table_name} WHERE {time_where} AND "onchaininfo_status" IN ('Deciding', 'ConfirmStarted', 'DecisionDepositPlaced', 'Submitted')) as active_referenda,
                (SELECT COUNT(*) FROM {self.table_name} WHERE {bounty_where}{time_suffix}) as total_bounties,
                (SELECT COUNT(*) FROM {self.table_name} WHERE {treasury_where}{time_suffix}) as treasury_proposals
        '''

