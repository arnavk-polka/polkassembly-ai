from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolParam, ParamType


class GetVoteStats(BaseTool):
    name = "get_vote_stats"
    description = "Get voting statistics for a specific proposal from voting_data table"
    category = "voting"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", required=True),
            ToolParam("decision", ParamType.ENUM, "Filter by vote decision", 
                     enum_values=["Aye", "Nay", "Abstain", "all"], default="all"),
        ]
    
    def build_sql(self, proposal_index: int, decision: str = "all", **kwargs) -> str:
        decision_filter = ""
        if decision and decision != "all":
            decision_filter = f'AND "decision" = \'{decision}\''
        
        return f'''
            SELECT 
                "decision",
                COUNT(*) as vote_count,
                SUM(CASE WHEN "is_delegated" = true THEN 1 ELSE 0 END) as delegated_count,
                SUM(CASE WHEN "is_delegated" = false THEN 1 ELSE 0 END) as direct_count
            FROM {self.table_name}
            WHERE "proposal_index" = {proposal_index}
            {decision_filter}
            GROUP BY "decision"
            ORDER BY vote_count DESC
        '''


class GetTopVoters(BaseTool):
    name = "get_top_voters"
    description = "Get the most active voters by participation count"
    category = "voting"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("time_window", ParamType.ENUM, "Time period to analyze", 
                     enum_values=self.VALID_TIME_WINDOWS, default="30d"),
            ToolParam("decision", ParamType.ENUM, "Filter by vote decision", 
                     enum_values=["Aye", "Nay", "Abstain", "all"], default="all"),
            ToolParam("limit", ParamType.INTEGER, "Number of top voters to return", default=10),
        ]
    
    def build_sql(self, time_window: str = "30d", decision: str = "all", 
                  limit: int = 10, **kwargs) -> str:
        
        time_filter = self._build_time_filter(time_window, "created_at")
        decision_filter = ""
        if decision and decision != "all":
            decision_filter = f'AND "decision" = \'{decision}\''
        
        where_clause = "1=1"
        if time_filter:
            where_clause = time_filter
        
        return f'''
            SELECT 
                "voter",
                COUNT(*) as total_votes,
                SUM(CASE WHEN "decision" = 'Aye' THEN 1 ELSE 0 END) as aye_votes,
                SUM(CASE WHEN "decision" = 'Nay' THEN 1 ELSE 0 END) as nay_votes,
                SUM(CASE WHEN "decision" = 'Abstain' THEN 1 ELSE 0 END) as abstain_votes,
                SUM(CASE WHEN "is_delegated" = true THEN 1 ELSE 0 END) as delegated_votes
            FROM {self.table_name}
            WHERE {where_clause}
            {decision_filter}
            GROUP BY "voter"
            ORDER BY total_votes DESC
            LIMIT {limit}
        '''


class GetVoterHistory(BaseTool):
    name = "get_voter_history"
    description = "Get voting history for a specific voter address"
    category = "voting"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("voter_address", ParamType.STRING, "The voter's account address", required=True),
            ToolParam("time_window", ParamType.ENUM, "Time period to analyze", 
                     enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=20),
        ]
    
    def build_sql(self, voter_address: str, time_window: str = "all", 
                  limit: int = 20, **kwargs) -> str:
        
        escaped_address = self._escape_string(voter_address)
        time_filter = self._build_time_filter(time_window, "created_at")
        
        where_parts = [f'"voter" = \'{escaped_address}\'']
        if time_filter:
            where_parts.append(time_filter)
        
        where_clause = " AND ".join(where_parts)
        
        return f'''
            SELECT 
                "proposal_index",
                "decision",
                "is_delegated",
                "delegated_to",
                "lock_period",
                "type",
                "created_at"
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "created_at" DESC NULLS LAST
            LIMIT {limit}
        '''


class GetDelegatedVotes(BaseTool):
    name = "get_delegated_votes"
    description = "Get delegated votes for a specific proposal or delegate"
    category = "voting"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", default=None),
            ToolParam("delegate_address", ParamType.STRING, "The delegate's account address", default=None),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=20),
        ]
    
    def build_sql(self, proposal_index: Optional[int] = None, 
                  delegate_address: Optional[str] = None, limit: int = 20, **kwargs) -> str:
        
        where_parts = ['"is_delegated" = true']
        
        if proposal_index is not None:
            where_parts.append(f'"proposal_index" = {proposal_index}')
        
        if delegate_address:
            escaped_address = self._escape_string(delegate_address)
            where_parts.append(f'"delegated_to" = \'{escaped_address}\'')
        
        where_clause = " AND ".join(where_parts)
        
        return f'''
            SELECT 
                "voter",
                "proposal_index",
                "decision",
                "delegated_to",
                "lock_period",
                "created_at"
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "created_at" DESC NULLS LAST
            LIMIT {limit}
        '''


class GetVotesByConviction(BaseTool):
    name = "get_votes_by_conviction"
    description = "Get votes filtered by conviction/lock period"
    category = "voting"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", default=None),
            ToolParam("lock_period", ParamType.INTEGER, "Lock period (0-6, where 6 = 6x conviction)", default=None),
            ToolParam("min_lock_period", ParamType.INTEGER, "Minimum lock period", default=None),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=20),
        ]
    
    def build_sql(self, proposal_index: Optional[int] = None, lock_period: Optional[int] = None,
                  min_lock_period: Optional[int] = None, limit: int = 20, **kwargs) -> str:
        
        where_parts = ['"lock_period" IS NOT NULL']
        
        if proposal_index is not None:
            where_parts.append(f'"proposal_index" = {proposal_index}')
        
        if lock_period is not None:
            where_parts.append(f'"lock_period" = {lock_period}')
        elif min_lock_period is not None:
            where_parts.append(f'"lock_period" >= {min_lock_period}')
        
        where_clause = " AND ".join(where_parts)
        
        return f'''
            SELECT 
                "voter",
                "proposal_index",
                "decision",
                "lock_period",
                "is_delegated",
                "created_at"
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "lock_period" DESC, "created_at" DESC NULLS LAST
            LIMIT {limit}
        '''


class CountVoters(BaseTool):
    name = "count_voters"
    description = "Count unique voters with optional filters"
    category = "voting"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", default=None),
            ToolParam("decision", ParamType.ENUM, "Filter by vote decision", 
                     enum_values=["Aye", "Nay", "Abstain", "all"], default="all"),
            ToolParam("time_window", ParamType.ENUM, "Time period to analyze", 
                     enum_values=self.VALID_TIME_WINDOWS, default="all"),
        ]
    
    def build_sql(self, proposal_index: Optional[int] = None, decision: str = "all",
                  time_window: str = "all", **kwargs) -> str:
        
        where_parts = []
        
        if proposal_index is not None:
            where_parts.append(f'"proposal_index" = {proposal_index}')
        
        if decision and decision != "all":
            where_parts.append(f'"decision" = \'{decision}\'')
        
        time_filter = self._build_time_filter(time_window, "created_at")
        if time_filter:
            where_parts.append(time_filter)
        
        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        
        return f'''
            SELECT 
                COUNT(DISTINCT "voter") as unique_voters,
                COUNT(*) as total_votes
            FROM {self.table_name}
            WHERE {where_clause}
        '''

