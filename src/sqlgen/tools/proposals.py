from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolParam, ParamType


class GetProposalById(BaseTool):
    name = "get_proposal_by_id"
    description = "Get detailed information about a specific proposal/referendum by its index number"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal", 
                     enum_values=self.VALID_PROPOSAL_TYPES, default="ReferendumV2"),
        ]
    
    def build_sql(self, proposal_index: int, network: str = "polkadot", 
                  proposal_type: str = "ReferendumV2", **kwargs) -> str:
        return f'''
            SELECT 
                "index", "title", "content", "source_network", "source_proposal_type",
                "onchaininfo_status", "onchaininfo_origin", "onchaininfo_proposer",
                "onchaininfo_beneficiaries_0_address", "onchaininfo_beneficiaries_0_amount",
                "onchaininfo_beneficiaries_0_assetid", "onchaininfo_reward",
                "onchaininfo_curator", "onchaininfo_description",
                "onchaininfo_votemetrics_aye_count", "onchaininfo_votemetrics_aye_value",
                "onchaininfo_votemetrics_nay_count", "onchaininfo_votemetrics_nay_value",
                "createdat", "updatedat",
                "metrics_comments", "metrics_reactions_like", "metrics_reactions_dislike",
                "publicuser_username", "hash"
            FROM {self.table_name}
            WHERE "index" = {proposal_index}
            AND "source_network" = '{network}'
            AND "source_proposal_type" = '{proposal_type}'
            LIMIT 1
        '''


class GetProposalStatus(BaseTool):
    name = "get_proposal_status"
    description = "Get the current status of a specific proposal/referendum"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal", 
                     enum_values=self.VALID_PROPOSAL_TYPES, default="ReferendumV2"),
        ]
    
    def build_sql(self, proposal_index: int, network: str = "polkadot", 
                  proposal_type: str = "ReferendumV2", **kwargs) -> str:
        return f'''
            SELECT 
                "index", "title", "source_network", "source_proposal_type",
                "onchaininfo_status", "onchaininfo_origin",
                "onchaininfo_votemetrics_aye_count", "onchaininfo_votemetrics_aye_value",
                "onchaininfo_votemetrics_nay_count", "onchaininfo_votemetrics_nay_value",
                "createdat", "updatedat"
            FROM {self.table_name}
            WHERE "index" = {proposal_index}
            AND "source_network" = '{network}'
            AND "source_proposal_type" = '{proposal_type}'
            LIMIT 1
        '''


class ListProposals(BaseTool):
    name = "list_proposals"
    description = "List proposals/referenda with optional filters for status, track, network, and time period"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
            ToolParam("track", ParamType.ARRAY, "Filter by track(s)/origin(s)", default=None),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal", 
                     enum_values=self.VALID_PROPOSAL_TYPES, default="ReferendumV2"),
            ToolParam("time_window", ParamType.ENUM, "Time period to filter", 
                     enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
            ToolParam("order_by", ParamType.ENUM, "Order results by", 
                     enum_values=["newest", "oldest", "most_votes"], default="newest"),
        ]
    
    def build_sql(self, network: str = "polkadot", status: Optional[List[str]] = None,
                  track: Optional[List[str]] = None, proposal_type: str = "ReferendumV2",
                  time_window: str = "all", limit: int = 10, order_by: str = "newest", **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            self._build_proposal_type_filter(proposal_type),
            self._build_status_filter(status, proposal_type),
            self._build_track_filter(track),
            self._build_time_filter(time_window),
            '"createdat" IS NOT NULL'
        ]
        
        where_clause = self._combine_filters(*filters)
        
        order_map = {
            "newest": '"createdat" DESC',
            "oldest": '"createdat" ASC',
            "most_votes": 'COALESCE("onchaininfo_votemetrics_aye_count", 0) + COALESCE("onchaininfo_votemetrics_nay_count", 0) DESC'
        }
        order = order_map.get(order_by, '"createdat" DESC')
        
        return f'''
            SELECT 
                "index", "title", "source_network", "source_proposal_type",
                "onchaininfo_status", "onchaininfo_origin", "onchaininfo_proposer",
                "onchaininfo_beneficiaries_0_amount", "onchaininfo_beneficiaries_0_assetid",
                "onchaininfo_votemetrics_aye_count", "onchaininfo_votemetrics_nay_count",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {order}
            LIMIT {limit}
        '''


class SearchProposals(BaseTool):
    name = "search_proposals"
    description = "Search proposals by title or content keywords"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("query", ParamType.STRING, "Search query for title/content", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal", 
                     enum_values=self.VALID_PROPOSAL_TYPES, default="ReferendumV2"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
        ]
    
    def build_sql(self, query: str, network: str = "polkadot", 
                  proposal_type: str = "ReferendumV2", limit: int = 10, **kwargs) -> str:
        
        escaped_query = self._escape_string(query)
        
        filters = [
            self._build_network_filter(network),
            self._build_proposal_type_filter(proposal_type),
            f'("title" ILIKE \'%{escaped_query}%\' OR "content" ILIKE \'%{escaped_query}%\')',
            '"createdat" IS NOT NULL'
        ]
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                "index", "title", "source_network", "source_proposal_type",
                "onchaininfo_status", "onchaininfo_origin",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "createdat" DESC
            LIMIT {limit}
        '''


class GetProposalVoteStats(BaseTool):
    name = "get_proposal_vote_stats"
    description = "Get voting statistics for a specific proposal"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
        ]
    
    def build_sql(self, proposal_index: int, network: str = "polkadot", **kwargs) -> str:
        return f'''
            SELECT 
                "index", "title", "source_network",
                "onchaininfo_status", "onchaininfo_origin",
                "onchaininfo_votemetrics_aye_count",
                "onchaininfo_votemetrics_aye_value",
                "onchaininfo_votemetrics_nay_count",
                "onchaininfo_votemetrics_nay_value",
                "onchaininfo_votemetrics_support_value",
                "onchaininfo_votemetrics_bareayes_value"
            FROM {self.table_name}
            WHERE "index" = {proposal_index}
            AND "source_network" = '{network}'
            AND "source_proposal_type" = 'ReferendumV2'
            LIMIT 1
        '''


class GetProposalsByProposer(BaseTool):
    name = "get_proposals_by_proposer"
    description = "Get proposals created by a specific proposer address"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposer_address", ParamType.STRING, "The proposer's blockchain address", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal (use 'all' to search all types)", 
                     enum_values=self.VALID_PROPOSAL_TYPES + ["all"], default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
        ]
    
    def build_sql(self, proposer_address: str, network: str = "polkadot", 
                  proposal_type: str = "all", limit: int = 10, **kwargs) -> str:
        escaped_address = self._escape_string(proposer_address)
        
        filters = [
            f'LOWER("onchaininfo_proposer") = LOWER(\'{escaped_address}\')',
            '"onchaininfo_proposer" IS NOT NULL',
            '"createdat" IS NOT NULL'
        ]
        
        if network and network != "both":
            filters.append(self._build_network_filter(network))
        
        if proposal_type and proposal_type != "all":
            filters.append(self._build_proposal_type_filter(proposal_type))
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                "index", "title", "source_network", "source_proposal_type",
                "onchaininfo_status", "onchaininfo_origin", "onchaininfo_proposer",
                "onchaininfo_beneficiaries_0_amount", "onchaininfo_beneficiaries_0_assetid",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "createdat" DESC
            LIMIT {limit}
        '''


class GetProposalFromUrl(BaseTool):
    name = "get_proposal_from_url"
    description = "Get proposal details from a Polkassembly URL (extracts network and index)"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal index from URL", required=True),
            ToolParam("network", ParamType.ENUM, "Network from URL domain", enum_values=["polkadot", "kusama"], required=True),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal from URL path", 
                     enum_values=["ReferendumV2", "TreasuryProposal", "Bounty", "Discussion"], default="ReferendumV2"),
        ]
    
    def build_sql(self, proposal_index: int, network: str, 
                  proposal_type: str = "ReferendumV2", **kwargs) -> str:
        return f'''
            SELECT 
                "index", "title", "content", "source_network", "source_proposal_type",
                "onchaininfo_status", "onchaininfo_origin", "onchaininfo_proposer",
                "onchaininfo_beneficiaries_0_address", "onchaininfo_beneficiaries_0_amount",
                "onchaininfo_beneficiaries_0_assetid", "onchaininfo_reward",
                "onchaininfo_curator", "onchaininfo_description",
                "onchaininfo_votemetrics_aye_count", "onchaininfo_votemetrics_aye_value",
                "onchaininfo_votemetrics_nay_count", "onchaininfo_votemetrics_nay_value",
                "createdat", "updatedat",
                "metrics_comments", "metrics_reactions_like", "metrics_reactions_dislike"
            FROM {self.table_name}
            WHERE "index" = {proposal_index}
            AND "source_network" = '{network}'
            LIMIT 1
        '''


class ListDiscussions(BaseTool):
    name = "list_discussions"
    description = "List discussion posts with optional filters"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="30d"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
        ]
    
    def build_sql(self, network: str = "polkadot", time_window: str = "30d", 
                  limit: int = 10, **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'Discussion\'',
            self._build_time_filter(time_window),
            '"createdat" IS NOT NULL'
        ]
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                "index", "title", "source_network",
                "publicuser_username",
                "metrics_comments", "metrics_reactions_like",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "createdat" DESC
            LIMIT {limit}
        '''


class ListTips(BaseTool):
    name = "list_tips"
    description = "List tip proposals with optional filters"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query (use 'both' to search both networks)", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
        ]
    
    def build_sql(self, network: str = "polkadot", status: Optional[List[str]] = None,
                  time_window: str = "all", limit: int = 10, **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'Tip\'',
            self._build_status_filter(status, "Tip"),
            self._build_time_filter(time_window),
            '"createdat" IS NOT NULL'
        ]
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                "index", "title", "source_network",
                "onchaininfo_status", "onchaininfo_reward",
                "onchaininfo_proposer",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "createdat" DESC
            LIMIT {limit}
        '''


class ListFellowshipReferenda(BaseTool):
    name = "list_fellowship_referenda"
    description = "List fellowship referenda with optional filters"
    category = "proposals"
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query (use 'both' to search both networks)", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("status", ParamType.ARRAY, "Filter by status(es)", default=None),
            ToolParam("time_window", ParamType.ENUM, "Time period", enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=10),
        ]
    
    def build_sql(self, network: str = "polkadot", status: Optional[List[str]] = None,
                  time_window: str = "all", limit: int = 10, **kwargs) -> str:
        
        filters = [
            self._build_network_filter(network),
            '"source_proposal_type" = \'FellowshipReferendum\'',
            self._build_status_filter(status, "FellowshipReferendum"),
            self._build_time_filter(time_window),
            '"createdat" IS NOT NULL'
        ]
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                "index", "title", "source_network",
                "onchaininfo_status", "onchaininfo_origin",
                "onchaininfo_proposer",
                "onchaininfo_votemetrics_aye_count", "onchaininfo_votemetrics_nay_count",
                "createdat",
                COUNT(*) OVER() as total_count
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY "createdat" DESC
            LIMIT {limit}
        '''

