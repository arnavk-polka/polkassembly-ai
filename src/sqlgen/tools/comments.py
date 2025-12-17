from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolParam, ParamType
import os


class GetCommentsByProposal(BaseTool):
    name = "get_comments_by_proposal"
    description = "Get all comments for a specific proposal/referendum by index and proposal type"
    category = "comments"
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        super().__init__(db_config, table_name, timeout)
        self.comments_table = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal/referendum index number", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal", 
                     enum_values=self.VALID_PROPOSAL_TYPES, default="ReferendumV2"),
            ToolParam("include_deleted", ParamType.BOOLEAN, "Include deleted comments", default=False),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=100),
            ToolParam("order_by", ParamType.ENUM, "Order results by", 
                     enum_values=["newest", "oldest"], default="newest"),
        ]
    
    def build_sql(self, proposal_index: int, network: str = "polkadot", 
                  proposal_type: str = "ReferendumV2", include_deleted: bool = False,
                  limit: int = 100, order_by: str = "newest", **kwargs) -> str:
        
        deleted_filter = "" if include_deleted else 'AND c."is_deleted" = false'
        order_clause = "DESC" if order_by == "newest" else "ASC"
        
        return f'''
            SELECT 
                c.id,
                c.network,
                c.proposal_type,
                c.index_or_hash,
                c.parent_comment_id,
                c.user_id,
                c.content,
                c.created_at,
                c.updated_at,
                c.is_deleted,
                c.data_source,
                c.author_address,
                c.ai_sentiment,
                c.history,
                c.public_user,
                c.children,
                c.reactions,
                g.title as proposal_title,
                g."onchaininfo_status" as proposal_status
            FROM {self.comments_table} c
            LEFT JOIN {self.table_name} g 
                ON c.index_or_hash::text = g."index"::text 
                AND c.network = g."source_network"
                AND c.proposal_type = g."source_proposal_type"
            WHERE c.index_or_hash::text = '{proposal_index}'
            AND c.network = '{network}'
            AND c.proposal_type = '{proposal_type}'
            {deleted_filter}
            ORDER BY c.created_at {order_clause}
            LIMIT {limit}
        '''


class GetCommentById(BaseTool):
    name = "get_comment_by_id"
    description = "Get a specific comment by its ID"
    category = "comments"
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        super().__init__(db_config, table_name, timeout)
        self.comments_table = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("comment_id", ParamType.INTEGER, "The comment ID", required=True),
        ]
    
    def build_sql(self, comment_id: int, **kwargs) -> str:
        return f'''
            SELECT 
                c.id,
                c.network,
                c.proposal_type,
                c.index_or_hash,
                c.parent_comment_id,
                c.user_id,
                c.content,
                c.created_at,
                c.updated_at,
                c.is_deleted,
                c.data_source,
                c.author_address,
                c.ai_sentiment,
                c.history,
                c.public_user,
                c.children,
                c.reactions,
                g.title as proposal_title,
                g."index" as proposal_index,
                g."onchaininfo_status" as proposal_status
            FROM {self.comments_table} c
            LEFT JOIN {self.table_name} g 
                ON c.index_or_hash::text = g."index"::text 
                AND c.network = g."source_network"
                AND c.proposal_type = g."source_proposal_type"
            WHERE c.id = {comment_id}
            LIMIT 1
        '''


class ListCommentsByUser(BaseTool):
    name = "list_comments_by_user"
    description = "List comments by a specific user (by user_id or author_address)"
    category = "comments"
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        super().__init__(db_config, table_name, timeout)
        self.comments_table = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("user_id", ParamType.INTEGER, "The user ID (if available)", default=None),
            ToolParam("author_address", ParamType.STRING, "The author's blockchain address", default=None),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("include_deleted", ParamType.BOOLEAN, "Include deleted comments", default=False),
            ToolParam("time_window", ParamType.ENUM, "Time period to filter", 
                     enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=50),
        ]
    
    def build_sql(self, user_id: Optional[int] = None, author_address: Optional[str] = None,
                  network: str = "polkadot", include_deleted: bool = False,
                  time_window: str = "all", limit: int = 50, **kwargs) -> str:
        
        if not user_id and not author_address:
            raise ValueError("Either user_id or author_address must be provided")
        
        filters = []
        
        if user_id:
            filters.append(f'c.user_id = {user_id}')
        
        if author_address:
            escaped_address = self._escape_string(author_address)
            filters.append(f'LOWER(c.author_address) = LOWER(\'{escaped_address}\')')
        
        if network and network != "both":
            filters.append(f"c.network = '{network}'")
        
        if not include_deleted:
            filters.append('c.is_deleted = false')
        
        time_filter = self._build_time_filter(time_window, "c.created_at")
        if time_filter:
            filters.append(time_filter)
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                c.id,
                c.network,
                c.proposal_type,
                c.index_or_hash,
                c.parent_comment_id,
                c.user_id,
                c.content,
                c.created_at,
                c.updated_at,
                c.is_deleted,
                c.data_source,
                c.author_address,
                c.ai_sentiment,
                c.public_user,
                g.title as proposal_title,
                g."index" as proposal_index
            FROM {self.comments_table} c
            LEFT JOIN {self.table_name} g 
                ON c.index_or_hash::text = g."index"::text 
                AND c.network = g."source_network"
                AND c.proposal_type = g."source_proposal_type"
            WHERE {where_clause}
            ORDER BY c.created_at DESC
            LIMIT {limit}
        '''


class SearchComments(BaseTool):
    name = "search_comments"
    description = "Search comments by content text"
    category = "comments"
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        super().__init__(db_config, table_name, timeout)
        self.comments_table = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("query", ParamType.STRING, "Search text to find in comment content", required=True),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Filter by proposal type", 
                     enum_values=self.VALID_PROPOSAL_TYPES + ["all"], default="all"),
            ToolParam("include_deleted", ParamType.BOOLEAN, "Include deleted comments", default=False),
            ToolParam("time_window", ParamType.ENUM, "Time period to filter", 
                     enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=50),
        ]
    
    def build_sql(self, query: str, network: str = "polkadot", 
                  proposal_type: str = "all", include_deleted: bool = False,
                  time_window: str = "all", limit: int = 50, **kwargs) -> str:
        
        escaped_query = self._escape_string(query)
        
        filters = [
            f"c.content ILIKE '%{escaped_query}%'"
        ]
        
        if network and network != "both":
            filters.append(f"c.network = '{network}'")
        
        if proposal_type and proposal_type != "all":
            filters.append(f"c.proposal_type = '{proposal_type}'")
        
        if not include_deleted:
            filters.append('c.is_deleted = false')
        
        time_filter = self._build_time_filter(time_window, "c.created_at")
        if time_filter:
            filters.append(time_filter)
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                c.id,
                c.network,
                c.proposal_type,
                c.index_or_hash,
                c.parent_comment_id,
                c.user_id,
                c.content,
                c.created_at,
                c.updated_at,
                c.is_deleted,
                c.data_source,
                c.author_address,
                c.ai_sentiment,
                c.public_user,
                g.title as proposal_title,
                g."index" as proposal_index
            FROM {self.comments_table} c
            LEFT JOIN {self.table_name} g 
                ON c.index_or_hash::text = g."index"::text 
                AND c.network = g."source_network"
                AND c.proposal_type = g."source_proposal_type"
            WHERE {where_clause}
            ORDER BY c.created_at DESC
            LIMIT {limit}
        '''


class GetCommentThread(BaseTool):
    name = "get_comment_thread"
    description = "Get a comment and all its replies (children) forming a thread"
    category = "comments"
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        super().__init__(db_config, table_name, timeout)
        self.comments_table = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("comment_id", ParamType.INTEGER, "The root comment ID", required=True),
            ToolParam("include_deleted", ParamType.BOOLEAN, "Include deleted comments", default=False),
        ]
    
    def build_sql(self, comment_id: int, include_deleted: bool = False, **kwargs) -> str:
        
        deleted_filter = "" if include_deleted else 'AND c.is_deleted = false'
        
        return f'''
            WITH RECURSIVE comment_thread AS (
                SELECT 
                    c.id,
                    c.network,
                    c.proposal_type,
                    c.index_or_hash,
                    c.parent_comment_id,
                    c.user_id,
                    c.content,
                    c.created_at,
                    c.updated_at,
                    c.is_deleted,
                    c.data_source,
                    c.author_address,
                    c.ai_sentiment,
                    c.public_user,
                    c.children,
                    c.reactions,
                    0 as depth
                FROM {self.comments_table} c
                WHERE c.id = {comment_id}
                {deleted_filter}
                
                UNION ALL
                
                SELECT 
                    c.id,
                    c.network,
                    c.proposal_type,
                    c.index_or_hash,
                    c.parent_comment_id,
                    c.user_id,
                    c.content,
                    c.created_at,
                    c.updated_at,
                    c.is_deleted,
                    c.data_source,
                    c.author_address,
                    c.ai_sentiment,
                    c.public_user,
                    c.children,
                    c.reactions,
                    ct.depth + 1
                FROM {self.comments_table} c
                INNER JOIN comment_thread ct ON c.parent_comment_id = ct.id
                WHERE c.is_deleted = false
            )
            SELECT 
                ct.*,
                g.title as proposal_title,
                g."index" as proposal_index
            FROM comment_thread ct
            LEFT JOIN {self.table_name} g 
                ON ct.index_or_hash::text = g."index"::text 
                AND ct.network = g."source_network"
                AND ct.proposal_type = g."source_proposal_type"
            ORDER BY ct.depth, ct.created_at ASC
        '''


class GetCommentsStats(BaseTool):
    name = "get_comments_stats"
    description = "Get statistics about comments for a proposal or overall"
    category = "comments"
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        super().__init__(db_config, table_name, timeout)
        self.comments_table = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("proposal_index", ParamType.INTEGER, "The proposal index (optional, omit for overall stats)", default=None),
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Type of proposal", 
                     enum_values=self.VALID_PROPOSAL_TYPES + ["all"], default="all"),
            ToolParam("time_window", ParamType.ENUM, "Time period to filter", 
                     enum_values=self.VALID_TIME_WINDOWS, default="all"),
        ]
    
    def build_sql(self, proposal_index: Optional[int] = None, network: str = "polkadot",
                  proposal_type: str = "all", time_window: str = "all", **kwargs) -> str:
        
        filters = []
        
        if proposal_index:
            filters.append(f"c.index_or_hash::text = '{proposal_index}'")
        
        if network and network != "both":
            filters.append(f"c.network = '{network}'")
        
        if proposal_type and proposal_type != "all":
            filters.append(f"c.proposal_type = '{proposal_type}'")
        
        time_filter = self._build_time_filter(time_window, "c.created_at")
        if time_filter:
            filters.append(time_filter)
        
        filters.append('c.is_deleted = false')
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                COUNT(*) as total_comments,
                COUNT(DISTINCT c.user_id) as unique_commenters,
                COUNT(DISTINCT c.index_or_hash) as proposals_with_comments,
                MIN(c.created_at) as first_comment_date,
                MAX(c.created_at) as last_comment_date,
                AVG(LENGTH(c.content)) as avg_comment_length
            FROM {self.comments_table} c
            WHERE {where_clause}
        '''


class GetTopCommenters(BaseTool):
    name = "get_top_commenters"
    description = "Get users who have made the most comments"
    category = "comments"
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        super().__init__(db_config, table_name, timeout)
        self.comments_table = os.getenv("POLKASSEMBLY_COMMENTS_TABLE", "governance_comments")
    
    def get_params(self) -> List[ToolParam]:
        return [
            ToolParam("network", ParamType.ENUM, "Network to query", enum_values=["polkadot", "kusama", "both"], default="polkadot"),
            ToolParam("proposal_type", ParamType.ENUM, "Filter by proposal type", 
                     enum_values=self.VALID_PROPOSAL_TYPES + ["all"], default="all"),
            ToolParam("time_window", ParamType.ENUM, "Time period to filter", 
                     enum_values=self.VALID_TIME_WINDOWS, default="all"),
            ToolParam("limit", ParamType.INTEGER, "Maximum results to return", default=20),
        ]
    
    def build_sql(self, network: str = "polkadot", proposal_type: str = "all",
                  time_window: str = "all", limit: int = 20, **kwargs) -> str:
        
        filters = []
        
        if network and network != "both":
            filters.append(f"c.network = '{network}'")
        
        if proposal_type and proposal_type != "all":
            filters.append(f"c.proposal_type = '{proposal_type}'")
        
        time_filter = self._build_time_filter(time_window, "c.created_at")
        if time_filter:
            filters.append(time_filter)
        
        filters.append('c.is_deleted = false')
        filters.append('c.user_id IS NOT NULL')
        
        where_clause = self._combine_filters(*filters)
        
        return f'''
            SELECT 
                c.user_id,
                c.author_address,
                MAX(c.public_user->>'username') as username,
                COUNT(*) as comment_count,
                MIN(c.created_at) as first_comment_date,
                MAX(c.created_at) as last_comment_date
            FROM {self.comments_table} c
            WHERE {where_clause}
            GROUP BY c.user_id, c.author_address
            ORDER BY comment_count DESC
            LIMIT {limit}
        '''

