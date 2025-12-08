from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import pandas as pd

from ..base.database import get_connection

logger = logging.getLogger(__name__)


class ToolError(Exception):
    def __init__(self, message: str, error_type: str = "validation_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


@dataclass
class ToolResult:
    success: bool
    data: List[Dict[str, Any]]
    total_count: int
    sql_query: str
    columns: List[str]
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "total_count": self.total_count,
            "sql_query": self.sql_query,
            "columns": self.columns,
            "error": self.error,
            "error_type": self.error_type,
            "metadata": self.metadata
        }


class ParamType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    ENUM = "enum"


@dataclass
class ToolParam:
    name: str
    param_type: ParamType
    description: str
    required: bool = False
    default: Any = None
    enum_values: Optional[List[str]] = None
    
    def to_schema(self) -> Dict[str, Any]:
        schema = {
            "type": self.param_type.value,
            "description": self.description
        }
        if self.enum_values:
            schema["enum"] = self.enum_values
        if self.default is not None:
            schema["default"] = self.default
        return schema


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    category: str = ""
    
    VALID_NETWORKS = ["polkadot", "kusama"]
    VALID_STATUSES = [
        "Submitted", "DecisionDepositPlaced", "ConfirmStarted", "Rejected",
        "TimedOut", "Deciding", "Executed", "Claimed", "Retracted", "Opened",
        "Closed", "Proposed", "Extended", "CuratorUnassigned", "Tabled",
        "Slashed", "Disapproved", "NotPassed", "Passed", "Added", "Awarded",
        "Started", "Active", "Approved", "Confirmed", "Cancelled", "Killed"
    ]
    VALID_PROPOSAL_TYPES = [
        "ReferendumV2", "ChildBounty", "Tip", "FellowshipReferendum", "Bounty",
        "DemocracyProposal", "CouncilMotion", "Referendum", "TechCommitteeProposal",
        "TreasuryProposal", "Discussion"
    ]
    VALID_TRACKS = [
        "Root", "WhitelistedCaller", "StakingAdmin", "Treasurer", "LeaseAdmin",
        "FellowshipAdmin", "GeneralAdmin", "AuctionAdmin", "ReferendumCanceller",
        "ReferendumKiller", "SmallTipper", "BigTipper", "SmallSpender",
        "MediumSpender", "BigSpender", "WishForChange"
    ]
    VALID_TIME_WINDOWS = ["7d", "30d", "90d", "180d", "365d", "all"]
    
    def __init__(self, db_config: Dict[str, Any], table_name: str = "governance_data", timeout: float = 30.0):
        self.db_config = db_config
        self.table_name = table_name
        self.timeout = timeout
    
    @abstractmethod
    def get_params(self) -> List[ToolParam]:
        pass
    
    @abstractmethod
    def build_sql(self, **kwargs) -> str:
        pass
    
    def validate_params(self, **kwargs) -> Dict[str, Any]:
        validated = {}
        params = {p.name: p for p in self.get_params()}
        
        for name, param in params.items():
            value = kwargs.get(name, param.default)
            
            if param.required and value is None:
                raise ToolError(f"Missing required parameter: {name}", "missing_param")
            
            if value is None:
                validated[name] = None
                continue
            
            if param.param_type == ParamType.INTEGER:
                try:
                    validated[name] = int(value)
                except (ValueError, TypeError):
                    raise ToolError(f"Parameter '{name}' must be an integer", "invalid_type")
            
            elif param.param_type == ParamType.FLOAT:
                try:
                    validated[name] = float(value)
                except (ValueError, TypeError):
                    raise ToolError(f"Parameter '{name}' must be a number", "invalid_type")
            
            elif param.param_type == ParamType.BOOLEAN:
                if isinstance(value, bool):
                    validated[name] = value
                elif isinstance(value, str):
                    validated[name] = value.lower() in ("true", "1", "yes")
                else:
                    validated[name] = bool(value)
            
            elif param.param_type == ParamType.ENUM:
                if param.enum_values and value not in param.enum_values:
                    raise ToolError(
                        f"Parameter '{name}' must be one of: {param.enum_values}",
                        "invalid_enum"
                    )
                validated[name] = value
            
            elif param.param_type == ParamType.ARRAY:
                if isinstance(value, str):
                    validated[name] = [v.strip() for v in value.split(",")]
                elif isinstance(value, list):
                    validated[name] = value
                else:
                    validated[name] = [value]
            
            else:
                validated[name] = str(value) if value is not None else None
        
        return validated
    
    def execute_sql(self, sql: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        try:
            with get_connection(self.db_config, self.timeout) as conn:
                df = pd.read_sql_query(sql, conn)
                return df.to_dict('records'), df.columns.tolist()
        except Exception as e:
            error_str = str(e).lower()
            if "timed out" in error_str or "connection" in error_str:
                raise ToolError(f"Database connection error: {e}", "connection_error")
            raise ToolError(f"SQL execution error: {e}", "sql_error")
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            validated = self.validate_params(**kwargs)
            sql = self.build_sql(**validated)
            logger.info(f"[{self.name}] Executing SQL: {sql}")
            
            results, columns = self.execute_sql(sql)
            
            total_count = len(results)
            if results and "total_count" in results[0]:
                total_count = results[0].get("total_count", len(results))
            
            return ToolResult(
                success=True,
                data=results,
                total_count=total_count,
                sql_query=sql,
                columns=columns,
                metadata={"tool": self.name, "params": validated}
            )
        
        except ToolError as e:
            logger.error(f"[{self.name}] Tool error: {e.message}")
            return ToolResult(
                success=False,
                data=[],
                total_count=0,
                sql_query="",
                columns=[],
                error=e.message,
                error_type=e.error_type,
                metadata={"tool": self.name}
            )
        
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}")
            return ToolResult(
                success=False,
                data=[],
                total_count=0,
                sql_query="",
                columns=[],
                error=str(e),
                error_type="unexpected_error",
                metadata={"tool": self.name}
            )
    
    def get_schema(self) -> Dict[str, Any]:
        params_schema = {}
        required = []
        
        for param in self.get_params():
            params_schema[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": {
                "type": "object",
                "properties": params_schema,
                "required": required
            }
        }
    
    def _escape_string(self, value: str) -> str:
        return value.replace("'", "''")
    
    def _build_network_filter(self, network: Optional[str]) -> str:
        if not network:
            return ""
        network_lower = network.lower()
        if network_lower == "both":
            return ""
        if network_lower in self.VALID_NETWORKS:
            return f'"source_network" = \'{network_lower}\''
        return ""
    
    def _build_time_filter(self, time_window: Optional[str], date_column: str = "createdat") -> str:
        if not time_window or time_window == "all":
            return ""
        
        days_map = {"7d": 7, "30d": 30, "90d": 90, "180d": 180, "365d": 365}
        if time_window in days_map:
            return f'"{date_column}" >= CURRENT_DATE - INTERVAL \'{days_map[time_window]} days\' AND "{date_column}" IS NOT NULL'
        return ""
    
    def _map_status_term(self, status: str, proposal_type: Optional[str] = None) -> List[str]:
        """Map user-friendly status terms to actual database status values."""
        status_lower = status.lower()
        
        # Active statuses vary by proposal type
        if status_lower in ["active", "voting", "deciding"]:
            if proposal_type in ["Bounty", "ChildBounty"]:
                return ["Active", "Added", "Approved", "CuratorProposed", "Extended", "Awarded"]
            else:
                # For referendums, fellowship referenda, treasury proposals, etc.
                return ["DecisionDepositPlaced", "Submitted", "Deciding", "ConfirmStarted", "ConfirmAborted"]
        
        if status_lower in ["passed", "executed", "approved", "confirmed"]:
            if proposal_type == "TreasuryProposal":
                return ["Awarded"]
            else:
                return ["Passed", "Executed", "Confirmed", "Approved"]
        
        if status_lower in ["rejected", "failed", "cancelled"]:
            return ["Rejected", "TimedOut", "Cancelled", "Killed", "ExecutionFailed"]
        
        if status_lower in ["closed", "completed"]:
            return ["Closed", "Executed", "Confirmed"]
        
        # If it's already a valid status and not a mapped term, return it as-is
        # But only if it wasn't matched by the mappings above
        if status in self.VALID_STATUSES:
            return [status]
        
        return []
    
    def _build_status_filter(self, statuses: Optional[List[str]], proposal_type: Optional[str] = None) -> str:
        if not statuses:
            return ""
        
        # Map user-friendly terms to actual statuses
        mapped_statuses = []
        for status in statuses:
            mapped = self._map_status_term(status, proposal_type)
            mapped_statuses.extend(mapped)
        
        # Remove duplicates while preserving order
        valid = list(dict.fromkeys([s for s in mapped_statuses if s in self.VALID_STATUSES]))
        
        if not valid:
            return ""
        if len(valid) == 1:
            return f'"onchaininfo_status" = \'{valid[0]}\''
        values = ", ".join(f"'{s}'" for s in valid)
        return f'"onchaininfo_status" IN ({values})'
    
    def _build_track_filter(self, tracks: Optional[List[str]]) -> str:
        if not tracks:
            return ""
        valid = [t for t in tracks if t in self.VALID_TRACKS]
        if not valid:
            return ""
        if len(valid) == 1:
            return f'"onchaininfo_origin" = \'{valid[0]}\''
        values = ", ".join(f"'{t}'" for t in valid)
        return f'"onchaininfo_origin" IN ({values})'
    
    def _build_proposal_type_filter(self, proposal_type: Optional[str]) -> str:
        if not proposal_type or proposal_type == "all":
            return ""
        if proposal_type in self.VALID_PROPOSAL_TYPES:
            return f'"source_proposal_type" = \'{proposal_type}\''
        return '"source_proposal_type" = \'ReferendumV2\''
    
    def _combine_filters(self, *filters: str) -> str:
        valid_filters = [f for f in filters if f]
        if not valid_filters:
            return "1=1"
        return " AND ".join(valid_filters)

