import os
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

def load_schema_info(schema_path_env: str) -> Dict[str, Dict[str, str]]:
    """Load schema information from schema_info.json"""
    schema_path_str = os.getenv(schema_path_env)
    if not schema_path_str:
        raise ValueError(f"{schema_path_env} environment variable is required")
    
    schema_path = Path(schema_path_str)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema info file not found at {schema_path}")
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        if 'columns' in schema_data:
            columns_data = schema_data['columns']
            logger.info(f"Loaded schema information (new format) from {schema_path}")
            return columns_data
        else:
            logger.info(f"Loaded schema information (old format) from {schema_path}")
            return schema_data
        
    except Exception as e:
        logger.error(f"Error loading schema info: {e}")
        raise

def get_table_schema_fallback(schema_info: Dict, table_name: str) -> str:
    """Fallback method to generate schema from loaded schema_info with expected values"""
    schema_parts = []
    schema_parts.append(f"Table: {table_name}")
    schema_parts.append("\nColumns:")
    
    enum_values = {
        'onchaininfo_origin': ['BigSpender', 'MediumSpender', 'SmallSpender', 'BigTipper', 'SmallTipper', 
                               'Root', 'Treasurer', 'GeneralAdmin', 'AuctionAdmin', 'LeaseAdmin', 
                               'StakingAdmin', 'FellowshipAdmin', 'ReferendumCanceller', 'ReferendumKiller',
                               'WhitelistedCaller', 'FastGeneralAdmin', 'WishForChange', 'Candidates',
                               'Members', 'Experts', 'Masters', 'GrandMasters', 'Fellows', 'SeniorFellows',
                               'SeniorExperts', 'SeniorMasters', 'Proficients'],
        'source_network': ['polkadot', 'kusama'],
        'source_proposal_type': ['ReferendumV2', 'TreasuryProposal', 'Bounty', 'ChildBounty', 'FellowshipReferendum'],
        'onchaininfo_status': ['Deciding', 'Confirming', 'Approved', 'Rejected', 'Cancelled', 'TimedOut',
                               'Killed', 'DecisionDepositPlaced', 'Submitted', 'ConfirmStarted', 'ConfirmAborted']
    }
    
    for column, info in schema_info.items():
        if isinstance(info, dict):
            data_type = info.get('type', info.get('data_type', 'unknown'))
            description = info.get('description', info.get('Description', 'No description'))
        else:
            data_type = 'unknown'
            description = str(info)
        
        if column in enum_values:
            expected_values = ', '.join([f"'{v}'" for v in enum_values[column][:10]])
            if len(enum_values[column]) > 10:
                expected_values += f", ... (and {len(enum_values[column]) - 10} more)"
            schema_parts.append(f"  - {column} ({data_type}): {description}")
            schema_parts.append(f"    Expected values: {expected_values}")
        else:
            schema_parts.append(f"  - {column} ({data_type}): {description}")
    
    return "\n".join(schema_parts)

def get_table_schema(schema_info: Dict, table_name: str, schema_path_env: str) -> str:
    """Load and format schema from JSON file with column descriptions and expected values"""
    try:
        schema_path_str = os.getenv(schema_path_env)
        if not schema_path_str:
            raise ValueError(f"{schema_path_env} environment variable not set")
        
        schema_path = Path(schema_path_str)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        logger.info(f"Loaded schema from JSON file: {schema_path}, formatting for LLM")
        return get_table_schema_fallback(schema_info, table_name)
        
    except Exception as e:
        logger.error(f"Error loading schema: {e}")
        logger.warning("Falling back to old schema generation method")
        return get_table_schema_fallback(schema_info, table_name)

