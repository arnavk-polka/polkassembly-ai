import logging
import re
from typing import List, Dict, Any, Optional

from ..utils.formatting import format_number_for_prompt

logger = logging.getLogger(__name__)


def format_bn_balance(
    value: str | int | float,
    network: str,
    asset_id: Optional[str | int] = None,
    number_after_comma: int = 2,
    with_unit: bool = True,
    with_thousand_delimiter: bool = True
) -> str:
    """
    Format blockchain balance using BN balance logic (string-based decimal splitting).
    
    Args:
        value: The raw balance value (string, int, or float)
        network: Network name ('polkadot' or 'kusama')
        asset_id: Optional asset ID for multi-asset support
        number_after_comma: Number of decimal places to show
        with_unit: Whether to include token symbol
        with_thousand_delimiter: Whether to add thousand separators
    
    Returns:
        Formatted balance string with token symbol
    """
    if value is None or str(value) in ['', 'None', 'NaN']:
        return ""
    
    value_string = str(value).split('.')[0]
    
    network_lower = str(network).lower() if network else 'polkadot'
    
    if asset_id is not None and str(asset_id) not in ['', 'None', 'NaN']:
        asset_id_int = int(float(asset_id))
        if asset_id_int == 1984:
            token_decimals = 6
            token_symbol = 'USDT'
        elif asset_id_int == 1337:
            token_decimals = 6
            token_symbol = 'USDC'
        elif asset_id_int == 30:
            token_decimals = 3
            token_symbol = 'DED'
        else:
            token_decimals = 10 if network_lower == 'polkadot' else 12
            token_symbol = 'DOT' if network_lower == 'polkadot' else 'KSM'
    else:
        token_decimals = 10 if network_lower == 'polkadot' else 12
        token_symbol = 'DOT' if network_lower == 'polkadot' else 'KSM'
    
    if len(value_string) > token_decimals:
        suffix = value_string[-token_decimals:]
        prefix = value_string[:-token_decimals]
    else:
        prefix = '0'
        suffix = value_string.zfill(token_decimals)
    
    if number_after_comma == 0 or not suffix:
        suffix = ''
    elif number_after_comma > 0:
        suffix = suffix[:number_after_comma]
    
    if with_thousand_delimiter:
        prefix = re.sub(r'\B(?=(\d{3})+(?!\d))', ',', prefix)
    
    if suffix:
        formatted_value = f"{prefix}.{suffix}"
    else:
        formatted_value = prefix
    
    if with_unit:
        return f"{formatted_value} {token_symbol}".strip()
    
    return formatted_value


def format_amount_by_asset_id(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format blockchain amounts using BN balance formatting.
    Handles both onchaininfo_beneficiaries_0_amount and onchaininfo_reward fields.
    """
    if not results:
        return results
    
    formatted_results = []
    
    for result in results:
        formatted_result = result.copy()
        
        network = None
        for key in result.keys():
            if 'network' in key.lower() and 'source' in key.lower():
                network = result.get(key)
                if network:
                    network = str(network).lower()
                break
        
        if not network:
            network = 'polkadot'
        
        amount_field = None
        asset_id_field = None
        
        for key in result.keys():
            if 'amount' in key.lower() and 'beneficiaries' in key.lower():
                amount_field = key
            elif 'assetid' in key.lower() and 'beneficiaries' in key.lower():
                asset_id_field = key
        
        if amount_field:
            amount_value = result.get(amount_field)
            asset_id_value = result.get(asset_id_field) if asset_id_field else None
            
            if amount_value is not None and str(amount_value) not in ['', 'None', 'NaN']:
                try:
                    formatted_display = format_bn_balance(amount_value, network, asset_id_value, with_unit=True)
                    formatted_value = format_bn_balance(amount_value, network, asset_id_value, with_unit=False)
                    
                    formatted_result[f"{amount_field}_formatted"] = formatted_value
                    formatted_result[f"{amount_field}_display"] = formatted_display
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not format amount {amount_value} with assetId {asset_id_value}: {e}")
        
        reward_field = None
        for key in result.keys():
            if key.lower() == 'onchaininfo_reward':
                reward_field = key
                break
        
        if reward_field:
            reward_value = result.get(reward_field)
            
            if reward_value is not None and str(reward_value) not in ['', 'None', 'NaN']:
                try:
                    formatted_display = format_bn_balance(reward_value, network, None, with_unit=True)
                    formatted_value = format_bn_balance(reward_value, network, None, with_unit=False)
                    
                    formatted_result[f"{reward_field}_formatted"] = formatted_value
                    formatted_result[f"{reward_field}_display"] = formatted_display
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not format reward {reward_value}: {e}")
        
        formatted_results.append(formatted_result)
    
    return formatted_results

def add_proposal_links(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add proposal links to results based on proposal type, network, and index/id.
    Link generation rules:
    - ReferendumV2 + polkadot: https://polkadot.polkassembly.io/referenda/{proposal_id}
    - Discussion + polkadot: https://polkadot.polkassembly.io/post/{proposal_id}
    - ReferendumV2 + kusama: https://kusama.polkassembly.io/referenda/{proposal_id}
    - Discussion + kusama: https://kusama.polkassembly.io/post/{proposal_id}
    """
    if not results:
        return results
    
    enhanced_results = []
    
    for result in results:
        enhanced_result = result.copy()
        
        proposal_id = None
        network = None
        proposal_type = None
        title = None
        
        for key in result.keys():
            key_lower = key.lower()
            if key_lower in ['objectid', 'object_id', 'firebase_id', '_id']:
                continue
            if key_lower in ['index', 'proposal_index']:
                proposal_id = result.get(key)
                break
        
        for key in result.keys():
            if 'network' in key.lower():
                network = result.get(key)
                if network:
                    network = str(network).lower()
                break
        
        if not network:
            network = 'polkadot'
            logger.debug(f"Network not found in result, defaulting to 'polkadot'")
        
        for key in result.keys():
            if 'proposal_type' in key.lower() or 'proposaltype' in key.lower() or key.lower() == 'type':
                proposal_type = result.get(key)
                break
        
        if not proposal_type:
            proposal_type = 'ReferendumV2'
            logger.debug(f"Proposal type not found in result, defaulting to 'ReferendumV2'")
        
        for key in result.keys():
            if key.lower() == 'title':
                title = result.get(key)
                break
        
        logger.debug(f"Link generation - ID: {proposal_id}, Network: {network}, Type: {proposal_type}, Title: {title}")
        
        if proposal_id is not None and network and proposal_type:
            try:
                if isinstance(proposal_id, (int, float)):
                    if isinstance(proposal_id, float) and proposal_id.is_integer():
                        proposal_id_clean = str(int(proposal_id))
                    elif isinstance(proposal_id, float):
                        proposal_id_clean = str(int(proposal_id))
                    else:
                        proposal_id_clean = str(proposal_id)
                else:
                    proposal_id_clean = str(proposal_id).strip()
                
                if proposal_id_clean and proposal_id_clean != 'None' and proposal_id_clean != 'NaN':
                    
                    link = None
                    
                    if network in ['polkadot'] and proposal_type in ['ReferendumV2']:
                        link = f"https://polkadot.polkassembly.io/referenda/{proposal_id_clean}"
                    elif network in ['polkadot'] and proposal_type in ['Discussion']:
                        link = f"https://polkadot.polkassembly.io/post/{proposal_id_clean}"
                    elif network in ['kusama'] and proposal_type in ['ReferendumV2']:
                        link = f"https://kusama.polkassembly.io/referenda/{proposal_id_clean}"
                    elif network in ['kusama'] and proposal_type in ['Discussion']:
                        link = f"https://kusama.polkassembly.io/post/{proposal_id_clean}"
                    
                    if link:
                        enhanced_result['proposal_link'] = link
                        
                        if title and str(title).strip() and str(title).strip() != 'None':
                            enhanced_result['proposal_link_display'] = f"[{title}]({link})"
                        else:
                            enhanced_result['proposal_link_display'] = f"[Proposal {proposal_id_clean}]({link})"
                        
                        logger.debug(f"Generated proposal link: {link}")
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not generate proposal link for ID {proposal_id}: {e}")
                pass
        
        enhanced_results.append(enhanced_result)
    
    return enhanced_results

