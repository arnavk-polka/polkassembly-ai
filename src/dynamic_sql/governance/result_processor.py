import logging
from typing import List, Dict, Any

from ..utils.formatting import format_number_for_prompt

logger = logging.getLogger(__name__)

def format_amount_by_asset_id(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format amounts based on assetId rules:
    - If assetId is NaN/None: keep amount as is, it is DOT
    - If assetId is 1984: remove 6 zeros (divide by 1,000,000) USDT
    - If assetId is 1337: remove 6 zeros (divide by 1,000,000)  USDC
    - If assetId is 30: remove 3 zeros (divide by 1,000) DED
    """
    if not results:
        return results
    
    formatted_results = []
    
    for result in results:
        formatted_result = result.copy()
        
        amount_field = None
        asset_id_field = None
        
        for key in result.keys():
            if 'amount' in key.lower() and 'beneficiaries' in key.lower():
                amount_field = key
            elif 'assetid' in key.lower() and 'beneficiaries' in key.lower():
                asset_id_field = key
        
        if amount_field and asset_id_field:
            amount_value = result.get(amount_field)
            asset_id_value = result.get(asset_id_field)
            
            if amount_value is not None and str(amount_value) not in ['', 'None', 'NaN']:
                try:
                    amount_float = float(amount_value)
                    
                    if asset_id_value is not None and str(asset_id_value) not in ['', 'None', 'NaN']:
                        asset_id_int = int(float(asset_id_value))
                        
                        if asset_id_int == 1984:
                            formatted_amount = amount_float / 1_000_000
                            formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                            formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} USDT"
                            
                        elif asset_id_int == 1337:
                            formatted_amount = amount_float / 1_000_000
                            formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                            formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} USDC"
                            
                        elif asset_id_int == 30:
                            formatted_amount = amount_float / 1_000
                            formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                            formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} DED"
                            
                        else:
                            formatted_result[f"{amount_field}_formatted"] = f"{amount_float:,.2f}"
                            formatted_result[f"{amount_field}_display"] = f"{amount_float:,.2f} (Asset ID: {asset_id_int})"
                    else:
                        formatted_result[f"{amount_field}_formatted"] = f"{amount_float:,.2f}"
                        formatted_result[f"{amount_field}_display"] = f"{amount_float:,.2f} DOT"
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not format amount {amount_value} with assetId {asset_id_value}: {e}")
                    pass
        
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

