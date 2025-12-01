"""
Natural Language Query to SQL Converter and Executor

This module converts natural language queries to SQL, executes them against PostgreSQL,
and returns natural language responses using OpenAI API.
"""

import os
import json
import logging
import psycopg2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from contextlib import contextmanager
import pandas as pd

# Load environment variables
load_dotenv()

# Add color support for logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    # Color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    def format(self, record):
        # Add color based on log level
        if record.levelno == logging.INFO:
            record.msg = f"{self.GREEN}{record.msg}{self.END}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{self.YELLOW}{record.msg}{self.END}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.END}"
        elif record.levelno == logging.DEBUG:
            record.msg = f"{self.CYAN}{record.msg}{self.END}"
        
        return super().format(record)

# Set up colored logging
def setup_colored_logging():
    """Set up colored logging for model calls"""
    logger = logging.getLogger()
    
    # Create a colored formatter
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add handler if not already present
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    return logger

# Helper function to print model usage in green
def print_model_usage(model_name: str, purpose: str):
    """Print model usage information in green color"""
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    END = '\033[0m'
    print(f"{GREEN}{BOLD}🤖 Using {model_name} for {purpose}{END}")

# Get Gemini model name and timeout from environment
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-pro')
GEMINI_MODEL_SQL = os.getenv('GEMINI_MODEL_SQL', 'gemini-2.5-pro')
GEMINI_TIMEOUT = float(os.getenv('GEMINI_TIMEOUT', '30'))
GEMINI_SQL_TIMEOUT = float(os.getenv('GEMINI_SQL_TIMEOUT', '120'))  # Longer timeout for SQL generation

# Add tiktoken for token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not available, using approximate token counting")

# Add Gemini client
try:
    from ..core.gemini_client import GeminiClient
except ImportError:
    GeminiClient = None
    print("Warning: Gemini client not available")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Query2SQL:
    def __init__(self, embedding_manager=None):
        """Initialize the Query2SQL converter with database and OpenAI connections
        
        Args:
            embedding_manager: Optional EmbeddingManager instance for dynamic Chroma collection
        """
        
        # Store embedding manager for contextual SQL generation
        self.embedding_manager = embedding_manager
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DATABASE'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Validate database configuration
        required_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # SQL Model configuration
        self.sql_model = os.getenv('SQL_MODEL', 'chatgpt').lower()
        logger.info(f"SQL Model configured: {self.sql_model}")
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = None
        
        # Gemini configuration
        self.gemini_client = None
        
        # Timeout configuration
        self.api_timeout = float(os.getenv('API_TIMEOUT', '10'))  # Default 10 seconds
        
        # Initialize clients based on SQL_MODEL preference
        if self.sql_model == 'chatgpt':
            # Initialize OpenAI as primary
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required when SQL_MODEL=chatgpt")
            self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.api_timeout)
            logger.info("OpenAI client initialized as primary SQL model")
            
            # Initialize Gemini as fallback
            if GeminiClient is not None:
                try:
                    self.gemini_client = GeminiClient(model_name=GEMINI_MODEL_SQL, timeout=GEMINI_SQL_TIMEOUT)
                    logger.info(f"Gemini {GEMINI_MODEL_SQL} initialized as fallback")
                except Exception as e:
                    logger.warning(f"Gemini fallback initialization failed: {e}")
        else:
            # Initialize Gemini as primary (default for non-chatgpt values)
            if GeminiClient is not None:
                try:
                    self.gemini_client = GeminiClient(model_name=GEMINI_MODEL_SQL, timeout=GEMINI_SQL_TIMEOUT)
                    logger.info(f"Gemini {GEMINI_MODEL_SQL} initialized as primary SQL model")
                except Exception as e:
                    logger.error(f"Gemini 2.5 Pro initialization failed: {e}")
                    raise ValueError("Failed to initialize Gemini 2.5 Pro. Please check GEMINI_API_KEY.")
            else:
                raise ValueError("Gemini client not available. Please install required dependencies.")
            
            # Initialize OpenAI as fallback
            if self.openai_api_key:
                self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.api_timeout)
                logger.info("OpenAI client initialized as fallback")
            else:
                logger.warning("OpenAI API key not provided, no fallback available")
        
        self.table_name = os.getenv('POSTGRES_TABLE_NAME', 'governance_data')
        
        # Load schema information
        self.schema_info = self._load_schema_info()
        self.table_schema = self._get_table_schema()
        
        logger.info(f"Initialized Query2SQL for table: {self.table_name}")
        logger.info(f"Loaded schema for {len(self.schema_info)} columns")
    
    def _load_schema_info(self) -> Dict[str, Dict[str, str]]:
        """Load schema information from schema_info.json"""
        schema_path_str = os.getenv('POSTGRES_SCHEMA_PATH')
        if not schema_path_str:
            raise ValueError("POSTGRES_SCHEMA_PATH environment variable is required")
        
        schema_path = Path(schema_path_str)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema info file not found at {schema_path}")
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Check if the schema has a 'columns' key (new format) or is direct (old format)
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
    
    def format_amount_by_asset_id(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            
            # Check if this result has amount and assetId fields
            amount_field = None
            asset_id_field = None
            
            # Find amount and assetId fields (they might have different names)
            for key in result.keys():
                if 'amount' in key.lower() and 'beneficiaries' in key.lower():
                    amount_field = key
                elif 'assetid' in key.lower() and 'beneficiaries' in key.lower():
                    asset_id_field = key
            
            # If we found both fields, apply formatting
            if amount_field and asset_id_field:
                amount_value = result.get(amount_field)
                asset_id_value = result.get(asset_id_field)
                
                # Only format if amount exists and is not None
                if amount_value is not None and str(amount_value) not in ['', 'None', 'NaN']:
                    try:
                        # Convert amount to float for processing
                        amount_float = float(amount_value)
                        
                        # Apply formatting based on assetId
                        if asset_id_value is not None and str(asset_id_value) not in ['', 'None', 'NaN']:
                            asset_id_int = int(float(asset_id_value))
                            
                            if asset_id_int == 1984:
                                # Remove 6 zeros (divide by 1,000,000) - USDT
                                formatted_amount = amount_float / 1_000_000
                                formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} USDT"
                                
                            elif asset_id_int == 1337:
                                # Remove 6 zeros (divide by 1,000,000) - USDC
                                formatted_amount = amount_float / 1_000_000
                                formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} USDC"
                                
                            elif asset_id_int == 30:
                                # Remove 3 zeros (divide by 1,000) - DED
                                formatted_amount = amount_float / 1_000
                                formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} DED"
                                
                            else:
                                # Unknown assetId, keep original
                                formatted_result[f"{amount_field}_formatted"] = f"{amount_float:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{amount_float:,.2f} (Asset ID: {asset_id_int})"
                        else:
                            # AssetId is NaN/None, keep original amount - DOT
                            formatted_result[f"{amount_field}_formatted"] = f"{amount_float:,.2f}"
                            formatted_result[f"{amount_field}_display"] = f"{amount_float:,.2f} DOT"
                            
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not format amount {amount_value} with assetId {asset_id_value}: {e}")
                        # Keep original values if formatting fails
                        pass
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    @staticmethod
    def _format_number_for_prompt(value: Any) -> str:
        """Format numeric values for readability while keeping the exact figure."""
        try:
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, (int, float)):
                abs_val = abs(value)
                if abs_val >= 1_000_000_000_000:
                    return f"{value:,.0f} ({value/1_000_000_000_000:.2f}T)"
                if abs_val >= 1_000_000_000:
                    return f"{value:,.0f} ({value/1_000_000_000:.2f}B)"
                if abs_val >= 1_000_000:
                    return f"{value:,.0f} ({value/1_000_000:.2f}M)"
                if 0 < abs_val < 0.001:
                    return f"{value:.6f}"
                if isinstance(value, float):
                    return f"{value:,.4f}".rstrip('0').rstrip('.')
                return f"{value:,}"
        except Exception:
            pass
        return str(value)
    
    def add_proposal_links(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            
            # Find relevant fields for link generation
            proposal_id = None
            network = None
            proposal_type = None
            title = None
            
            # Extract proposal ID - ONLY use index fields, never Firebase IDs
            # Priority: index > proposal_index (explicitly exclude 'id' and 'proposal_id' to avoid Firebase IDs)
            for key in result.keys():
                key_lower = key.lower()
                # Skip Firebase IDs and other non-index identifiers
                if key_lower in ['objectid', 'object_id', 'firebase_id', '_id']:
                    continue
                if key_lower in ['index', 'proposal_index']:
                    proposal_id = result.get(key)
                    break
            
            # Extract network (try multiple field names)
            for key in result.keys():
                if 'network' in key.lower():
                    network = result.get(key)
                    if network:
                        network = str(network).lower()
                    break
            
            # If network not found in results, default to polkadot (most common)
            if not network:
                network = 'polkadot'
                logger.debug(f"Network not found in result, defaulting to 'polkadot'")
            
            # Extract proposal type (try multiple field names)
            for key in result.keys():
                if 'proposal_type' in key.lower() or 'proposaltype' in key.lower() or key.lower() == 'type':
                    proposal_type = result.get(key)
                    break
            
            # If proposal type not found in results, default to ReferendumV2 (most common)
            if not proposal_type:
                proposal_type = 'ReferendumV2'
                logger.debug(f"Proposal type not found in result, defaulting to 'ReferendumV2'")
            
            # Extract title
            for key in result.keys():
                if key.lower() == 'title':
                    title = result.get(key)
                    break
            
            # Debug logging
            logger.debug(f"Link generation - ID: {proposal_id}, Network: {network}, Type: {proposal_type}, Title: {title}")
            
            # Generate link if we have the required information
            if proposal_id is not None and network and proposal_type:
                try:
                    # Clean and validate proposal_id - convert to int if numeric to avoid float formatting
                    if isinstance(proposal_id, (int, float)):
                        # Convert to int if it's a whole number (e.g., 1793.0 -> 1793)
                        if isinstance(proposal_id, float) and proposal_id.is_integer():
                            proposal_id_clean = str(int(proposal_id))
                        elif isinstance(proposal_id, float):
                            proposal_id_clean = str(int(proposal_id))
                        else:
                            proposal_id_clean = str(proposal_id)
                    else:
                        proposal_id_clean = str(proposal_id).strip()
                    
                    if proposal_id_clean and proposal_id_clean != 'None' and proposal_id_clean != 'NaN':
                        
                        # Generate link based on type and network
                        link = None
                        
                        if network in ['polkadot'] and proposal_type in ['ReferendumV2']:
                            link = f"https://polkadot.polkassembly.io/referenda/{proposal_id_clean}"
                        elif network in ['polkadot'] and proposal_type in ['Discussion']:
                            link = f"https://polkadot.polkassembly.io/post/{proposal_id_clean}"
                        elif network in ['kusama'] and proposal_type in ['ReferendumV2']:
                            link = f"https://kusama.polkassembly.io/referenda/{proposal_id_clean}"
                        elif network in ['kusama'] and proposal_type in ['Discussion']:
                            link = f"https://kusama.polkassembly.io/post/{proposal_id_clean}"
                        
                        # Add link to result if generated
                        if link:
                            enhanced_result['proposal_link'] = link
                            
                            # Also create a display version with title if available
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
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken or approximate counting
        """
        if tiktoken:
            try:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Error with tiktoken: {e}, using approximate counting")
        
        # Approximate token counting (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def trim_prompt_to_fit_tokens(self, system_prompt: str, max_tokens: int = 20000, completion_tokens: int = 1000, buffer_tokens: int = 500) -> str:
        """
        Trim the system prompt to fit within token limits
        
        Args:
            system_prompt: The full system prompt
            max_tokens: Maximum tokens allowed by the model (default: 20000)
            completion_tokens: Tokens reserved for completion (default: 1000)
            buffer_tokens: Safety buffer tokens (default: 500)
        
        Returns:
            Trimmed system prompt that fits within limits
        """
        # Count current tokens
        current_tokens = self.count_tokens(system_prompt)
        
        # If within limits, return as-is
        if current_tokens <= max_tokens:
            logger.info(f"Token analysis - Current: {current_tokens}, Max: {max_tokens} - No trimming needed")
            return system_prompt
        
        # Calculate target tokens (95% of max_tokens)
        target_tokens = int(max_tokens * 0.95)
        
        logger.info(f"Token analysis - Current: {current_tokens}, Max: {max_tokens}, Target: {target_tokens} - Trimming needed")
        
        # Calculate target length based on token ratio
        target_length = int(len(system_prompt) * (target_tokens / current_tokens))
        
        # Find good places to cut (preserve important sections)
        lines = system_prompt.split('\n')
        
        # Priority sections to keep (in order of importance)
        essential_sections = [
            'COLUMN SELECTION STRATEGY:',
            'EXAMPLE QUERIES:',
            'CRITICAL NULL VALUE HANDLING:',
            'NaN VALUE HANDLING:',
            'Very very Important Rule:'
        ]
        
        # Build trimmed prompt by keeping essential sections
        trimmed_lines = []
        current_length = 0
        
        # First pass: add essential sections
        for line in lines:
            if any(section in line for section in essential_sections):
                # Found essential section, add it and some context
                section_start = lines.index(line)
                # Add this section and next 10 lines (or until next section)
                for i in range(section_start, min(section_start + 10, len(lines))):
                    if lines[i] not in trimmed_lines:
                        test_length = current_length + len(lines[i]) + 1
                        if test_length < target_length:
                            trimmed_lines.append(lines[i])
                            current_length = test_length
                        else:
                            break
        
        # Second pass: add other important lines if space allows
        for line in lines:
            if line not in trimmed_lines and current_length + len(line) + 1 < target_length:
                # Skip very long lines (likely examples that can be shortened)
                if len(line) < 200:
                    trimmed_lines.append(line)
                    current_length += len(line) + 1
        
        trimmed_prompt = '\n'.join(trimmed_lines)
        
        # Final token check
        final_tokens = self.count_tokens(trimmed_prompt)
        logger.info(f"Prompt trimmed: {current_tokens} -> {final_tokens} tokens (target: {target_tokens})")
        
        return trimmed_prompt
    
    def _generate_sql_with_model_deterministic(self, system_prompt: str) -> str:
        """Generate SQL using deterministic settings (temperature=0, top_p=1, seed=42)"""
        try:
            from ..prompts.sql_generation_system_prompt import PROMPT as sql_generation_system_prompt
            if self.sql_model == 'chatgpt' and self.openai_client:
                print_model_usage("GPT-4", "SQL generation (governance data, deterministic)")
                logger.debug("Using ChatGPT for deterministic SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": sql_generation_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            elif self.gemini_client:
                print_model_usage(f"{GEMINI_MODEL_SQL}", "SQL generation (governance data, deterministic)")
                logger.debug("Using Gemini for deterministic SQL generation")
                from ..prompts.sql_generation_system_prompt import PROMPT as sql_generation_system_prompt
                full_prompt = f"""{sql_generation_system_prompt}

{system_prompt}"""
                try:
                    response = self.gemini_client.get_response(full_prompt)
                    return response.strip()
                except Exception as e:
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                        logger.warning(f"Gemini SQL model overloaded (503 error), falling back to general Gemini model: {e}")
                        try:
                            print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (governance data)")
                            fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_SQL_TIMEOUT)
                            response = fallback_client.get_response(full_prompt)
                            logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for SQL generation")
                            return response.strip()
                        except Exception as fallback_error:
                            logger.error(f"Fallback Gemini model also failed: {fallback_error}")
                            raise e
                    else:
                        raise e
            elif self.openai_client:
                print_model_usage("GPT-4", "SQL generation fallback (governance data, deterministic)")
                logger.debug("Using ChatGPT as fallback for deterministic SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": sql_generation_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("No SQL generation model available")
        except Exception as e:
            logger.error(f"Error in deterministic SQL model, trying fallback: {e}")
            if self.sql_model != 'chatgpt' and self.openai_client:
                logger.info("Falling back to ChatGPT for deterministic SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": sql_generation_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            elif self.sql_model == 'chatgpt' and self.gemini_client:
                logger.info("Falling back to Gemini for deterministic SQL generation")
                from ..prompts.sql_generation_system_prompt import PROMPT as sql_generation_system_prompt
                full_prompt = f"""{sql_generation_system_prompt}

{system_prompt}"""
                response = self.gemini_client.get_response(full_prompt)
                return response.strip()
            else:
                raise e

    def _generate_sql_with_model(self, system_prompt: str, user_message: str = None) -> str:
        """Generate SQL using the configured model (Gemini or ChatGPT) with fallback for 503 errors"""
        try:
            if self.sql_model == 'chatgpt' and self.openai_client:
                # Use ChatGPT as primary
                print_model_usage("GPT-4", "SQL generation (governance data)")
                logger.debug("Using ChatGPT for SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": sql_generation_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
                
            elif self.gemini_client:
                # Use Gemini as primary (or fallback for ChatGPT)
                print_model_usage(f"{GEMINI_MODEL_SQL}", "SQL generation (governance data)")
                logger.debug("Using Gemini for SQL generation")
                
                # Construct the full prompt for Gemini
                from ..prompts.sql_generation_system_prompt import PROMPT as sql_generation_system_prompt
                full_prompt = f"""{sql_generation_system_prompt}

{system_prompt}"""
                
                try:
                    response = self.gemini_client.get_response(full_prompt)
                    return response.strip()
                except Exception as e:
                    # Check if it's a 503 error (model overloaded)
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                        logger.warning(f"Gemini SQL model overloaded (503 error), falling back to general Gemini model: {e}")
                        # Create a fallback Gemini client with the general model
                        try:
                            print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (governance data)")
                            fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_SQL_TIMEOUT)
                            response = fallback_client.get_response(full_prompt)
                            logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for SQL generation")
                            return response.strip()
                        except Exception as fallback_error:
                            logger.error(f"Fallback Gemini model also failed: {fallback_error}")
                            raise e  # Re-raise original error if fallback fails
                    else:
                        # Re-raise non-503 errors
                        raise e
                
            elif self.openai_client:
                # Fallback to OpenAI if Gemini fails
                print_model_usage("GPT-4", "SQL generation fallback (governance data)")
                logger.debug("Using ChatGPT as fallback for SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": sql_generation_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("No SQL generation model available")
                
        except Exception as e:
            logger.error(f"Error in primary SQL model, trying fallback: {e}")
            
            # Try fallback model
            if self.sql_model != 'chatgpt' and self.openai_client:
                # Gemini failed, try ChatGPT
                logger.info("Falling back to ChatGPT for SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": sql_generation_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            elif self.sql_model == 'chatgpt' and self.gemini_client:
                # ChatGPT failed, try Gemini
                logger.info("Falling back to Gemini 2.5 Pro for SQL generation")
                from ..prompts.sql_generation_system_prompt import PROMPT as sql_generation_system_prompt
                full_prompt = f"""{sql_generation_system_prompt}

{system_prompt}"""
                response = self.gemini_client.get_response(full_prompt)
                return response.strip()
            else:
                raise e
    
    def _get_table_schema(self) -> str:
        """Load and format schema from JSON file with column descriptions and expected values"""
        try:
            schema_path_str = os.getenv('POSTGRES_SCHEMA_PATH')
            if not schema_path_str:
                raise ValueError("POSTGRES_SCHEMA_PATH environment variable not set")
            
            schema_path = Path(schema_path_str)
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
            # Load the entire JSON file
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
            
            # Use the formatted schema method which includes descriptions
            # This provides better context to the LLM about column types and expected values
            logger.info(f"Loaded schema from JSON file: {schema_path}, formatting for LLM")
            return self._get_table_schema_fallback()
            
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            # Fallback to old method if loading fails
            logger.warning("Falling back to old schema generation method")
            return self._get_table_schema_fallback()
    
    def _get_table_schema_fallback(self) -> str:
        """Fallback method to generate schema from loaded schema_info with expected values"""
        schema_parts = []
        schema_parts.append(f"Table: {self.table_name}")
        schema_parts.append("\nColumns:")
        
        # Known enum columns with expected values
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
        
        for column, info in self.schema_info.items():
            # Handle both old and new schema formats
            if isinstance(info, dict):
                # New format: {"type": "TEXT", "description": "..."}
                data_type = info.get('type', info.get('data_type', 'unknown'))
                description = info.get('description', info.get('Description', 'No description'))
            else:
                # Fallback for unexpected format
                data_type = 'unknown'
                description = str(info)
            
            # Add expected values for enum-like columns
            if column in enum_values:
                expected_values = ', '.join([f"'{v}'" for v in enum_values[column][:10]])  # Show first 10
                if len(enum_values[column]) > 10:
                    expected_values += f", ... (and {len(enum_values[column]) - 10} more)"
                schema_parts.append(f"  - {column} ({data_type}): {description}")
                schema_parts.append(f"    Expected values: {expected_values}")
            else:
                schema_parts.append(f"  - {column} ({data_type}): {description}")
        
        return "\n".join(schema_parts)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with timeout"""
        conn = None
        try:
            # Add timeout to database connection
            db_config_with_timeout = self.db_config.copy()
            db_config_with_timeout['connect_timeout'] = int(self.api_timeout)
            
            conn = psycopg2.connect(**db_config_with_timeout)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def generate_sql_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Convert natural language query to SQL using OpenAI - can return multiple queries with error correction"""

        # print("Schema of table is", self.table_schema)

        try:
            # Generate initial SQL queries
            sql_queries = self._generate_sql_with_retry(natural_query, conversation_history)
            return sql_queries
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            raise

    
    def execute_sql_queries(self, sql_queries: List[str]) -> List[Tuple[List[Dict[str, Any]], List[str]]]:
        """Execute multiple SQL queries against PostgreSQL database"""
        all_results = []
        
        try:
            with self.get_connection() as conn:
                for i, sql_query in enumerate(sql_queries):
                    logger.info(f"Executing query {i+1}/{len(sql_queries)}: {sql_query}")
                    
                    # Use pandas for easier data handling
                    df = pd.read_sql_query(sql_query, conn)
                    
                    # Convert DataFrame to list of dictionaries
                    results = df.to_dict('records')
                    columns = df.columns.tolist()
                    
                    all_results.append((results, columns))
                    logger.info(f"Query {i+1} executed successfully. Retrieved {len(results)} rows")
                
                return all_results
                
        except Exception as e:
            logger.error(f"Error executing SQL queries: {e}")
            logger.error(f"Queries: {sql_queries}")
            raise
    
    def execute_sql_query(self, sql_query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute single SQL query - wrapper for backwards compatibility"""
        results = self.execute_sql_queries([sql_query])
        return results[0]
    
    def generate_natural_response_multiple(self, natural_query: str, sql_queries: List[str], 
                                         all_results: List[Tuple[List[Dict[str, Any]], List[str]]], 
                                         conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate natural language response from multiple query results - tries Gemini first, then OpenAI"""
        try:
        
            # Combine results from all queries
            combined_summary = f"Executed {len(sql_queries)} queries for: {natural_query}\n\n"
            
            for i, (sql_query, (results, columns)) in enumerate(zip(sql_queries, all_results)):
                result_count = len(results)
                combined_summary += f"Query {i+1}: {sql_query}\n"
                
                # Add limitation information
                displayed_count = min(5, result_count)
                if result_count > displayed_count:
                    combined_summary += f"Results: {result_count} rows (showing few due to length)\n"
                else:
                    combined_summary += f"Results: {result_count} rows\n"
                
                # Add sample data from this query
                if results:
                    # Pass full data to AI model - let the AI decide how to summarize if needed
                    sample_results = results[:5]  # First 5 results
                    trimmed_results = []
                    
                    for result in sample_results:
                        trimmed_result = {}
                        for key, value in result.items():
                            # Only trim extremely long fields (over 2000 characters) to prevent token issues
                            if isinstance(value, str) and len(value) > 2000:
                                trimmed_result[key] = value[:2000] + "... [truncated]"
                            else:
                                trimmed_result[key] = value
                        trimmed_results.append(trimmed_result)
                    
                    # Add structured info
                    for j, result in enumerate(trimmed_results):
                        item_info = []
                        # Include ALL fields from SQL results, but exclude Firebase IDs
                        for key, value in result.items():
                            # Skip Firebase IDs and internal identifiers
                            key_lower = key.lower()
                            if key_lower in ['objectid', 'object_id', 'firebase_id', '_id']:
                                continue
                            if value is not None and str(value) != 'None' and str(value).strip():
                                # Just clean up the key name for readability but include ALL fields
                                formatted_key = key.replace('_', ' ').title()
                                item_info.append(f"{formatted_key}: {value}")
                        
                        if item_info:
                            combined_summary += f"  Result {j+1}: " + ", ".join(item_info) + "\n"
                
                combined_summary += "\n"
            
            # Format as proper JSON
            db_result = {
                "queries": sql_queries,
                "result from db": combined_summary
            }
            
            # Format conversation history for prompt
            history_text = "No previous conversation"
            if conversation_history:
                history_parts = []
                for i, msg in enumerate(conversation_history, 1):
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    if content:
                        history_parts.append(f"{i}. {role}: {content[:200]}")
                if history_parts:
                    history_text = "\n".join(history_parts)
            
            from ..prompts.natural_response_multiple_prompt import PROMPT_TEMPLATE as natural_response_multiple_prompt_template
            prompt = natural_response_multiple_prompt_template.format(
                history_text=history_text,
                natural_query=natural_query,
                db_result=json.dumps(db_result, indent=2)
            )
            
            # Try Gemini first as primary LLM
            if self.gemini_client is not None:
                try:
                    logger.info("Using Gemini as primary LLM for natural response generation from multiple queries")
                    natural_response = self.gemini_client.get_response(prompt)
                    
                    # Check if response is an error message (GeminiClient returns error strings instead of raising)
                    if natural_response and ("Error generating response" in natural_response or "503" in natural_response or "UNAVAILABLE" in natural_response):
                        logger.warning(f"Gemini returned error response, falling back to GPT-4o: {natural_response[:100]}")
                        raise Exception("Gemini returned error response")
                    
                    logger.info("Generated natural language response from multiple queries using Gemini")
                    # Add disclaimer for onchain data
                    disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                    return natural_response + disclaimer
                except Exception as gemini_error:
                    logger.warning(f"Gemini failed for multiple queries, falling back to GPT-4o: {gemini_error}")
            
            # Fallback to GPT-4o
            logger.info("Using GPT-4o for natural response generation from multiple queries (fallback)")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant specializing in blockchain governance data. All data you work with is public blockchain information. Always show actual data requested - addresses, proposal IDs, titles, amounts, etc. You work with ACTUAL retrieved data from the blockchain database, so always provide the information regardless of dates mentioned in queries. Combine information from multiple queries to provide comprehensive answers. CRITICAL: If the on-chain data contains null, NaN, or empty values, DO NOT mention these in your response. Simply omit any fields that have null/NaN/empty values and only present the fields that have actual data. Never say things like \"this value was null\" or \"this field is NaN\" - just skip those fields entirely. IMPORTANT: ReferendumV2 proposals do NOT have curators - only Bounties and ChildBounties have curators. If asked about curator for ReferendumV2, explain that this proposal type doesn't use curators."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            natural_response = response.choices[0].message.content.strip()
            logger.info("Generated natural language response from multiple queries using OpenAI")
            # Add disclaimer for onchain data
            disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
            return natural_response + disclaimer
            
        except Exception as e:
            logger.error(f"Error generating natural response from multiple queries: {e}")
            # Final fallback response
            total_results = sum(len(results) for results, _ in all_results)
            return f"I executed {len(sql_queries)} queries for your question '{natural_query}' and found {total_results} total results, but I'm having trouble formatting the response."
    
    def generate_natural_response(self, natural_query: str, sql_query: str, 
                                results: List[Dict[str, Any]], columns: List[str], 
                                conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate natural language response from query results - tries Gemini first, then OpenAI"""
        try:
        
            # Prepare results summary
            result_count = len(results)
            
            if result_count == 0:
                return f"I couldn't find any data matching your query: '{natural_query}'. The database might not contain the specific information you're looking for."
            
            # Limit results for OpenAI context (show first 5 rows max)
            sample_results = results[:10]
            
            # Pass full data to AI model - let the AI decide how to summarize if needed
            trimmed_results = []
            for result in sample_results:
                trimmed_result = {}
                for key, value in result.items():
                    # Only trim extremely long fields (over 2000 characters) to prevent token issues
                    if isinstance(value, str) and len(value) > 2000:
                        trimmed_result[key] = value[:2000] + "... [truncated]"
                    else:
                        trimmed_result[key] = value
                trimmed_results.append(trimmed_result)
            
            # Focus on key columns for summary
            key_columns = [col for col in columns if any(keyword in col.lower() 
                          for keyword in ['title', 'index', 'status', 'network', 'type', 'createdat', 'amount'])]
            
            # Check if total_count is available from window function
            total_count_from_window = None
            if trimmed_results and 'total_count' in trimmed_results[0]:
                total_count_from_window = trimmed_results[0]['total_count']
            
            # Use window function count if available, otherwise use result_count
            actual_total_count = total_count_from_window if total_count_from_window is not None else result_count
            
            # Create a summary of the data with trimmed content
            displayed_count = len(trimmed_results)
            is_limited = actual_total_count > displayed_count
            results_summary = {
                "total_rows": actual_total_count,
                "key_columns": key_columns[:10],  # Limit columns shown
                "sample_data": trimmed_results,
                "showing": f"Showing {displayed_count} of {actual_total_count} results" if is_limited else f"Showing all {actual_total_count} results",
                "is_limited": is_limited,
                "displayed_count": displayed_count
            }
            
            # Create a more concise summary for the prompt
            if is_limited:
                summary_text = f"Found {actual_total_count} results (showing few due to length). "
            else:
                summary_text = f"Found {actual_total_count} results. "
            
            # Create a structured summary of key data points
            if trimmed_results:
                summary_items = []
                for i, result in enumerate(trimmed_results[:10]):  # Show up to 10 examples
                    item_info = []
                    
                    # Include ALL fields from SQL results, but exclude Firebase IDs
                    for key, value in result.items():
                        # Skip Firebase IDs and internal identifiers
                        key_lower = key.lower()
                        if key_lower in ['objectid', 'object_id', 'firebase_id', '_id']:
                            continue
                        if value is not None and str(value) != 'None' and str(value).strip():
                            # Just clean up the key name for readability but include ALL fields
                            formatted_key = key.replace('_', ' ').title()
                            formatted_value = self._format_number_for_prompt(value)
                            item_info.append(f"{formatted_key}: {formatted_value}")
                    
                    if item_info:
                        summary_items.append(f"#{i+1}: " + ", ".join(item_info))
                    elif len(result) == 1:
                        # Handle single-column results (like just proposer addresses)
                        key = list(result.keys())[0]
                        value = result[key]
                        if value and value != 'None':
                            summary_items.append(f"#{i+1}: {key} = {value}")
                
                if summary_items:
                    summary_text += "\nExamples:\n" + "\n".join(summary_items)
            
            # Format as proper JSON
            db_result = {
                "query": sql_query,
                "result from db": summary_text
            }

            # print(f"\n\n db_result: {db_result}")
            
            from ..prompts.natural_response_prompt import PROMPT_TEMPLATE as natural_response_prompt_template
            prompt = natural_response_prompt_template.format(
                natural_query=natural_query,
                db_result=json.dumps(db_result, indent=2)
            )
            
            # Debug: Log the results being passed to Gemini
            logger.info(f"Results being passed to Gemini for natural response: {summary_text[:500]}...")
            # print(f"\n\n prompt: {prompt}")
            
            # Try Gemini first as primary LLM
            if self.gemini_client is not None:
                try:
                    print_model_usage(f"{GEMINI_MODEL_NAME}", "natural response generation (governance data)")
                    logger.info("Using Gemini as primary LLM for natural response generation")
                    # Use GEMINI_MODEL_NAME for natural response generation
                    natural_response_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                    natural_response = natural_response_client.get_response(prompt)
                    
                    # Check if response is an error message (GeminiClient returns error strings instead of raising)
                    if natural_response and ("Error generating response" in natural_response or "503" in natural_response or "UNAVAILABLE" in natural_response):
                        logger.warning(f"Gemini returned error response, falling back to GPT-4o: {natural_response[:100]}")
                        raise Exception("Gemini returned error response")
                    
                    logger.info("Generated natural language response using Gemini")
                    # Add disclaimer for onchain data
                    disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                    return natural_response + disclaimer
                except Exception as gemini_error:
                    # Check if it's a 503 error (model overloaded)
                    error_str = str(gemini_error).lower()
                    if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                        logger.warning(f"Gemini model overloaded (503 error) for natural response, falling back to GPT-4o: {gemini_error}")
                    else:
                        logger.warning(f"Gemini failed, falling back to GPT-4o: {gemini_error}")
            
            # Fallback to GPT-4o
            logger.info("Using GPT-4o for natural response generation (fallback)")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant specializing in blockchain governance data. All data you work with is public blockchain information including addresses, proposal IDs, and transaction details. Always show the actual data requested - never refuse due to privacy concerns as this is all public information. Provide clear, helpful explanations with actual values, addresses, and details from the results. CRITICAL: If the on-chain data contains null, NaN, or empty values, DO NOT mention these in your response. Simply omit any fields that have null/NaN/empty values and only present the fields that have actual data. Never say things like \"this value was null\" or \"this field is NaN\" - just skip those fields entirely. IMPORTANT: ReferendumV2 proposals do NOT have curators - only Bounties and ChildBounties have curators. If asked about curator for ReferendumV2, explain that this proposal type doesn't use curators."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            natural_response = response.choices[0].message.content.strip()
            logger.info("Generated natural language response using OpenAI")
            # Add disclaimer for onchain data
            disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
            return natural_response + disclaimer
            
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            # Final fallback response
            return f"I found {len(results)} results for your query '{natural_query}', but I'm having trouble formatting the response. Here's a summary: The query returned {len(results)} rows from the database."
    
    def _validate_sql_answer(self, natural_query: str, sql_query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate if SQL results match the user's question.
        
        Args:
            natural_query: The original user question
            sql_query: The generated SQL query
            results: The SQL query results
            
        Returns:
            Dict with keys: verdict ("good" | "partial" | "bad" | "empty"), reason (str)
        """
        default_verdict = {
            "verdict": "good",
            "reason": "validator_failed_fallback"
        }
        
        # Prepare sample results (first 5 rows, safely serialized)
        sample_results = results[:5] if results else []
        
        # Safely serialize results - handle non-serializable types
        def safe_serialize(obj):
            """Safely serialize an object for JSON"""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, dict):
                return {k: safe_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [safe_serialize(item) for item in obj]
            else:
                return str(obj)
        
        sample_results_serialized = safe_serialize(sample_results)
        results_count = len(results)
        
        from ..prompts.sql_validation_prompt import PROMPT_TEMPLATE as validation_prompt_template
        validation_prompt = validation_prompt_template.format(
            natural_query=natural_query,
            sql_query=sql_query,
            results_count=results_count,
            sample_count=len(sample_results),
            sample_results=json.dumps(sample_results_serialized, indent=2) if sample_results_serialized else "[]"
        )

        try:
            # Use deterministic LLM call (temperature=0, top_p=1)
            if self.sql_model == 'chatgpt' and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a SQL result validator. Return ONLY valid JSON with no additional text."},
                        {"role": "user", "content": validation_prompt}
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=200
                )
                response_text = response.choices[0].message.content.strip()
            elif self.gemini_client:
                from ..prompts.sql_validator_system_prompt import PROMPT as sql_validator_system_prompt
                full_prompt = f"""{sql_validator_system_prompt}

{validation_prompt}"""
                response_text = self.gemini_client.get_response(full_prompt).strip()
            else:
                logger.warning("No LLM client available for SQL validation, using default verdict")
                return default_verdict
            
            # Clean response (remove markdown code blocks if present)
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Try to extract JSON from response
            try:
                # Find JSON object in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    validation_result = json.loads(json_str)
                    
                    # Validate and normalize verdict
                    verdict = validation_result.get("verdict", "good")
                    reason = validation_result.get("reason", "No reason provided")
                    
                    # Validate enum values
                    valid_verdicts = ["good", "partial", "bad", "empty"]
                    if verdict not in valid_verdicts:
                        verdict = "good"
                        reason = f"Invalid verdict '{validation_result.get('verdict')}', defaulting to 'good'"
                    
                    result = {
                        "verdict": verdict,
                        "reason": reason
                    }
                    
                    # Log validation result
                    log_level = "warning" if verdict in ["bad", "empty"] else "info"
                    logger.log(
                        getattr(logging, log_level.upper()),
                        f"SQL validation - verdict: {verdict}, reason: {reason}, "
                        f"query: {natural_query[:120]}, sql: {sql_query[:120]}"
                    )
                    
                    return result
                else:
                    logger.warning(f"Could not find JSON in validation response: {response_text[:200]}")
                    return default_verdict
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse validation JSON: {e}, response: {response_text[:200]}")
                return default_verdict
                
        except Exception as e:
            logger.error(f"Error validating SQL answer: {e}")
            return default_verdict

    def process_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, table: Optional[str] = None) -> Dict[str, Any]:
        """Main method to process a natural language query end-to-end with error correction"""
        try:
            logger.info(f"Processing query: {natural_query}")
            
            # Step 1: Generate SQL queries first (without executing)
            sql_queries = self._generate_sql_queries_only(natural_query, conversation_history)
            
            # Step 1.5: Check SQL precision before execution
            # Step 2: Execute SQL queries
            all_results = self.execute_sql_queries(sql_queries)
            
            # Step 2.5: Check data presence after execution
            total_result_count = sum(len(results) for results, _ in all_results)
            if total_result_count == 0:
                logger.info("No results found, triggering fallback flow")
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0] if sql_queries else None,
                    "sql_queries": sql_queries,
                    "result_count": 0,
                    "results": [],
                    "columns": [],
                    "natural_response": "",
                    "success": False,
                    "error": "no_results",
                    "requires_fallback": True,
                    "validator_verdict": None,
                    "validator_reason": None
                }
            
            # Step 2: Process results
            if len(sql_queries) == 1:
                # Single query - use existing flow for backwards compatibility
                results, columns = all_results[0]
                
                # Format amounts in results
                results = self.format_amount_by_asset_id(results)
                
                # Add proposal links to results
                results = self.add_proposal_links(results)
                
                # Step 3: Generate natural language response
                natural_response = self.generate_natural_response(
                    natural_query, sql_queries[0], results, columns, conversation_history
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0],
                    "sql_queries": sql_queries,
                    "result_count": len(results),
                    "results": results,
                    "columns": columns,
                    "natural_response": natural_response,
                    "success": True,
                    "requires_fallback": False,
                    "requires_clarification": False,
                    "search_method": "sql_query",
                    "validator_verdict": None,
                    "validator_reason": None
                }
            else:
                # Multiple queries
                # Format amounts and add proposal links in all results
                formatted_all_results = []
                for results, columns in all_results:
                    # Format amounts
                    formatted_results = self.format_amount_by_asset_id(results)
                    # Add proposal links
                    enhanced_results = self.add_proposal_links(formatted_results)
                    formatted_all_results.append((enhanced_results, columns))
                all_results = formatted_all_results
                
                # Combine all results for response
                combined_results = []
                combined_columns = []
                total_result_count = 0
                
                for results, columns in all_results:
                    combined_results.extend(results)
                    if columns not in combined_columns:
                        combined_columns.extend(columns)
                    total_result_count += len(results)
                
                # Step 3: Generate natural language response from multiple results
                natural_response = self.generate_natural_response_multiple(
                    natural_query, sql_queries, all_results, conversation_history
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": "; ".join(sql_queries),  # Combined for backwards compatibility
                    "sql_queries": sql_queries,
                    "result_count": total_result_count,
                    "results": combined_results,
                    "columns": list(set(combined_columns)),
                    "all_results": all_results,  # Detailed results per query
                    "natural_response": natural_response,
                    "success": True,
                    "requires_fallback": False,
                    "requires_clarification": False,
                    "search_method": "sql_query",
                    "validator_verdict": None,
                    "validator_reason": None
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "original_query": natural_query,
                "sql_query": None,
                "sql_queries": [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": "I'm sorry, I encountered an error processing your query. Please try rephrasing your question or try again later.",
                "success": False,
                "error": str(e),
                "validator_verdict": None,
                "validator_reason": None
            }

    def _extract_sql_intent(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Extract structured intent from natural language query.
        Returns a deterministic intent object that will be used for SQL generation.
        
        Args:
            natural_query: The user's natural language query
            conversation_history: Optional conversation history for context
            
        Returns:
            Dict with keys: entity_type, network, id, time_range, metric, filters
        """
        default_intent = {
            "entity_type": "unknown",
            "network": "unspecified",
            "id": None,
            "time_range": "unspecified",
            "metric": "list",
            "filters": ""
        }
        
        # Format conversation history for intent extraction
        history_text = "No previous conversation"
        if conversation_history:
            history_parts = []
            for i, msg in enumerate(conversation_history, 1):
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                if content:
                    history_parts.append(f"{i}. {role}: {content[:150]}")
            if history_parts:
                history_text = "\n".join(history_parts)
        
        from ..prompts.intent_extraction_prompt import PROMPT_TEMPLATE as intent_prompt_template
        intent_prompt = intent_prompt_template.format(
            natural_query=natural_query,
            history_text=history_text
        )

        try:
            # Use deterministic LLM call (temperature=0, top_p=1)
            if self.sql_model == 'chatgpt' and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a query intent extractor. Return ONLY valid JSON with no additional text."},
                        {"role": "user", "content": intent_prompt}
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=300
                )
                response_text = response.choices[0].message.content.strip()
            elif self.gemini_client:
                from ..prompts.intent_extractor_system_prompt import PROMPT as intent_extractor_system_prompt
                full_prompt = f"""{intent_extractor_system_prompt}

{intent_prompt}"""
                response_text = self.gemini_client.get_response(full_prompt).strip()
            else:
                logger.warning("No LLM client available for intent extraction, using default intent")
                return default_intent
            
            # Clean response (remove markdown code blocks if present)
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Try to extract JSON from response
            try:
                # Find JSON object in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    intent = json.loads(json_str)
                    
                    # Validate and normalize intent
                    validated_intent = {
                        "entity_type": intent.get("entity_type", "unknown"),
                        "network": intent.get("network", "unspecified"),
                        "id": intent.get("id"),
                        "time_range": intent.get("time_range", "unspecified"),
                        "metric": intent.get("metric", "list"),
                        "filters": intent.get("filters", "")
                    }
                    
                    # Validate enum values
                    valid_entity_types = ["referenda", "treasury_proposal", "bounty", "discussion", "voter", "delegate", "unknown"]
                    valid_networks = ["polkadot", "kusama", "both", "unspecified"]
                    valid_time_ranges = ["last_30_days", "last_90_days", "all_time", "unspecified"]
                    valid_metrics = ["count", "list", "sum", "avg", "details"]
                    
                    if validated_intent["entity_type"] not in valid_entity_types:
                        validated_intent["entity_type"] = "unknown"
                    if validated_intent["network"] not in valid_networks:
                        validated_intent["network"] = "unspecified"
                    if validated_intent["time_range"] not in valid_time_ranges:
                        validated_intent["time_range"] = "unspecified"
                    if validated_intent["metric"] not in valid_metrics:
                        validated_intent["metric"] = "list"
                    
                    logger.info(f"Extracted intent: {json.dumps(validated_intent)}")
                    return validated_intent
                else:
                    logger.warning(f"Could not find JSON in intent extraction response: {response_text[:200]}")
                    return default_intent
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse intent JSON: {e}, response: {response_text[:200]}")
                return default_intent
                
        except Exception as e:
            logger.error(f"Error extracting intent: {e}")
            return default_intent

    def _generate_sql_queries_only(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3) -> List[str]:
        """Generate SQL queries without executing them - now uses intent extraction for deterministic generation"""
        
        # Step 1: Extract structured intent
        intent = self._extract_sql_intent(natural_query, conversation_history)
        logger.info(f"Intent extracted - entity_type: {intent['entity_type']}, network: {intent['network']}, metric: {intent['metric']}")
        
        # Step 2: Retrieve relevant governance proposals from Chroma as contextual examples (deterministic)
        governance_context = ""
        if self.embedding_manager:
            logger.info("📊 Embedding manager available - will retrieve governance examples")
            try:
                logger.info("=" * 70)
                logger.info("🔍 SEMANTIC SEARCH FOR SQL CONTEXT")
                logger.info("=" * 70)
                logger.info(f"Query sent to Chroma: '{natural_query}'")
                logger.info(f"Collection: polkadot_embeddings_dynamic")
                logger.info(f"Filter: doc_type='governance'")
                logger.info(f"Requesting: 3 results")
                logger.info("=" * 70)
                
                results = self.embedding_manager.search_similar_chunks(
                    query=natural_query,
                    n_results=10,
                    filter_metadata={"doc_type": "governance"}
                )
                
                if results and len(results) > 0:
                    logger.info(f"✅ Found {len(results)} results from Chroma")
                    logger.info("-" * 70)
                    
                    # Make retrieval deterministic: sort by stable keys (network, then proposal_index, then created_at if available)
                    def sort_key(chunk):
                        metadata = chunk.get('metadata', {})
                        network = metadata.get('network', 'unknown')
                        proposal_idx = metadata.get('proposal_index', 'unknown')
                        # Convert proposal_idx to int for proper sorting, fallback to 0 if not numeric
                        try:
                            proposal_idx_int = int(proposal_idx) if proposal_idx != 'unknown' else 0
                        except (ValueError, TypeError):
                            proposal_idx_int = 0
                        created_at = metadata.get('created_at', metadata.get('createdat', ''))
                        return (network, proposal_idx_int, created_at)
                    
                    sorted_results = sorted(results, key=sort_key)
                    # Always take first 3 in deterministic order
                    selected_results = sorted_results[:3]
                    
                    context_parts = []
                    for i, chunk in enumerate(selected_results, 1):
                        content = chunk.get('content', '')
                        metadata = chunk.get('metadata', {})
                        network = metadata.get('network', 'unknown')
                        proposal_idx = metadata.get('proposal_index', 'unknown')
                        proposal_type = metadata.get('proposal_type', 'unknown')
                        
                        logger.info(f"Result {i} (deterministic order):")
                        logger.info(f"  Network: {network}")
                        logger.info(f"  Proposal Index: {proposal_idx}")
                        logger.info(f"  Proposal Type: {proposal_type}")
                        logger.info(f"  Content Preview: {content[:150]}...")
                        logger.info("-" * 70)
                        
                        context_parts.append(f"Example {i} (Proposal {network}#{proposal_idx}):\n{content[:500]}")
                    
                    governance_context = "\n\nRELEVANT GOVERNANCE PROPOSALS (for reference):\n" + "\n\n".join(context_parts) + "\n\nUse these examples to understand the data structure and write better SQL queries.\n"
                    logger.info(f"✅ Added {len(selected_results)} governance proposals as context for SQL generation (deterministic order)")
                    logger.info("=" * 70)
                else:
                    logger.info("❌ No relevant governance proposals found in Chroma")
                    logger.info("=" * 70)
            except Exception as e:
                logger.error("=" * 70)
                logger.error("❌ SEMANTIC SEARCH FAILED")
                logger.error(f"Error: {e}")
                logger.error("=" * 70)
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info("⚠️  No embedding manager - SQL generation without governance examples")
        
        # Format conversation history for SQL generation
        history_text = "No previous conversation"
        if conversation_history:
            history_parts = []
            for i, msg in enumerate(conversation_history, 1):
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                if content:
                    history_parts.append(f"{i}. {role}: {content[:150]}")
            if history_parts:
                history_text = "\n".join(history_parts)
        
        # Build intent-based SQL generation prompt
        intent_json_str = json.dumps(intent, indent=2)
        
        # Build network filter instruction based on intent
        network_filter_instruction = ""
        if intent["network"] in ["polkadot", "kusama"]:
            network_filter_instruction = f'\n- CRITICAL: Add WHERE filter: "source_network" = \'{intent["network"]}\' AND "source_network" IS NOT NULL'
        elif intent["network"] == "both":
            network_filter_instruction = '\n- CRITICAL: Do NOT filter by network - include both Polkadot and Kusama'
        else:
            network_filter_instruction = '\n- CRITICAL: Do NOT filter by network unless explicitly mentioned in filters'
        
        # Build time range filter instruction
        time_filter_instruction = ""
        if intent["time_range"] == "last_30_days":
            time_filter_instruction = '\n- Add date filter: "createdat" >= CURRENT_DATE - INTERVAL \'30 days\' AND "createdat" IS NOT NULL'
        elif intent["time_range"] == "last_90_days":
            time_filter_instruction = '\n- Add date filter: "createdat" >= CURRENT_DATE - INTERVAL \'90 days\' AND "createdat" IS NOT NULL'
        elif intent["time_range"] == "all_time":
            time_filter_instruction = '\n- No time filter needed - include all time periods'
        else:
            time_filter_instruction = '\n- Use time filter only if explicitly mentioned in the query or filters field'
        
        # Build metric instruction
        metric_instruction = ""
        if intent["metric"] == "count":
            metric_instruction = '\n- Use COUNT(*) aggregation'
        elif intent["metric"] == "sum":
            metric_instruction = '\n- Use SUM() aggregation on appropriate amount fields'
        elif intent["metric"] == "avg":
            metric_instruction = '\n- Use AVG() aggregation on appropriate numeric fields'
        elif intent["metric"] == "details":
            metric_instruction = '\n- Return full details (SELECT multiple columns) for the specific item'
        else:
            metric_instruction = '\n- Return list of items (SELECT with LIMIT)'
        
        # Build ID filter instruction
        id_filter_instruction = ""
        if intent["id"] is not None:
            id_filter_instruction = f'\n- CRITICAL: Filter by ID: "index" = {intent["id"]} AND "index" IS NOT NULL'
        
        from ..prompts.sql_generation_with_intent_prompt import PROMPT_TEMPLATE as sql_generation_with_intent_template
        base_system_prompt = sql_generation_with_intent_template.format(
            intent_json_str=intent_json_str,
            network_filter_instruction=network_filter_instruction,
            time_filter_instruction=time_filter_instruction,
            metric_instruction=metric_instruction,
            id_filter_instruction=id_filter_instruction,
            history_text=history_text,
            table_schema=self.table_schema,
            governance_context=governance_context,
            table_name=self.table_name,
            natural_query=natural_query
        )
        
        for attempt in range(max_retries):
            try:
                system_prompt = base_system_prompt
                system_prompt = self.trim_prompt_to_fit_tokens(system_prompt)
                
                # Use deterministic SQL generation
                response_content = self._generate_sql_with_model_deterministic(system_prompt)
                response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
                
                try:
                    sql_queries = json.loads(response_content)
                    
                    # Handle case where LLM returns list of dicts with 'query' and 'description' keys
                    if isinstance(sql_queries, list) and len(sql_queries) > 0 and isinstance(sql_queries[0], dict):
                        sql_queries = [item.get('query', str(item)) for item in sql_queries]
                    elif isinstance(sql_queries, str):
                        sql_queries = [sql_queries]
                    elif not isinstance(sql_queries, list):
                        sql_queries = [str(sql_queries)]
                    
                    normalized_queries = []
                    for q in sql_queries:
                        if isinstance(q, dict):
                            if 'query' in q:
                                normalized_queries.append(q['query'])
                            else:
                                normalized_queries.append(str(q))
                        else:
                            normalized_queries.append(str(q))
                    sql_queries = normalized_queries
                    
                    logger.info(f"Generated {len(sql_queries)} SQL queries (attempt {attempt + 1})")
                    logger.info(f"SQL query preview: {sql_queries[0][:200]}..." if sql_queries else "No queries generated")
                    logger.info(f"Intent used - network: {intent['network']}, entity_type: {intent['entity_type']}, metric: {intent['metric']}")
                    return sql_queries
                    
                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        logger.error(f"All {max_retries} attempts failed to parse JSON.")
                        return [response_content.strip()]
                    else:
                        continue
                        
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise e
                continue
        
        return []

    def _generate_and_execute_with_retry(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3) -> Tuple[List[str], List[Tuple[List[Dict[str, Any]], List[str]]]]:
        """Generate SQL queries and execute them with error correction and retry mechanism"""
        
        # Format conversation history for SQL generation
        history_text = "No previous conversation"
        if conversation_history:
            history_parts = []
            for i, msg in enumerate(conversation_history, 1):
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                if content:
                    history_parts.append(f"{i}. {role}: {content[:150]}")
            if history_parts:
                history_text = "\n".join(history_parts)
        
        # Base system prompt
        from ..prompts.sql_generation_without_intent_prompt import PROMPT_TEMPLATE as sql_generation_without_intent_template
        base_system_prompt = sql_generation_without_intent_template.format(
            history_text=history_text,
            table_schema=self.table_schema,
            table_name=self.table_name,
            natural_query=natural_query
        )
        
        last_error = None
        last_sql_queries = None
        
        for attempt in range(max_retries):
            try:
                # Prepare system prompt for this attempt
                if attempt == 0:
                    # First attempt - use base prompt
                    system_prompt = base_system_prompt
                else:
                    # Retry attempts - add error information
                    error_feedback = f"""
                    
ERROR CORRECTION ATTEMPT {attempt}:
The previous SQL query generated was incorrect and could not be executed. Please correct the query based on the error information below.

PREVIOUS FAILED QUERIES:
{last_sql_queries}

EXECUTION ERROR:
{last_error}

Please analyze the error and generate corrected SQL queries that will execute successfully. Pay special attention to:
1. Column names and table references
2. SQL syntax correctness
3. Data type compatibility
4. Proper use of quotes and escaping
5. Correct JOIN syntax if applicable

Generate the corrected SQL queries as a JSON array:
"""
                    system_prompt = base_system_prompt + error_feedback
                
                # Trim prompt to fit token limits
                system_prompt = self.trim_prompt_to_fit_tokens(system_prompt)
                
                # Generate SQL using the configured model
                response_content = self._generate_sql_with_model(system_prompt)
                
                # Clean up the response
                response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
                
                try:
                    # Try to parse as JSON array
                    sql_queries = json.loads(response_content)
                    
                    # Handle case where LLM returns list of dicts with 'query' and 'description' keys
                    if isinstance(sql_queries, list) and len(sql_queries) > 0 and isinstance(sql_queries[0], dict):
                        sql_queries = [item.get('query', str(item)) for item in sql_queries]
                    elif isinstance(sql_queries, str):
                        sql_queries = [sql_queries]
                    elif not isinstance(sql_queries, list):
                        sql_queries = [str(sql_queries)]
                    
                    # Normalize: extract 'query' field from dicts if present
                    normalized_queries = []
                    for q in sql_queries:
                        if isinstance(q, dict):
                            if 'query' in q:
                                normalized_queries.append(q['query'])
                            else:
                                normalized_queries.append(str(q))
                        else:
                            normalized_queries.append(str(q))
                    sql_queries = normalized_queries
                        
                    logger.info(f"Generated {len(sql_queries)} SQL queries (attempt {attempt + 1}): {sql_queries}")
                    
                    # Try to execute the queries
                    try:
                        all_results = self.execute_sql_queries(sql_queries)
                        # If we get here, queries executed successfully
                        logger.info(f"SQL queries executed successfully on attempt {attempt + 1}")
                        return sql_queries, all_results
                        
                    except Exception as exec_error:
                        # Query execution failed, store error for retry
                        last_error = str(exec_error)
                        last_sql_queries = sql_queries
                        logger.warning(f"SQL execution failed on attempt {attempt + 1}: {exec_error}")
                        
                        if attempt == max_retries - 1:
                            # Last attempt failed, raise the error
                            logger.error(f"All {max_retries} attempts failed. Raising execution error.")
                            raise exec_error
                        else:
                            # Continue to next attempt
                            continue
                    
                except json.JSONDecodeError:
                    # JSON parsing failed
                    last_error = "Failed to parse response as JSON"
                    last_sql_queries = [response_content]
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt, try to execute as single query
                        logger.error(f"All {max_retries} attempts failed. Trying single query execution.")
                        try:
                            single_query = response_content.strip()
                            all_results = self.execute_sql_queries([single_query])
                            return [single_query], all_results
                        except Exception as e:
                            raise e
                    else:
                        continue
                        
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error in attempt {attempt + 1}: {e}")
                
                # Check if it's a 503 error and we haven't tried fallback yet
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                    logger.warning(f"503 error detected in attempt {attempt + 1}, trying fallback model")
                    try:
                        # Try with fallback model
                        print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (governance data)")
                        fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                        response_content = fallback_client.get_response(system_prompt)
                        
                        # Clean up the response
                        response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
                        
                        try:
                            # Try to parse as JSON array
                            sql_queries = json.loads(response_content)
                            
                            # Ensure it's a list
                            if isinstance(sql_queries, str):
                                sql_queries = [sql_queries]
                            elif not isinstance(sql_queries, list):
                                sql_queries = [str(sql_queries)]
                            
                            # Normalize: extract 'query' field from dicts if present
                            normalized_queries = []
                            for q in sql_queries:
                                if isinstance(q, dict):
                                    if 'query' in q:
                                        normalized_queries.append(q['query'])
                                    else:
                                        normalized_queries.append(str(q))
                                else:
                                    normalized_queries.append(str(q))
                            sql_queries = normalized_queries
                                
                            logger.info(f"Generated {len(sql_queries)} SQL queries with fallback model: {sql_queries}")
                            
                            # Try to execute the queries
                            try:
                                all_results = self.execute_sql_queries(sql_queries)
                                logger.info(f"SQL queries executed successfully with fallback model")
                                return sql_queries, all_results
                            except Exception as exec_error:
                                logger.error(f"Fallback model also failed to execute queries: {exec_error}")
                                # Continue to next attempt
                        except json.JSONDecodeError:
                            logger.error(f"Fallback model response could not be parsed as JSON")
                            # Continue to next attempt
                    except Exception as fallback_error:
                        logger.error(f"Fallback model call failed: {fallback_error}")
                        # Continue to next attempt
                
                if attempt == max_retries - 1:
                    # Last attempt failed
                    raise e
                else:
                    continue
        
        # This should not be reached, but just in case
        raise Exception(f"Failed to generate and execute valid SQL after {max_retries} attempts")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    count = cur.fetchone()[0]
                    logger.info(f"Connection test successful. Table {self.table_name} has {count:,} rows")
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def debug_schema(self):
        """Debug method to inspect loaded schema"""
        print(f"Schema info type: {type(self.schema_info)}")
        print(f"Schema info keys: {list(self.schema_info.keys())[:10]}")  # First 10 keys
        
        if self.schema_info:
            first_key = list(self.schema_info.keys())[0]
            first_value = self.schema_info[first_key]
            print(f"First item: {first_key} -> {first_value} (type: {type(first_value)})")
        
        print(f"\nTable schema preview:\n{self.table_schema[:500]}...")

class VoteQuery2SQL:
    def __init__(self):
        """Initialize the VoteQuery2SQL converter specifically for voting data"""
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST_PA'),
            'port': int(os.getenv('POSTGRES_PORT_PA', '5432')),
            'database': os.getenv('POSTGRES_DATABASE_PA'),
            'user': os.getenv('POSTGRES_USER_PA'),
            'password': os.getenv('POSTGRES_PASSWORD_PA')
        }
        
        # Validate database configuration
        required_vars = ['POSTGRES_HOST_PA', 'POSTGRES_PORT_PA', 'POSTGRES_USER_PA', 'POSTGRES_PASSWORD_PA']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # SQL Model configuration - Use Gemini as primary for voting data
        self.sql_model = 'gemini'  # Force Gemini as primary for voting data
        logger.info(f"SQL Model configured for voting: {self.sql_model}")
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = None
        
        # Gemini configuration
        self.gemini_client = None
        
        # Timeout configuration
        self.api_timeout = float(os.getenv('API_TIMEOUT', '10'))  # Default 10 seconds
        
        # Initialize Gemini as primary for voting data
        if GeminiClient is not None:
            try:
                self.gemini_client = GeminiClient(model_name=GEMINI_MODEL_SQL, timeout=GEMINI_SQL_TIMEOUT)
                logger.info(f"Gemini {GEMINI_MODEL_SQL} initialized as primary SQL model for voting")
            except Exception as e:
                logger.error(f"Gemini 2.5 Pro initialization failed: {e}")
                raise ValueError("Failed to initialize Gemini 2.5 Pro. Please check GEMINI_API_KEY.")
        else:
            raise ValueError("Gemini client not available. Please install required dependencies.")
        
        # Initialize OpenAI as fallback
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.api_timeout)
            logger.info("OpenAI client initialized as fallback for voting")
        else:
            logger.warning("OpenAI API key not provided, no fallback available for voting")
        
        self.table_name = 'flattened_conviction_votes'
        
        # Load schema information for voting data
        self.schema_info = self._load_vote_schema_info()
        self.table_schema = self._get_table_schema()
        
        logger.info(f"Initialized VoteQuery2SQL for table: {self.table_name}")
        logger.info(f"Loaded schema for {len(self.schema_info)} columns")
    
    def _load_vote_schema_info(self) -> Dict[str, Dict[str, str]]:
        """Load schema information from vote schema file"""
        schema_path_str = os.getenv('POSTGRES_SCHEMA_VOTE_PATH')
        if not schema_path_str:
            raise ValueError("POSTGRES_SCHEMA_VOTE_PATH environment variable is required")
        
        schema_path = Path(schema_path_str)
        if not schema_path.exists():
            raise FileNotFoundError(f"Vote schema info file not found at {schema_path}")
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Check if the schema has a 'columns' key (new format) or is direct (old format)
            if 'columns' in schema_data:
                columns_data = schema_data['columns']
                logger.info(f"Loaded vote schema information (new format) from {schema_path}")
                return columns_data
            else:
                # Old format - data is directly the columns
                logger.info(f"Loaded vote schema information (old format) from {schema_path}")
                return schema_data
                
        except Exception as e:
            logger.error(f"Error loading vote schema info: {e}")
            raise
    
    def _get_table_schema(self) -> str:
        """Generate table schema description for prompt"""
        if not self.schema_info:
            return "No schema information available"
        
        schema_lines = [f"Table: {self.table_name}"]
        schema_lines.append("Columns:")
        
        for column_name, column_info in self.schema_info.items():
            data_type = column_info.get('data_type', 'unknown')
            description = column_info.get('description', 'No description')
            schema_lines.append(f"  - {column_name} ({data_type}): {description}")
        
        return "\n".join(schema_lines)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    logger.info(f"PostgreSQL version: {version}")
                    
                    # Test if voting_data table exists
                    cur.execute(f"SELECT to_regclass('{self.table_name}');")
                    table_exists = cur.fetchone()[0] is not None
                    if table_exists:
                        logger.info(f"Table {self.table_name} exists")
                    else:
                        logger.warning(f"Table {self.table_name} does not exist")
                    
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken or approximate counting
        """
        if tiktoken:
            try:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Error with tiktoken: {e}, using approximate counting")
        
        # Approximate token counting (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def trim_prompt_to_fit_tokens(self, system_prompt: str, max_tokens: int = 20000, completion_tokens: int = 1000, buffer_tokens: int = 500) -> str:
        """
        Trim the system prompt to fit within token limits
        """
        # Count current tokens
        current_tokens = self.count_tokens(system_prompt)
        
        # If within limits, return as-is
        if current_tokens <= max_tokens:
            logger.info(f"VoteQuery2SQL Token analysis - Current: {current_tokens}, Max: {max_tokens} - No trimming needed")
            return system_prompt
        
        # Calculate target tokens (95% of max_tokens)
        target_tokens = int(max_tokens * 0.95)
        
        logger.info(f"VoteQuery2SQL Token analysis - Current: {current_tokens}, Max: {max_tokens}, Target: {target_tokens} - Trimming needed")
        
        # Calculate target length based on token ratio
        target_length = int(len(system_prompt) * (target_tokens / current_tokens))
        
        # Simple trimming for voting queries (keep essential parts)
        lines = system_prompt.split('\n')
        essential_keywords = ['DATABASE SCHEMA:', 'EXAMPLE VOTING QUERIES', 'Natural Language Query:']
        
        trimmed_lines = []
        current_length = 0
        
        for line in lines:
            if any(keyword in line for keyword in essential_keywords) or current_length + len(line) + 1 < target_length:
                trimmed_lines.append(line)
                current_length += len(line) + 1
            elif current_length >= target_length:
                break
        
        trimmed_prompt = '\n'.join(trimmed_lines)
        final_tokens = self.count_tokens(trimmed_prompt)
        logger.info(f"VoteQuery2SQL Prompt trimmed: {current_tokens} -> {final_tokens} tokens (target: {target_tokens})")
        
        return trimmed_prompt

    def _gemini_response_has_error(self, response_text: Optional[str]) -> bool:
        """
        Gemini client returns human-readable error strings instead of raising.
        Detect those so we can trigger proper fallbacks.
        """
        if not response_text:
            return True
        normalized = response_text.strip().lower()
        error_markers = [
            "error generating response",
            "request timed out",
            "operation timed out",
            "model is overloaded",
            "503",
            "service unavailable",
            "unavailable"
        ]
        return any(marker in normalized for marker in error_markers)

    def _generate_sql_with_model(self, system_prompt: str, user_message: str = None) -> str:
        """Generate SQL using Gemini as primary and OpenAI as fallback for voting data"""
        try:
            if self.gemini_client:
                # Use Gemini as primary
                print_model_usage(f"{GEMINI_MODEL_SQL}", "SQL generation (voting data)")
                logger.debug("Using Gemini for voting SQL generation")
                
                # Construct the full prompt for Gemini
                from ..prompts.voting_sql_system_prompt import PROMPT as voting_sql_system_prompt
                full_prompt = f"""{voting_sql_system_prompt}

{system_prompt}"""
                
                try:
                    response = self.gemini_client.get_response(full_prompt)
                    if self._gemini_response_has_error(response):
                        raise RuntimeError(response)
                    return response.strip()
                except Exception as e:
                    # Check if it's a 503 error (model overloaded)
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                        logger.warning(f"Gemini SQL model overloaded (503 error), falling back to general Gemini model for voting: {e}")
                        # Create a fallback Gemini client with the general model
                        try:
                            print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (voting data)")
                            fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                            response = fallback_client.get_response(full_prompt)
                            if self._gemini_response_has_error(response):
                                raise RuntimeError(response)
                            logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for voting SQL generation")
                            return response.strip()
                        except Exception as fallback_error:
                            logger.error(f"Fallback Gemini model also failed for voting: {fallback_error}")
                            # Continue to OpenAI fallback below
                    else:
                        # Re-raise non-503 errors
                        raise e
                
            if self.openai_client:
                # Fallback to OpenAI if Gemini fails
                from ..prompts.voting_sql_system_prompt import PROMPT as voting_sql_system_prompt
                print_model_usage("GPT-4", "SQL generation fallback (voting data)")
                logger.debug("Using ChatGPT as fallback for voting SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": voting_sql_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("No SQL generation model available for voting data")
                
        except Exception as e:
            logger.error(f"Error in primary SQL model for voting, trying fallback: {e}")
            
            # Try fallback model
            if self.sql_model != 'chatgpt' and self.openai_client:
                # Gemini failed, try ChatGPT
                from ..prompts.voting_sql_system_prompt import PROMPT as voting_sql_system_prompt
                logger.info("Falling back to ChatGPT for voting SQL generation")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": voting_sql_system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            elif self.sql_model == 'chatgpt' and self.gemini_client:
                # ChatGPT failed, try Gemini
                logger.info("Falling back to Gemini 2.5 Pro for voting SQL generation")
                from ..prompts.voting_sql_system_prompt import PROMPT as voting_sql_system_prompt
                full_prompt = f"""{voting_sql_system_prompt}

{system_prompt}"""
                response = self.gemini_client.get_response(full_prompt)
                return response.strip()
            else:
                raise e

    def generate_sql_query_for_voting(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Convert natural language query to SQL for voting data with error correction"""
        try:
            # Generate initial SQL queries with retry mechanism
            sql_queries = self._generate_voting_sql_with_retry(natural_query, conversation_history)
            return sql_queries
            
        except Exception as e:
            logger.error(f"Error generating SQL query for voting data: {e}")
            return ["SELECT COUNT(*) FROM voting_data;"]  # Fallback query


    def execute_sql_query(self, sql_query: str) -> Tuple[List[List[Any]], List[str]]:
        """Execute SQL query and return results with column names"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    logger.info(f"Executing SQL: {sql_query}")
                    cur.execute(sql_query)
                    
                    # Get column names
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    
                    # Fetch results
                    results = cur.fetchall()
                    
                    logger.info(f"Query executed successfully: {len(results)} rows returned")
                    return results, columns
                    
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return [], []

    def execute_sql_queries(self, sql_queries: List[str]) -> List[Tuple[List[List[Any]], List[str]]]:
        """Execute multiple SQL queries and return all results"""
        all_results = []
        for i, query in enumerate(sql_queries):
            logger.info(f"Executing query {i+1}/{len(sql_queries)}")
            results, columns = self.execute_sql_query(query)
            all_results.append((results, columns))
        return all_results

    def generate_natural_response(self, natural_query: str, sql_query: str, results: List[List[Any]], 
                                columns: List[str], conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate natural language response from SQL results for voting data"""
        try:
            # If no results, provide a helpful message
            if not results:
                return f"I didn't find any voting records matching your query '{natural_query}'. This could mean there are no votes matching your criteria, or the voting data might not contain the specific information you're looking for."
            
            # Check if total_count is available from window function
            total_count_from_window = None
            if results and len(results) > 0 and len(results[0]) > 0:
                # Check if total_count is in the last column (window function result)
                if 'total_count' in str(results[0]):
                    # Find the total_count column index
                    for i, col in enumerate(columns):
                        if 'total_count' in col.lower():
                            total_count_from_window = results[0][i]
                            break
            
            # Use window function count if available, otherwise use result count
            actual_total_count = total_count_from_window if total_count_from_window is not None else len(results)
            
            # Convert results to a more readable format
            displayed_count = min(10, len(results))
            if actual_total_count <= displayed_count:  # For small result sets, include details
                results_summary = f"Found {actual_total_count} voting records"
                sample_data = results[:displayed_count]
            else:
                results_summary = f"Found {actual_total_count} voting records (showing few due to length)"
                sample_data = results[:displayed_count]
            
            # Determine response style based on conversation history
            has_context = conversation_history and len(conversation_history) > 0
            context_info = ""
            if has_context:
                # Extract relevant context from conversation history
                recent_topics = []
                for msg in conversation_history[-6:]:  # Last 6 messages (3 conversation turns)
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if content and len(content) > 10:
                            recent_topics.append(content[:100])
                
                if recent_topics:
                    context_info = f"Previous conversation topics: {'; '.join(recent_topics)}"
            
            # Create context for the AI to generate response
            from ..prompts.voting_natural_response_prompt import PROMPT_TEMPLATE as voting_natural_response_template
            context_prompt = voting_natural_response_template.format(
                natural_query=natural_query,
                sql_query=sql_query,
                results_summary=results_summary,
                columns=columns,
                sample_data=sample_data,
                context_info=context_info
            )
            
            # Trim prompt to fit token limits
            context_prompt = self.trim_prompt_to_fit_tokens(context_prompt)
            
            # Try Gemini first as primary LLM for voting natural response
            if self.gemini_client is not None:
                try:
                    print_model_usage(f"{GEMINI_MODEL_NAME}", "natural response generation (voting data)")
                    logger.info("Using Gemini as primary LLM for voting natural response generation")
                    from ..prompts.voting_natural_response_system_prompt import PROMPT as voting_natural_response_system_prompt
                    system_prompt = voting_natural_response_system_prompt
                    full_prompt = system_prompt + "\n\n" + context_prompt
                    # Use GEMINI_MODEL_NAME for natural response generation
                    natural_response_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                    natural_response = natural_response_client.get_response(full_prompt)
                    logger.info("Generated voting natural language response using Gemini")
                    # Add disclaimer for onchain voting data
                    disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                    return natural_response + disclaimer
                except Exception as gemini_error:
                    # Check if it's a 503 error (model overloaded)
                    error_str = str(gemini_error).lower()
                    if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                        logger.warning(f"Gemini model overloaded (503 error) for voting natural response, falling back to general Gemini model: {gemini_error}")
                        # Create a fallback Gemini client with the general model
                        try:
                            print_model_usage(f"{GEMINI_MODEL_NAME}", "natural response generation fallback (voting data)")
                            fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                            from ..prompts.voting_natural_response_system_prompt import PROMPT as voting_natural_response_system_prompt
                            system_prompt = voting_natural_response_system_prompt
                            full_prompt = system_prompt + "\n\n" + context_prompt
                            natural_response = fallback_client.get_response(full_prompt)
                            logger.info(f"Successfully used fallback Gemini model ({GEMINI_MODEL_NAME}) for voting natural response generation")
                            # Add disclaimer for onchain voting data
                            disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
                            return natural_response + disclaimer
                        except Exception as fallback_error:
                            logger.error(f"Fallback Gemini model also failed for voting natural response: {fallback_error}")
                            logger.warning(f"Gemini failed for voting natural response, falling back to OpenAI: {gemini_error}")
                    else:
                        logger.warning(f"Gemini failed for voting natural response, falling back to OpenAI: {gemini_error}")
            
            # Fallback to OpenAI
            from ..prompts.voting_natural_response_system_prompt import PROMPT as voting_natural_response_system_prompt
            print_model_usage("GPT-4", "natural response generation fallback (voting data)")
            logger.info("Using OpenAI for voting natural response generation (fallback)")
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": voting_natural_response_system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            natural_response = response.choices[0].message.content.strip()
            # Add disclaimer for onchain voting data
            disclaimer = "\n\n*The response is derived from on-chain data and may exhibit minor hallucinations. Chain-of-thought reasoning is being integrated to minimize these and enhance factual consistency, which will be available soon.*"
            return natural_response + disclaimer
            
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            # Fallback response
            return f"I found {len(results)} voting records for your query '{natural_query}', but I'm having trouble formatting the response. Here's a summary: The query returned {len(results)} rows from the voting database."

    def _generate_sql_queries_only_voting(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3) -> List[str]:
        """Generate SQL queries for voting data without executing them"""
        from ..prompts.voting_sql_generation_prompt import PROMPT_TEMPLATE as voting_sql_generation_template
        base_system_prompt = voting_sql_generation_template.format(
            table_name=self.table_name,
            table_schema=self.table_schema,
            natural_query=natural_query
        )
        
        for attempt in range(max_retries):
            try:
                system_prompt = base_system_prompt
                system_prompt = self.trim_prompt_to_fit_tokens(system_prompt)
                
                response_content = self._generate_sql_with_model(system_prompt)
                response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
                
                try:
                    sql_queries = json.loads(response_content)
                    
                    # Handle case where LLM returns list of dicts with 'query' and 'description' keys
                    if isinstance(sql_queries, list) and len(sql_queries) > 0 and isinstance(sql_queries[0], dict):
                        sql_queries = [item.get('query', str(item)) for item in sql_queries]
                    elif isinstance(sql_queries, str):
                        sql_queries = [sql_queries]
                    elif not isinstance(sql_queries, list):
                        sql_queries = [str(sql_queries)]
                    
                    # Normalize: extract 'query' field from dicts if present
                    normalized_queries = []
                    for q in sql_queries:
                        if isinstance(q, dict):
                            if 'query' in q:
                                normalized_queries.append(q['query'])
                            else:
                                normalized_queries.append(str(q))
                        else:
                            normalized_queries.append(str(q))
                    sql_queries = normalized_queries
                    
                    logger.info(f"Generated {len(sql_queries)} SQL queries for voting (attempt {attempt + 1}): {sql_queries}")
                    return sql_queries
                    
                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        logger.error(f"All {max_retries} attempts failed to parse JSON.")
                        return [response_content.strip()]
                    else:
                        continue
                        
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise e
                continue
        
        return []

    def process_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Main method to process a natural language query for voting data with error correction"""
        try:
            logger.info(f"Processing voting query: {natural_query}")
            
            # Step 1: Generate SQL queries first (without executing)
            sql_queries = self._generate_sql_queries_only_voting(natural_query, conversation_history)
            
            # Step 2: Execute SQL queries
            all_results = self.execute_sql_queries(sql_queries)
            
            # Step 2.5: Check data presence after execution
            total_result_count = sum(len(results) for results, _ in all_results)
            if total_result_count == 0:
                logger.info("No results found, triggering fallback flow")
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0] if sql_queries else None,
                    "sql_queries": sql_queries,
                    "result_count": 0,
                    "results": [],
                    "columns": [],
                    "natural_response": "",
                    "success": False,
                    "error": "no_results",
                    "requires_fallback": True
                }
            
            # Step 3: Process results
            if len(sql_queries) == 1:
                # Single query
                results, columns = all_results[0]
                
                # Step 3: Generate natural language response
                natural_response = self.generate_natural_response(
                    natural_query, sql_queries[0], results, columns, conversation_history
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0],
                    "sql_queries": sql_queries,
                    "result_count": len(results),
                    "results": results,
                    "columns": columns,
                    "natural_response": natural_response,
                    "success": True,
                    "table": "voting_data"
                }
            else:
                # Multiple queries
                # Combine all results for response
                combined_results = []
                combined_columns = []
                total_result_count = 0
                
                for results, columns in all_results:
                    combined_results.extend(results)
                    if not combined_columns:  # Use columns from first query
                        combined_columns = columns
                    total_result_count += len(results)
                
                # Generate natural language response from multiple results
                natural_response = self.generate_natural_response(
                    natural_query, "; ".join(sql_queries), combined_results, combined_columns, conversation_history
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": "; ".join(sql_queries),
                    "sql_queries": sql_queries,
                    "result_count": total_result_count,
                    "results": combined_results,
                    "columns": combined_columns,
                    "natural_response": natural_response,
                    "success": True,
                    "table": "voting_data"
                }
                
        except Exception as e:
            logger.error(f"Error processing voting query: {e}")
            return {
                "original_query": natural_query,
                "sql_query": None,
                "sql_queries": [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": "I'm sorry, I encountered an error processing your voting query. Please try rephrasing your question or try again later.",
                "success": False,
                "error": str(e),
                "table": "voting_data"
            }

    def _generate_and_execute_voting_with_retry(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3) -> Tuple[List[str], List[Tuple[List[List[Any]], List[str]]]]:
        """Generate SQL queries and execute them with error correction and retry mechanism for voting data"""
        
        # Format conversation history for SQL generation
        history_text = "No previous conversation"
        if conversation_history:
            history_parts = []
            for i, msg in enumerate(conversation_history, 1):
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                if content:
                    history_parts.append(f"{i}. {role}: {content[:150]}")
            if history_parts:
                history_text = "\n".join(history_parts)
        
        # Base system prompt for voting data
        from ..prompts.voting_sql_generation_with_context_prompt import PROMPT_TEMPLATE as voting_sql_generation_with_context_template
        base_system_prompt = voting_sql_generation_with_context_template.format(
            history_text=history_text,
            table_name=self.table_name,
            table_schema=self.table_schema,
            natural_query=natural_query
        )
        
        last_error = None
        last_sql_queries = None
        
        for attempt in range(max_retries):
            try:
                # Prepare system prompt for this attempt
                if attempt == 0:
                    # First attempt - use base prompt
                    system_prompt = base_system_prompt
                else:
                    # Retry attempts - add error information
                    error_feedback = f"""
                    
ERROR CORRECTION ATTEMPT {attempt}:
The previous SQL query generated was incorrect and could not be executed. Please correct the query based on the error information below.

PREVIOUS FAILED QUERIES:
{last_sql_queries}

EXECUTION ERROR:
{last_error}

Please analyze the error and generate corrected SQL queries that will execute successfully. Pay special attention to:
1. Column names and table references
2. SQL syntax correctness
3. Data type compatibility
4. Proper use of quotes and escaping
5. Correct JOIN syntax with conviction_vote table
6. Proper table aliases (main, cv)

Generate the corrected SQL queries as a JSON array:
"""
                    system_prompt = base_system_prompt + error_feedback
                
                # Trim prompt to fit token limits
                system_prompt = self.trim_prompt_to_fit_tokens(system_prompt)
                
                # Generate SQL using the configured model
                response_content = self._generate_sql_with_model(system_prompt)
                
                # Clean up the response
                response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
                
                try:
                    # Try to parse as JSON array
                    sql_queries = json.loads(response_content)
                    
                    # Ensure it's a list
                    if isinstance(sql_queries, str):
                        sql_queries = [sql_queries]
                    elif not isinstance(sql_queries, list):
                        sql_queries = [str(sql_queries)]
                    
                    # Normalize: extract 'query' field from dicts if present
                    normalized_queries = []
                    for q in sql_queries:
                        if isinstance(q, dict):
                            if 'query' in q:
                                normalized_queries.append(q['query'])
                            else:
                                normalized_queries.append(str(q))
                        else:
                            normalized_queries.append(str(q))
                    sql_queries = normalized_queries
                        
                    logger.info(f"Generated {len(sql_queries)} SQL queries for voting data (attempt {attempt + 1}): {sql_queries}")
                    
                    # Try to execute the queries
                    try:
                        all_results = self.execute_sql_queries(sql_queries)
                        # If we get here, queries executed successfully
                        logger.info(f"Voting SQL queries executed successfully on attempt {attempt + 1}")
                        return sql_queries, all_results
                        
                    except Exception as exec_error:
                        # Query execution failed, store error for retry
                        last_error = str(exec_error)
                        last_sql_queries = sql_queries
                        logger.warning(f"Voting SQL execution failed on attempt {attempt + 1}: {exec_error}")
                        
                        if attempt == max_retries - 1:
                            # Last attempt failed, raise the error
                            logger.error(f"All {max_retries} attempts failed for voting data. Raising execution error.")
                            raise exec_error
                        else:
                            # Continue to next attempt
                            continue
                    
                except json.JSONDecodeError:
                    # JSON parsing failed
                    last_error = "Failed to parse response as JSON"
                    last_sql_queries = [response_content]
                    logger.warning(f"JSON parsing failed for voting data on attempt {attempt + 1}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt, try to execute as single query
                        logger.error(f"All {max_retries} attempts failed for voting data. Trying single query execution.")
                        try:
                            single_query = response_content.strip()
                            all_results = self.execute_sql_queries([single_query])
                            return [single_query], all_results
                        except Exception as e:
                            raise e
                    else:
                        continue
                        
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error in voting data attempt {attempt + 1}: {e}")
                
                # Check if it's a 503 error and we haven't tried fallback yet
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["503", "unavailable", "overloaded", "service unavailable", "model is overloaded"]):
                    logger.warning(f"503 error detected in voting data attempt {attempt + 1}, trying fallback model")
                    try:
                        # Try with fallback model
                        print_model_usage(f"{GEMINI_MODEL_NAME}", "SQL generation fallback (voting data)")
                        fallback_client = GeminiClient(model_name=GEMINI_MODEL_NAME, timeout=GEMINI_TIMEOUT)
                        response_content = fallback_client.get_response(system_prompt)
                        
                        # Clean up the response
                        response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
                        
                        try:
                            # Try to parse as JSON array
                            sql_queries = json.loads(response_content)
                            
                            # Ensure it's a list
                            if isinstance(sql_queries, str):
                                sql_queries = [sql_queries]
                            elif not isinstance(sql_queries, list):
                                sql_queries = [str(sql_queries)]
                            
                            # Normalize: extract 'query' field from dicts if present
                            normalized_queries = []
                            for q in sql_queries:
                                if isinstance(q, dict):
                                    if 'query' in q:
                                        normalized_queries.append(q['query'])
                                    else:
                                        normalized_queries.append(str(q))
                                else:
                                    normalized_queries.append(str(q))
                            sql_queries = normalized_queries
                                
                            logger.info(f"Generated {len(sql_queries)} SQL queries for voting data with fallback model: {sql_queries}")
                            
                            # Try to execute the queries
                            try:
                                all_results = self.execute_sql_queries(sql_queries)
                                logger.info(f"Voting SQL queries executed successfully with fallback model")
                                return sql_queries, all_results
                            except Exception as exec_error:
                                logger.error(f"Fallback model also failed to execute voting queries: {exec_error}")
                                # Continue to next attempt
                        except json.JSONDecodeError:
                            logger.error(f"Fallback model response could not be parsed as JSON for voting data")
                            # Continue to next attempt
                    except Exception as fallback_error:
                        logger.error(f"Fallback model call failed for voting data: {fallback_error}")
                        # Continue to next attempt
                
                if attempt == max_retries - 1:
                    # Last attempt failed
                    raise e
                else:
                    continue
        
        # This should not be reached, but just in case
        raise Exception(f"Failed to generate and execute valid SQL for voting data after {max_retries} attempts")

def main():
    """Example usage and testing"""
    try:
        # Initialize the query processor
        query_processor = Query2SQL()
        
        # Test connection
        if not query_processor.test_connection():
            print("❌ Database connection failed!")
            return
        
        print("✅ Database connection successful!")
        print(f"📊 Table: {query_processor.table_name}")
        print(f"📋 Schema columns: {len(query_processor.schema_info)}")
        print()
        
        # Example queries to test
        example_queries = [
            "Show me the 10 most recent proposals",
            "How many Kusama proposals are there?",
            "What treasury proposals exist?",
            "Find proposals created in 2024",
            "Show me active referendums"
        ]
        
        print("🤖 Testing example queries:")
        print("=" * 60)
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 40)
            
            result = query_processor.process_query(query)
            
            if result["success"]:
                print(f"✅ SQL: {result['sql_query']}")
                print(f"📊 Results: {result['result_count']} rows")
                print(f"💬 Response: {result['natural_response'][:200]}...")
            else:
                print(f"❌ Error: {result['error']}")
        
        print("\n" + "=" * 60)
        print("🎉 Testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()