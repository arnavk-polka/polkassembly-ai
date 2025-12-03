#!/home/ubuntu/venv/bin/python3
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import logging
import fcntl
import os
import time
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations.slack_bot import SlackBot

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH", "/home/ubuntu/pa-ai/polkassembly-ai")

PROJECT_ROOT = Path(BASE_PATH)
LOG_DIR = PROJECT_ROOT / "logs"
LOCK_FILE = PROJECT_ROOT / ".update_governance_data.lock"

CLEAN_DIRS = [
    PROJECT_ROOT / "data/onchain_data",
    PROJECT_ROOT / "onchain_data/onchain_first_pull/all_csv",
    PROJECT_ROOT / "onchain_data/onchain_first_pull/one_table",
    PROJECT_ROOT / "onchain_data/onchain_first_pull/one_table/filter_data",
]

SCRIPTS = [
    PROJECT_ROOT / "src/ingestion/onchain/onchain_data.py",
    PROJECT_ROOT / "src/ingestion/onchain/flatten_all_data.py",
    PROJECT_ROOT / "src/ingestion/onchain/create_one_table.py",
    PROJECT_ROOT / "src/ingestion/onchain/filter_data.py",
    PROJECT_ROOT / "src/ingestion/onchain/insert_into_postgres.py",
]

SLEEP_INTERVAL = 24 * 60 * 60

LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger():
    """Setup logger with timestamp-based log file"""
    log_file = LOG_DIR / f"update_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True
    )
    
    return logger, log_file

logger = logging.getLogger(__name__)

try:
    slack_bot = SlackBot()
    logger.info("Slack bot initialized successfully")
except Exception as e:
    logger.warning("Failed to initialize Slack bot: %s", e)
    slack_bot = None

def acquire_lock(lock_path: Path):
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.info("Acquired lock: %s", lock_path)
        return lock_fd
    except BlockingIOError:
        logger.info("Another update is already running; skipping this iteration.")
        os.close(lock_fd)
        return None

def clean_directories():
    logger.info("Ensuring directories exist and cleaning *.csv / *.json files...")
    for d in CLEAN_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("  Cleaning: %s", d)
        for pattern in ("*.csv", "*.json"):
            for file_path in d.glob(pattern):
                logger.info("    Deleting: %s", file_path)
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning("    Failed to delete %s: %s", file_path, e)
                    if slack_bot:
                        slack_bot.post_error_to_slack(
                            f"Failed to delete file during cleanup: {file_path}",
                            context={
                                "error": str(e),
                                "directory": str(d),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

def run_script(script_path: Path):
    """Run a script and stream its output to both terminal and log file in real-time."""
    if not script_path.is_file():
        logger.error("Script not found: %s", script_path)
        error_msg = f"Script not found: {script_path}"
        if slack_bot:
            slack_bot.post_error_to_slack(
                error_msg,
                context={
                    "script_path": str(script_path),
                    "project_root": str(PROJECT_ROOT),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        raise FileNotFoundError(error_msg)
    
    logger.info("="*60)
    logger.info("Running: %s", script_path)
    logger.info("="*60)
    
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output_lines = []
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(line)
                output_lines.append(line)
        
        return_code = process.wait()
        
        if return_code != 0:
            error_output = "\n".join(output_lines[-50:])
            raise subprocess.CalledProcessError(
                return_code, 
                [sys.executable, str(script_path)],
                output=error_output
            )
        
        logger.info("Completed: %s (exit code: 0)", script_path.name)
        logger.info("-"*60)
        
    except Exception as e:
        process.kill()
        raise

def run_update_cycle(log_file):
    """Run one complete update cycle"""
    start_time = datetime.now()
    logger.info(f"===== Starting governance update at {PROJECT_ROOT} =====")
    os.chdir(PROJECT_ROOT)

    lock_fd = acquire_lock(LOCK_FILE)
    
    if lock_fd is None:
        logger.info("Skipping this cycle due to lock")
        return

    try:
        clean_directories()

        for idx, script in enumerate(SCRIPTS, 1):
            logger.info(f"\n[{idx}/{len(SCRIPTS)}] Executing script...")
            run_script(script)

        logger.info("===== Update completed successfully =====")
        
        if slack_bot:
            duration = (datetime.now() - start_time).total_seconds()
            slack_bot.post_to_slack({
                "event": "Governance Data Update",
                "status": "SUCCESS ✅",
                "duration_seconds": round(duration, 2),
                "scripts_executed": len(SCRIPTS),
                "timestamp": datetime.now().isoformat(),
                "log_file": str(log_file),
            })
            
    except subprocess.CalledProcessError as e:
        error_msg = f"Script failed with exit code {e.returncode}"
        logger.error(
            "Script failed with exit code %s: %s\nOutput:\n%s",
            e.returncode,
            e.cmd,
            e.output if e.output else                     "No output captured",
        )
        
        if slack_bot:
            slack_bot.post_error_to_slack(
                error_msg,
                context={
                    "exit_code": e.returncode,
                    "command": str(e.cmd),
                    "output": e.output[:1000] if e.output else "No output",
                    "timestamp": datetime.now().isoformat(),
                    "log_file": str(log_file),
                }
            )
        
    except Exception as e:
        error_msg = f"Unexpected error during governance data update: {str(e)}"
        logger.exception("Unexpected error: %s", e)
        
        if slack_bot:
            slack_bot.post_error_to_slack(
                error_msg,
                context={
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "log_file": str(log_file),
                }
            )
        
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except Exception as lock_error:
                logger.warning("Failed to release lock: %s", lock_error)

def main():
    """Main loop that runs updates every 24 hours"""
    logger.info("="*80)
    logger.info("Starting governance data update daemon (runs every 24 hours)")
    logger.info("="*80)
    
    if slack_bot:
        slack_bot.post_to_slack({
            "event": "Governance Update Daemon Started",
            "status": "RUNNING 🚀",
            "interval": "Every 24 hours",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(PROJECT_ROOT),
        })
    
    iteration = 0
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting iteration #{iteration} at {datetime.now()}")
            logger.info(f"{'='*80}\n")
            
            current_logger, log_file = setup_logger()
            
            run_update_cycle(log_file)
            
            next_run = datetime.now() + timedelta(seconds=SLEEP_INTERVAL)
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration #{iteration} completed. Next run at: {next_run}")
            logger.info(f"Sleeping for 24 hours...")
            logger.info(f"{'='*80}\n")
            
            time.sleep(SLEEP_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("\n" + "="*80)
            logger.info("Received shutdown signal. Stopping daemon gracefully...")
            logger.info("="*80)
            
            if slack_bot:
                slack_bot.post_to_slack({
                    "event": "Governance Update Daemon Stopped",
                    "status": "SHUTDOWN 🛑",
                    "total_iterations": iteration,
                    "timestamp": datetime.now().isoformat(),
                })
            break
            
        except Exception as e:
            logger.exception("Unexpected error in main loop: %s", e)
            
            if slack_bot:
                slack_bot.post_error_to_slack(
                    f"Main loop error (continuing): {str(e)}",
                    context={
                        "error_type": type(e).__name__,
                        "iteration": iteration,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            
            logger.info("Continuing to next iteration despite error...")
            time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()