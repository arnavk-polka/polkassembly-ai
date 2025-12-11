#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import logging
import fcntl
import os
import time
from dotenv import load_dotenv

# ----------------- LOAD ENV -----------------
load_dotenv()

##############################################
#             PROJECT ROOT RESOLUTION        #
##############################################

# 1️⃣ If BASE_PATH is defined in .env, use it
BASE_PATH = os.getenv("BASE_PATH")

if BASE_PATH:
    PROJECT_ROOT = Path(BASE_PATH).expanduser().resolve()
else:
    # 2️⃣ Auto-detect project root from file location:
    # update_governance_data.py → /project/src/scripts/
    # PROJECT_ROOT = parent of parent of parent
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

print(f"[INFO] Using PROJECT_ROOT = {PROJECT_ROOT}")

##############################################
#                PATH DEFINITIONS            #
##############################################

LOG_DIR = PROJECT_ROOT / "logs"
LOCK_FILE = PROJECT_ROOT / ".update_governance_data.lock"

CLEAN_DIRS = [
    PROJECT_ROOT / "data/onchain_data",
    PROJECT_ROOT / "onchain_data/onchain_first_pull/all_csv",
    PROJECT_ROOT / "onchain_data/onchain_first_pull/one_table",
    PROJECT_ROOT / "onchain_data/onchain_first_pull/one_table/filter_data",
]

SCRIPTS = [
    PROJECT_ROOT / "src/data/onchain_data.py",
    PROJECT_ROOT / "src/texttosql/flatten_all_data.py",
    PROJECT_ROOT / "src/texttosql/create_one_table.py",
    PROJECT_ROOT / "src/texttosql/filter_data.py",
    PROJECT_ROOT / "src/texttosql/insert_into_postgres.py",
]

# Daemon run frequency (24 hours)
SLEEP_INTERVAL = 24 * 60 * 60

# ----------------- LOGGING SETUP -----------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger():
    """Setup logger with rotating logfile."""
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

# ----------------- SLACK BOT -----------------

try:
    # Load SlackBot dynamically from project root
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils.slack_bot import SlackBot
    slack_bot = SlackBot()
    logger.info("Slack bot initialized successfully")
except Exception as e:
    logger.warning("Failed to initialize Slack bot: %s", e)
    slack_bot = None

# ----------------- LOCKING -----------------

def acquire_lock(lock_path: Path):
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.info("Acquired lock: %s", lock_path)
        return lock_fd
    except BlockingIOError:
        logger.info("Another update is already running; skipping.")
        os.close(lock_fd)
        return None

# ----------------- CLEAN DIRECTORIES -----------------

def clean_directories():
    logger.info("Cleaning CSV/JSON directories...")
    for d in CLEAN_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cleaning: {d}")
        for pattern in ("*.csv", "*.json"):
            for file_path in d.glob(pattern):
                try:
                    file_path.unlink()
                    logger.info(f"  Deleted {file_path}")
                except Exception as e:
                    logger.warning(f"Failed deleting {file_path}: {e}")
                    if slack_bot:
                        slack_bot.post_error_to_slack(
                            f"Failed to delete file: {file_path}",
                            context={"error": str(e)}
                        )

# ----------------- RUN CHILD SCRIPTS -----------------

def run_script(script_path: Path):
    """Runs a child script and streams real-time output."""
    if not script_path.is_file():
        msg = f"Script not found: {script_path}"
        logger.error(msg)
        if slack_bot:
            slack_bot.post_error_to_slack(msg)
        raise FileNotFoundError(msg)

    logger.info("=" * 60)
    logger.info(f"Running: {script_path}")
    logger.info("=" * 60)

    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    output_lines = []
    for line in process.stdout:
        print(line.rstrip())
        output_lines.append(line.rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code, [sys.executable, str(script_path)],
            output="\n".join(output_lines[-50:])
        )

    logger.info(f"Finished: {script_path.name} (exit 0)")

# ----------------- UPDATE CYCLE -----------------

def run_update_cycle(log_file):
    logger.info(f"===== Starting governance update at {PROJECT_ROOT} =====")

    os.chdir(PROJECT_ROOT)
    lock_fd = acquire_lock(LOCK_FILE)

    if lock_fd is None:
        return

    try:
        clean_directories()

        for idx, script in enumerate(SCRIPTS, 1):
            logger.info(f"[{idx}/{len(SCRIPTS)}] Running script...")
            run_script(script)

        logger.info("===== Update completed successfully =====")

        if slack_bot:
            slack_bot.post_to_slack({
                "event": "Governance Update",
                "status": "SUCCESS",
                "scripts": len(SCRIPTS),
                "timestamp": datetime.now().isoformat(),
                "log_file": str(log_file),
            })

    except subprocess.CalledProcessError as e:
        errmsg = f"Script failed (exit {e.returncode}): {e.output}"
        logger.error(errmsg)
        if slack_bot:
            slack_bot.post_error_to_slack(errmsg)

    except Exception as e:
        errmsg = f"Unexpected error: {e}"
        logger.exception(errmsg)
        if slack_bot:
            slack_bot.post_error_to_slack(errmsg)

    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except Exception as e:
            logger.warning(f"Failed releasing lock: {e}")

# ----------------- MAIN LOOP -----------------

def main():
    logger.info("=" * 80)
    logger.info("Governance update daemon starting...")
    logger.info("=" * 80)

    if slack_bot:
        slack_bot.post_to_slack({
            "event": "Daemon Started",
            "status": "RUNNING",
            "interval": "24 hours",
            "timestamp": datetime.now().isoformat(),
        })

    iteration = 0

    while True:
        iteration += 1
        logger.info(f"\n===== Iteration #{iteration} at {datetime.now()} =====")

        current_logger, log_file = setup_logger()
        run_update_cycle(log_file)

        next_run = datetime.now() + timedelta(seconds=SLEEP_INTERVAL)
        logger.info(f"Next run at {next_run}. Sleeping...")

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
