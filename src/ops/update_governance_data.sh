#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/pa-ai/polkassembly-ai"
LOG_DIR="$PROJECT_ROOT/logs"
LOCK_FILE="$PROJECT_ROOT/.update_governance_data.lock"
VENV_PATH="/home/ubuntu/venv"

CLEAN_DIRS=(
  "/home/ubuntu/pa-ai/polkassembly-ai/data/onchain_data"
  "/home/ubuntu/pa-ai/polkassembly-ai/onchain_data/onchain_first_pull/all_csv"
  "/home/ubuntu/pa-ai/polkassembly-ai/onchain_data/onchain_first_pull/one_table"
  "/home/ubuntu/pa-ai/polkassembly-ai/onchain_data/onchain_first_pull/one_table/filter_data"
)

SCRIPTS=(
  "/home/ubuntu/pa-ai/polkassembly-ai/src/ingestion/onchain/onchain_data.py"
  "/home/ubuntu/pa-ai/polkassembly-ai/src/ingestion/onchain/flatten_all_data.py"
  "/home/ubuntu/pa-ai/polkassembly-ai/src/ingestion/onchain/create_one_table.py"
  "/home/ubuntu/pa-ai/polkassembly-ai/src/ingestion/onchain/filter_data.py"
  "/home/ubuntu/pa-ai/polkassembly-ai/src/ingestion/onchain/insert_into_postgres.py"
)

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/update_$(date +'%Y%m%d_%H%M%S').log"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  echo "[INFO] Activating virtual environment at $VENV_PATH" | tee -a "$LOG_FILE"
  source "$VENV_PATH/bin/activate"
else
  echo "[ERROR] Virtual environment not found at $VENV_PATH" | tee -a "$LOG_FILE"
  exit 1
fi

PYTHON="$(command -v python3)"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date) Another update is already running; exiting." | tee -a "$LOG_FILE"
  exit 0
fi

echo "===== $(date) Starting nightly governance update =====" | tee -a "$LOG_FILE"
cd "$PROJECT_ROOT"

echo "[CLEAN] Ensuring directories exist and deleting *.csv and *.json..." | tee -a "$LOG_FILE"
for d in "${CLEAN_DIRS[@]}"; do
  mkdir -p "$d"
  echo "  - Cleaning: $d" | tee -a "$LOG_FILE"
  find "$d" -type f \( -name '*.csv' -o -name '*.json' \) -print -delete | tee -a "$LOG_FILE" || true
done

for script in "${SCRIPTS[@]}"; do
  if [[ -f "$script" ]]; then
    echo "[RUN] $script" | tee -a "$LOG_FILE"
    "$PYTHON" -u "$script" | tee -a "$LOG_FILE"
  else
    echo "[WARN] Script not found: $script" | tee -a "$LOG_FILE"
    exit 1
  fi
done

echo "===== $(date) Update completed successfully =====" | tee -a "$LOG_FILE"


