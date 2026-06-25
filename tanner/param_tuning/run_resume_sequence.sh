#!/usr/bin/env bash
set -u
cd "$HOME/dynamic-honey/tanner/param_tuning" || exit 1
source ../.venv/bin/activate

LOG_DIR="$HOME/dynamic-honey/tanner/param_tuning/resume_logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
SUMMARY="$LOG_DIR/summary-$STAMP.log"

echo "Resume sequence started: $(date)" | tee "$SUMMARY"

run_step () {
    local label="$1"
    shift
    local logfile="$LOG_DIR/${label}-$STAMP.log"
    echo "=== [$label] START: $(date) ===" | tee -a "$SUMMARY"
    echo "cmd: $*" | tee -a "$SUMMARY"
    "$@" > "$logfile" 2>&1
    local rc=$?
    echo "=== [$label] EXIT CODE: $rc (log: $logfile) ===" | tee -a "$SUMMARY"
    return 0
}

run_step "run2_familyA"  python3 run_param_matrix.py --families A   --run-name run2 --resume --execute
run_step "run3_familyB"  python3 run_param_matrix.py --families B   --run-name run3 --resume --execute
run_step "run4_familyCD" python3 run_param_matrix.py --families C,D --run-name run4 --resume --execute

echo "Resume sequence finished: $(date)" | tee -a "$SUMMARY"
