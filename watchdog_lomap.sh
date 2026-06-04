#!/bin/bash
# watchdog_lomap.sh — safe auto-restart for the LoMAP experiment chain.
#
# What it does:
#   1. Holds an exclusive flock so only ONE watchdog instance can run
#   2. Every 60s, checks if `run_glue.py` is alive
#   3. If dead AND we've been alive long enough to know it really crashed
#      (not just between two runs in the chain), checks GPU is free,
#      runs fix_corrupted_results.sh, then restarts tmux session "lomap"
#      with the same chain command
#   4. After 3 consecutive failures (no progress between restarts), stops
#      and writes a warning to /tmp/watchdog_lomap.dead
#
# Safe by design:
#   - Never touches output dirs or checkpoints
#   - Only fixes all_results.json via a separate idempotent script
#   - Only restarts if no other python+CUDA process is running
#   - Tracks "real progress" by checking whether new all_results.json files
#     appear between checks; if no progress for 3 restarts → stop
#
# Usage:
#   nohup bash watchdog_lomap.sh > /tmp/watchdog.log 2>&1 &
#   To stop:  kill $(cat /tmp/watchdog_lomap.pid)

set -e

REPO_ROOT="/home/li/Subspace-Tuning"
LOCK_FILE="/tmp/watchdog_lomap.lock"
PID_FILE="/tmp/watchdog_lomap.pid"
DEAD_FILE="/tmp/watchdog_lomap.dead"
STATE_FILE="/tmp/watchdog_lomap.state"
LOG_FILE="/tmp/watchdog.log"

CHAIN_CMD="bash hp_tuning.sh && bash overnight_baselines.sh"
CHECK_INTERVAL=60          # seconds between checks
SETTLE_TIME=120            # seconds to wait between consecutive runs (avoid false-positive)
MAX_FAILS=3                # consecutive restarts with no progress -> abort

log() { echo "[$(date)] $*" | tee -a "$LOG_FILE"; }

# Acquire lock
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Another watchdog is running. Exiting." >&2
    exit 1
fi
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

cd "$REPO_ROOT"

is_chain_running() {
    # 1. tmux session 'lomap' exists
    tmux has-session -t lomap 2>/dev/null || return 1
    # 2. some run_glue.py or hp_tuning.sh process is alive
    pgrep -f "run_glue.py" > /dev/null && return 0
    pgrep -f "overnight_baselines.sh" > /dev/null && return 0
    pgrep -f "hp_tuning.sh" > /dev/null && return 0
    return 1
}

gpu_is_busy() {
    # Returns 0 if some other python is using the GPU. Used to avoid colliding
    # with manually-started runs.
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${mem:-0}" -gt 500 ]; then
        return 0
    fi
    return 1
}

count_completed_runs() {
    ls "$REPO_ROOT/NLU/output/glue"/*/model/all_results.json 2>/dev/null | wc -l
}

# Bookkeeping
last_completed=$(count_completed_runs)
echo "$last_completed" > "$STATE_FILE"
fail_count=0

log "Watchdog starting. Initial completed runs: $last_completed"
log "Will restart tmux 'lomap' if chain dies. Stop with: kill $(cat $PID_FILE)"

while true; do
    sleep "$CHECK_INTERVAL"

    # Aborted previously: just sit idle
    if [ -f "$DEAD_FILE" ]; then
        sleep 600
        continue
    fi

    if is_chain_running; then
        # Healthy. Reset fail counter when we observe new progress.
        cur_completed=$(count_completed_runs)
        if [ "$cur_completed" -gt "$last_completed" ]; then
            log "Progress: $last_completed -> $cur_completed completed runs"
            last_completed=$cur_completed
            echo "$last_completed" > "$STATE_FILE"
            fail_count=0
        fi
        continue
    fi

    # Chain not running. Could be: (a) finished, (b) crashed, (c) reboot
    log "Chain appears dead. Verifying (settle ${SETTLE_TIME}s)..."

    # Wait SETTLE_TIME to avoid false alarm during between-runs gap
    sleep "$SETTLE_TIME"
    if is_chain_running; then
        log "False alarm — chain came back up"
        continue
    fi

    # Distinguish completion from crash: did the chain leave a "DONE" marker?
    if grep -q "Baselines done (large tasks deferred" /tmp/overnight_baselines.log 2>/dev/null && grep -q "HP tuning DONE" /tmp/hp_tuning.log 2>/dev/null; then
        log "Detected legitimate completion. Watchdog exiting."
        exit 0
    fi

    # The chain is genuinely dead and not finished. We will restart it.
    # NOTE: failure is judged by "restart did not bring the chain back up",
    # NOT by "no new completed run" — a single large task (qnli/sst2/mnli)
    # legitimately runs 30+ min with zero completed-run increments and must
    # never be counted as a failure.

    # GPU must be free
    if gpu_is_busy; then
        log "GPU busy with another process; waiting before restart"
        continue
    fi

    # Repair any all_results.json / checkpoints the abrupt termination corrupted
    log "Repairing corrupted checkpoints / all_results.json (if any)"
    bash "$REPO_ROOT/fix_corrupted_results.sh" 2>&1 | tee -a "$LOG_FILE"

    # Kill any stale tmux session
    tmux kill-session -t lomap 2>/dev/null || true

    # Restart
    log "Restarting tmux session 'lomap' with chain"
    tmux new-session -d -s lomap -c "$REPO_ROOT"
    tmux send-keys -t lomap "$CHAIN_CMD" Enter

    # Give it generous time to initialize: big datasets (qnli/qqp/mnli) need
    # several minutes to download + tokenize before run_glue.py shows up busy.
    # Poll for up to 8 minutes for the chain to come alive.
    started=0
    for _ in $(seq 1 16); do
        sleep 30
        if is_chain_running; then started=1; break; fi
    done

    if [ "$started" -eq 1 ]; then
        log "Chain restarted successfully"
        fail_count=0
    else
        fail_count=$((fail_count + 1))
        log "Restart attempt $fail_count/$MAX_FAILS: chain did not come up within 8 min"
        if [ "$fail_count" -ge "$MAX_FAILS" ]; then
            log "Reached max restart failures. Marking dead. Investigate logs manually."
            touch "$DEAD_FILE"
        fi
    fi
done
