#!/bin/bash
# boot_resume.sh — invoked by cron @reboot on every system boot.
# Waits for the system to settle, then ensures both the training chain and
# the watchdog are running inside detached tmux sessions.
#
# Idempotent: if either is already alive, leaves it alone.

set -e

REPO_ROOT="/home/li/Subspace-Tuning"
LOG="/tmp/boot_resume.log"

# cron starts jobs with SHELL=/bin/sh; tmux inherits this when creating panes.
export SHELL=/bin/bash

echo "[$(date)] boot_resume invoked" >> "$LOG"

# 1. Wait until the GPU is ready (nvidia-smi can fail right after boot)
for i in {1..30}; do
    if nvidia-smi >/dev/null 2>&1; then break; fi
    sleep 5
done
echo "[$(date)] GPU ready after ${i} attempts" >> "$LOG"

# 2. Wait for filesystem to fully mount + repo dir to be reachable
for i in {1..10}; do
    if [ -d "$REPO_ROOT" ] && [ -x "$REPO_ROOT/.venv/bin/python" ]; then break; fi
    sleep 5
done

# 3. Ensure tmux server is up
tmux start-server 2>/dev/null || true

# 4. Repair anything reboot-corrupted before either process starts
cd "$REPO_ROOT"
echo "[$(date)] running fix_corrupted_results.sh" >> "$LOG"
bash fix_corrupted_results.sh >> "$LOG" 2>&1 || true

# 5. Start training chain in tmux 'lomap' if not running
if ! tmux has-session -t lomap 2>/dev/null; then
    echo "[$(date)] starting tmux 'lomap'" >> "$LOG"
    tmux new-session -d -s lomap -c "$REPO_ROOT"
    tmux send-keys -t lomap "bash hp_tuning.sh && bash overnight_baselines.sh" Enter
fi

# 6. Start watchdog in tmux 'watchdog' if not running.
# IMPORTANT: only clean stale pid/lock/dead files when the recorded PID is gone.
# Deleting them while a live watchdog holds them defeats flock and could
# allow a second watchdog to start.
if [ -f /tmp/watchdog_lomap.pid ]; then
    OLD_WD_PID=$(cat /tmp/watchdog_lomap.pid 2>/dev/null || echo "")
    if [ -n "$OLD_WD_PID" ] && kill -0 "$OLD_WD_PID" 2>/dev/null; then
        echo "[$(date)] watchdog already alive (pid $OLD_WD_PID); skipping" >> "$LOG"
    else
        echo "[$(date)] stale pid file (pid $OLD_WD_PID dead); cleaning" >> "$LOG"
        rm -f /tmp/watchdog_lomap.pid /tmp/watchdog_lomap.lock /tmp/watchdog_lomap.dead
    fi
fi

if ! tmux has-session -t watchdog 2>/dev/null; then
    # Final guard: ensure no stale watchdog process exists with a different pid file
    if pgrep -f "bash.*watchdog_lomap.sh" >/dev/null; then
        echo "[$(date)] watchdog process exists outside expected pid file; not starting another" >> "$LOG"
    else
        echo "[$(date)] starting tmux 'watchdog'" >> "$LOG"
        tmux new-session -d -s watchdog -c "$REPO_ROOT"
        tmux send-keys -t watchdog "bash watchdog_lomap.sh" Enter
    fi
fi

echo "[$(date)] boot_resume done" >> "$LOG"
