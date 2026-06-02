#!/bin/bash
# run_nlu_continue.sh — run remaining NLU tasks after cola is done
# Designed to be launched after the initial cola 3-seed run completes.

set -e
GPU=${1:-0}
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "================================================"
echo "Continuing NLU suite (skip cola, start sst2)"
echo "GPU=$GPU"
echo "================================================"

# LoMAP r=2, remaining tasks after cola
for task in sst2 mrpc rte stsb qnli qqp mnli; do
    echo ""; echo ">>> LoMAP r=2  task=$task  seeds=6,7,8"
    bash "$REPO_ROOT/run_nlu_local.sh" map 2 4 "6,7,8" "$task" "$GPU"
done

# LoRA r=2, all 8 tasks
for task in cola sst2 mrpc rte stsb qnli qqp mnli; do
    echo ""; echo ">>> LoRA r=2  task=$task  seeds=6,7,8"
    bash "$REPO_ROOT/run_nlu_local.sh" lora 2 4 "6,7,8" "$task" "$GPU"
done

echo ""
echo "================================================"
echo "All r=2 experiments done. Summary:"
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/summarize_nlu.py"
echo "================================================"
