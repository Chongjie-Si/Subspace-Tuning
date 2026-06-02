#!/bin/bash
# run_full_nlu_suite.sh — run complete NLU suite: LoMAP + LoRA, r=2, 3 seeds, all 8 tasks
# This is the full experiment suite to reproduce/verify paper Table 1.
#
# Usage:
#   bash run_full_nlu_suite.sh [gpu]    # default GPU=0
#
# Estimated time: ~6-8 hours (serial on 1 GPU)
# Results appear live via: .venv/bin/python summarize_nlu.py

set -e
GPU=${1:-0}
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "================================================"
echo "Full NLU suite: LoMAP + LoRA  r=2  seeds=6,7,8"
echo "GPU=$GPU"
echo "================================================"

TASKS="cola sst2 mrpc rte stsb qnli qqp mnli"

# LoMAP r=2, all tasks, 3 seeds
for task in $TASKS; do
    echo ""
    echo ">>> LoMAP r=2  task=$task  seeds=6,7,8"
    bash "$REPO_ROOT/run_nlu_local.sh" map 2 4 "6,7,8" "$task" "$GPU"
done

# LoRA r=2, all tasks, 3 seeds
for task in $TASKS; do
    echo ""
    echo ">>> LoRA r=2  task=$task  seeds=6,7,8"
    bash "$REPO_ROOT/run_nlu_local.sh" lora 2 4 "6,7,8" "$task" "$GPU"
done

echo ""
echo "================================================"
echo "All experiments done. Final summary:"
echo "================================================"
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/summarize_nlu.py"
