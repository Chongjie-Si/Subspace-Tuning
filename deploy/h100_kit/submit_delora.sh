#!/bin/bash
# submit_delora.sh — submit all DeLoRA comparison experiments on H100.
#
# Usage (on H100 head node, after h100_setup.sh):
#   bash deploy/h100_kit/submit_delora.sh
#
# What this runs (in parallel on 8 GPUs):
#   1. DeLoRA NLU  DeBERTa-base  r=2,  8 tasks × 3 seeds  (~15 GPU·h)
#   2. DeLoRA CR   LLaMA-7B      r=16 + r=32              (~20 GPU·h)
#
# Total wall-clock on 8×H100: ~4-5 h (both run concurrently on separate GPUs).
# Total GPU·h: ~35.
#
# Results land in:
#   NLU/output/glue/<size>_delora_<task>_r<R>_seed<S>/model/all_results.json
#   CR_MR/output/llama-7b_delora_r<R>_a<A>/<bench>.txt

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS="$REPO_ROOT/deploy/h100_kit/scripts"

source "$REPO_ROOT/.venv/bin/activate"

# Upgrade peft to >= 0.14 for DeLoRA support (idempotent)
echo ">>> Ensuring peft >= 0.14 ..."
pip install "peft>=0.14" --quiet --upgrade

export PYTHONPATH="$REPO_ROOT/NLU/src:$REPO_ROOT/loralib:$PYTHONPATH"

echo ""
echo "================================================"
echo "Launching DeLoRA NLU (GPUs 0-3) in background"
echo "================================================"
bash "$SCRIPTS/run_delora_nlu_grid.sh" base 2 6,7,8 0,1,2,3 &
NLU_PID=$!

echo ""
echo "================================================"
echo "Launching DeLoRA CR  (GPUs 4-7) in background"
echo "================================================"
bash "$SCRIPTS/run_delora_cr.sh" 4,5,6,7 &
CR_PID=$!

echo ""
echo "Both jobs launched. Waiting..."
wait $NLU_PID && echo "NLU done." || echo "NLU FAILED (check logs/delora_nlu_failures.log)"
wait $CR_PID  && echo "CR done."  || echo "CR FAILED  (check logs/cr_failures.log)"

echo ""
echo "================================================"
echo "All DeLoRA experiments complete."
echo "Summary:"
echo "  NLU: python summarize_nlu.py"
echo "  CR:  python CR_MR/scripts_for_baselines/aggregate_results.py CR_MR/output"
echo "================================================"
