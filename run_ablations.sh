#!/bin/bash
# run_ablations.sh — run all MAP ablation experiments (CR, LLaMA-7B)
#
# Requires: conda activate lomap-cr
# Run from repo root.
#
# Usage:
#   bash run_ablations.sh all    0    # all ablations, GPU=0
#   bash run_ablations.sh eps    0    # only epsilon ablation
#   bash run_ablations.sh beta   0
#   bash run_ablations.sh rank   0
#   bash run_ablations.sh norm   0
#   bash run_ablations.sh detach 0

set -e
TARGET=${1:-all}
GPU=${2:-0}
SEED=42

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CR_DIR="$REPO_ROOT/CR_MR"

cd "$CR_DIR"

run_ablation() {
    echo ""; echo "========== Ablation: $1 =========="
    bash "scripts_for_ablation/$1" "$GPU" "$SEED"
}

case "$TARGET" in
    eps)    run_ablation ablation_epsilon.sh ;;
    beta)   run_ablation ablation_beta0.sh ;;
    rank)   run_ablation ablation_rank.sh ;;
    norm)   run_ablation ablation_norm_scope.sh ;;
    detach) run_ablation ablation_detach.sh ;;
    all)
        run_ablation ablation_epsilon.sh
        run_ablation ablation_beta0.sh
        run_ablation ablation_norm_scope.sh
        run_ablation ablation_detach.sh
        run_ablation ablation_rank.sh
        ;;
    *)
        echo "Usage: bash run_ablations.sh [eps|beta|rank|norm|detach|all] [GPU_ID]"
        exit 1 ;;
esac

echo ""; echo "All ablations done. Aggregate with:"
echo "  python scripts_for_baselines/aggregate_results.py --method ablation/eps_1e-6 --rank 16 --seeds $SEED"
