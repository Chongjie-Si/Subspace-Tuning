#!/bin/bash
# submit_all.sh — submit the full LoMAP AAAI experiment suite to PBS.
#
# Usage:
#   bash submit_all.sh [REPO_ROOT_ON_CLUSTER]
#
# Submission order (qsub returns immediately; PBS schedules):
#   1. nlu_base_r2.pbs        — NLU Table 1 (DeBERTa-v3-base, r=2)
#   2. nlu_base_r8.pbs        — NLU Table 1 (DeBERTa-v3-base, r=8)
#   3. nlu_large_r2.pbs       — NLU Table 1 (DeBERTa-v3-large, r=2)
#   4. nlu_large_r8.pbs       — NLU Table 1 (DeBERTa-v3-large, r=8)
#   5. cr_llama7b.pbs         — CR Table 2 (LLaMA-7B, r=4/8/16/32)
#   6. cr_llama3_8b.pbs       — CR Table 2 (LLaMA3-8B, r=16/32)
#   7. ablation.pbs           — §5 ablations (β₀/ε/detach/norm_scope)
#
# Total wall time per job: 12-24h on 8×H100. Submit all in parallel — PBS handles queueing.

set -e
REPO_ROOT=${1:-$(pwd)}
KIT="$REPO_ROOT/deploy/h100_kit/pbs_templates"

mkdir -p "$REPO_ROOT/logs"

submit_with_root() {
    local pbs="$1"
    qsub -v REPO_ROOT="$REPO_ROOT" "$pbs"
}

# NLU Table 1
J1=$(submit_with_root "$KIT/nlu_base_r2.pbs")
J2=$(submit_with_root "$KIT/nlu_base_r8.pbs")
J3=$(submit_with_root "$KIT/nlu_large_r2.pbs")
J4=$(submit_with_root "$KIT/nlu_large_r8.pbs")

# CR Table 2
J5=$(submit_with_root "$KIT/cr_llama7b.pbs")
J6=$(submit_with_root "$KIT/cr_llama3_8b.pbs")

# §5 ablations
J7=$(submit_with_root "$KIT/ablation.pbs")

echo ""
echo "Submitted jobs:"
echo "  NLU base r=2:    $J1"
echo "  NLU base r=8:    $J2"
echo "  NLU large r=2:   $J3"
echo "  NLU large r=8:   $J4"
echo "  CR LLaMA-7B:     $J5"
echo "  CR LLaMA3-8B:    $J6"
echo "  Ablations:       $J7"
echo ""
echo "Watch with: qstat -u $USER"
echo "Logs in:    $REPO_ROOT/logs/"
echo "Summary:    .venv/bin/python $REPO_ROOT/summarize_nlu.py"
