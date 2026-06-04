#!/bin/bash
# overnight_baselines.sh — fill in LoRA baselines + missing LoMAP runs
# on the 4080. Designed to run unattended overnight.
#
# Order: small/fast tasks first (gives results quickly even if interrupted)
# Each run skips automatically if all_results.json already exists.

set -e
cd /home/li/Subspace-Tuning
LOG=/tmp/overnight_baselines.log
echo "[$(date)] === overnight_baselines start ===" | tee -a $LOG

is_done() {
    local f="NLU/output/glue/${1}_${2}_r2_seed${3}/model/all_results.json"
    [ -f "$f" ]
}

run_one() {
    local method=$1 task=$2 seed=$3
    if is_done "$method" "$task" "$seed"; then
        echo "[$(date)] SKIP $method $task seed=$seed (done)" | tee -a $LOG
        return 0
    fi
    echo "[$(date)] RUN  $method $task seed=$seed" | tee -a $LOG
    bash run_nlu_local.sh "$method" 2 4 "$seed" "$task" 0 >> $LOG 2>&1
}

# Phase 1: small tasks, all seeds — fastest, most informative
# LoRA on cola, mrpc, rte, stsb × 3 seeds = 12 runs (~2.2h)
# LoMAP fill-in: stsb 3 seeds, rte seed 7,8 (~55min)
PHASE1=(
    "lora cola 6" "lora cola 7" "lora cola 8"
    "lora mrpc 6" "lora mrpc 7" "lora mrpc 8"
    "lora rte  6" "lora rte  7" "lora rte  8"
    "lora stsb 6" "lora stsb 7" "lora stsb 8"
    "map  rte  7" "map  rte  8"
    "map  stsb 6" "map  stsb 7" "map  stsb 8"
    "lora sst2 6" "lora sst2 7" "lora sst2 8"
)

for entry in "${PHASE1[@]}"; do
    read -r m t s <<< "$entry"
    run_one "$m" "$t" "$s"
done

echo "[$(date)] === Phase 1 done ===" | tee -a $LOG
.venv/bin/python summarize_nlu.py | tee -a $LOG

# Phase 2 & 3 (qnli, sst2 LoRA) are DEFERRED TO H100.
# On the 16GB 4080, qnli (seq 512) needs train_bs=8 + accum=4 → ~4h/run.
# That's 24h just for qnli×6 + huge time for sst2 LoRA×3 — not worth local time.
# These large-task baselines run on H100 via deploy/h100_kit/.
# The 4080's remaining time is better spent on HP tuning (Phase A+B), which is
# the highest-value, fastest-iterating work for the paper.
#
# To re-enable locally, uncomment the blocks below.

# PHASE2=( "lora qnli 6" "lora qnli 7" "lora qnli 8" "map qnli 6" "map qnli 7" "map qnli 8" )
# for entry in "${PHASE2[@]}"; do read -r m t s <<< "$entry"; run_one "$m" "$t" "$s"; done

# PHASE3=( "lora sst2 6" "lora sst2 7" "lora sst2 8" )
# for entry in "${PHASE3[@]}"; do read -r m t s <<< "$entry"; run_one "$m" "$t" "$s"; done

echo "[$(date)] === Baselines done (large tasks deferred to H100) ===" | tee -a $LOG
.venv/bin/python summarize_nlu.py | tee -a $LOG
