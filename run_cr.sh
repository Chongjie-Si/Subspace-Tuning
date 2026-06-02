#!/bin/bash
# run_cr.sh — launch LoMAP / baseline experiments for Commonsense Reasoning
#
# Must be run from the repo root, or pass REPO_ROOT explicitly.
# Requires: conda activate lomap-cr
#
# Usage:
#   bash run_cr.sh lomap  16 32 0 42        # LoMAP r=16, GPU=0, seed=42
#   bash run_cr.sh lora   16 32 0 42        # LoRA baseline
#   bash run_cr.sh delora 16 32 0 42        # DeLoRA baseline
#   bash run_cr.sh loraga 16 32 0 42        # LoRA-GA baseline
#   bash run_cr.sh randlora 16 32 0 42
#   bash run_cr.sh gralora  16 32 0 42
#
# Multi-seed (3 seeds, sequential):
#   for SEED in 6 42 123; do bash run_cr.sh lomap 16 32 0 $SEED; done
#
# After all seeds finish, aggregate:
#   python CR_MR/scripts_for_baselines/aggregate_results.py \
#       --method lomap --rank 16 --seeds 6 42 123

set -e
ADAPTER=${1:-lomap}
RANK=${2:-16}
ALPHA=${3:-32}
GPU=${4:-0}
SEED=${5:-42}

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CR_DIR="$REPO_ROOT/CR_MR"
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="$CR_DIR/commonsense_170k.json"
OUTPUT_DIR="$CR_DIR/output/${ADAPTER}/llama7b_r${RANK}_seed${SEED}"
DATASETS="boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa"

mkdir -p "$OUTPUT_DIR"
cd "$CR_DIR"

echo "================================================"
echo "Task : Commonsense Reasoning"
echo "Adapter: $ADAPTER  rank=$RANK  alpha=$ALPHA"
echo "GPU=$GPU  seed=$SEED"
echo "Output: $OUTPUT_DIR"
echo "================================================"

# ── Training ────────────────────────────────────
if [ "$ADAPTER" = "lomap" ] || [ "$ADAPTER" = "lora" ] || [ "$ADAPTER" = "dora" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python finetune.py \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 200 \
        --adapter_name "$ADAPTER" \
        --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]' \
        --lora_r "$RANK" --lora_alpha "$ALPHA" \
        --use_gradient_checkpointing \
        --seed "$SEED"
else
    # DeLoRA, LoRA-GA, RandLoRA, GraLoRA via HF PEFT
    CUDA_VISIBLE_DEVICES=$GPU python finetune_peft.py \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 200 \
        --adapter_name "$ADAPTER" \
        --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]' \
        --lora_r "$RANK" --lora_alpha "$ALPHA" \
        --use_gradient_checkpointing \
        --seed "$SEED"
fi

# ── Evaluation ──────────────────────────────────
# Map adapter name to the --adapter flag expected by commonsense_evaluate.py
case "$ADAPTER" in
    lomap)   EVAL_ADAPTER="LoMAP" ;;
    lora)    EVAL_ADAPTER="LoRA" ;;
    dora)    EVAL_ADAPTER="DoRA" ;;
    delora)  EVAL_ADAPTER="DeLoRA" ;;
    loraga)  EVAL_ADAPTER="LoRA-GA" ;;
    randlora) EVAL_ADAPTER="RandLoRA" ;;
    gralora) EVAL_ADAPTER="GraLoRA" ;;
    *)       EVAL_ADAPTER="LoRA" ;;
esac

echo "--- Evaluating with --adapter $EVAL_ADAPTER ---"
for DATASET in $DATASETS; do
    CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
        --model LLaMA-7B \
        --adapter "$EVAL_ADAPTER" \
        --dataset "$DATASET" \
        --base_model "$BASE_MODEL" \
        --batch_size 1 \
        --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
done

echo "Done. Results in $OUTPUT_DIR"
