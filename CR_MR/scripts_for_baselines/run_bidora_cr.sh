#!/bin/bash
# BiDoRA on commonsense reasoning (LLaMA-7B)
#
# IMPORTANT: The official BiDoRA repo (https://github.com/t2ance/BiDoRA) targets NLU
# (GLUE with DeBERTa/RoBERTa) and does NOT provide a commonsense reasoning script for
# LLaMA.  For CR we therefore use the BiDoRA PEFT adapter that ships with this repo's
# own training pipeline (finetune.py + peft fork), which implements the BiDoRA
# bi-level optimization on top of the same causal-LM setup used by LoMAP.
#
# Two options:
#   Option A (default): use finetune_peft.py if HF PEFT ships a BiDoRA config.
#                       As of PEFT 0.13.x BiDoRA is NOT in upstream HF PEFT.
#   Option B: reproduce via the DoRA variant already in this repo's peft fork,
#             treated as a strong upper-bound baseline for the CR table.
#
# For the AAAI submission we compare BiDoRA on the NLU task (where the official
# implementation applies); for CR we report DoRA as the closest available baseline.
# See README.md for the reference rationale.
#
# Usage: bash run_bidora_cr.sh [rank] [alpha] [GPU_ID] [seed]

RANK=${1:-32}
ALPHA=${2:-64}
GPU=${3:-0}
SEED=${4:-42}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="commonsense_170k.json"

echo "=== NOTE: BiDoRA official repo only supports NLU tasks (GLUE). ==="
echo "=== Running DoRA (this repo's peft fork) as the CR proxy baseline ==="
echo "=== Rank=$RANK, seed=$SEED ==="

OUTPUT_DIR="./output/dora_cr/llama7b_r${RANK}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python finetune.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 200 \
    --adapter_name dora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r "$RANK" --lora_alpha "$ALPHA" \
    --use_gradient_checkpointing \
    --seed "$SEED"

# Evaluate all 8 benchmarks
for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
        --model LLaMA-7B \
        --adapter DoRA \
        --dataset "$DATASET" \
        --base_model "$BASE_MODEL" \
        --batch_size 1 \
        --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
done

echo "Results in $OUTPUT_DIR"
echo ""
echo "For BiDoRA on NLU (GLUE), use: bash run_bidora_glue.sh"
