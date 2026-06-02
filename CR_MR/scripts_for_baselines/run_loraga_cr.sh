#!/bin/bash
# Run LoRA-GA on commonsense reasoning
# Official repo: https://github.com/Outsider565/LoRA-GA
#
# Setup:
#   git clone https://github.com/Outsider565/LoRA-GA
#   (follow their installation instructions)
#
# Usage: bash run_loraga_cr.sh [rank] [alpha] [GPU_ID]

RANK=${1:-16}
ALPHA=${2:-32}
GPU=${3:-0}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="$(pwd)/commonsense_170k.json"

LORAGA_DIR="$(dirname "$0")/../../LoRA-GA"

if [ ! -d "$LORAGA_DIR" ]; then
    echo "LoRA-GA not found at $LORAGA_DIR"
    echo "Please run: git clone https://github.com/Outsider565/LoRA-GA $LORAGA_DIR"
    exit 1
fi

OUTPUT_DIR="$(pwd)/output/loraga/llama7b_r${RANK}"
mkdir -p "$OUTPUT_DIR"

echo "=== LoRA-GA on LLaMA-7B commonsense reasoning, r=$RANK ==="

cd "$LORAGA_DIR"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --lora_r $RANK --lora_alpha $ALPHA \
    --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]'

cd -

for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
        --model LLaMA-7B \
        --adapter LoRA \
        --dataset $DATASET \
        --base_model "$BASE_MODEL" \
        --batch_size 1 \
        --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
done
