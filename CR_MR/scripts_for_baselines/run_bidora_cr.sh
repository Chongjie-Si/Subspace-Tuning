#!/bin/bash
# Run BiDoRA on commonsense reasoning
# Uses the official BiDoRA implementation: https://github.com/t2ance/BiDoRA
#
# Setup:
#   git clone https://github.com/t2ance/BiDoRA
#   pip install -e BiDoRA/
#   (or copy BiDoRA's training script and use the same data/hyper-params below)
#
# Usage: bash run_bidora_cr.sh [rank] [alpha] [GPU_ID]

RANK=${1:-32}
ALPHA=${2:-64}
GPU=${3:-0}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="$(pwd)/commonsense_170k.json"

# Assumes BiDoRA was cloned to ../../BiDoRA relative to this file
BIDORA_DIR="$(dirname "$0")/../../BiDoRA"

if [ ! -d "$BIDORA_DIR" ]; then
    echo "BiDoRA not found at $BIDORA_DIR"
    echo "Please run: git clone https://github.com/t2ance/BiDoRA $BIDORA_DIR"
    exit 1
fi

OUTPUT_DIR="$(pwd)/output/bidora/llama7b_r${RANK}"
mkdir -p "$OUTPUT_DIR"

echo "=== BiDoRA on LLaMA-7B commonsense reasoning, r=$RANK ==="

# BiDoRA training — adjust the script name to match their repo
cd "$BIDORA_DIR"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --lora_r $RANK --lora_alpha $ALPHA \
    --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]'

cd -

# Evaluate
for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
        --model LLaMA-7B \
        --adapter LoRA \
        --dataset $DATASET \
        --base_model "$BASE_MODEL" \
        --batch_size 1 \
        --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
done
