#!/bin/bash
# Run DeLoRA on commonsense reasoning using HuggingFace PEFT's built-in DeLoRA support
# Reference: https://github.com/ExplainableML/DeLoRA + https://huggingface.co/docs/peft/package_reference/delora
#
# DeLoRA is available in PEFT >= 0.12.0 via peft_type="DELORA"
# Install: pip install peft>=0.12.0
#
# Usage: bash run_delora_cr.sh [rank] [alpha] [GPU_ID] [seed]
# Defaults: rank=16, alpha=32, GPU=0, seed=42

RANK=${1:-16}
ALPHA=${2:-32}
GPU=${3:-0}
SEED=${4:-42}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="commonsense_170k.json"

OUTPUT_DIR="./output/delora_cr/llama7b_r${RANK}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"
echo "=== DeLoRA on LLaMA-7B commonsense reasoning, r=$RANK, seed=$SEED ==="

CUDA_VISIBLE_DEVICES=$GPU python finetune_peft.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 200 \
    --adapter_name delora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r "$RANK" --lora_alpha "$ALPHA" \
    --use_gradient_checkpointing \
    --seed "$SEED"

# Evaluate all 8 benchmarks
for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
        --model LLaMA-7B \
        --adapter DeLoRA \
        --dataset "$DATASET" \
        --base_model "$BASE_MODEL" \
        --batch_size 1 \
        --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
done

echo "Results in $OUTPUT_DIR"
