#!/bin/bash
# Run RandLoRA and GraLoRA (via HF PEFT) on commonsense reasoning
# Both are available in PEFT >= 0.13.0
# Reference:
#   RandLoRA: https://huggingface.co/docs/peft/package_reference/randlora
#   GraLoRA:  https://huggingface.co/docs/peft/package_reference/gralora
#
# Usage: bash run_randlora_gralora_cr.sh [rank] [GPU_ID]

RANK=${1:-16}
GPU=${2:-0}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="commonsense_170k.json"

for ADAPTER in randlora gralora; do
    OUTPUT_DIR="./output/${ADAPTER}/llama7b_r${RANK}"
    echo "=== $ADAPTER on LLaMA-7B commonsense reasoning, r=$RANK ==="
    CUDA_VISIBLE_DEVICES=$GPU python finetune_peft.py \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 \
        --adapter_name $ADAPTER \
        --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]' \
        --lora_r $RANK --lora_alpha $((RANK*2)) \
        --use_gradient_checkpointing

    for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
        CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
            --model LLaMA-7B \
            --adapter LoRA \
            --dataset $DATASET \
            --base_model "$BASE_MODEL" \
            --batch_size 1 \
            --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
    done
done
