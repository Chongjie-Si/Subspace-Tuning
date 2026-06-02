#!/bin/bash
# LoRA-GA on commonsense reasoning (LLaMA-7B)
# Official repo: https://github.com/Outsider565/LoRA-GA
# Reference: Wang et al. "LoRA-GA: Low-Rank Adaptation with Gradient Approximation" (NeurIPS 2024)
#
# LoRA-GA initializes A and B matrices using gradient information (SVD of the
# estimated gradient of W at step 0), rather than Kaiming/zero initialization.
# This is available in HF PEFT >= 0.12.0 via:
#   LoraConfig(init_lora_weights="lora-ga")
#
# The official repo (Outsider565/LoRA-GA) uses Hydra config management and targets
# NLU/instruction-tuning benchmarks. For commonsense reasoning on LLaMA-7B we use
# HF PEFT's LoRA-GA implementation inside our standard finetune_peft.py pipeline.
#
# Usage: bash run_loraga_cr.sh [rank] [alpha] [GPU_ID] [seed]

RANK=${1:-16}
ALPHA=${2:-32}
GPU=${3:-0}
SEED=${4:-42}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="commonsense_170k.json"

OUTPUT_DIR="./output/loraga_cr/llama7b_r${RANK}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "=== LoRA-GA on LLaMA-7B commonsense reasoning, r=$RANK, seed=$SEED ==="
echo "=== Using HF PEFT init_lora_weights=lora-ga (PEFT >= 0.12.0) ==="

# finetune_peft.py supports lora-ga via --adapter_name loraga
CUDA_VISIBLE_DEVICES=$GPU python finetune_peft.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 200 \
    --adapter_name loraga \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r "$RANK" --lora_alpha "$ALPHA" \
    --use_gradient_checkpointing \
    --seed "$SEED"

# Evaluate all 8 benchmarks
for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
        --model LLaMA-7B \
        --adapter LoRA-GA \
        --dataset "$DATASET" \
        --base_model "$BASE_MODEL" \
        --batch_size 1 \
        --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
done

echo "Results in $OUTPUT_DIR"
