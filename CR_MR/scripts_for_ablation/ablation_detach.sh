#!/bin/bash
# Ablation: detach vs non-detach denominator in MAP normalization
# "detach" means the Frobenius norm in the denominator is computed with stop_gradient
# Usage: bash ablation_detach.sh [GPU_ID]

GPU=${1:-0}
SEED=${2:-42}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="commonsense_170k.json"
RANK=16
ALPHA=32

for DETACH in false true; do
    OUTPUT_DIR="./output/ablation/detach_${DETACH}"
    echo "=== detach_denominator=$DETACH ==="
    CUDA_VISIBLE_DEVICES=$GPU python finetune.py \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 --adapter_name lomap \
        --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
        --lora_r $RANK --lora_alpha $ALPHA \
        --map_detach_denom $DETACH \
        --use_gradient_checkpointing \
        --seed $SEED

    for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
        CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
            --model LLaMA-7B \
            --adapter LoMAP \
            --dataset $DATASET \
            --base_model "$BASE_MODEL" \
            --batch_size 1 \
            --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
    done
done
