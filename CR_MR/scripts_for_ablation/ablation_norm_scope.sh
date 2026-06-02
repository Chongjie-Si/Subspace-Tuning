#!/bin/bash
# Ablation: normalization scope comparison
# global Frobenius vs column-wise vs row-wise vs row+column
# This requires patching the normalization in loralib/loralib/layers.py
# Usage: bash ablation_norm_scope.sh [GPU_ID]

GPU=${1:-0}
SEED=${2:-42}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="commonsense_170k.json"
RANK=16
ALPHA=32

# Run with map_norm_scope argument (requires corresponding code support in finetune.py / LoraConfig)
for NORM_SCOPE in global column row row_column; do
    OUTPUT_DIR="./output/ablation/norm_${NORM_SCOPE}"
    echo "=== norm_scope=$NORM_SCOPE ==="
    CUDA_VISIBLE_DEVICES=$GPU python finetune.py \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 --adapter_name lomap \
        --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
        --lora_r $RANK --lora_alpha $ALPHA \
        --map_norm_scope $NORM_SCOPE \
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
