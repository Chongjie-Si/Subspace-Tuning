#!/bin/bash
# Ablation: rank sweep r in {2, 4, 8, 16, 32, 64}
# Compares LoRA vs LoMAP at each rank on LLaMA-7B commonsense reasoning
# Usage: bash ablation_rank.sh [GPU_ID]

GPU=${1:-0}
SEED=${2:-42}
BASE_MODEL="huggyllama/llama-7b"
DATA_PATH="commonsense_170k.json"

for RANK in 2 4 8 16 32 64; do
    ALPHA=$((RANK * 2))

    for ADAPTER in lora lomap; do
        OUTPUT_DIR="./output/ablation/rank_${RANK}_${ADAPTER}"
        echo "=== rank=$RANK adapter=$ADAPTER ==="
        CUDA_VISIBLE_DEVICES=$GPU python finetune.py \
            --base_model "$BASE_MODEL" \
            --data_path "$DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
            --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
            --eval_step 80 --save_step 80 --adapter_name $ADAPTER \
            --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
            --lora_r $RANK --lora_alpha $ALPHA \
            --use_gradient_checkpointing \
            --seed $SEED

        # Evaluate: LoRA adapter uses "LoRA"; LoMAP uses "LoMAP"
        EVAL_ADAPTER="LoRA"
        if [ "$ADAPTER" = "lomap" ]; then EVAL_ADAPTER="LoMAP"; fi

        for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
            CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
                --model LLaMA-7B \
                --adapter $EVAL_ADAPTER \
                --dataset $DATASET \
                --base_model "$BASE_MODEL" \
                --batch_size 1 \
                --lora_weights "$OUTPUT_DIR" | tee -a "${OUTPUT_DIR}/${DATASET}.txt"
        done
    done
done
