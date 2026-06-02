#!/bin/bash
# BiDoRA on GLUE NLU (DeBERTa/RoBERTa)
# Official implementation: https://github.com/t2ance/BiDoRA
# Reference: Qin et al. "BiDoRA: Bi-level Optimization-based Weight-decomposed
#            Low-Rank Adaptation" (EMNLP 2024)
#
# IMPORTANT: BiDoRA uses a bi-level optimization scheme with its own training
# pipeline. It does NOT use the standard HF Trainer. You must clone the official
# repo and run their script.
#
# Setup:
#   git clone https://github.com/t2ance/BiDoRA ../../BiDoRA
#   pip install -r ../../BiDoRA/requirements.txt
#
# The official BiDoRA training script is:
#   examples/NLU/examples/text-classification/run_glue_bilevel.py
#
# Usage: bash run_bidora_glue.sh [rank] [GPU_ID] [seed]

RANK=${1:-2}
GPU=${2:-0}
SEED=${3:-6}
MODEL="roberta-base"   # BiDoRA paper uses RoBERTa; change to deberta-v3-base for cross-comparison

BIDORA_DIR="$(dirname "$0")/../../BiDoRA"
BIDORA_SCRIPT="$BIDORA_DIR/examples/NLU/examples/text-classification/run_glue_bilevel.py"

if [ ! -f "$BIDORA_SCRIPT" ]; then
    echo "ERROR: BiDoRA not found at $BIDORA_DIR"
    echo "Please run: git clone https://github.com/t2ance/BiDoRA $BIDORA_DIR"
    exit 1
fi

# BiDoRA hyper-params from their paper (RoBERTa-base, Table 3)
declare -A TASKS
# task -> "max_seq_len lr arch_lr train_iters retrain_iters"
TASKS[cola]="512 2e-4 1e-4 1000 1000"
TASKS[mrpc]="512 2e-4 1e-4 500  500"
TASKS[rte]="512  2e-4 1e-4 500  500"
TASKS[stsb]="512 2e-4 1e-4 1000 1000"
TASKS[sst2]="512 2e-4 1e-4 5000 5000"
TASKS[qnli]="512 2e-4 1e-4 5000 5000"
TASKS[qqp]="512  1e-4 1e-4 5000 5000"
TASKS[mnli]="512 2e-4 1e-4 5000 5000"

for TASK in "${!TASKS[@]}"; do
    read -r SEQ_LEN LR ARCH_LR TRAIN_ITERS RETRAIN_ITERS <<< "${TASKS[$TASK]}"
    OUTPUT_DIR="./output/glue/bidora_${TASK}_r${RANK}_seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    echo "=== BiDoRA on $TASK (r=$RANK, seed=$SEED) ==="

    CUDA_VISIBLE_DEVICES=$GPU python "$BIDORA_SCRIPT" \
        --model_name_or_path "$MODEL" \
        --task_name "$TASK" \
        --work_dir "$OUTPUT_DIR" \
        --lora_r "$RANK" \
        --lora_alpha $((RANK * 2)) \
        --lora_type bidora \
        --lr "$LR" \
        --arch_lr "$ARCH_LR" \
        --max_seq_length "$SEQ_LEN" \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --retrain_train_batch_size 32 \
        --train_iters "$TRAIN_ITERS" \
        --retrain_iters "$RETRAIN_ITERS" \
        --seed "$SEED"
done

echo "BiDoRA GLUE results in ./output/glue/bidora_*_r${RANK}_seed${SEED}/"
