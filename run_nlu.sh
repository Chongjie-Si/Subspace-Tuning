#!/bin/bash
# run_nlu.sh — launch LoMAP / baseline experiments for GLUE NLU
#
# Requires: conda activate lomap-nlu
# Run from repo root.
#
# Usage:
#   bash run_nlu.sh map  2  4  6     # LoMAP r=2, alpha=4, seed=6
#   bash run_nlu.sh lora 8 16 6      # LoRA r=8, alpha=16, seed=6
#   bash run_nlu.sh delora 2 4 6     # DeLoRA (HF PEFT)
#
# Runs all 8 GLUE tasks sequentially on a single GPU.
# For parallel: set TASKS below and pipe to xargs -P N.

set -e
LORA_TYPE=${1:-map}
RANK=${2:-2}
ALPHA=${3:-4}
SEED=${4:-6}
GPU=${5:-0}

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
NLU_DIR="$REPO_ROOT/NLU"
MODEL="microsoft/deberta-v3-base"
MODULES="query,key,value,intermediate,layer.output,attention.output"

# Per-task hyperparameters (from paper Table — DeBERTa-v3-base)
declare -A SEQ_LEN LR EPOCHS WARMUP CLS_DROP
SEQ_LEN=([mnli]=256  [sst2]=128 [cola]=64  [qqp]=320 [qnli]=512 [rte]=320 [mrpc]=320 [stsb]=128)
LR=([mnli]=5e-4 [sst2]=8e-4 [cola]=8e-4 [qqp]=1e-3 [qnli]=5e-4 [rte]=1.2e-3 [mrpc]=1e-3 [stsb]=5e-4)
EPOCHS=([mnli]=12 [sst2]=24 [cola]=25 [qqp]=5 [qnli]=5 [rte]=50 [mrpc]=30 [stsb]=25)
WARMUP=([mnli]=100 [sst2]=1000 [cola]=100 [qqp]=1000 [qnli]=1000 [rte]=200 [mrpc]=100 [stsb]=100)
CLS_DROP=([mnli]=0.10 [sst2]=0.00 [cola]=0.10 [qqp]=0.10 [qnli]=0.10 [rte]=0.20 [mrpc]=0.10 [stsb]=0.10)

echo "================================================"
echo "Task : GLUE NLU  lora_type=$LORA_TYPE  r=$RANK  alpha=$ALPHA  seed=$SEED"
echo "================================================"

cd "$NLU_DIR"

for TASK in mnli sst2 cola qqp qnli rte mrpc stsb; do
    OUT="./output/glue/${LORA_TYPE}_${TASK}_r${RANK}_seed${SEED}"
    echo "--- $TASK ---"
    CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch \
        --master_port=8679 --nproc_per_node=1 \
        examples/text-classification/run_glue.py \
        --model_name_or_path "$MODEL" \
        --task_name "$TASK" \
        --apply_lora --lora_type "$LORA_TYPE" \
        --lora_r "$RANK" --lora_module "$MODULES" --lora_alpha "$ALPHA" \
        --do_train --do_eval \
        --max_seq_length "${SEQ_LEN[$TASK]}" \
        --per_device_train_batch_size 32 \
        --learning_rate "${LR[$TASK]}" \
        --num_train_epochs "${EPOCHS[$TASK]}" \
        --warmup_steps "${WARMUP[$TASK]}" \
        --cls_dropout "${CLS_DROP[$TASK]}" \
        --weight_decay 0.00 \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 10000 \
        --logging_steps 10 \
        --tb_writter_loginterval 100 \
        --report_to tensorboard \
        --seed "$SEED" \
        --root_output_dir "$OUT" \
        --overwrite_output_dir
done

echo "Done. Results in NLU/output/glue/${LORA_TYPE}_*_r${RANK}_seed${SEED}/"
