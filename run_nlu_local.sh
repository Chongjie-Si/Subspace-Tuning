#!/bin/bash
# run_nlu_local.sh — run NLU experiments using the local venv (no conda needed)
#
# Usage:
#   bash run_nlu_local.sh [lora_type] [rank] [alpha] [seed] [task] [gpu]
#
# Examples:
#   bash run_nlu_local.sh map  2 4 6 cola 0    # LoMAP r=2, CoLA (quick ~5min)
#   bash run_nlu_local.sh lora 2 4 6 cola 0    # LoRA baseline
#   bash run_nlu_local.sh map  2 4 6 all  0    # LoMAP all 8 GLUE tasks
#   bash run_nlu_local.sh map  8 16 6 all 0    # LoMAP r=8
#
# Tensorboard:
#   tensorboard --logdir NLU/output/glue/
#
# Resume: re-run the same command — Trainer auto-detects last checkpoint.

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO_ROOT/.venv/bin/python"

if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found at $REPO_ROOT/.venv"
    echo "Run first: python3 -m venv $REPO_ROOT/.venv --system-site-packages"
    echo "           $REPO_ROOT/.venv/bin/pip install sacremoses sentencepiece scikit-learn datasets tensorboard tensorboardX"
    echo "           $REPO_ROOT/.venv/bin/pip install -e $REPO_ROOT/loralib/"
    exit 1
fi

LORA_TYPE=${1:-map}
RANK=${2:-2}
ALPHA=${3:-4}
SEED=${4:-6}
TASK=${5:-cola}
GPU=${6:-0}

NLU_DIR="$REPO_ROOT/NLU"
MODEL="microsoft/deberta-v3-base"
MODULES="query,key,value,intermediate,layer.output,attention.output"

# Per-task hyperparameters
declare -A SEQ_LEN LR EPOCHS WARMUP CLS_DROP
SEQ_LEN=([mnli]=256  [sst2]=128 [cola]=64  [qqp]=320 [qnli]=512 [rte]=320 [mrpc]=320 [stsb]=128)
LR=([mnli]=5e-4 [sst2]=8e-4 [cola]=8e-4 [qqp]=1e-3 [qnli]=5e-4 [rte]=1.2e-3 [mrpc]=1e-3 [stsb]=5e-4)
EPOCHS=([mnli]=12 [sst2]=24 [cola]=25 [qqp]=5 [qnli]=5 [rte]=50 [mrpc]=30 [stsb]=25)
WARMUP=([mnli]=100 [sst2]=1000 [cola]=100 [qqp]=1000 [qnli]=1000 [rte]=200 [mrpc]=100 [stsb]=100)
CLS_DROP=([mnli]=0.10 [sst2]=0.00 [cola]=0.10 [qqp]=0.10 [qnli]=0.10 [rte]=0.20 [mrpc]=0.10 [stsb]=0.10)

run_task() {
    local t=$1
    local out="$NLU_DIR/output/glue/${LORA_TYPE}_${t}_r${RANK}_seed${SEED}"
    echo ""; echo "=== $t  lora_type=$LORA_TYPE  r=$RANK  seed=$SEED ==="
    echo "Output: $out"

    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO_ROOT/NLU/src:$REPO_ROOT/loralib" \
    "$VENV" \
        "$NLU_DIR/examples/text-classification/run_glue.py" \
        --model_name_or_path "$MODEL" \
        --task_name "$t" \
        --apply_lora --lora_type "$LORA_TYPE" \
        --lora_r "$RANK" --lora_module "$MODULES" --lora_alpha "$ALPHA" \
        --do_train --do_eval \
        --max_seq_length "${SEQ_LEN[$t]}" \
        --per_device_train_batch_size 32 \
        --learning_rate "${LR[$t]}" \
        --num_train_epochs "${EPOCHS[$t]}" \
        --warmup_steps "${WARMUP[$t]}" \
        --cls_dropout "${CLS_DROP[$t]}" \
        --weight_decay 0.00 \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 200 \
        --save_total_limit 3 \
        --load_best_model_at_end \
        --logging_steps 10 \
        --tb_writter_loginterval 50 \
        --report_to tensorboard \
        --seed "$SEED" \
        --root_output_dir "$out"

    echo "Done: $t → $out"
    echo "Tensorboard: tensorboard --logdir $out/log"
}

echo "================================================"
echo "NLU local experiment"
echo "lora_type=$LORA_TYPE  r=$RANK  alpha=$ALPHA  seed=$SEED"
echo "task=$TASK  GPU=$GPU"
echo "venv=$VENV"
echo "================================================"

cd "$NLU_DIR"

if [ "$TASK" = "all" ]; then
    for t in cola stsb mrpc rte sst2 qnli qqp mnli; do
        run_task "$t"
    done
    echo ""; echo "All tasks done."
    echo "Tensorboard: tensorboard --logdir $NLU_DIR/output/glue/"
else
    run_task "$TASK"
fi
