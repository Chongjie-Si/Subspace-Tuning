#!/bin/bash
# run_nlu_local.sh — run NLU experiments using the local venv (no conda needed)
#
# Usage:
#   bash run_nlu_local.sh [lora_type] [rank] [alpha] [seed] [task] [gpu]
#
# Examples:
#   bash run_nlu_local.sh map  2 4 6 cola 0         # LoMAP r=2, CoLA, single seed
#   bash run_nlu_local.sh lora 2 4 6 cola 0         # LoRA baseline
#   bash run_nlu_local.sh map  2 4 "6,7,8" cola 0   # LoMAP r=2, CoLA, 3 seeds → avg
#   bash run_nlu_local.sh map  2 4 all cola 0        # LoMAP, seeds 6/7/8
#   bash run_nlu_local.sh map  2 4 6 all  0          # LoMAP all 8 GLUE tasks
#   bash run_nlu_local.sh map  8 16 6 all 0          # LoMAP r=8
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
SEED_ARG=${4:-6}
TASK=${5:-cola}
GPU=${6:-0}

NLU_DIR="$REPO_ROOT/NLU"
MODEL="microsoft/deberta-v3-base"
MODULES="query,key,value,intermediate,layer.output,attention.output"

# Expand seed list: "all" → 6 7 8; "6,7,8" → 6 7 8; "6" → 6
if [ "$SEED_ARG" = "all" ]; then
    SEEDS=(6 7 8)
else
    IFS=',' read -ra SEEDS <<< "$SEED_ARG"
fi

# Per-task hyperparameters (matched to paper: warmup_ratio=0.1, no warmup_steps)
declare -A SEQ_LEN LR EPOCHS CLS_DROP BEST_METRIC GREATER_IS_BETTER
SEQ_LEN=([mnli]=256  [sst2]=128 [cola]=64  [qqp]=320 [qnli]=512 [rte]=320 [mrpc]=320 [stsb]=128)
LR=([mnli]=5e-4 [sst2]=8e-4 [cola]=8e-4 [qqp]=1e-3 [qnli]=5e-4 [rte]=1.2e-3 [mrpc]=1e-3 [stsb]=5e-4)
EPOCHS=([mnli]=12 [sst2]=24 [cola]=25 [qqp]=5 [qnli]=5 [rte]=50 [mrpc]=30 [stsb]=25)
CLS_DROP=([mnli]=0.10 [sst2]=0.00 [cola]=0.10 [qqp]=0.10 [qnli]=0.10 [rte]=0.20 [mrpc]=0.10 [stsb]=0.10)
# Task-specific best-model metric (use task metric, not loss, to select checkpoint)
BEST_METRIC=([mnli]=accuracy [sst2]=accuracy [cola]=matthews_correlation \
             [qqp]=accuracy [qnli]=accuracy [rte]=accuracy [mrpc]=accuracy [stsb]=pearson)
GREATER_IS_BETTER=([mnli]=True [sst2]=True [cola]=True \
                   [qqp]=True [qnli]=True [rte]=True [mrpc]=True [stsb]=True)

run_task() {
    local t=$1
    local seed=$2
    local out="$NLU_DIR/output/glue/${LORA_TYPE}_${t}_r${RANK}_seed${seed}"
    echo ""; echo "=== $t  lora_type=$LORA_TYPE  r=$RANK  seed=$seed ==="
    echo "Output: $out"

    # NLU fork uses internal names: frd = standard LoRA, svd = AdaLoRA, map = LoMAP.
    # User-facing values lora/adalora/map get translated here.
    local internal_lora_type
    case "$LORA_TYPE" in
        lora)    internal_lora_type="frd" ;;
        adalora) internal_lora_type="svd" ;;
        map)     internal_lora_type="map" ;;
        frd|svd) internal_lora_type="$LORA_TYPE" ;;
        *) echo "ERROR: unknown lora_type=$LORA_TYPE (use lora|adalora|map)"; return 1 ;;
    esac

    # Save/eval cadence: keep evaluation frequent (paper-style), but save sparsely
    # to avoid I/O thrash (each ckpt = ~715MB of full model state). Long tasks
    # (mnli/qqp) save every 1000 steps; short tasks every 200.
    local save_steps=200
    case "$t" in
        mnli|qqp) save_steps=1000 ;;
        qnli)     save_steps=500  ;;
    esac

    # Eval batch size: scale down for long-sequence tasks to avoid OOM on 16GB GPUs.
    # qnli uses max_seq_length=512, rte/qqp=320 — eval batch 64 at seq 512 OOMs a 4080.
    local eval_bs=64
    case "$t" in
        qnli)          eval_bs=16 ;;
        rte|qqp|mrpc)  eval_bs=32 ;;
        mnli)          eval_bs=32 ;;
    esac

    # Train batch size + grad accumulation: DeBERTa-v2 disentangled attention is
    # O(seq^2) in memory. At seq 512 (qnli), train batch 32 OOMs a 16GB GPU in the
    # attention forward. Use smaller micro-batch + accumulation to keep effective
    # batch = 32 (paper setting) while fitting in memory.
    local train_bs=32 accum=1
    case "$t" in
        qnli)      train_bs=8;  accum=4 ;;   # 8*4 = 32, seq 512
        qqp|mnli)  train_bs=16; accum=2 ;;   # 16*2 = 32, seq 320/256
        rte)       train_bs=16; accum=2 ;;   # seq 320
    esac

    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$REPO_ROOT/NLU/src:$REPO_ROOT/loralib" \
    "$VENV" \
        "$NLU_DIR/examples/text-classification/run_glue.py" \
        --model_name_or_path "$MODEL" \
        --task_name "$t" \
        --apply_lora --lora_type "$internal_lora_type" \
        --lora_r "$RANK" --lora_module "$MODULES" --lora_alpha "$ALPHA" \
        --do_train --do_eval \
        --max_seq_length "${SEQ_LEN[$t]}" \
        --per_device_train_batch_size "$train_bs" \
        --gradient_accumulation_steps "$accum" \
        --per_device_eval_batch_size "$eval_bs" \
        --learning_rate "${LR[$t]}" \
        --num_train_epochs "${EPOCHS[$t]}" \
        --warmup_ratio 0.1 \
        --cls_dropout "${CLS_DROP[$t]}" \
        --weight_decay 0.00 \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps "$save_steps" \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --metric_for_best_model "${BEST_METRIC[$t]}" \
        --greater_is_better "${GREATER_IS_BETTER[$t]}" \
        --logging_steps 50 \
        --tb_writter_loginterval 50 \
        --report_to tensorboard \
        --skip_memory_metrics \
        --seed "$seed" \
        --root_output_dir "$out"

    echo "Done: $t seed=$seed → $out"
}

# Aggregate results across seeds for a single task
summarize_task() {
    local t=$1
    echo ""
    echo "--- Summary: $t  lora_type=$LORA_TYPE  r=$RANK ---"
    for seed in "${SEEDS[@]}"; do
        local res="$NLU_DIR/output/glue/${LORA_TYPE}_${t}_r${RANK}_seed${seed}/model/all_results.json"
        if [ -f "$res" ]; then
            python3 -c "
import json, sys
d = json.load(open('$res'))
metric = [v for k,v in d.items() if k.startswith('eval_') and 'loss' not in k and 'runtime' not in k and 'sample' not in k and 'mem' not in k][0]
key    = [k for k,v in d.items() if k.startswith('eval_') and 'loss' not in k and 'runtime' not in k and 'sample' not in k and 'mem' not in k][0]
print(f'  seed=$seed  {key}={metric:.4f}')
" 2>/dev/null || echo "  seed=$seed  (result not found)"
        else
            echo "  seed=$seed  (not yet run)"
        fi
    done

    # Compute average if multiple seeds all have results
    if [ "${#SEEDS[@]}" -gt 1 ]; then
        python3 -c "
import json, os, glob
seeds = [${SEEDS[*]}]
vals = []
for s in seeds:
    p = '$NLU_DIR/output/glue/${LORA_TYPE}_${t}_r${RANK}_seed'+str(s)+'/model/all_results.json'
    if os.path.exists(p):
        d = json.load(open(p))
        v = [v for k,v in d.items() if k.startswith('eval_') and 'loss' not in k and 'runtime' not in k and 'sample' not in k and 'mem' not in k]
        if v: vals.append(v[0])
if vals:
    import statistics
    avg = sum(vals)/len(vals)
    std = statistics.stdev(vals) if len(vals)>1 else 0
    print(f'  AVERAGE over {len(vals)} seeds: {avg:.4f} ± {std:.4f}')
" 2>/dev/null
    fi
}

echo "================================================"
echo "NLU local experiment"
echo "lora_type=$LORA_TYPE  r=$RANK  alpha=$ALPHA  seeds=${SEEDS[*]}"
echo "task=$TASK  GPU=$GPU"
echo "venv=$VENV"
echo "================================================"

cd "$NLU_DIR"

TASKS_TO_RUN=()
if [ "$TASK" = "all" ]; then
    TASKS_TO_RUN=(cola stsb mrpc rte sst2 qnli qqp mnli)
else
    TASKS_TO_RUN=("$TASK")
fi

for t in "${TASKS_TO_RUN[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_task "$t" "$seed"
    done
    summarize_task "$t"
done

echo ""
echo "================================================"
echo "All done."
echo "Tensorboard: tensorboard --logdir $NLU_DIR/output/glue/"
echo "================================================"
