#!/bin/bash
# run_nlu_grid.sh — runs the full {method × task × seed} grid on N GPUs in parallel.
# Each (task, seed, method) becomes a single Python invocation; jobs are dispatched
# round-robin onto GPUs from a pool with a fixed concurrency cap.
#
# Usage:
#   bash run_nlu_grid.sh <size> <rank> <seeds_csv> <gpus_csv>
#   e.g. bash run_nlu_grid.sh base 2 6,7,8 0,1,2,3,4,5,6,7
#        bash run_nlu_grid.sh large 8 6,7,8 0,1,2,3
#
# Skips runs whose all_results.json already exists (auto-resume).

set -e
SIZE=${1:-base}        # base | large
RANK=${2:-2}
SEEDS_CSV=${3:-6,7,8}
GPUS_CSV=${4:-0}

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NLU_DIR="$REPO_ROOT/NLU"
VENV="$REPO_ROOT/.venv/bin/python"

case "$SIZE" in
    base)  MODEL="microsoft/deberta-v3-base";  CLS_DIM=768  ;;
    large) MODEL="microsoft/deberta-v3-large"; CLS_DIM=1024 ;;
    *) echo "Bad size: $SIZE"; exit 1 ;;
esac

# alpha=2*rank (paper convention)
ALPHA=$((RANK * 2))

# DoRA uses the same target modules
MODULES="query,key,value,intermediate,layer.output,attention.output"

# Per-task hyperparams (paper Table 4)
declare -A SEQ_LEN LR EPOCHS CLS_DROP BEST_METRIC
SEQ_LEN=([mnli]=256  [sst2]=128 [cola]=64  [qqp]=320 [qnli]=512 [rte]=320 [mrpc]=320 [stsb]=128)
LR=([mnli]=5e-4 [sst2]=8e-4 [cola]=8e-4 [qqp]=1e-3 [qnli]=5e-4 [rte]=1.2e-3 [mrpc]=1e-3 [stsb]=5e-4)
EPOCHS=([mnli]=12 [sst2]=24 [cola]=25 [qqp]=5 [qnli]=5 [rte]=50 [mrpc]=30 [stsb]=25)
CLS_DROP=([mnli]=0.10 [sst2]=0.00 [cola]=0.10 [qqp]=0.10 [qnli]=0.10 [rte]=0.20 [mrpc]=0.10 [stsb]=0.10)
BEST_METRIC=([mnli]=accuracy [sst2]=accuracy [cola]=matthews_correlation \
             [qqp]=accuracy [qnli]=accuracy [rte]=accuracy [mrpc]=accuracy [stsb]=pearson)

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra GPUS  <<< "$GPUS_CSV"
NGPU=${#GPUS[@]}

# Run grid: methods × tasks × seeds
METHODS=(map lora)   # DoRA needs separate flag; add 'dora' if you implement it
TASKS=(cola sst2 mrpc rte stsb qnli qqp mnli)

mkdir -p "$NLU_DIR/output/glue" logs

# Build job list
JOBS_FILE=$(mktemp)
for method in "${METHODS[@]}"; do
    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            out="$NLU_DIR/output/glue/${SIZE}_${method}_${task}_r${RANK}_seed${seed}"
            done_marker="$out/model/all_results.json"
            if [ -f "$done_marker" ]; then
                # Skip already-done; verify via best_metric in trainer_state
                continue
            fi
            echo "$method $task $seed $out" >> "$JOBS_FILE"
        done
    done
done

NJOBS=$(wc -l < "$JOBS_FILE")
echo "Total jobs to run: $NJOBS  (skipped existing)"
echo "Parallel GPUs: ${GPUS[*]}"

# Dispatch round-robin: GPU i ← every Ni'th job
for (( i=0; i<NGPU; i++ )); do
    GPU=${GPUS[$i]}
    (
        idx=0
        while IFS= read -r line; do
            if (( idx % NGPU == i )); then
                read -r method task seed out <<< "$line"
                t=$task
                LOG="logs/${SIZE}_${method}_${t}_r${RANK}_seed${seed}_gpu${GPU}.log"
                echo "[GPU $GPU] $method $t seed=$seed -> $out" | tee -a logs/grid_dispatch.log

                save_steps=200
                case "$t" in
                    mnli|qqp) save_steps=2000 ;;
                    qnli)     save_steps=1000 ;;
                esac

                # Translate user-facing method to NLU fork's internal name
                # (frd = standard LoRA, svd = AdaLoRA, map = LoMAP)
                case "$method" in
                    lora)    internal_lora_type="frd" ;;
                    adalora) internal_lora_type="svd" ;;
                    map)     internal_lora_type="map" ;;
                    *) echo "ERROR unknown method=$method"; continue ;;
                esac

                CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO_ROOT/NLU/src:$REPO_ROOT/loralib" \
                "$VENV" \
                    "$NLU_DIR/examples/text-classification/run_glue.py" \
                    --model_name_or_path "$MODEL" \
                    --task_name "$t" \
                    --apply_lora --lora_type "$internal_lora_type" \
                    --lora_r "$RANK" --lora_module "$MODULES" --lora_alpha "$ALPHA" \
                    --do_train --do_eval \
                    --max_seq_length "${SEQ_LEN[$t]}" \
                    --per_device_train_batch_size 32 \
                    --per_device_eval_batch_size 64 \
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
                    --greater_is_better True \
                    --logging_steps 50 \
                    --tb_writter_loginterval 50 \
                    --report_to tensorboard \
                    --skip_memory_metrics \
                    --seed "$seed" \
                    --root_output_dir "$out" \
                    > "$LOG" 2>&1 || echo "FAIL: $method $t seed=$seed" >> logs/grid_failures.log
            fi
            idx=$((idx+1))
        done < "$JOBS_FILE"
    ) &
done

wait
rm -f "$JOBS_FILE"

echo ""
echo "================================================"
echo "Grid done.  Summary:"
"$VENV" "$REPO_ROOT/summarize_nlu.py" || true
