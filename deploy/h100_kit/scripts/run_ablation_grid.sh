#!/bin/bash
# run_ablation_grid.sh — runs the §5 sensitivity sweep on N GPUs in parallel.
# Sweeps β₀, ε, detach_denom, norm_scope on a single representative task
# (LLaMA-7B CR @ r=16) per the paper convention.
#
# Usage: bash run_ablation_grid.sh <gpus_csv>

set -e
GPUS_CSV=${1:-0,1,2,3,4,5,6,7}
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CR_DIR="$REPO_ROOT/CR_MR"
VENV="$REPO_ROOT/.venv/bin/python"
BASE="huggyllama/llama-7b"
DATA="$CR_DIR/commonsense_170k.json"
RANK=16
ALPHA=32
LR=2e-4
SEED=42

IFS=',' read -ra GPUS <<< "$GPUS_CSV"
NGPU=${#GPUS[@]}

mkdir -p "$CR_DIR/output/ablation" "$REPO_ROOT/logs"

# Ablation jobs: (label, extra_flags)
JOBS_FILE=$(mktemp)
# β₀ sweep
for b in 0.0 0.0001 0.001 0.01 1.0; do
    echo "beta0_${b}|--map_beta_init $b" >> "$JOBS_FILE"
done
# ε sweep
for e in 1e-3 1e-5 1e-6 1e-8; do
    echo "eps_${e}|--map_eps $e" >> "$JOBS_FILE"
done
# detach_denom
echo "detach_off|--map_detach_denom False" >> "$JOBS_FILE"
echo "detach_on|--map_detach_denom True"   >> "$JOBS_FILE"
# norm scope
for s in global column row row_column; do
    echo "norm_${s}|--map_norm_scope $s" >> "$JOBS_FILE"
done

NJOBS=$(wc -l < "$JOBS_FILE")
echo "Ablation jobs: $NJOBS  on GPUs ${GPUS[*]}"

# Train phase
for (( i=0; i<NGPU; i++ )); do
    GPU=${GPUS[$i]}
    (
        idx=0
        cd "$CR_DIR"
        while IFS= read -r line; do
            if (( idx % NGPU == i )); then
                label="${line%%|*}"
                flags="${line#*|}"
                out="$CR_DIR/output/ablation/${label}"
                if [ -f "$out/adapter_model.bin" ]; then
                    sz=$(stat -c %s "$out/adapter_model.bin" 2>/dev/null || echo 0)
                    if [ "$sz" -ge 1024 ]; then idx=$((idx+1)); continue; fi
                    rm -f "$out/adapter_model.bin"   # corrupt/truncated → rerun
                fi

                LOG="$REPO_ROOT/logs/abl_${label}_gpu${GPU}.log"
                echo "[GPU $GPU] ablation $label" | tee -a "$REPO_ROOT/logs/abl_dispatch.log"

                CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                "$VENV" finetune.py \
                    --base_model "$BASE" \
                    --data_path "$DATA" \
                    --output_dir "$out" \
                    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
                    --learning_rate "$LR" --cutoff_len 256 --val_set_size 120 \
                    --eval_step 80 --save_step 80 --adapter_name lomap \
                    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
                    --lora_r "$RANK" --lora_alpha "$ALPHA" \
                    --use_gradient_checkpointing \
                    --seed "$SEED" \
                    $flags \
                    > "$LOG" 2>&1 || echo "FAIL $label" >> "$REPO_ROOT/logs/abl_failures.log"
            fi
            idx=$((idx+1))
        done < "$JOBS_FILE"
    ) &
done
wait

# Eval all
echo ""
echo "Evaluating ablations on 8 CR benchmarks..."
# commonsense_evaluate.py uses relative dataset/<bench>/test.json — must run with
# cwd = CR_MR/ (the training subshells cd'd in, but this main-shell loop did not).
cd "$CR_DIR"
DATASETS=(boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa)
for label_line in $(cat "$JOBS_FILE"); do
    label="${label_line%%|*}"
    out="$CR_DIR/output/ablation/${label}"
    [ -f "$out/adapter_model.bin" ] || continue
    for ds in "${DATASETS[@]}"; do
        [ -f "$out/${ds}.txt" ] && continue
        CUDA_VISIBLE_DEVICES=${GPUS[0]} "$VENV" "$CR_DIR/commonsense_evaluate.py" \
            --model LLaMA-7B \
            --adapter LoMAP \
            --dataset "$ds" \
            --base_model "$BASE" \
            --batch_size 1 \
            --lora_weights "$out" >> "$out/${ds}.txt" 2>&1 || true
    done
done
rm -f "$JOBS_FILE"
echo "Ablation grid done."
