#!/bin/bash
# run_delora_cr.sh — DeLoRA on LLaMA-7B commonsense reasoning (H100).
#
# Usage:
#   bash run_delora_cr.sh <gpus_csv>
#   e.g. bash run_delora_cr.sh 0,1,2,3,4,5,6,7
#
# Runs DeLoRA at r=16 (the most important row in Table 2) plus r=32 for
# completeness. Skips already-finished runs.

set -e
GPUS_CSV=${1:-0,1,2,3,4,5,6,7}

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CR_DIR="$REPO_ROOT/CR_MR"
VENV="$REPO_ROOT/.venv/bin/python"
BASE="huggyllama/llama-7b"
MODEL_TAG="LLaMA-7B"
DATA="$CR_DIR/commonsense_170k.json"

IFS=',' read -ra GPUS <<< "$GPUS_CSV"
NGPU=${#GPUS[@]}

RANKS=(16 32)

mkdir -p "$CR_DIR/output" "$REPO_ROOT/logs"

# Build job list
JOBS_FILE=$(mktemp)
for rank in "${RANKS[@]}"; do
    case "$rank" in
        16) alpha=32 ;;
        32) alpha=64 ;;
        *)  alpha=$((rank * 2)) ;;
    esac
    out="$CR_DIR/output/llama-7b_delora_r${rank}_a${alpha}"
    adapter_done="$out/adapter_model.bin"
    if [ -f "$adapter_done" ]; then
        sz=$(stat -c %s "$adapter_done" 2>/dev/null || echo 0)
        if [ "$sz" -lt 1024 ]; then
            echo "  corrupt adapter ($sz B), re-queue: $out"
            rm -f "$adapter_done"
        else
            echo "  skip (done): $out"
            continue
        fi
    fi
    echo "delora $rank $alpha $out" >> "$JOBS_FILE"
done

NJOBS=$(wc -l < "$JOBS_FILE" 2>/dev/null || echo 0)
echo "DeLoRA CR jobs to run: $NJOBS"

for (( i=0; i<NGPU; i++ )); do
    GPU=${GPUS[$i]}
    (
        idx=0
        cd "$CR_DIR"
        while IFS= read -r line; do
            if (( idx % NGPU == i )); then
                read -r method rank alpha out <<< "$line"
                # r=16 → lr=2e-4 (paper Table 5); r=32 → lr=3e-4
                case "$rank" in
                    16) lr=2e-4 ;;
                    32) lr=3e-4 ;;
                    *)  lr=2e-4 ;;
                esac
                LOG="$REPO_ROOT/logs/cr_llama-7b_delora_r${rank}_gpu${GPU}.log"
                echo "[GPU $GPU] DeLoRA train r=$rank -> $out" | tee -a "$REPO_ROOT/logs/cr_dispatch.log"

                CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                "$VENV" finetune.py \
                    --base_model "$BASE" \
                    --data_path "$DATA" \
                    --output_dir "$out" \
                    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
                    --learning_rate "$lr" --cutoff_len 256 --val_set_size 120 \
                    --eval_step 80 --save_step 80 --adapter_name "delora" \
                    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
                    --lora_r "$rank" --lora_alpha "$alpha" \
                    --use_gradient_checkpointing \
                    > "$LOG" 2>&1 || echo "FAIL delora r=$rank gpu=$GPU" >> "$REPO_ROOT/logs/cr_failures.log"
            fi
            idx=$((idx+1))
        done < "$JOBS_FILE"
    ) &
done
wait
rm -f "$JOBS_FILE"

# Eval phase
echo ""
echo "Evaluating DeLoRA adapters..."
DATASETS=(boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa)
cd "$CR_DIR"
for rank in "${RANKS[@]}"; do
    case "$rank" in
        16) alpha=32 ;; 32) alpha=64 ;;
    esac
    out="$CR_DIR/output/llama-7b_delora_r${rank}_a${alpha}"
    [ -f "$out/adapter_model.bin" ] || { echo "skip eval $out (no adapter)"; continue; }
    for ds in "${DATASETS[@]}"; do
        done_marker="$out/${ds}.txt"
        [ -f "$done_marker" ] && continue
        CUDA_VISIBLE_DEVICES=${GPUS[0]} "$VENV" "$CR_DIR/commonsense_evaluate.py" \
            --model "$MODEL_TAG" \
            --adapter "DeLoRA" \
            --dataset "$ds" \
            --base_model "$BASE" \
            --batch_size 1 \
            --lora_weights "$out" >> "$out/${ds}.txt" 2>&1 || true
    done
done

echo "DeLoRA CR done."
"$VENV" "$CR_DIR/scripts_for_baselines/aggregate_results.py" "$CR_DIR/output" || true
