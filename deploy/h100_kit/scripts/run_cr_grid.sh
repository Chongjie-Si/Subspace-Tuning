#!/bin/bash
# run_cr_grid.sh — LLaMA commonsense reasoning grid on N GPUs.
#
# Usage:
#   bash run_cr_grid.sh <model> <ranks_csv> <methods_csv> <gpus_csv>
#   e.g. bash run_cr_grid.sh llama-7b 4,8,16,32 lora,lomap 0,1,2,3,4,5,6,7
#        bash run_cr_grid.sh llama3-8b 16,32 lora,lomap,dora 0,1,2,3,4,5,6,7
#
# Skips runs whose adapter_model.bin already exists.

set -e
MODEL_KEY=${1:-llama-7b}
RANKS_CSV=${2:-16,32}
METHODS_CSV=${3:-lomap,lora,dora}
GPUS_CSV=${4:-0,1,2,3,4,5,6,7}

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CR_DIR="$REPO_ROOT/CR_MR"
VENV="$REPO_ROOT/.venv/bin/python"

case "$MODEL_KEY" in
    llama-7b)   BASE="huggyllama/llama-7b";        MODEL_TAG="LLaMA-7B"  ;;
    llama2-7b)  BASE="meta-llama/Llama-2-7b-hf";   MODEL_TAG="LLaMA2-7B" ;;
    llama3-8b)  BASE="meta-llama/Meta-Llama-3-8B"; MODEL_TAG="LLaMA3-8B" ;;
    *) echo "Unknown model: $MODEL_KEY"; exit 1 ;;
esac

DATA="$CR_DIR/commonsense_170k.json"
[ -f "$DATA" ] || { echo "ERROR: $DATA missing. Pull commonsense_170k.json"; exit 1; }

# Paper Table 5 hyperparameters
declare -A LR_FOR_RANK
LR_FOR_RANK=([4]=3e-4 [8]=3e-4 [16]=2e-4 [32]=3e-4)

IFS=',' read -ra RANKS   <<< "$RANKS_CSV"
IFS=',' read -ra METHODS <<< "$METHODS_CSV"
IFS=',' read -ra GPUS    <<< "$GPUS_CSV"
NGPU=${#GPUS[@]}

mkdir -p "$CR_DIR/output" "$REPO_ROOT/logs"

# Build job list
JOBS_FILE=$(mktemp)
for method in "${METHODS[@]}"; do
    for rank in "${RANKS[@]}"; do
        # Paper alpha schedule:  r=4 → α=32 ; r=8 → α=64 ; r=16 → α=32 ; r=32 → α=64
        case "$rank" in
            4)  alpha=32 ;;
            8)  alpha=64 ;;
            16) alpha=32 ;;
            32) alpha=64 ;;
            *)  alpha=$((rank * 2)) ;;
        esac
        out="$CR_DIR/output/${MODEL_KEY}_${method}_r${rank}_a${alpha}"
        adapter_done="$out/adapter_model.bin"
        # Treat a present-but-tiny adapter as corrupt (truncated by preemption):
        # a real LoRA adapter is at least tens of KB. Re-queue if zero/missing/tiny.
        if [ -f "$adapter_done" ]; then
            sz=$(stat -c %s "$adapter_done" 2>/dev/null || echo 0)
            if [ "$sz" -lt 1024 ]; then
                echo "  corrupt adapter ($sz B), re-queue: $out"
                rm -f "$adapter_done"
            else
                continue
            fi
        fi
        echo "$method $rank $alpha $out" >> "$JOBS_FILE"
    done
done

NJOBS=$(wc -l < "$JOBS_FILE")
echo "CR jobs to run: $NJOBS  (model=$MODEL_KEY, methods=${METHODS[*]}, ranks=${RANKS[*]})"

# Train phase: round-robin across GPUs
for (( i=0; i<NGPU; i++ )); do
    GPU=${GPUS[$i]}
    (
        idx=0
        cd "$CR_DIR"
        while IFS= read -r line; do
            if (( idx % NGPU == i )); then
                read -r method rank alpha out <<< "$line"
                lr=${LR_FOR_RANK[$rank]:-2e-4}
                LOG="$REPO_ROOT/logs/cr_${MODEL_KEY}_${method}_r${rank}_gpu${GPU}.log"
                echo "[GPU $GPU] CR train $method r=$rank → $out" | tee -a "$REPO_ROOT/logs/cr_dispatch.log"

                CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                "$VENV" finetune.py \
                    --base_model "$BASE" \
                    --data_path "$DATA" \
                    --output_dir "$out" \
                    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
                    --learning_rate "$lr" --cutoff_len 256 --val_set_size 120 \
                    --eval_step 80 --save_step 80 --adapter_name "$method" \
                    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
                    --lora_r "$rank" --lora_alpha "$alpha" \
                    --use_gradient_checkpointing \
                    > "$LOG" 2>&1 || echo "FAIL $method r=$rank gpu=$GPU" >> "$REPO_ROOT/logs/cr_failures.log"
            fi
            idx=$((idx+1))
        done < "$JOBS_FILE"
    ) &
done
wait
rm -f "$JOBS_FILE"

# Eval phase: 8 commonsense benchmarks per (method, rank)
echo ""
echo "================================================"
echo "Evaluating all trained adapters..."
echo "================================================"
DATASETS=(boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa)
for method in "${METHODS[@]}"; do
    for rank in "${RANKS[@]}"; do
        case "$rank" in
            4) alpha=32 ;; 8) alpha=64 ;; 16) alpha=32 ;; 32) alpha=64 ;;
            *) alpha=$((rank*2)) ;;
        esac
        out="$CR_DIR/output/${MODEL_KEY}_${method}_r${rank}_a${alpha}"
        [ -f "$out/adapter_model.bin" ] || { echo "skip $out (no adapter)"; continue; }
        for ds in "${DATASETS[@]}"; do
            done_marker="$out/${ds}.txt"
            [ -f "$done_marker" ] && continue
            # Use first GPU for sequential eval
            CUDA_VISIBLE_DEVICES=${GPUS[0]} "$VENV" "$CR_DIR/commonsense_evaluate.py" \
                --model "$MODEL_TAG" \
                --adapter "$(echo $method | tr '[:lower:]' '[:upper:]')" \
                --dataset "$ds" \
                --base_model "$BASE" \
                --batch_size 1 \
                --lora_weights "$out" >> "$out/${ds}.txt" 2>&1 || true
        done
    done
done

echo "Done CR grid for $MODEL_KEY."
"$VENV" "$CR_DIR/scripts_for_baselines/aggregate_results.py" "$CR_DIR/output" || true
