#!/bin/bash
# run_delora_nlu_grid.sh — DeLoRA grid on DeBERTaV3 GLUE (H100).
#
# Usage:
#   bash run_delora_nlu_grid.sh <size> <rank> <seeds_csv> <gpus_csv>
#   e.g. bash run_delora_nlu_grid.sh base 2 6,7,8 0,1,2,3,4,5,6,7
#        bash run_delora_nlu_grid.sh base 8 6,7,8 0,1,2,3,4,5,6,7
#
# Requires peft >= 0.14 (DeLoRA support) — installed by h100_setup.sh via
# CR_MR/requirements.txt (peft>=0.13; upgrade to 0.14+ once on the node).
# If not yet installed: pip install "peft>=0.14" --upgrade
#
# Skips runs whose all_results.json already exists (auto-resume).

set -e
SIZE=${1:-base}
RANK=${2:-2}
SEEDS_CSV=${3:-6,7,8}
GPUS_CSV=${4:-0,1,2,3,4,5,6,7}

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$REPO_ROOT/deploy/h100_kit/scripts/run_delora_nlu.py"
VENV="$REPO_ROOT/.venv/bin/python"
OUT_ROOT="$REPO_ROOT/NLU/output/glue"

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra GPUS  <<< "$GPUS_CSV"
NGPU=${#GPUS[@]}

TASKS=(cola sst2 mrpc rte stsb qnli qqp mnli)

mkdir -p "$REPO_ROOT/logs"

# Build job list
JOBS_FILE=$(mktemp)
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        done_marker="$OUT_ROOT/${SIZE}_delora_${task}_r${RANK}_seed${seed}/model/all_results.json"
        if [ -f "$done_marker" ]; then
            echo "  skip (done): $done_marker"
            continue
        fi
        echo "$task $seed" >> "$JOBS_FILE"
    done
done

NJOBS=$(wc -l < "$JOBS_FILE")
echo "DeLoRA NLU jobs: $NJOBS  (size=$SIZE rank=$RANK seeds=${SEEDS[*]})"

for (( i=0; i<NGPU; i++ )); do
    GPU=${GPUS[$i]}
    (
        idx=0
        while IFS= read -r line; do
            if (( idx % NGPU == i )); then
                read -r task seed <<< "$line"
                LOG="$REPO_ROOT/logs/nlu_${SIZE}_delora_${task}_r${RANK}_seed${seed}_gpu${GPU}.log"
                echo "[GPU $GPU] DeLoRA $SIZE $task r=$RANK seed=$seed" | tee -a "$REPO_ROOT/logs/delora_nlu_dispatch.log"

                CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                PYTHONPATH="$REPO_ROOT/NLU/src:$REPO_ROOT/loralib" \
                "$VENV" "$SCRIPT" \
                    --task "$task" \
                    --seed "$seed" \
                    --rank "$RANK" \
                    --size "$SIZE" \
                    --output_root "$OUT_ROOT" \
                    > "$LOG" 2>&1 || echo "FAIL delora $task seed=$seed" >> "$REPO_ROOT/logs/delora_nlu_failures.log"
            fi
            idx=$((idx+1))
        done < "$JOBS_FILE"
    ) &
done

wait
rm -f "$JOBS_FILE"

echo ""
echo "DeLoRA NLU grid done."
"$VENV" "$REPO_ROOT/summarize_nlu.py" || true
