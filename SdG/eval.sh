#!/bin/bash
# Evaluate all trained DreamBooth models on CLIP-T, CLIP-I, DINO, LPIPS
# Usage: bash eval.sh [subject] [seed]

SUBJECT=${1:-cat}
SEED=${2:-42}
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
INSTANCE_DIR="./$SUBJECT"

for METHOD in lora lora-dash lomap; do
    if [ "$METHOD" = "lora" ]; then
        MODEL_PATH="./lora-trained-xl-$SUBJECT"
    elif [ "$METHOD" = "lora-dash" ]; then
        MODEL_PATH="./lora-trained-xl-dash-$SUBJECT"
    elif [ "$METHOD" = "lomap" ]; then
        MODEL_PATH="./lora-trained-xl-lomap-$SUBJECT"
    fi

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH, skipping."
        continue
    fi

    echo "=== Evaluating $METHOD on subject '$SUBJECT' ==="
    python eval_dreambooth.py \
        --model_path "$MODEL_PATH" \
        --subject "$SUBJECT" \
        --instance_dir "$INSTANCE_DIR" \
        --base_model "$MODEL" \
        --seed $SEED \
        --results_file "${MODEL_PATH}/eval_results_seed${SEED}.json"
done

# Aggregate results
python - <<'EOF'
import json, glob, os

for method_dir in glob.glob("./lora-trained-xl-*"):
    results_files = glob.glob(f"{method_dir}/eval_results_*.json")
    if not results_files:
        continue
    all_results = [json.load(open(f)) for f in results_files]
    method = os.path.basename(method_dir)
    print(f"\n{method}:")
    for key in ["clip_t", "clip_i", "dino", "lpips_diversity"]:
        vals = [r[key] for r in all_results if key in r]
        if vals:
            import numpy as np
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
EOF
