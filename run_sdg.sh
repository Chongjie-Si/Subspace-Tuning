#!/bin/bash
# run_sdg.sh — launch LoMAP / LoRA DreamBooth training + quantitative eval
#
# Requires: conda activate lomap-sdg
# Run from repo root.
#
# Usage:
#   bash run_sdg.sh lora    cat   0     # LoRA, subject=cat, GPU=0
#   bash run_sdg.sh lomap   cat   0     # LoMAP
#   bash run_sdg.sh dash    cat   0     # LoRA-Dash
#   bash run_sdg.sh all     cat   0     # train all three, then eval
#
# Subjects: cat dog rc_car teapot ... (folder names in SdG/)

set -e
METHOD=${1:-lomap}
SUBJECT=${2:-cat}
GPU=${3:-0}

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SDG_DIR="$REPO_ROOT/SdG"
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
INSTANCE_DIR="$SDG_DIR/$SUBJECT"
STEPS=500

if [ ! -d "$INSTANCE_DIR" ]; then
    echo "ERROR: Subject directory not found: $INSTANCE_DIR"
    echo "Available subjects: $(ls $SDG_DIR | grep -v '\.py\|\.sh\|\.txt\|README\|peft\|__pycache__')"
    exit 1
fi

cd "$SDG_DIR"

train_method() {
    local method=$1
    local out_dir="$SDG_DIR/output/${method}-${SUBJECT}"
    mkdir -p "$out_dir"
    echo "--- Training $method on subject: $SUBJECT ---"

    local extra_args=""
    case "$method" in
        lomap) extra_args="--lora_use_map" ;;
        dash)  extra_args="--lora_use_dash" ;;
        lora)  extra_args="" ;;
        *)     echo "Unknown method: $method"; exit 1 ;;
    esac

    CUDA_VISIBLE_DEVICES=$GPU accelerate launch train.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --instance_data_dir="$INSTANCE_DIR" \
        --pretrained_vae_model_name_or_path="$VAE_PATH" \
        --output_dir="$out_dir" \
        --mixed_precision="fp16" \
        --instance_prompt="a photo of ${SUBJECT}" \
        --resolution=1024 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --learning_rate=1e-4 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=$STEPS \
        --validation_prompt="A photo of ${SUBJECT} in a park" \
        --validation_epochs=100 \
        --seed=42 \
        $extra_args

    echo "Model saved to $out_dir"
}

eval_method() {
    local method=$1
    local out_dir="$SDG_DIR/output/${method}-${SUBJECT}"
    local gen_dir="$SDG_DIR/output/${method}-${SUBJECT}-generated"
    mkdir -p "$gen_dir"

    echo "--- Generating images: $method on $SUBJECT ---"
    CUDA_VISIBLE_DEVICES=$GPU python infer.py \
        --model_dir "$out_dir" \
        --subject "$SUBJECT" \
        --output_dir "$gen_dir" \
        --num_images 25

    echo "--- Evaluating: $method ---"
    python eval_dreambooth.py \
        --subject "$SUBJECT" \
        --generated_dir "$gen_dir" \
        --reference_dir "$INSTANCE_DIR" \
        --prompts "a photo of ${SUBJECT} in a park" \
                  "a photo of ${SUBJECT} on a table" \
                  "a photo of ${SUBJECT} in the snow" \
                  "a photo of ${SUBJECT} in a bucket" \
                  "a photo of ${SUBJECT} in a jungle" \
        | tee "$out_dir/eval_results.txt"
}

echo "================================================"
echo "Task : Subject-driven Generation"
echo "Method=$METHOD  Subject=$SUBJECT  GPU=$GPU"
echo "================================================"

if [ "$METHOD" = "all" ]; then
    for M in lora lomap dash; do
        train_method "$M"
        eval_method "$M"
    done
    echo "--- Summary ---"
    for M in lora lomap dash; do
        echo "=== $M ==="; cat "$SDG_DIR/output/${M}-${SUBJECT}/eval_results.txt" 2>/dev/null || echo "(no results)"
    done
else
    train_method "$METHOD"
    eval_method "$METHOD"
fi

echo "Done."
