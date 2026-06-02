#!/bin/bash
# setup_envs.sh — one-shot environment setup for all three tasks
#
# Creates three isolated conda environments:
#   lomap-cr   : Commonsense / Math Reasoning (LLaMA, finetune.py)
#   lomap-nlu  : GLUE / NLU (DeBERTa, run_glue.py)
#   lomap-sdg  : Subject-driven Generation (SDXL, diffusers)
#
# Usage:
#   bash setup_envs.sh            # set up all three
#   bash setup_envs.sh cr         # only CR
#   bash setup_envs.sh nlu        # only NLU
#   bash setup_envs.sh sdg        # only SdG
#
# After setup, activate with:
#   conda activate lomap-cr | lomap-nlu | lomap-sdg

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET=${1:-all}

# ────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────
check_conda() {
    if ! command -v conda &>/dev/null; then
        echo "ERROR: conda not found. Install Miniconda first."
        exit 1
    fi
}

# ────────────────────────────────────────────────
# CR — Commonsense & Math Reasoning
# ────────────────────────────────────────────────
setup_cr() {
    echo "========================================="
    echo "Setting up lomap-cr (CR / Math Reasoning)"
    echo "========================================="
    conda create -y -n lomap-cr python=3.10
    conda run -n lomap-cr pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    conda run -n lomap-cr pip install -r "$REPO_ROOT/CR_MR/requirements.txt"
    # Install repo's PEFT fork (contains LoMAP)
    conda run -n lomap-cr pip install -e "$REPO_ROOT/CR_MR/peft/"
    echo "lomap-cr ready. Activate with: conda activate lomap-cr"
}

# ────────────────────────────────────────────────
# NLU — GLUE benchmark (DeBERTa)
# ────────────────────────────────────────────────
setup_nlu() {
    echo "========================================="
    echo "Setting up lomap-nlu (NLU / GLUE)"
    echo "========================================="
    conda create -y -n lomap-nlu python=3.8
    # NLU uses older PyTorch (Python 3.7/3.8 + CUDA 11.1 originally)
    # Use a newer compatible combination for modern GPUs
    conda run -n lomap-nlu pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --index-url https://download.pytorch.org/whl/cu117
    # Install NLU-specific deps (skip tensorflow — not needed for inference)
    conda run -n lomap-nlu pip install \
        numpy tqdm datasets sentencepiece scikit-learn scipy \
        pandas requests dill mlflow tensorboardX \
        "transformers>=4.4.2,<4.5.0"
    # Install customised transformers (contains DeBERTa + loralib hooks)
    conda run -n lomap-nlu pip install -e "$REPO_ROOT/NLU/"
    # Install loralib (contains LoMAPLinear used by modeling_deberta_v2.py)
    conda run -n lomap-nlu pip install -e "$REPO_ROOT/loralib/"
    echo "lomap-nlu ready. Activate with: conda activate lomap-nlu"
}

# ────────────────────────────────────────────────
# SdG — Subject-driven Generation (SDXL + diffusers)
# ────────────────────────────────────────────────
setup_sdg() {
    echo "========================================="
    echo "Setting up lomap-sdg (Subject-driven Gen)"
    echo "========================================="
    conda create -y -n lomap-sdg python=3.10
    conda run -n lomap-sdg pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    conda run -n lomap-sdg pip install -r "$REPO_ROOT/SdG/requirements.txt"
    # Install diffusers from source (required for LoMAP DreamBooth training)
    TMP=$(mktemp -d)
    git clone --depth=1 https://github.com/huggingface/diffusers "$TMP/diffusers"
    conda run -n lomap-sdg pip install -e "$TMP/diffusers"
    # Install SdG peft fork
    conda run -n lomap-sdg pip install -e "$REPO_ROOT/SdG/peft/"
    # DreamBooth eval dependencies
    conda run -n lomap-sdg pip install open_clip_torch lpips
    echo "lomap-sdg ready. Activate with: conda activate lomap-sdg"
    echo "NOTE: Run 'conda activate lomap-sdg && accelerate config' before training."
}

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
check_conda

case "$TARGET" in
    cr)  setup_cr  ;;
    nlu) setup_nlu ;;
    sdg) setup_sdg ;;
    all)
        setup_cr
        setup_nlu
        setup_sdg
        echo ""
        echo "All environments created:"
        echo "  conda activate lomap-cr   # CR / Math"
        echo "  conda activate lomap-nlu  # GLUE / NLU"
        echo "  conda activate lomap-sdg  # DreamBooth / SdG"
        ;;
    *)
        echo "Usage: bash setup_envs.sh [cr|nlu|sdg|all]"
        exit 1
        ;;
esac
