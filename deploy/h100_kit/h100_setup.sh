#!/bin/bash
# h100_setup.sh — one-shot environment + data setup on a fresh H100 node
# Run this ONCE on the head node (or in your job's first step) before submitting any training job.
#
# Assumes: CUDA 12.x, conda or python3, ~200GB free disk for HF cache + checkpoints
#
# Usage:
#   bash h100_setup.sh [HF_TOKEN]   # HF_TOKEN needed for LLaMA gated models

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HF_TOKEN=${1:-}

echo "================================================"
echo "LoMAP H100 setup"
echo "Repo root: $REPO_ROOT"
echo "================================================"

cd "$REPO_ROOT"

# 1. Create venv
if [ ! -d ".venv" ]; then
    echo ">>> Creating venv"
    python3 -m venv .venv
fi
VENV="$REPO_ROOT/.venv/bin/python"
PIP="$REPO_ROOT/.venv/bin/pip"

# 2. Install PyTorch 2.4 (matches paper, stable on H100)
echo ">>> Installing PyTorch 2.4 + cu121"
"$PIP" install --quiet torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install NLU stack (HF Transformers patched fork)
echo ">>> Installing NLU dependencies"
"$PIP" install --quiet sacremoses sentencepiece scikit-learn datasets==2.21.0 \
    tensorboard tensorboardX accelerate==0.34.2 evaluate==0.4.3
"$PIP" install --quiet -e "$REPO_ROOT/loralib/"

# 4. Install CR_MR PEFT fork
echo ">>> Installing CR_MR (PEFT fork with LoMAP)"
"$PIP" install --quiet -e "$REPO_ROOT/CR_MR/peft/"
"$PIP" install --quiet -r "$REPO_ROOT/CR_MR/requirements.txt"

# 5. Install SdG (DreamBooth) deps
if [ -f "$REPO_ROOT/SdG/requirements.txt" ]; then
    echo ">>> Installing SdG requirements"
    "$PIP" install --quiet -r "$REPO_ROOT/SdG/requirements.txt" || echo "SdG deps optional; ignore failures"
fi

# 6. HF login
if [ -n "$HF_TOKEN" ]; then
    echo ">>> HF login"
    "$VENV" -c "from huggingface_hub import login; login('$HF_TOKEN', add_to_git_credential=False)"
fi

# 7. Pre-download datasets
echo ">>> Pre-downloading GLUE"
"$VENV" -c "
from datasets import load_dataset
for t in ['cola','sst2','mrpc','qqp','qnli','rte','stsb','mnli']:
    print('GLUE:', t)
    load_dataset('glue', t)
" || echo "GLUE pre-download had issues (may need network); training will retry"

# 8. Pre-download model configs (small files, full weights pulled lazily)
echo ">>> Pre-downloading model configs"
"$VENV" -c "
from transformers import AutoConfig, AutoTokenizer
for m in ['microsoft/deberta-v3-base', 'microsoft/deberta-v3-large']:
    AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m)
    print('OK', m)
" || true

echo ""
echo "================================================"
echo "Setup done."
echo "Next: submit a job with"
echo "  qsub deploy/h100_kit/pbs_templates/nlu_base_r2.pbs"
echo "================================================"
