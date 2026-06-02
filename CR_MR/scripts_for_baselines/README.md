# Baseline Reproduction Guide

This directory contains scripts for reproducing baseline comparisons required for the AAAI 2026 paper.

## Setup Principles

All baselines use:
- **Same base model**: LLaMA-7B (`huggyllama/llama-7b`) for CR, DeBERTa-v3-base for GLUE
- **Same training data**: `commonsense_170k.json` for CR
- **Same rank**: match the LoMAP rank you are comparing against
- **Same seed**: run ≥3 seeds and report mean ± std
- **Same evaluation**: `commonsense_evaluate.py` for CR, `run_glue.py` for GLUE

## Script Overview

| Script | Method | Tasks | Source |
|--------|--------|-------|--------|
| `run_delora_cr.sh` | DeLoRA | CR (LLaMA-7B) | HF PEFT >= 0.12.0 |
| `run_delora_glue.sh` | DeLoRA | GLUE (DeBERTa-v3) | HF PEFT >= 0.12.0 |
| `run_bidora_cr.sh` | BiDoRA | CR (LLaMA-7B) | github.com/t2ance/BiDoRA |
| `run_loraga_cr.sh` | LoRA-GA | CR (LLaMA-7B) | github.com/Outsider565/LoRA-GA |
| `run_randlora_gralora_cr.sh` | RandLoRA, GraLoRA | CR (LLaMA-7B) | HF PEFT >= 0.13.0 |

## Installation

```bash
# DeLoRA, RandLoRA, GraLoRA via HF PEFT
pip install "peft>=0.12.0"

# BiDoRA — official implementation
git clone https://github.com/t2ance/BiDoRA ../../BiDoRA
pip install -e ../../BiDoRA/

# LoRA-GA — official implementation
git clone https://github.com/Outsider565/LoRA-GA ../../LoRA-GA
# follow LoRA-GA's own installation guide
```

## Multi-seed Evaluation

Run with 3 seeds and aggregate:
```bash
for SEED in 6 42 123; do
    bash run_delora_cr.sh 16 32 0 $SEED
done

python aggregate_results.py --method delora --output output/delora/
```

## Parameter Budget Matching

For a fair comparison with LoMAP:
- **Rank-matched**: same `r` value
- **Parameter-matched**: adjust `r` so total trainable params ≈ LoMAP params

LoMAP adds 2 extra scalar params per adapted layer (map_alpha, map_beta).
At rank r with N adapted layers: LoMAP params ≈ LoRA params + 2N (negligible difference).

DoRA adds `out_features` per adapted layer, which is significant.
For DeBERTa-v3-base (hidden=768): DoRA at r=2 adds 768 params/layer vs LoMAP's 2.
