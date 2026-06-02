# Baseline Reproduction Guide

This directory contains scripts for reproducing baseline comparisons required for the AAAI 2026 paper.

## Setup Principles

All baselines use:
- **Same base model**: LLaMA-7B (`huggyllama/llama-7b`) for CR, DeBERTa-v3-base for GLUE
- **Same training data**: `commonsense_170k.json` for CR
- **Same rank**: match the LoMAP rank you are comparing against
- **Same seed**: run ≥3 seeds and report mean ± std
- **Same evaluation**: `commonsense_evaluate.py` for CR, `run_glue.py` for GLUE

## Data Leakage Note (Known Issue in Established Protocol)

The `commonsense_170k.json` training set follows the protocol of Hu et al. (2023) "LLM-Adapters".
A fraction of the test-set questions appear verbatim in the training mixture:
- HellaSwag: ~29 % of test examples
- WinoGrande: ~26 % of test examples
- BoolQ: ~3.5 %
- Other tasks: < 1 %

This is a **known property of the LLM-Adapters benchmark** used by all prior work
(LoRA, DoRA, PiSSA, MiLoRA, FLoRA, etc.). Because every method trains on the same data,
the comparison remains fair within the benchmark. However, absolute numbers on HellaSwag
and WinoGrande are inflated relative to truly held-out evaluation.

**For the paper**: do not claim absolute SoTA on HellaSwag/WinoGrande; instead highlight
relative gains over baselines, and note the established protocol in the experimental setup.

## Script Overview

| Script | Method | Tasks | Source |
|--------|--------|-------|--------|
| `run_delora_cr.sh` | DeLoRA | CR (LLaMA-7B) | HF PEFT >= 0.12.0 |
| `run_delora_glue.sh` | DeLoRA | GLUE (DeBERTa-v3) | HF PEFT >= 0.12.0 |
| `run_bidora_cr.sh` | DoRA proxy (CR) | CR (LLaMA-7B) | This repo's peft fork |
| `run_bidora_glue.sh` | BiDoRA | GLUE (RoBERTa) | github.com/t2ance/BiDoRA |
| `run_loraga_cr.sh` | LoRA-GA | CR (LLaMA-7B) | HF PEFT >= 0.12.0 (`init_lora_weights="lora-ga"`) |
| `run_randlora_gralora_cr.sh` | RandLoRA, GraLoRA | CR (LLaMA-7B) | HF PEFT >= 0.13.0 |

## Note on BiDoRA

The official BiDoRA repo (https://github.com/t2ance/BiDoRA) targets NLU tasks (GLUE with
RoBERTa/DeBERTa) and does **not** provide a commonsense reasoning script for LLaMA.
- For CR: `run_bidora_cr.sh` runs DoRA (the closest available baseline) as a proxy.
- For GLUE/NLU: `run_bidora_glue.sh` uses the official BiDoRA repo.

## Note on LoRA-GA

The official LoRA-GA repo (https://github.com/Outsider565/LoRA-GA) uses Hydra config
management. For commonsense reasoning we use HF PEFT's `init_lora_weights="lora-ga"`
inside our standard `finetune_peft.py` pipeline, which implements the same
gradient-approximation initialization.

## Installation

```bash
# DeLoRA, RandLoRA, GraLoRA, LoRA-GA via HF PEFT
pip install "peft>=0.13.0"

# BiDoRA — official implementation (for GLUE only)
git clone https://github.com/t2ance/BiDoRA ../../BiDoRA
pip install -r ../../BiDoRA/requirements.txt
```

## Multi-seed Evaluation

Run with 3 seeds and aggregate:
```bash
for SEED in 6 42 123; do
    bash run_delora_cr.sh 16 32 0 $SEED
done

python aggregate_results.py --method delora_cr --rank 16 --seeds 6 42 123
```

## Parameter Budget Matching

For a fair comparison with LoMAP:
- **Rank-matched**: same `r` value
- **Parameter-matched**: adjust `r` so total trainable params ≈ LoMAP params

LoMAP adds 2 extra scalar params per adapted layer (map_alpha, map_beta).
At rank r with N adapted layers: LoMAP params ≈ LoRA params + 2N (negligible difference).

DoRA adds `out_features` per adapted layer, which is significant.
For DeBERTa-v3-base (hidden=768): DoRA at r=2 adds 768 params/layer vs LoMAP's 2.
