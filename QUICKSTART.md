# QUICKSTART — MAP / LoMAP Experiments (AAAI 2026)

## 1. Clone & Setup

```bash
git clone https://github.com/haoruilee/Subspace-Tuning.git
cd Subspace-Tuning
git checkout codex/add-lomap

# Install all three conda environments (or pick one)
bash setup_envs.sh all      # all tasks
bash setup_envs.sh cr       # CR / Math only
bash setup_envs.sh nlu      # GLUE only
bash setup_envs.sh sdg      # DreamBooth only
```

GPU requirements per task:
| Task | Min VRAM | Recommended |
|------|----------|-------------|
| CR (LLaMA-7B) | 32 GB | A100 40GB |
| NLU (DeBERTa-v3-base) | 16 GB | A100 40GB |
| SdG (SDXL) | 24 GB | RTX 3090 / A100 |

---

## 2. Commonsense Reasoning

```bash
conda activate lomap-cr
cd /path/to/Subspace-Tuning

# LoMAP (3 seeds)
for SEED in 6 42 123; do
    bash run_cr.sh lomap 16 32 0 $SEED
done

# Baselines (same 3 seeds each)
for SEED in 6 42 123; do
    bash run_cr.sh lora    16 32 0 $SEED
    bash run_cr.sh delora  16 32 0 $SEED
    bash run_cr.sh loraga  16 32 0 $SEED
done

# Aggregate results
python CR_MR/scripts_for_baselines/aggregate_results.py \
    --method lomap lora delora_cr loraga_cr \
    --rank 16 --seeds 6 42 123
```

---

## 3. GLUE / NLU

```bash
conda activate lomap-nlu
cd /path/to/Subspace-Tuning

# LoMAP r=2 and r=8
bash run_nlu.sh map  2 4  6 0
bash run_nlu.sh map  8 16 6 0

# LoRA baselines
bash run_nlu.sh lora 2 4  6 0
bash run_nlu.sh lora 8 16 6 0

# DeLoRA (via HF PEFT — uses run_delora_glue.sh)
bash CR_MR/scripts_for_baselines/run_delora_glue.sh 2 6
bash CR_MR/scripts_for_baselines/run_delora_glue.sh 8 6
```

---

## 4. Subject-driven Generation (DreamBooth)

```bash
conda activate lomap-sdg
cd /path/to/Subspace-Tuning

# Must run once: accelerate config
accelerate config

# Train all three methods on a subject, then quantitative eval
bash run_sdg.sh all cat 0

# Individual
bash run_sdg.sh lomap cat 0
bash run_sdg.sh lora  cat 0
```

---

## 5. Ablation Experiments

```bash
conda activate lomap-cr

bash run_ablations.sh eps    0    # epsilon sensitivity
bash run_ablations.sh beta   0    # beta_0 sensitivity
bash run_ablations.sh norm   0    # norm_scope (global / column / row / row+col)
bash run_ablations.sh detach 0    # detach denominator
bash run_ablations.sh rank   0    # rank sweep r in {2,4,8,16,32,64}
bash run_ablations.sh all    0    # run everything
```

---

## 6. Collect Results

```bash
# CR multi-seed summary → LaTeX row + CSV
python CR_MR/scripts_for_baselines/aggregate_results.py \
    --method lomap lora delora_cr \
    --rank 16 --seeds 6 42 123 \
    --output_root CR_MR/output

# Tensorboard
tensorboard --logdir CR_MR/output/lomap/llama7b_r16_seed42/runs
```

---

## 7. What's on this branch (`codex/add-lomap`)

| Component | Where |
|-----------|-------|
| LoMAP core (loralib) | `loralib/loralib/layers.py` |
| LoMAP core (PEFT fork) | `CR_MR/peft/src/peft/tuners/lora.py` |
| LoMAP for GLUE (DeBERTa) | `NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py` |
| LoMAP for SdG (SDXL) | `SdG/train.py` |
| Baseline scripts | `CR_MR/scripts_for_baselines/` |
| Ablation scripts | `CR_MR/scripts_for_ablation/` |
| DreamBooth eval | `SdG/eval_dreambooth.py` |
| Result aggregation | `CR_MR/scripts_for_baselines/aggregate_results.py` |
| Paper (AAAI 2026) | `paper-aaai/AAAI_MAP/` |
