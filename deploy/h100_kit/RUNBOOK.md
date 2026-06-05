# H100 Experiment Runbook

End-to-end guide to run the remaining LoMAP experiments on a PBS H100 cluster,
with the exact numbers we expect so anyone can tell at a glance whether a run is
healthy or broken.

---

## 0. Pre-flight (once)

```bash
# On a machine with the repo:
rsync -av --exclude='.venv' --exclude='NLU/output' --exclude='CR_MR/output' \
      --exclude='*.pyc' --exclude='__pycache__' \
      ~/Subspace-Tuning/  USER@HEADNODE:~/Subspace-Tuning/

# On the H100 head node:
cd ~/Subspace-Tuning
bash deploy/h100_kit/h100_setup.sh $HF_TOKEN     # HF_TOKEN only needed for LLaMA-3-8B
```

`h100_setup.sh` installs torch 2.4+cu121, the patched NLU transformers, the CR
PEFT fork, pre-downloads GLUE, and logs into HF. CR data (`commonsense_170k.json`,
96 MB) ships in the repo. `huggyllama/llama-7b` is ungated; only LLaMA-3-8B needs
the token.

### Sanity smoke test (10 min, 1 GPU) before submitting the full battery
```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/NLU/src:$PWD/loralib
bash deploy/h100_kit/scripts/run_nlu_grid.sh base 2 6 0    # cola+... seed 6 only, GPU 0
# Expect: cola LoMAP ≈ 71-72, runs to completion, all_results.json written.
```

---

## 1. Submit everything

```bash
bash deploy/h100_kit/submit_all.sh ~/Subspace-Tuning
qstat -u $USER          # watch the queue
tail -f logs/grid_dispatch.log
```

Seven PBS jobs, each `select=1:ngpus=8`. They are **idempotent**: re-submitting
skips finished runs, repairs corrupted checkpoints, and resumes. A preemption or
reboot loses at most one in-flight run.

| PBS job | Content | ~wall (8×H100) |
|---------|---------|----------------|
| nlu_base_r2  | DeBERTa-base r=2, 8 tasks × {LoMAP,LoRA} × 3 seeds | 4 h |
| nlu_base_r8  | DeBERTa-base r=8, same grid | 6 h |
| nlu_large_r2 | DeBERTa-large r=2, same grid | 10 h |
| nlu_large_r8 | DeBERTa-large r=8, same grid | 14 h |
| cr_llama7b   | LLaMA-7B r∈{4,8,16,32} × {LoMAP,LoRA,DoRA} + 8-bench eval | 18 h |
| cr_llama3_8b | LLaMA-3-8B r∈{16,32} × {LoMAP,LoRA,DoRA} + eval | 12 h |
| ablation     | LLaMA-7B r=16, sweep β₀/ε/detach/norm_scope | 6 h |

All in parallel ≈ **14-18 h wall-clock** on a well-provisioned queue.

---

## 2. What we expect (health-check targets)

### NLU — DeBERTa-v3-base r=2  (anchored by our local 4080 runs)
These already ran locally; H100 only adds qnli/qqp/mnli. Use as integrity check —
H100 numbers should land within ±0.5 of these (FP32, same seeds):

| Task | LoMAP (local) | LoRA (local) | Paper LoMAP | Paper LoRA |
|------|---------------|--------------|-------------|------------|
| CoLA | 71.5 | 70.5 | 70.38 | 69.15 |
| SST-2 | 96.1 | (pending) | 95.91 | 93.92 |
| MRPC | 90.7 | 91.0 | 91.67 | 90.19 |
| RTE | 88.8 | 88.3 | 89.16 | 87.01 |
| STS-B | 91.9 | 91.5 | 92.14 | 90.75 |
| **QNLI** | **? (expect ~94)** | ? (~93.5) | 94.31 | 93.37 |
| **QQP** | **? (expect ~91-92)** | ? (~91) | 91.83 | 90.61 |
| **MNLI** | **? (expect ~90.5)** | ? (~90) | 90.52 | 90.03 |

> Red flag: any LoMAP task >1.5 below paper, or LoMAP < LoRA by >0.5 on CoLA/SST-2.
> Expected overall: LoMAP ≈ LoRA (+0.3–0.5 mean), LoMAP variance ≤ LoRA.

### NLU r=8 / large
Expect the same ordering as r=2; absolute scores rise ~0.5–1.0 with r=8 and
~1.5–2.5 with large. Paper large r=2 LoMAP avg = 91.07, large r=8 = 91.18.

### CR — LLaMA-7B (paper Table, avg over 8 commonsense benchmarks)

| r | Paper LoMAP avg | Paper LoRA avg | Paper DoRA |
|---|-----------------|----------------|------------|
| 4  | 75.9 | ~70 | — |
| 8  | 76.4 | — | — |
| 16 | 77.9 | 70.9 | (n/a) |
| 32 | 78.8 | — | 78.4 |

LLaMA-3-8B: paper LoMAP r=32 = 85.8 (vs DoRA 85.2, AdaLoRA 82.1).

> Per-benchmark sanity (LLaMA-7B r=16 LoMAP): BoolQ ~70, PIQA ~83, SIQA ~80,
> HellaSwag ~91, WinoGrande ~84, ARC-e ~85, ARC-c ~71, OBQA ~83.
> Red flag: any benchmark near chance (BoolQ ~62, PIQA ~50) ⇒ adapter not merged
> or wrong base model.

### Ablation (LLaMA-7B r=16, single seed)
Expect **flat** curves: β₀∈{0,1e-4,1e-3,1e-2,1.0}, ε∈{1e-3..1e-8}, detach on/off,
norm_scope global/column/row all within ~±0.5 of each other. The story is
*robustness* (default config is near-optimal), consistent with our local CoLA
finding that these knobs are insensitive.

---

## 3. Collect results

```bash
# NLU summary table (LoMAP vs LoRA vs paper, all cells):
.venv/bin/python summarize_nlu.py

# CR aggregation (mean over 8 benchmarks per method/rank):
.venv/bin/python CR_MR/scripts_for_baselines/aggregate_results.py CR_MR/output
```

Per-run artifacts:
- NLU: `NLU/output/glue/<size>_<method>_<task>_r<R>_seed<S>/model/all_results.json`
- CR:  `CR_MR/output/<model>_<method>_r<R>_a<A>/<benchmark>.txt`

---

## 4. Failure playbook

| Symptom | Cause | Action |
|---------|-------|--------|
| Job dies at walltime | task didn't finish in window | re-`qsub` same .pbs — resumes from checkpoint |
| `EOFError`/torch.load on resume | torn checkpoint from preemption | auto-handled: grid runs fix_corrupted_results.sh first |
| CR eval all ~chance | wrong base model / unmerged adapter | check HF_TOKEN, confirm adapter_model.bin >1KB |
| OOM on large+qnli | seq 512 + big model | grid already uses bs16+accum2 for large/qnli |
| LoMAP ≈ LoRA exactly | lora_type not translated | verify `--lora_type map` reached run_glue (grid maps lora→frd, map→map) |
| `--local-rank` argparse error | someone used torch.distributed.launch | grids call python directly — don't wrap in launch |

---

## 5. Decision gate after results land

The local data already shows LoMAP's edge over a well-tuned LoRA is mild (+0.4
mean, lower variance). After H100 fills in qnli/qqp/mnli + r=8 + large + CR:

- **If CR shows the paper's large gaps (LoMAP +5–7 over LoRA at r=16):** that's
  the strong result — lead the paper with commonsense reasoning, NLU as support.
- **If CR gap is also mild:** reframe around parameter-minimal design + robustness
  + the provable adaptation-radius bound; do NOT claim large accuracy deltas.

CR is the higher-stakes, higher-signal track — watch it first.
