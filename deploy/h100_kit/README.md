# LoMAP H100 Deployment Kit

Drop-in package to reproduce the AAAI submission's full experiment battery on a PBS H100 cluster. **Tested: PBS Pro / TORQUE.**

## Quick start

```bash
# 1. rsync repo to cluster (or git clone)
rsync -av --exclude='.venv' --exclude='NLU/output' --exclude='CR_MR/output' \
    /home/li/Subspace-Tuning/ user@cluster:~/Subspace-Tuning/

# 2. SSH to head node, set up env (one-time, ~10 min)
cd ~/Subspace-Tuning
bash deploy/h100_kit/h100_setup.sh $HF_TOKEN     # HF_TOKEN needed for LLaMA-3

# 3. Submit all jobs (parallel queue on H100)
bash deploy/h100_kit/submit_all.sh ~/Subspace-Tuning

# 4. Watch progress
qstat -u $USER
tail -f logs/grid_dispatch.log

# 5. Summary as runs complete
.venv/bin/python summarize_nlu.py
```

## What gets run

| PBS job | Hours (8×H100) | Output |
|---------|----------------|--------|
| `nlu_base_r2.pbs`   | ~4   | NLU Table 1, DeBERTa-base, r=2 |
| `nlu_base_r8.pbs`   | ~6   | NLU Table 1, DeBERTa-base, r=8 |
| `nlu_large_r2.pbs`  | ~10  | NLU Table 1, DeBERTa-large, r=2 |
| `nlu_large_r8.pbs`  | ~14  | NLU Table 1, DeBERTa-large, r=8 |
| `cr_llama7b.pbs`    | ~18  | CR Table 2, LLaMA-7B, r=4/8/16/32 |
| `cr_llama3_8b.pbs`  | ~12  | CR Table 2, LLaMA3-8B, r=16/32 |
| `ablation.pbs`      | ~6   | §5: β₀, ε, detach, norm_scope |

**All 7 jobs in parallel ≈ 14 h wall-clock** (longest job dominates) on a sufficiently provisioned queue.

## Internals

- **Method dispatch.** Each grid script (`run_*_grid.sh`) lists every `(method, task, seed)` triple, dispatches them round-robin onto the GPU pool. A single Python process per GPU; concurrent training on all 8 cards.
- **Auto-resume.** If `all_results.json` (NLU) or `adapter_model.bin` (CR) already exists for a config, the grid skips it. Re-submitting any job continues where the previous one left off.
- **Checkpoint policy.** Long tasks (MNLI/QQP) use `save_steps=2000`, others `save_steps=200` — bounded I/O so large queues don't thrash disks.
- **Memory hygiene.** All NLU runs use `--skip_memory_metrics` (no `tracemalloc`); checkpoints capped at `save_total_limit=2`; HF best-checkpoint reload happens at the end of each run.
- **Best-metric vs final-eval.** Each task's `metric_for_best_model` is set explicitly (mcc / pearson / accuracy), so the saved checkpoint is the right one even after preemption.

## Files

```
deploy/h100_kit/
├── README.md                    # this file
├── h100_setup.sh                # one-shot env + dataset prep
├── submit_all.sh                # qsub the 7 jobs
├── pbs_templates/
│   ├── nlu_base_r2.pbs / nlu_base_r8.pbs
│   ├── nlu_large_r2.pbs / nlu_large_r8.pbs
│   ├── cr_llama7b.pbs / cr_llama3_8b.pbs
│   └── ablation.pbs
└── scripts/
    ├── run_nlu_grid.sh          # dispatcher for NLU
    ├── run_cr_grid.sh           # dispatcher for Commonsense Reasoning
    └── run_ablation_grid.sh     # §5 sensitivity sweep
```

## Tuning per cluster

If your PBS queue requires different resource specs, edit the top of each `.pbs` file:
- `#PBS -l select=1:ncpus=16:ngpus=8:mem=128gb`  → adjust ncpus/ngpus/mem
- `#PBS -l walltime=24:00:00`                    → adjust walltime
- Add `#PBS -q <queuename>` if needed
- Add `#PBS -A <project>` for charging

If you have fewer than 8 GPUs per node, edit the `GPUS_CSV` argument in the matching `.pbs` to a shorter list (e.g. `0,1,2,3` for 4 GPUs); the grid script will detect and dispatch round-robin.
