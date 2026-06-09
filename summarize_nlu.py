#!/usr/bin/env python3
"""
summarize_nlu.py — print a summary table of NLU GLUE results.

Usage:
  python summarize_nlu.py [output_dir]

Default output_dir: NLU/output/glue/
"""
import json
import os
import sys
import statistics
from pathlib import Path

OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("NLU/output/glue")

TASKS = ["cola", "sst2", "mrpc", "qqp", "qnli", "rte", "stsb", "mnli"]

# Task → primary eval metric key (as it appears in all_results.json)
TASK_METRIC = {
    "cola":  "eval_matthews_correlation",
    "sst2":  "eval_accuracy",
    "mrpc":  "eval_accuracy",
    "qqp":   "eval_accuracy",
    "qnli":  "eval_accuracy",
    "rte":   "eval_accuracy",
    "stsb":  "eval_pearson",
    "mnli":  "eval_accuracy",
}

PAPER_LOMAP_R2 = {
    "cola": 70.38, "sst2": 95.91, "mrpc": 91.67, "qqp": 91.83,
    "qnli": 94.31, "rte": 89.16, "stsb": 92.14, "mnli": 90.52,
}
PAPER_LORA_R2 = {
    "cola": 69.15, "sst2": 93.92, "mrpc": 90.19, "qqp": 90.61,
    "qnli": 93.37, "rte": 87.01, "stsb": 90.75, "mnli": 90.03,
}

def load_result(method, task, rank, seed):
    p = OUTPUT_DIR / f"{method}_{task}_r{rank}_seed{seed}" / "model" / "all_results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    metric_key = TASK_METRIC[task]
    val = d.get(metric_key)
    if val is None:
        return None
    return round(val * 100, 2)

def method_stats(method, task, rank, seeds=(6, 7, 8)):
    vals = [v for s in seeds if (v := load_result(method, task, rank, s)) is not None]
    if not vals:
        return None, None, None
    avg = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return vals, avg, std

def print_table(rank=2, seeds=(6, 7, 8)):
    methods = [("map", "LoMAP"), ("lora", "LoRA"), ("delora", "DeLoRA")]
    paper_refs = {"map": PAPER_LOMAP_R2, "lora": PAPER_LORA_R2}

    print(f"\n{'='*90}")
    print(f"  NLU Results  r={rank}  seeds={list(seeds)}")
    print(f"{'='*90}")
    print(f"{'Method':<12} {'Task':<8} {'Seeds':<22} {'Avg':>7} {'±Std':>6} {'Paper':>7} {'Gap':>7}")
    print(f"{'-'*90}")

    for key, name in methods:
        task_avgs = []
        for task in TASKS:
            vals, avg, std = method_stats(key, task, rank, seeds)
            if vals is None:
                print(f"{name:<12} {task:<8} {'(not run)':^22}  {'—':>7}  {'—':>5}  {paper_refs[key].get(task, 0.0):>7.2f}  {'—':>7}")
                continue
            seed_str = "  ".join(f"{v:.2f}" for v in vals)
            paper_val = paper_refs[key].get(task, None)
            gap_str = f"{avg - paper_val:+.2f}" if paper_val else "—"
            print(f"{name:<12} {task:<8} {seed_str:<22}  {avg:>7.2f}  {std:>5.2f}  {paper_val if paper_val else 0:>7.2f}  {gap_str:>7}")
            task_avgs.append(avg)

        if task_avgs:
            overall = statistics.mean(task_avgs)
            paper_overall = statistics.mean(paper_refs[key].values())
            print(f"{name:<12} {'ALL':<8} {'':^22}  {overall:>7.2f}  {'':>5}  {paper_overall:>7.2f}  {overall-paper_overall:>+7.2f}")
        print()

if __name__ == "__main__":
    print_table(rank=2)
    print_table(rank=8)
