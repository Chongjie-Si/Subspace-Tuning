#!/usr/bin/env python3
"""
aggregate_results.py — parse per-seed, per-dataset evaluation output files and
compute mean ± std across seeds for commonsense reasoning experiments.

Each evaluation run produces a file like:
    output/{method}/llama7b_r{rank}_seed{seed}/{dataset}.txt

The file contains lines from commonsense_evaluate.py, e.g.:
    test: 100/100 | accuracy 68  0.68

This script finds the last accuracy line in each file and aggregates.

Usage:
    # Aggregate a single method at rank 16 across 3 seeds
    python aggregate_results.py --method delora_cr --rank 16 --seeds 6 42 123

    # Aggregate multiple methods
    python aggregate_results.py --method lora lomap delora_cr --rank 16 --seeds 6 42 123

    # Point to a specific output root
    python aggregate_results.py --method lomap --rank 16 --seeds 6 42 123 \
        --output_root ./output

Output:
    Prints a LaTeX-ready table row + a CSV summary to stdout.
"""

import argparse
import os
import re
import json
from typing import Dict, List, Optional, Tuple
import statistics


DATASETS = [
    "boolq", "piqa", "social_i_qa", "hellaswag",
    "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa",
]

DATASET_SHORT = {
    "boolq": "BoolQ",
    "piqa": "PIQA",
    "social_i_qa": "SIQA",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "ARC-Challenge": "ARC-c",
    "ARC-Easy": "ARC-e",
    "openbookqa": "OBQA",
}


def parse_accuracy_from_file(filepath: str) -> Optional[float]:
    """
    Extract the final accuracy from a commonsense_evaluate.py output file.
    Lines look like:
        test:100/100 | accuracy 68  0.68
    We want the float accuracy (last field on the accuracy line).
    Falls back to parsing the JSON experiment file if .txt gives no result.
    """
    if not os.path.exists(filepath):
        return None

    acc = None
    with open(filepath, "r") as f:
        for line in f:
            # Pattern: "accuracy 68  0.68" or "accuracy 0.68"
            m = re.search(r"accuracy\s+\d+\s+([\d.]+)", line)
            if m:
                acc = float(m.group(1)) * 100  # store as percent
            # Also accept bare float at end
            m2 = re.search(r"accuracy\s+([\d.]+)$", line.strip())
            if m2:
                v = float(m2.group(1))
                acc = v * 100 if v <= 1.0 else v
    return acc


def parse_accuracy_from_json(json_path: str) -> Optional[float]:
    """Fall back: compute accuracy from the JSON prediction file."""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
        if not data:
            return None
        correct = sum(1 for d in data if d.get("flag", False))
        return correct / len(data) * 100
    except Exception:
        return None


def collect_seed_results(
    output_root: str,
    method: str,
    rank: int,
    seeds: List[int],
) -> Dict[str, List[float]]:
    """
    Returns a dict: dataset -> list of accuracy values (one per seed).
    Missing files produce None entries (excluded from stats).
    """
    results: Dict[str, List[float]] = {ds: [] for ds in DATASETS}

    for seed in seeds:
        run_dir = os.path.join(output_root, method, f"llama7b_r{rank}_seed{seed}")
        for ds in DATASETS:
            txt = os.path.join(run_dir, f"{ds}.txt")
            json_file = os.path.join("experiment", f"LLaMA-7B-{method.replace('_cr','').upper()}-{ds}.json")
            acc = parse_accuracy_from_file(txt)
            if acc is None:
                acc = parse_accuracy_from_json(json_file)
            if acc is not None:
                results[ds].append(acc)

    return results


def summarize(values: List[float]) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return f"{mean:.1f}±{std:.1f}"


def print_latex_row(method: str, rank: int, results: Dict[str, List[float]]):
    all_accs = [v for vals in results.values() for v in vals]
    if not all_accs:
        print(f"% No results found for {method} r={rank}")
        return

    # compute per-dataset mean, then macro-average
    per_ds_means = []
    cols = []
    for ds in DATASETS:
        vals = results[ds]
        if vals:
            m = statistics.mean(vals)
            per_ds_means.append(m)
            cols.append(f"{m:.1f}")
        else:
            cols.append("--")

    avg = statistics.mean(per_ds_means) if per_ds_means else 0.0
    cols.append(f"{avg:.1f}")
    print(f"% {method} r={rank}")
    print(" & ".join([method] + cols) + " \\\\")


def print_csv(method: str, rank: int, results: Dict[str, List[float]]):
    header = ["method", "rank"] + [DATASET_SHORT[ds] for ds in DATASETS] + ["Avg"]
    print(",".join(header))
    per_ds = []
    row = [method, str(rank)]
    for ds in DATASETS:
        vals = results[ds]
        s = summarize(vals)
        row.append(s)
        if vals:
            per_ds.append(statistics.mean(vals))
    avg = f"{statistics.mean(per_ds):.1f}" if per_ds else "N/A"
    row.append(avg)
    print(",".join(row))


def main():
    parser = argparse.ArgumentParser(description="Aggregate commonsense reasoning results across seeds.")
    parser.add_argument("--method", nargs="+", required=True,
                        help="Method name(s), matching subdirectory in --output_root. "
                             "E.g. lora lomap delora_cr loraga_cr")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seeds", type=int, nargs="+", default=[6, 42, 123])
    parser.add_argument("--output_root", default="./output",
                        help="Root directory containing method subdirectories")
    parser.add_argument("--format", choices=["latex", "csv", "both"], default="both")
    args = parser.parse_args()

    for method in args.method:
        results = collect_seed_results(args.output_root, method, args.rank, args.seeds)
        n_found = sum(len(v) for v in results.values())
        print(f"\n=== {method} r={args.rank} (seeds={args.seeds}, {n_found} data points found) ===")

        if args.format in ("latex", "both"):
            print("\n[LaTeX row]")
            print_latex_row(method, args.rank, results)

        if args.format in ("csv", "both"):
            print("\n[CSV]")
            print_csv(method, args.rank, results)

        print("\n[Per-dataset breakdown]")
        for ds in DATASETS:
            vals = results[ds]
            print(f"  {DATASET_SHORT[ds]:12s}: {summarize(vals):12s}  (n={len(vals)})")


if __name__ == "__main__":
    main()
