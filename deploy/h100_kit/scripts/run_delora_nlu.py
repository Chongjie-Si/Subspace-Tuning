#!/usr/bin/env python3
"""run_delora_nlu.py — DeLoRA on DeBERTaV3 GLUE (H100).

Uses standard HF transformers + PEFT (>= 0.14, which ships DeLoRA support).
Mirrors the exact hyperparameters from the LoMAP paper (Table 4) so results
are directly comparable with the NLU Table.

Usage:
    # Single task/seed on GPU 0:
    CUDA_VISIBLE_DEVICES=0 python run_delora_nlu.py \
        --task cola --seed 6 --rank 2 --size base \
        --output_root NLU/output/glue

    # Full grid via the shell wrapper (run_delora_nlu_grid.sh):
    bash deploy/h100_kit/scripts/run_delora_nlu_grid.sh base 2 6,7,8 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import sys
import math
import numpy as np

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate

# ---------------------------------------------------------------------------
# Per-task hyperparameters (paper Table 4)
# ---------------------------------------------------------------------------
TASK_HPS = {
    "mnli":  dict(seq_len=256, lr=5e-4, epochs=12, cls_drop=0.10, metric="accuracy",  num_labels=3),
    "sst2":  dict(seq_len=128, lr=8e-4, epochs=24, cls_drop=0.00, metric="accuracy",  num_labels=2),
    "cola":  dict(seq_len=64,  lr=8e-4, epochs=25, cls_drop=0.10, metric="matthews_correlation", num_labels=2),
    "qqp":   dict(seq_len=320, lr=1e-3, epochs=5,  cls_drop=0.10, metric="accuracy",  num_labels=2),
    "qnli":  dict(seq_len=512, lr=5e-4, epochs=5,  cls_drop=0.10, metric="accuracy",  num_labels=2),
    "rte":   dict(seq_len=320, lr=1.2e-3, epochs=50, cls_drop=0.20, metric="accuracy", num_labels=2),
    "mrpc":  dict(seq_len=320, lr=1e-3, epochs=30, cls_drop=0.10, metric="accuracy",  num_labels=2),
    "stsb":  dict(seq_len=128, lr=5e-4, epochs=25, cls_drop=0.10, metric="pearson",   num_labels=1),
}

GLUE_KEY = {
    "mnli": "mnli", "sst2": "sst2", "cola": "cola",
    "qqp": "qqp", "qnli": "qnli", "rte": "rte",
    "mrpc": "mrpc", "stsb": "stsb",
}

TARGET_MODULES = ["query_projections", "key_projections", "value_projections",
                  "intermediate.dense", "output.dense", "self.out_proj"]

# DeBERTa-v3 actual module names (matched by suffix)
DEBERTA_TARGET_MODULES = [
    "query_proj", "key_proj", "value_proj",
    "intermediate.dense", "output.dense", "out_proj",
]

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics_fn(task, is_regression):
    glue_metric = hf_evaluate.load("glue", GLUE_KEY[task])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if is_regression:
            preds = logits.squeeze()
        else:
            preds = logits.argmax(axis=-1)
        result = glue_metric.compute(predictions=preds, references=labels)
        return result

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_HPS.keys()))
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--size", choices=["base", "large"], default="base")
    parser.add_argument("--output_root", default="NLU/output/glue")
    args = parser.parse_args()

    task = args.task
    hp = TASK_HPS[task]
    is_regression = (task == "stsb")
    rank = args.rank
    alpha = rank * 2  # paper convention
    size = args.size

    model_name = (
        "microsoft/deberta-v3-base" if size == "base"
        else "microsoft/deberta-v3-large"
    )
    out_dir = os.path.join(
        args.output_root,
        f"{size}_delora_{task}_r{rank}_seed{args.seed}",
    )
    results_path = os.path.join(out_dir, "model", "all_results.json")

    if os.path.exists(results_path):
        print(f"Already done: {results_path} — skipping.")
        sys.exit(0)

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    raw = load_dataset("glue", GLUE_KEY[task])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        if task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"],
                             truncation=True, max_length=hp["seq_len"])
        elif task in ("rte", "mrpc", "qqp", "qnli"):
            key1, key2 = {
                "rte": ("sentence1", "sentence2"),
                "mrpc": ("sentence1", "sentence2"),
                "qqp": ("question1", "question2"),
                "qnli": ("question", "sentence"),
            }[task]
            return tokenizer(examples[key1], examples[key2],
                             truncation=True, max_length=hp["seq_len"])
        elif task == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"],
                             truncation=True, max_length=hp["seq_len"])
        else:
            key = {"cola": "sentence", "sst2": "sentence"}[task]
            return tokenizer(examples[key], truncation=True, max_length=hp["seq_len"])

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names
                        if task != "mnli" else
                        [c for c in raw["train"].column_names if c != "label"])

    label_col = "label"
    train_ds = tokenized["train"]
    eval_key = "validation_matched" if task == "mnli" else "validation"
    eval_ds = tokenized[eval_key]

    collator = DataCollatorWithPadding(tokenizer)

    # -----------------------------------------------------------------------
    # Model + DeLoRA
    # -----------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=hp["num_labels"],
        hidden_dropout_prob=hp["cls_drop"],
        attention_probs_dropout_prob=hp["cls_drop"],
    )

    # Find actual target module names for this model
    found_targets = []
    for name, _ in model.named_modules():
        for t in DEBERTA_TARGET_MODULES:
            if name.endswith(t) and name not in found_targets:
                found_targets.append(t)
                break
    # Deduplicate while preserving order
    seen = set()
    unique_targets = []
    for t in found_targets:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=unique_targets or DEBERTA_TARGET_MODULES,
        use_dora=True,   # DeLoRA uses DoRA-style decoupled scaling in PEFT >= 0.14
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    best_metric = hp["metric"]
    save_steps = 200 if task not in ("mnli", "qqp", "qnli") else \
                 2000 if task in ("mnli", "qqp") else 1000

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=hp["lr"],
        warmup_ratio=0.1,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to="tensorboard",
        fp16=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(task, is_regression),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

    # -----------------------------------------------------------------------
    # Save results in NLU-compatible format
    # -----------------------------------------------------------------------
    metrics = trainer.evaluate()
    os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results: {metrics}")
    print(f"Saved to: {results_path}")


if __name__ == "__main__":
    main()
