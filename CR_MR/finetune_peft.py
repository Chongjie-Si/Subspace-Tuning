#!/usr/bin/env python3
"""
finetune_peft.py — wrapper that uses HuggingFace PEFT directly (not this repo's fork)
to reproduce baselines: DeLoRA, RandLoRA, GraLoRA, LoRA-GA, etc.

This is intentionally a minimal standalone script so that each baseline
is run in the same training setup as finetune.py (same data, same tokenizer,
same hyper-parameters), but uses standard HF PEFT configs for the adapter.

Supported adapter_name values:
  lora        — standard LoRA via PEFT
  delora      — DeLoRA (PEFT >= 0.12.0, peft_type=DELORA)
  randlora    — RandLoRA (PEFT >= 0.13.0)
  gralora     — GraLoRA (PEFT >= 0.13.0)

Usage:
    python finetune_peft.py \
        --base_model huggyllama/llama-7b \
        --data_path commonsense_170k.json \
        --output_dir ./output/delora/llama7b_r16 \
        --adapter_name delora \
        --lora_r 16 --lora_alpha 32 \
        --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]' \
        --num_epochs 3 --batch_size 16 --micro_batch_size 16 \
        --learning_rate 2e-4
"""

import os
import sys
import json
import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional

# Use system PEFT (not this repo's fork)
import peft
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

PEFT_VERSION = tuple(int(x) for x in peft.__version__.split(".")[:2])


def _make_config(adapter_name: str, lora_r: int, lora_alpha: int,
                 lora_dropout: float, target_modules: List[str]):
    if adapter_name in ("lora", "delora", "randlora", "gralora"):
        kwargs = dict(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if adapter_name == "delora":
            from peft import DeLoraConfig
            return DeLoraConfig(**kwargs)
        elif adapter_name == "randlora":
            from peft import RandLoraConfig
            return RandLoraConfig(**kwargs)
        elif adapter_name == "gralora":
            from peft import GraLoraConfig
            return GraLoraConfig(**kwargs)
        else:
            return LoraConfig(**kwargs)
    else:
        raise ValueError(f"Unknown adapter_name: {adapter_name}")


def train(
    base_model: str = "",
    data_path: str = "commonsense_170k.json",
    output_dir: str = "./output",
    adapter_name: str = "lora",
    batch_size: int = 16,
    micro_batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    cutoff_len: int = 256,
    val_set_size: int = 120,
    eval_step: int = 80,
    save_step: int = 80,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = False,
    seed: int = 42,
):
    assert base_model, "Please specify --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size
    transformers.set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        trust_remote_code=True,
    )

    if "Llama-3" in base_model or "llama-3" in base_model.lower():
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    elif model.config.model_type == "llama":
        try:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    config = _make_config(adapter_name, lora_r, lora_alpha, lora_dropout,
                          target_modules or ["q_proj", "v_proj"])
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    def tokenize(prompt):
        result = tokenizer(
            prompt, truncation=True, max_length=cutoff_len,
            padding=False, return_tensors=None,
        )
        if (result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_prompt(data_point):
        if data_point.get("input", ""):
            return (f"Below is an instruction that describes a task, paired with an input that provides further context. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{data_point['instruction']}\n\n"
                    f"### Input:\n{data_point['input']}\n\n### Response:\n{data_point['output']}")
        return (f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{data_point['instruction']}\n\n### Response:\n{data_point['output']}")

    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    train_data = data["train"].shuffle(seed=seed).map(
        lambda x: tokenize(generate_prompt(x)), remove_columns=data["train"].column_names
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=eval_step,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            report_to="none",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
