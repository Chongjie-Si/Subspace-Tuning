
# Adapting Pre-trained Models for Commonsense Reasoning and Math Reasoning Tasks

This folder contains the implementations for commonsense reasoning and math reasoning tasks.

## Setup Environment

### Create and Activate the Conda Environment

```bash
conda create -n Reasoning python=3.10
conda activate Reasoning
```

### Install the Pre-requisites

```bash
pip install -r requirements.txt
```

## Code Structure

- [tuners/](./peft/src/peft/tuners/) contains the PEFT methods used for this task. For more PEFT methods, you can use the implementations in [loralib/](../loralib/loralib/) to modify the corresponding codes.
- [dataset/](./dataset/) contains the datasets for testing.
- [peft/](./peft/) contains all the codes related to PEFT.
- ``scripts_for_../`` contains the training and evaluation scripts for different tasks.
- [finetune](./finetune.py) is the training code.
- [commonsense_evaluate](./commonsense_evaluate.py) is the evaluation code for commonsense reasoning tasks.
- [math_evaluate](./math_evaluate.py) is the evaluation code for math reasoning tasks.

## Adapting LLAMA / LLAMA2 / LLAMA3 for CR Tasks

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Fine-tuning and Evaluating LLAMA with LoRA</span></strong></summary>

```bash
# Fine-tuning
# llama.sh

CUDA_VISIBLE_DEVICES=$4 python finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path 'commonsense_170k.json' \
    --output_dir $3 \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing
```

```bash
sh llama.sh 32 64 ./finetune/lora_r=32/ 0
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

- `$1`: the rank of LoRA.
- `$2`: the corresponding alpha of LoRA.
- `$3`: where to save the fine-tuned model.
- `$4`: GPU number.
- `--adapter_name`: the method used for fine-tuning.
- `--target_modules`: which modules for fine-tuning.

```bash
# Evaluating
# part of llama_eval.sh

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset boolq \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/boolq.txt

    ...
    ...
```

```bash
sh llama_eval.sh ./finetune/lora_r=32/ 0
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

- `$1`: the location of fine-tuned weights
- `$2`: GPU number

 </details>

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Fine-tuning and Evaluating LLAMA2 with DoRA</span></strong></summary>

```bash
sh llama2.sh 32 64 ./finetune/dora_r=32/ 0
```

```bash
sh llama2_eval.sh ./finetune/dora_r=32/ 0
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

See first example for details on hyper-parameters.

 </details>

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Fine-tuning and Evaluating LLAMA3 with LoRA-Dash</span></strong></summary>

First, please replace the codes in [lora.py](./peft/src/peft/tuners/lora.py) with those in [lora-dash.py](./peft/src/peft/tuners/lora-dash.py). Then, run the scripts as follows:

```bash
sh llama3.sh 32 64 ./finetune/lora-dash_r=32/ 0
```

```bash
sh llama3_eval.sh ./finetune/lora-dash_r=32/ 0
```

For other methods, please follow a similar pipeline.

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

See first example for details on hyper-parameters.

 </details>

## Adapting LLAMA / BLOOMz / GPT for MR Tasks

The setup for the MR task is basically the same as for the CR task. You can directly run the corresponding script.

## GPU Memory

Considering that you may have different computing resources, we provide a simple table of the GPU resources required for the different models.

| Model Name | GPU Requirement |
|------------|-----------------|
| LLAMA | A100 (80G) |
| LLAMA2 | A100 (80G) |
| LLAMA3 | A100 (80G) |
| BLOOMz | A100 (80G) |
| GPT | A100 (80G) |

## Q&A

If you encounter any issues, please refer to these two links: [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters/issues) and [DoRA](https://github.com/NVlabs/DoRA/issues). These resources cover the majority of problems you may encounter. Additionally, you are also welcome to contact us by submitting an [issue](https://github.com/Chongjie-Si/Subspace-Tuning/issues) or via [email](mailto:chongjiesi@sjtu.edu.cn).

## Acknowledgement

This directory is modified based on [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters) and [DoRA](https://github.com/NVlabs/DoRA). We greatly appreciate their remarkable works.
