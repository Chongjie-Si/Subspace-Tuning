
# Adapting Pre-trained Models for Subject-driven Generation Task

This folder contains the implementations for subject-driven generation task.

## Setup Environment

### Create and Activate the Conda Environment

```bash
conda create -n SdG python=3.10
conda activate SdG
```

### Install the Pre-requisites

```bash
pip install -r requirements.txt
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

## Initialize the Accelerate

```bash
accelerate config
```

## Code Structure

- [cat/](./cat/) contains the training images. It is one folder from the [DreamBooth dataset](https://github.com/google/dreambooth/tree/main/dataset). You can also change [cat/](./cat/) with another folder in DreamBooth dataset.
- [peft/](./peft/) contains all the codes related to PEFT.
- [train.sh](./train.sh) is the training script.
- [infer.py](./infer.py) is the inference code.

## Adapting SDXL for SdG Tasks

The training script provides some PEFT methods, such as LoRA and LoRA-Dash, to fine-tune SDXL model for this task.

```bash
bash train.sh
```

### Hyper-parameter Setup

- `instance_prompt`: the prompt for the input images.
- `validation_prompt`: the prompt for validation.
- `lora_use_dash`: whether to use LoRA-Dash.

After training, run the following commands to generate the images:

```bash
python infer.py
```

## GPU Memory

Considering that you may have different computing resources, we have tested that this task can be conducted on one RTX 3090 GPU.

## Q&A

If you encounter any issues, please refer to this link: [huggingface/diffusers/examples](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md). It covers the majority of problems you may encounter. Additionally, you are also welcome to contact us by submitting an [issue](https://github.com/Chongjie-Si/Subspace-Tuning/issues) or via [email](mailto:chongjiesi@sjtu.edu.cn).

## Acknowledgement

This directory is modified based on [huggingface/diffusers/examples](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md). We greatly appreciate their remarkable works.
