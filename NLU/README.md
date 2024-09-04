# Adapting Pre-trained Models for NLU Task

This folder contains the implementations for NLU task.

## Setup Environment

### Create and Activate the Conda Environment

```bash
conda create -n NLU python=3.7
conda activate NLU 
```

### Install Pytorch

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install the Pre-requisites

Install dependencies:

```bash
pip install -r requirements.txt
```

Install `transformers` (version: `v4.4.2`):

```bash
pip install -e . 
```

Install `loralib`:

```bash
pip install -e ../loralib/
```

## Code Structure

- [loralib/](../loralib/loralib/) contains the PEFT methods used for this task.
- [examples/](./examples/) contains the training codes.
- [scripts/](./scripts/) contains the training scripts for different datasets.

## Adapting Pre-trained Models on GLUE benchmark

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Adapting RoBERTa-base with BitFit on the MNLI Dataset</span></strong></summary>

```bash
python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name mnli \
--apply_bitfit \
--do_train --do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 32 --learning_rate 5e-4 --num_train_epochs 7 \
--warmup_steps 1000 \
--cls_dropout 0.15 --weight_decay 0 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 30000 \
--logging_steps 500 \
--seed 42 \
--root_output_dir ./output/glue/mnli \
--overwrite_output_dir
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

+ `model_name_or_path`: Apply pre-trained models. `roberta-base` for RoBERTa-base (125M).
+ `apply_bitfit`: Apply BitFit for the model.

</details>

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Adapting DeBERTaV3-base with Adapter on the CoLA Dataset</span></strong></summary>

```bash
python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name cola \
--apply_adapter --adapter_type houlsby --adapter_size 64 \
--do_train --do_eval --max_seq_length 64 \
--per_device_train_batch_size 32 --learning_rate 8e-4 \
--num_train_epochs 25 --warmup_steps 100 \
--cls_dropout 0.10 --weight_decay 0.00 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 \
--tb_writter_loginterval 100 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/glue/cola \
--overwrite_output_dir
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

+ `model_name_or_path`: Apply pre-trained models. `microsoft/deberta-v3-base` for DeBERTaV3-base (184M).
+ `apply_adapter`: Apply Adapter for the model.
+ `adapter_type`: Specify the type of Adapter. `houlsby` for Houlsby Adapter and `pfeiffer` for Pfeiffer Adapter.
+ `adapter_size`: Specify the size of Adapter.

</details>

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Adapting RoBERTa-Large with AdaLoRA on the MRPC Dataset</span></strong></summary>

```bash
python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-large \
--task_name mrpc \
--apply_lora --apply_adalora --lora_type svd \
--target_rank 1   --lora_r 2   \
--reg_orth_coef 0.1 \
--init_warmup 600 --final_warmup 1800 --mask_interval 1 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 320 \
--per_device_train_batch_size 32 --learning_rate 1e-3 \
--num_train_epochs 30 --warmup_ratio 0.1 \
--cls_dropout 0.0 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 300 \
--save_strategy steps --save_steps 3000 \
--logging_steps 100 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/debertav3-base/mrpc \
--overwrite_output_dir
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

+ `model_name_or_path`: Apply pre-trained models. `roberta-large` for RoBERTa-Large (355M).
+ `apply_lora, apply_adalora`: Apply AdaLoRA for the model.
+ `lora_type`: `svd` for usage of SVDLinear in [adalora](../loralib/loralib/adalora.py).
+ Other parameters: See the hyper-parameter settings in [AdaLoRA](https://github.com/QingruZhang/AdaLoRA/tree/main).

</details>

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Adapting RoBERTa-Large with Other Methods on the RTE Dataset</span></strong></summary>
We here take LoRA as an example.

```bash
python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-large \
--task_name rte \
--apply_lora --lora_type frd \
--lora_r 2 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 4 \
--do_train --do_eval --max_seq_length 320 \
--per_device_train_batch_size 32 --learning_rate 1.2e-3 \
--num_train_epochs 50 --warmup_steps 200 \
--cls_dropout 0.20 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 --report_to tensorboard \
--seed 6 \
--root_output_dir ./output/glue/rte \
--overwrite_output_dir 
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

+ `apply_lora`: Apply LoRA and other methods for the model.
+ `lora_type`: `frd` for the utilization of the Linear module in [layers](../loralib/loralib/layers.py). If an alternative approach is preferred, you may replace the contents of [layers](../loralib/loralib/layers.py) directly with the contents from other files, such us [TriLoRA](../loralib/loralib/layers_TriLoRA.py).

</details>

## GPU Memory

Considering that you may have different computing resources, we provide a simple table of the GPU resources required for the different models.

| Model Name | GPU Requirement |
|------------|-----------------|
| RoBERTa-base | RTX 3090 (24G)|
| DeBERTaV3-base | RTX 3090 (24G) & A100 (80G) |
| RoBERTa-Large | A100 (80G) |
| DeBERTa-XXL | A100 (80G) |
|More..|...|

## Q&A

If you encounter any issues, please refer to these two links: [LoRA](https://github.com/microsoft/LoRA/issues) and [AdaLoRA](https://github.com/QingruZhang/AdaLoRA/issues). These resources cover the majority of problems you may encounter. Additionally, you are also welcome to contact us by submitting an [issue](https://github.com/Chongjie-Si/Subspace-Tuning/issues) or via [email](mailto:chongjiesi@sjtu.edu.cn).

## Acknowledgement

This directory is modified based on [LoRA](https://github.com/microsoft/LoRA) and [AdaLoRA](https://github.com/QingruZhang/AdaLoRA). We greatly appreciate their remarkable works.
