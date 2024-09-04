# Adapting Pre-trained Models for NLG and QA tasks

This folder contains the implementations for NLG and QA tasks.

## Setup Environment

### Create and Activate the Conda Environment

```bash
conda create -n NLG python=3.7
conda activate NLG 
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

Install `transformers` (version: `v4.21.0`):

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

## Adapting Pre-trained Models on NLG benchmark

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Adapting BART with AdaLoRA on Xsum Dataset</span></strong></summary>

```bash
accelerate launch --multi_gpu --num_machine=1 --num_processes=8 \
--main_process_port=8679 --mixed_precision="no" \
examples/summarization/run_summarization_no_trainer.py \
--model_name_or_path facebook/bart-large \
--dataset_name xsum \
--apply_lora --apply_adalora \
--lora_type svd --target_rank 8 --lora_r 12 \
--lora_alpha 32 \
--reg_orth_coef 0.1 \
--init_warmup 6000 --final_warmup 25000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--per_device_train_batch_size 8 --learning_rate 5e-4 \
--num_train_epochs 25 --num_warmup_steps 3000 \
--max_source_length 768 --max_target_length 64 --max_length 768 \
--pad_to_max_length --num_beams 8 \
--per_device_eval_batch_size 8 \
--seed 9 \
--with_tracking \
--tb_writter_loginterval 500 \
--output_dir ./output/bart-large/xsum 
```

<strong><span style="font-size: 1em;">Hyper-parameter Setup</span></strong>

+ `apply_lora`: Apply LoRA to the target model.
+ `lora_type`: Config the low-rank parameterization, `frd` for low-rank decomposition and `svd` for SVD decomposition. Use `svd` for AdaLoRA and `frd` for LoRA or other methods.
+ `apply_adalora`: Further apply rank allocator in AdaLoRA for the model that have been modified by LoRA.
+ `lora_module`: The types of modules updated by LoRA.
+ `lora_r`: The initial rank of each incremental matrix.
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix.
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `final_warmup`: The steps of final warmup for budget scheduler.
+ `mask_interval`: The time interval between two budget allocations.
+ `beta1` and `beta2`: The coefficient of exponential moving average when updating importance scores.
+ `reg_orth_coef`: The weight of orthogonal regularization.

You can see [here](../NLU/README.md) for more explanations on the hyper-parameters.

</details>

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Adapting BART with AdaLoRA on CNN/DailyMail Dataset</span></strong></summary>

```bash
accelerate launch --multi_gpu --num_machine=1 --num_processes=8 --main_process_port=8675 --mixed_precision="no" \
examples/summarization/run_summarization_no_trainer.py \
--model_name_or_path facebook/bart-large \
--dataset_name cnn_dailymail --dataset_config "3.0.0" \
--apply_lora --apply_rankselector \
--lora_type svd --target_rank 2 --lora_r 4 \
--lora_alpha 32 \
--reg_orth_coef 0.1 \
--init_warmup 5000 --final_warmup 85000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--per_device_train_batch_size 4 --learning_rate 5e-4   \
--num_train_epochs 15 --num_warmup_steps 3000 \
--max_source_length 1024 --max_target_length 160 --max_length 1024 \
--pad_to_max_length --num_beams 4 \
--per_device_eval_batch_size 4 \
--seed 9 \
--with_tracking \
--tb_writter_loginterval 500 \
--output_dir ./output/bart-large/cnn_dailymail  
```

</details>

## Adapting Pre-trained Models on QA benchmark

<details>
  <summary><strong><span style="font-size: 1.2em;">Example: Adapting DeBERTaV3-base with AdaLoRA on SQuADv2.0 Dataset</span></strong></summary>

```bash
python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/question-answering/run_qa.py \
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad_v2 \
--apply_lora --apply_adalora \
--lora_type svd --target_rank 1 --lora_r 2 \
--reg_orth_coef 0.1 \
--init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train --do_eval --version_2_with_negative \
--max_seq_length 384 --doc_stride 128 \
--per_device_train_batch_size 16 \
--learning_rate 1e-3 \
--num_train_epochs 12 \
--max_step 300 \
--warmup_steps 1000 --per_device_eval_batch_size 128 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 100000 \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 9 \
--root_output_dir ./output/debertav3-base/squadv2 \
--overwrite_output_dir 

```

</details>

## Q&A

If you encounter any issues, please refer to these two links: [LoRA](https://github.com/microsoft/LoRA/issues) and [AdaLoRA](https://github.com/QingruZhang/AdaLoRA/issues). These resources cover the majority of problems you may encounter. Additionally, you are also welcome to contact us by submitting an [issue](https://github.com/Chongjie-Si/Subspace-Tuning/issues) or via [email](mailto:chongjiesi@sjtu.edu.cn).

## Acknowledgement

This directory is modified based on [LoRA](https://github.com/microsoft/LoRA) and [AdaLoRA](https://github.com/QingruZhang/AdaLoRA). We greatly appreciate their remarkable works.
