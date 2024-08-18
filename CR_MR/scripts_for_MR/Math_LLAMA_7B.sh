# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

CUDA_VISIBLE_DEVICES=$4 python finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path 'math_10k.json' \
    --output_dir $3 \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 0 \
    --eval_step 80 --save_step 80  --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights $3|tee -a $3/gsm8k.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset AQuA \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights $3|tee -a $3/AQuA.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights $3|tee -a $3/MultiArith.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights $3|tee -a $3/SVAMP.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset SingleEq \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights $3|tee -a $3/SingleEq.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset AddSub \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights $3|tee -a $3/AddSub.txt
