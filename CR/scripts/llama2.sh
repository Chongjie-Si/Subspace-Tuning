# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
CUDA_VISIBLE_DEVICES=$4 python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' \
    --output_dir $3 \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name dora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/boolq.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset piqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/piqa.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset social_i_qa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/hellaswag.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/winogrande.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/ARC-Challenge.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/ARC-Easy.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/openbookqa.txt