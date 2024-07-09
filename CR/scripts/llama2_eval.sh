# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/boolq.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset piqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/piqa.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset social_i_qa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/hellaswag.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/winogrande.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/ARC-Challenge.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/ARC-Easy.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/openbookqa.txt