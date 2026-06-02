#!/bin/bash
# Run LoMAP on all 8 GLUE tasks with DeBERTa-v3-base
# Usage: bash run_lomap_glue_all.sh [rank] [alpha] [seed]
# Defaults: rank=2, alpha=4, seed=6
RANK=${1:-2}
ALPHA=${2:-4}
SEED=${3:-6}
MODEL="microsoft/deberta-v3-base"
MODULES="query,key,value,intermediate,layer.output,attention.output"

declare -A TASKS
# task -> "max_seq_len lr epochs warmup cls_dropout"
TASKS[mnli]="256 5e-4 12 100 0.10"
TASKS[sst2]="128 8e-4 24 1000 0.00"
TASKS[cola]="64 8e-4 25 100 0.10"
TASKS[qqp]="320 1e-3 5 1000 0.10"
TASKS[qnli]="512 5e-4 5 1000 0.10"
TASKS[rte]="320 1.2e-3 50 200 0.20"
TASKS[mrpc]="320 1e-3 30 100 0.10"
TASKS[stsb]="128 5e-4 25 100 0.10"

for TASK in "${!TASKS[@]}"; do
    read -r SEQ_LEN LR EPOCHS WARMUP CLS_DROP <<< "${TASKS[$TASK]}"
    echo "=== Running LoMAP on $TASK (r=$RANK, alpha=$ALPHA, seed=$SEED) ==="
    python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
    examples/text-classification/run_glue.py \
    --model_name_or_path $MODEL \
    --task_name $TASK \
    --apply_lora --lora_type map \
    --lora_r $RANK \
    --lora_module $MODULES \
    --lora_alpha $ALPHA \
    --do_train --do_eval --max_seq_length $SEQ_LEN \
    --per_device_train_batch_size 32 --learning_rate $LR \
    --num_train_epochs $EPOCHS --warmup_steps $WARMUP \
    --cls_dropout $CLS_DROP --weight_decay 0.00 \
    --evaluation_strategy steps --eval_steps 100 \
    --save_strategy steps --save_steps 10000 \
    --logging_steps 10 \
    --tb_writter_loginterval 100 \
    --report_to tensorboard \
    --seed $SEED \
    --root_output_dir ./output/glue/lomap_${TASK}_r${RANK}_seed${SEED} \
    --overwrite_output_dir
done
