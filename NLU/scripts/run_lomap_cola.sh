python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name cola \
--apply_lora --lora_type map \
--lora_r ${1:-2} \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha ${2:-4} \
--do_train --do_eval --max_seq_length 64 \
--per_device_train_batch_size 32 --learning_rate 8e-4 \
--num_train_epochs 25 --warmup_steps 100 \
--cls_dropout 0.10 --weight_decay 0.00 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 \
--tb_writter_loginterval 100 \
--report_to tensorboard \
--seed ${3:-6} \
--root_output_dir ./output/glue/lomap_cola_r${1:-2} \
--overwrite_output_dir
