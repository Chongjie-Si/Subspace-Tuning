#!/bin/bash
# preflight_verify.sh — three quick verification runs on the 4080 before
# committing to H100. ~70 minutes total.
#
# Run 1: SST-2 LoMAP single-seed FP16 — verify mixed precision doesn't crash/regress
# Run 2: CoLA  LoRA (frd) single-seed FP32 — verify --lora_type=lora→frd path works
#                                              and anchor LoRA baseline number
# Run 3: SST-2 LoRA  single-seed FP32 — bigger task LoRA anchor (we already have
#                                              SST-2 LoMAP for direct comparison)

set -e
cd /home/li/Subspace-Tuning
LOG_DIR=/tmp/preflight
mkdir -p "$LOG_DIR"

# Run 1: LoMAP SST-2 FP16 (only 4 epochs, just to confirm no NaN / no major drop)
echo "[$(date)] === Run 1: LoMAP SST-2 FP16 (4 epochs only) ==="
rm -rf NLU/output/glue/preflight_map_sst2_fp16
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)/NLU/src:$(pwd)/loralib" \
.venv/bin/python NLU/examples/text-classification/run_glue.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --task_name sst2 \
    --apply_lora --lora_type map \
    --lora_r 2 --lora_module "query,key,value,intermediate,layer.output,attention.output" --lora_alpha 4 \
    --do_train --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
    --learning_rate 8e-4 --num_train_epochs 4 \
    --warmup_ratio 0.1 --cls_dropout 0.0 --weight_decay 0.0 \
    --evaluation_strategy steps --eval_steps 200 \
    --save_strategy steps --save_steps 1000 --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --logging_steps 50 --tb_writter_loginterval 50 --report_to tensorboard \
    --skip_memory_metrics \
    --fp16 \
    --seed 6 --root_output_dir NLU/output/glue/preflight_map_sst2_fp16 \
    > "$LOG_DIR/run1_lomap_sst2_fp16.log" 2>&1
echo "[$(date)] Run 1 done."

# Run 2: LoRA CoLA FP32 — anchor baseline
echo "[$(date)] === Run 2: LoRA CoLA FP32 (full 25 epochs) ==="
rm -rf NLU/output/glue/preflight_lora_cola
bash run_nlu_local.sh lora 2 4 6 cola 0 > "$LOG_DIR/run2_lora_cola.log" 2>&1
echo "[$(date)] Run 2 done."

# Run 3: LoRA SST-2 single seed (faster than full grid; only 6 epochs to anchor)
echo "[$(date)] === Run 3: LoRA SST-2 FP32 (6 epochs only, anchor) ==="
rm -rf NLU/output/glue/preflight_lora_sst2
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)/NLU/src:$(pwd)/loralib" \
.venv/bin/python NLU/examples/text-classification/run_glue.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --task_name sst2 \
    --apply_lora --lora_type frd \
    --lora_r 2 --lora_module "query,key,value,intermediate,layer.output,attention.output" --lora_alpha 4 \
    --do_train --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
    --learning_rate 8e-4 --num_train_epochs 6 \
    --warmup_ratio 0.1 --cls_dropout 0.0 --weight_decay 0.0 \
    --evaluation_strategy steps --eval_steps 200 \
    --save_strategy steps --save_steps 1000 --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --logging_steps 50 --tb_writter_loginterval 50 --report_to tensorboard \
    --skip_memory_metrics \
    --seed 6 --root_output_dir NLU/output/glue/preflight_lora_sst2 \
    > "$LOG_DIR/run3_lora_sst2.log" 2>&1
echo "[$(date)] Run 3 done."

# Summary
echo ""
echo "=================================================="
echo "Preflight summary"
echo "=================================================="
for d in preflight_map_sst2_fp16 preflight_lora_cola preflight_lora_sst2; do
    f="NLU/output/glue/$d/model/all_results.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
m = [(k,v) for k,v in d.items() if k.startswith('eval_') and all(x not in k for x in ['loss','runtime','sample','mem'])]
if m: print(f'  $d: {m[0][0].replace(\"eval_\",\"\")}={m[0][1]*100:.2f}')
" 2>/dev/null
    else
        echo "  $d: NO RESULT (check log)"
    fi
done
echo ""
echo "Logs in: $LOG_DIR"
