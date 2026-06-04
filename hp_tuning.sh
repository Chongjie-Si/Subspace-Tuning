#!/bin/bash
# hp_tuning.sh — targeted hyperparameter sweep for LoMAP on cola+mrpc.
# Goal: find a config that reliably beats LoRA on small tasks.
#
# Phase A (~4h): conservative knobs — lr, beta_init, detach_denom
#   Stage 1: lr scan (0.5x, 1.0x, 2.0x base lr)
#   Stage 2: at winning lr, scan map_beta_init in {0.01, 0.1, 1.0}
#   Stage 3: at winning (lr, beta0), detach_denom on/off
#   Stage 4: confirm best config × 3 seeds × cola+mrpc
#
# Phase B (~6-8h): advanced knobs — most likely to push past LoRA
#   B1: norm_scope ∈ {global, column, row, row_column}
#   B2: map_lr_scale ∈ {0.001, 0.01, 0.1, 1.0}     ← grad-magnitude mismatch fix
#   B3: freeze_alpha vs learn_alpha                 ← stability vs flexibility
#   B4: lora_dropout + weight_decay regularization  ← anti-overfitting on small tasks
#   B5: confirm globally-best config × 3 seeds × cola+mrpc+rte

set -e
cd /home/li/Subspace-Tuning
VENV="$(pwd)/.venv/bin/python"
NLU="NLU/examples/text-classification/run_glue.py"
PYPATH="$(pwd)/NLU/src:$(pwd)/loralib"
MODEL="microsoft/deberta-v3-base"
MODULES="query,key,value,intermediate,layer.output,attention.output"
LOG=/tmp/hp_tuning.log

run_lomap() {
    # Args (all required, pass empty string to use default):
    #   <task> <seed> <lr> <beta_init> <detach: True|False> <name>
    #   [norm_scope] [lr_scale] [freeze_alpha: True|False] [lora_dropout] [weight_decay]
    local task=$1 seed=$2 lr=$3 beta=$4 detach=$5 name=$6
    local norm_scope=${7:-global}
    local lr_scale=${8:-1.0}
    local freeze_alpha=${9:-False}
    local lora_dropout=${10:-0.0}
    local weight_decay=${11:-0.0}
    local out="NLU/output/glue/hp_${name}_${task}_seed${seed}"

    if [ -f "$out/model/all_results.json" ]; then
        echo "[$(date)] SKIP $name $task seed=$seed" | tee -a $LOG
        return 0
    fi
    echo "[$(date)] RUN  $name $task seed=$seed" \
         "(lr=$lr beta0=$beta detach=$detach scope=$norm_scope" \
         "lr_scale=$lr_scale freeze_a=$freeze_alpha drop=$lora_dropout wd=$weight_decay)" | tee -a $LOG

    case "$task" in
        cola) seq=64;  ep=25; metric=matthews_correlation; drop=0.10 ;;
        mrpc) seq=320; ep=30; metric=accuracy;             drop=0.10 ;;
        rte)  seq=320; ep=50; metric=accuracy;             drop=0.20 ;;
        *) echo "Unknown task $task"; return 1 ;;
    esac

    local save_steps=200
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYPATH" "$VENV" "$NLU" \
        --model_name_or_path "$MODEL" --task_name "$task" \
        --apply_lora --lora_type map --lora_r 2 --lora_module "$MODULES" --lora_alpha 4 \
        --map_beta_init "$beta" --map_detach_denom "$detach" \
        --map_norm_scope "$norm_scope" --map_lr_scale "$lr_scale" \
        --map_freeze_alpha "$freeze_alpha" \
        --do_train --do_eval --max_seq_length "$seq" \
        --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
        --learning_rate "$lr" --num_train_epochs "$ep" \
        --warmup_ratio 0.1 --cls_dropout "$drop" --weight_decay "$weight_decay" \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps "$save_steps" --save_total_limit 2 \
        --load_best_model_at_end --metric_for_best_model "$metric" --greater_is_better True \
        --logging_steps 50 --tb_writter_loginterval 50 --report_to tensorboard \
        --skip_memory_metrics --seed "$seed" --root_output_dir "$out" >> $LOG 2>&1
}

read_metric() {
    local out=$1
    python3 -c "
import json,sys,os
ar = '$out/model/all_results.json'
ts = '$out/model/trainer_state.json'
if not os.path.exists(ar): print('NA'); sys.exit()
r = json.load(open(ar))
m_key = [k for k in r if k.startswith('eval_') and all(x not in k for x in ['loss','runtime','sample','mem'])][0]
val = r[m_key]
# sanity check vs trainer_state best
if os.path.exists(ts):
    s = json.load(open(ts))
    if s.get('best_metric') and s['best_metric'] > val + 1e-4:
        val = s['best_metric']
print(f'{val*100:.2f}')
"
}

echo "[$(date)] === HP tuning start ===" | tee -a $LOG

# ---------------- Stage 1: lr scan (cola only, fast) ----------------
echo "[$(date)] >>> Stage 1: lr scan on cola" | tee -a $LOG
# paper cola lr=8e-4. Try 0.5x=4e-4, 1.0x=8e-4, 2.0x=1.6e-3
for lr in 4e-4 8e-4 1.6e-3; do
    name="lr_${lr}"
    run_lomap cola 42 "$lr" 1.0 False "$name"
done

echo "[$(date)] Stage 1 results:" | tee -a $LOG
BEST_LR=8e-4; BEST_LR_VAL=0
for lr in 4e-4 8e-4 1.6e-3; do
    val=$(read_metric "NLU/output/glue/hp_lr_${lr}_cola_seed42")
    echo "  lr=$lr  cola=$val" | tee -a $LOG
    if [ "$val" != "NA" ]; then
        cmp=$(python3 -c "print(1 if $val > $BEST_LR_VAL else 0)")
        if [ "$cmp" = "1" ]; then BEST_LR=$lr; BEST_LR_VAL=$val; fi
    fi
done
echo "[$(date)] Best lr = $BEST_LR (cola=$BEST_LR_VAL)" | tee -a $LOG

# ---------------- Stage 2: map_beta_init scan ----------------
echo "[$(date)] >>> Stage 2: map_beta_init scan at lr=$BEST_LR" | tee -a $LOG
for b in 0.01 0.1 1.0; do
    name="b${b}_lr${BEST_LR}"
    run_lomap cola 42 "$BEST_LR" "$b" False "$name"
done

BEST_BETA=1.0; BEST_BETA_VAL=$BEST_LR_VAL
for b in 0.01 0.1 1.0; do
    val=$(read_metric "NLU/output/glue/hp_b${b}_lr${BEST_LR}_cola_seed42")
    echo "  beta_init=$b  cola=$val" | tee -a $LOG
    if [ "$val" != "NA" ]; then
        cmp=$(python3 -c "print(1 if $val > $BEST_BETA_VAL else 0)")
        if [ "$cmp" = "1" ]; then BEST_BETA=$b; BEST_BETA_VAL=$val; fi
    fi
done
echo "[$(date)] Best beta_init = $BEST_BETA (cola=$BEST_BETA_VAL)" | tee -a $LOG

# ---------------- Stage 3: detach_denom toggle ----------------
echo "[$(date)] >>> Stage 3: detach_denom scan at lr=$BEST_LR beta=$BEST_BETA" | tee -a $LOG
for d in False True; do
    name="d${d}_b${BEST_BETA}_lr${BEST_LR}"
    run_lomap cola 42 "$BEST_LR" "$BEST_BETA" "$d" "$name"
done

BEST_DETACH=False; BEST_DETACH_VAL=$BEST_BETA_VAL
for d in False True; do
    val=$(read_metric "NLU/output/glue/hp_d${d}_b${BEST_BETA}_lr${BEST_LR}_cola_seed42")
    echo "  detach=$d  cola=$val" | tee -a $LOG
    if [ "$val" != "NA" ]; then
        cmp=$(python3 -c "print(1 if $val > $BEST_DETACH_VAL else 0)")
        if [ "$cmp" = "1" ]; then BEST_DETACH=$d; BEST_DETACH_VAL=$val; fi
    fi
done
echo "[$(date)] Best detach = $BEST_DETACH (cola=$BEST_DETACH_VAL)" | tee -a $LOG

# ---------------- Stage 4: confirm best config × 3 seeds × 2 tasks ----------------
echo "[$(date)] >>> Stage 4: best config confirmation" | tee -a $LOG
echo "[$(date)] BEST: lr=$BEST_LR  beta_init=$BEST_BETA  detach=$BEST_DETACH" | tee -a $LOG
for task in cola mrpc; do
    for seed in 6 7 8; do
        name="best_lr${BEST_LR}_b${BEST_BETA}_d${BEST_DETACH}"
        run_lomap "$task" "$seed" "$BEST_LR" "$BEST_BETA" "$BEST_DETACH" "$name"
    done
done

echo "[$(date)] === Phase A DONE ===" | tee -a $LOG
echo "[$(date)] Phase A best config summary:" | tee -a $LOG
A_BEST_VAL=0
for task in cola mrpc; do
    for seed in 6 7 8; do
        name="best_lr${BEST_LR}_b${BEST_BETA}_d${BEST_DETACH}"
        val=$(read_metric "NLU/output/glue/hp_${name}_${task}_seed${seed}")
        echo "  $task seed=$seed  $val" | tee -a $LOG
    done
done

# =============================================================================
# Phase B: advanced knobs that target the gradient-magnitude / regularization
# weak points exposed by the analysis (map_alpha drift huge, beta drift small).
# Each scan uses single seed=42 and locks in the Phase-A winners as baseline.
# =============================================================================
echo "[$(date)] === Phase B start ===" | tee -a $LOG

# ---------------- B1: map_norm_scope ∈ {global, column, row, row_column} ----
echo "[$(date)] >>> B1: norm_scope scan at lr=$BEST_LR beta=$BEST_BETA detach=$BEST_DETACH" | tee -a $LOG
for scope in global column row row_column; do
    name="B1_scope${scope}"
    run_lomap cola 42 "$BEST_LR" "$BEST_BETA" "$BEST_DETACH" "$name" "$scope"
done

BEST_SCOPE=global; BEST_SCOPE_VAL=$BEST_DETACH_VAL
for scope in global column row row_column; do
    val=$(read_metric "NLU/output/glue/hp_B1_scope${scope}_cola_seed42")
    echo "  scope=$scope  cola=$val" | tee -a $LOG
    if [ "$val" != "NA" ]; then
        cmp=$(python3 -c "print(1 if $val > $BEST_SCOPE_VAL else 0)")
        if [ "$cmp" = "1" ]; then BEST_SCOPE=$scope; BEST_SCOPE_VAL=$val; fi
    fi
done
echo "[$(date)] Best scope = $BEST_SCOPE (cola=$BEST_SCOPE_VAL)" | tee -a $LOG

# ---------------- B2: map_lr_scale ∈ {0.001, 0.01, 0.1, 1.0} ----
# Hypothesis: map_alpha grad ~ O(10^2), lora_A grad ~ O(10^-2). Letting them share
# one lr means α takes huge steps while A barely moves. Scaling α/β lr down should
# stabilize — most likely big win.
echo "[$(date)] >>> B2: map_lr_scale scan" | tee -a $LOG
for scale in 0.001 0.01 0.1 1.0; do
    name="B2_lrscale${scale}"
    run_lomap cola 42 "$BEST_LR" "$BEST_BETA" "$BEST_DETACH" "$name" "$BEST_SCOPE" "$scale"
done

BEST_SCALE=1.0; BEST_SCALE_VAL=$BEST_SCOPE_VAL
for scale in 0.001 0.01 0.1 1.0; do
    val=$(read_metric "NLU/output/glue/hp_B2_lrscale${scale}_cola_seed42")
    echo "  lr_scale=$scale  cola=$val" | tee -a $LOG
    if [ "$val" != "NA" ]; then
        cmp=$(python3 -c "print(1 if $val > $BEST_SCALE_VAL else 0)")
        if [ "$cmp" = "1" ]; then BEST_SCALE=$scale; BEST_SCALE_VAL=$val; fi
    fi
done
echo "[$(date)] Best lr_scale = $BEST_SCALE (cola=$BEST_SCALE_VAL)" | tee -a $LOG

# ---------------- B3: freeze_alpha vs learn_alpha ----
echo "[$(date)] >>> B3: freeze_alpha toggle" | tee -a $LOG
for fa in False True; do
    name="B3_freeze${fa}"
    run_lomap cola 42 "$BEST_LR" "$BEST_BETA" "$BEST_DETACH" "$name" \
              "$BEST_SCOPE" "$BEST_SCALE" "$fa"
done

BEST_FREEZE=False; BEST_FREEZE_VAL=$BEST_SCALE_VAL
for fa in False True; do
    val=$(read_metric "NLU/output/glue/hp_B3_freeze${fa}_cola_seed42")
    echo "  freeze_alpha=$fa  cola=$val" | tee -a $LOG
    if [ "$val" != "NA" ]; then
        cmp=$(python3 -c "print(1 if $val > $BEST_FREEZE_VAL else 0)")
        if [ "$cmp" = "1" ]; then BEST_FREEZE=$fa; BEST_FREEZE_VAL=$val; fi
    fi
done
echo "[$(date)] Best freeze_alpha = $BEST_FREEZE (cola=$BEST_FREEZE_VAL)" | tee -a $LOG

# ---------------- B4: regularization (lora_dropout × weight_decay) ----
# Small grid: (drop, wd) ∈ {(0.0, 0.0), (0.05, 0.0), (0.0, 1e-4), (0.05, 1e-4)}
echo "[$(date)] >>> B4: regularization (dropout × weight_decay)" | tee -a $LOG
for cfg in "0.0:0.0" "0.05:0.0" "0.0:1e-4" "0.05:1e-4"; do
    drop=${cfg%:*}; wd=${cfg#*:}
    name="B4_drop${drop}_wd${wd}"
    run_lomap cola 42 "$BEST_LR" "$BEST_BETA" "$BEST_DETACH" "$name" \
              "$BEST_SCOPE" "$BEST_SCALE" "$BEST_FREEZE" "$drop" "$wd"
done

BEST_DROP=0.0; BEST_WD=0.0; BEST_REG_VAL=$BEST_FREEZE_VAL
for cfg in "0.0:0.0" "0.05:0.0" "0.0:1e-4" "0.05:1e-4"; do
    drop=${cfg%:*}; wd=${cfg#*:}
    val=$(read_metric "NLU/output/glue/hp_B4_drop${drop}_wd${wd}_cola_seed42")
    echo "  drop=$drop wd=$wd  cola=$val" | tee -a $LOG
    if [ "$val" != "NA" ]; then
        cmp=$(python3 -c "print(1 if $val > $BEST_REG_VAL else 0)")
        if [ "$cmp" = "1" ]; then BEST_DROP=$drop; BEST_WD=$wd; BEST_REG_VAL=$val; fi
    fi
done
echo "[$(date)] Best (drop, wd) = ($BEST_DROP, $BEST_WD)  cola=$BEST_REG_VAL" | tee -a $LOG

# ---------------- B5: globally-best config × 3 seeds × {cola, mrpc, rte} ----
echo "[$(date)] >>> B5: globally-best config confirmation × 3 tasks × 3 seeds" | tee -a $LOG
echo "[$(date)] GLOBAL BEST: lr=$BEST_LR beta=$BEST_BETA detach=$BEST_DETACH" \
     "scope=$BEST_SCOPE lr_scale=$BEST_SCALE freeze_a=$BEST_FREEZE drop=$BEST_DROP wd=$BEST_WD" | tee -a $LOG

GLOBAL_NAME="GLOBAL_lr${BEST_LR}_b${BEST_BETA}_d${BEST_DETACH}_s${BEST_SCOPE}_ls${BEST_SCALE}_fa${BEST_FREEZE}_dr${BEST_DROP}_wd${BEST_WD}"
for task in cola mrpc rte; do
    for seed in 6 7 8; do
        run_lomap "$task" "$seed" "$BEST_LR" "$BEST_BETA" "$BEST_DETACH" "$GLOBAL_NAME" \
                  "$BEST_SCOPE" "$BEST_SCALE" "$BEST_FREEZE" "$BEST_DROP" "$BEST_WD"
    done
done

echo "[$(date)] === HP tuning DONE (Phase A + B) ===" | tee -a $LOG
echo "[$(date)] Final summary: globally-best 3-seed averages" | tee -a $LOG
for task in cola mrpc rte; do
    vals=""
    for seed in 6 7 8; do
        v=$(read_metric "NLU/output/glue/hp_${GLOBAL_NAME}_${task}_seed${seed}")
        vals="$vals $v"
    done
    echo "  $task seeds=6,7,8: $vals" | tee -a $LOG
done
