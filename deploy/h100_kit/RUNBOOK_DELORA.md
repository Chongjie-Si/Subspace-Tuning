# DeLoRA 对比实验 Runbook

**目的**：在 H100 上跑 DeLoRA baseline，填进 AAAI 论文 Table 2（CR）和 Table 4（NLU），与已有的 LoMAP/LoRA/DoRA 数字对比。

---

## 0. 前置条件确认

在 H100 节点上执行：

```bash
nvidia-smi            # 确认 8 张 H100 可用
python3 --version     # 需要 3.9+
df -h /               # 需要至少 200 GB 剩余（模型 + 数据集缓存）
```

---

## 1. 拉代码

```bash
git clone https://github.com/haoruilee/Subspace-Tuning.git
cd Subspace-Tuning
git checkout codex/add-lomap
```

验证关键文件存在：

```bash
ls deploy/h100_kit/submit_delora.sh
ls deploy/h100_kit/scripts/run_delora_nlu.py
ls deploy/h100_kit/scripts/run_delora_cr.sh
```

---

## 2. 安装环境

```bash
# 建 venv（只需一次）
python3 -m venv .venv
source .venv/bin/activate

# PyTorch 2.4 + CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 基础依赖
pip install accelerate==0.34.2 datasets==2.21.0 evaluate==0.4.3 \
    transformers==4.44.2 tensorboard scikit-learn sentencepiece \
    sacremoses scipy fire bitsandbytes

# LoMAP CR PEFT fork（含 DeLoRA 实现）
pip install -e CR_MR/peft/

# NLU loralib fork
pip install -e loralib/

# HF PEFT >= 0.19（DeloraConfig 从此版本引入）
pip install "peft>=0.19" --upgrade
```

安装后验证：

```bash
python -c "from peft import DeloraConfig; print('DeloraConfig OK')"
python -c "from peft.tuners.lora import LoraConfig; import peft; print(peft.__version__)"
```

若 `DeloraConfig` 导入失败，说明 peft 版本不够，再跑一次 `pip install "peft>=0.19" --upgrade`。

---

## 3. 下载数据

### 3.1 NLU（GLUE）

GLUE 数据集自动从 HuggingFace Hub 下载，提前缓存：

```bash
python - <<'EOF'
from datasets import load_dataset
tasks = ['cola','sst2','mrpc','qqp','qnli','rte','stsb','mnli']
for t in tasks:
    print(f"downloading glue/{t} ...")
    load_dataset('glue', t)
print("GLUE done")
EOF
```

缓存默认落在 `~/.cache/huggingface/datasets/`，约 2 GB。

### 3.2 NLU 预训练模型（DeBERTa-v3-base）

```bash
python - <<'EOF'
from transformers import AutoTokenizer, AutoConfig
for m in ['microsoft/deberta-v3-base']:
    AutoConfig.from_pretrained(m)
    AutoTokenizer.from_pretrained(m)
    print(f"config/tokenizer cached: {m}")
print("Model weights will download automatically at first training run (~180 MB)")
EOF
```

### 3.3 CR 数据（commonsense_170k.json）

文件随 repo 一起，确认存在即可：

```bash
ls -lh CR_MR/commonsense_170k.json
# 期望：约 96 MB
```

若缺失（例如 git 用了 LFS 但未拉取）：

```bash
git lfs pull --include="CR_MR/commonsense_170k.json"
```

### 3.4 CR 预训练模型（LLaMA-7B）

LLaMA-7B（`huggyllama/llama-7b`）是无门控模型，无需 token：

```bash
python - <<'EOF'
from transformers import AutoConfig
AutoConfig.from_pretrained("huggyllama/llama-7b")
print("LLaMA-7B config ok; weights (~13 GB) will pull at first run")
EOF
```

若集群无外网，手动下载后修改脚本里的 `BASE` 变量指向本地路径：

```bash
# 在有网机器上：
# huggingface-cli download huggyllama/llama-7b --local-dir /data/models/llama-7b
# 然后在 run_delora_cr.sh 里改：
# BASE=/data/models/llama-7b
```

---

## 4. 运行实验

### 一键启动（推荐）

```bash
source .venv/bin/activate
bash deploy/h100_kit/submit_delora.sh
```

这会并行启动两组作业：
- **GPU 0–3**：NLU（DeBERTa-base，r=2，8任务×3种子）
- **GPU 4–7**：CR（LLaMA-7B，r=16 + r=32）

实时监控：

```bash
# 总进度
tail -f logs/delora_nlu_dispatch.log
tail -f logs/cr_dispatch.log

# 某个具体 run 的训练 loss
tail -f logs/nlu_base_delora_cola_r2_seed6_gpu0.log
tail -f logs/cr_llama-7b_delora_r16_gpu4.log
```

### 单独运行 NLU

```bash
# 单任务调试（在提交前先跑 cola 确认环境正常）
CUDA_VISIBLE_DEVICES=0 python deploy/h100_kit/scripts/run_delora_nlu.py \
    --task cola --seed 6 --rank 2 --size base \
    --output_root NLU/output/glue

# 完整网格（8任务×3种子，分配到4块GPU）
bash deploy/h100_kit/scripts/run_delora_nlu_grid.sh base 2 6,7,8 0,1,2,3
```

### 单独运行 CR

```bash
bash deploy/h100_kit/scripts/run_delora_cr.sh 4,5,6,7
```

---

## 5. 保存训练曲线

### TensorBoard 日志路径

| 实验 | 日志目录 |
|------|----------|
| NLU（每个 run）| `NLU/output/glue/<size>_delora_<task>_r<R>_seed<S>/runs/` |
| CR（每个 run） | `CR_MR/output/llama-7b_delora_r<R>_a<A>/runs/` |

### 启动 TensorBoard

```bash
# NLU 所有 run
tensorboard --logdir NLU/output/glue --port 6006 --bind_all &

# CR 所有 run
tensorboard --logdir CR_MR/output --port 6007 --bind_all &
```

在本机 SSH 端口转发后访问：

```bash
# 本机执行（把 H100_HOST 替换为节点 IP 或域名）
ssh -L 6006:localhost:6006 -L 6007:localhost:6007 USER@H100_HOST
# 然后浏览器打开 http://localhost:6006
```

### 导出曲线为文件（推荐，防止节点断连丢失）

实验跑完后把所有 TensorBoard 事件文件打包：

```bash
# 打包 NLU 曲线
tar -czf delora_nlu_tb_logs.tar.gz \
    NLU/output/glue/*/runs/

# 打包 CR 曲线
tar -czf delora_cr_tb_logs.tar.gz \
    CR_MR/output/llama-7b_delora_*/runs/

# 下载到本机
scp USER@H100_HOST:~/Subspace-Tuning/delora_*_tb_logs.tar.gz ./
```

### 用 Python 导出 CSV（投稿用，不依赖 TensorBoard 界面）

```bash
python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv, os, glob

out_csv = "delora_curves.csv"
rows = []
# 找所有 events 文件
for events_dir in glob.glob("NLU/output/glue/*/runs") + \
                  glob.glob("CR_MR/output/llama-7b_delora_*/runs"):
    run_name = events_dir.split("/")[-2]
    ea = EventAccumulator(events_dir)
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        for e in ea.Scalars(tag):
            rows.append({"run": run_name, "tag": tag, "step": e.step, "value": e.value})

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run","tag","step","value"])
    writer.writeheader(); writer.writerows(rows)
print(f"Exported {len(rows)} data points → {out_csv}")
EOF
```

---

## 6. 查看结果

实验完成后：

```bash
# NLU 汇总表（LoMAP vs LoRA vs DeLoRA，8任务×3种子均值±std vs 论文值）
python summarize_nlu.py

# CR per-rank 均值（8个 benchmark 平均）
# run_delora_cr.sh 结束时已自动打印，也可手动重跑：
python - <<'EOF'
import os, re
cr_dir = "CR_MR"
datasets = ["boolq","piqa","social_i_qa","hellaswag",
            "winogrande","ARC-Challenge","ARC-Easy","openbookqa"]
for rank, alpha in [(16, 32), (32, 64)]:
    out = f"{cr_dir}/output/llama-7b_delora_r{rank}_a{alpha}"
    accs = []
    for ds in datasets:
        f = f"{out}/{ds}.txt"
        if os.path.exists(f):
            m = re.findall(r"accuracy\s+\d+\s+([0-9.]+)", open(f).read())
            if m:
                accs.append(float(m[-1]) * 100)
    avg = sum(accs)/len(accs) if accs else 0
    print(f"DeLoRA LLaMA-7B r={rank}: {len(accs)}/8 benchmarks  avg={avg:.1f}%")
    for ds, v in zip(datasets, accs):
        print(f"  {ds:<20} {v:.1f}%")
EOF
```

---

## 7. 预期数字（健康性检查）

### NLU — DeBERTa-base r=2（对比 Table 4）

DeLoRA 是 2025 ICLR 方法，DeBERTa 上的公开数字较少。以 LoRA 为下界，LoMAP 为上界：

| Task | LoRA (paper) | LoMAP (paper) | DeLoRA 预期范围 |
|------|-------------|---------------|----------------|
| CoLA | 69.15 | 70.38 | 68–71 |
| SST-2 | 93.92 | 95.91 | 93–96 |
| MRPC | 90.19 | 91.67 | 89–92 |
| RTE  | 87.01 | 89.16 | 86–90 |
| STS-B | 90.75 | 92.14 | 90–93 |
| QNLI | 93.37 | 94.31 | 93–95 |
| QQP  | 90.61 | 91.83 | 90–92 |
| MNLI | 90.03 | 90.52 | 89–91 |

**红灯**：任意任务比 LoRA paper 值低 >1.5，说明 DeloraConfig 超参有问题（`delora_lambda` 或 lr）。

### CR — LLaMA-7B（对比 Table 2）

| r | DeLoRA 预期平均 | LoMAP paper | LoRA paper |
|---|----------------|-------------|------------|
| 16 | 75–78 | 77.9 | 70.9 |
| 32 | 76–79 | 78.8 | — |

**红灯**：r=16 平均低于 72，说明 adapter 没有正确 merge（检查 `commonsense_evaluate.py` 里 DeLoRA 的 merge 路径）。

---

## 8. 故障处理

| 现象 | 原因 | 处理 |
|------|------|------|
| `ImportError: cannot import name 'DeloraConfig' from 'peft'` | peft 版本 < 0.19 | `pip install "peft>=0.19" --upgrade` |
| NLU 训练 loss 不下降 | lr 可能过小/大 | DeloraConfig 的 lr 使用 LoRA 同款（已写入脚本），若异常尝试 ×10 |
| CR eval 精度接近随机（BoolQ ~62%） | adapter 未 merge | 确认 `commonsense_evaluate.py` 里 `"DeLoRA"` 拼写与 lora.py 里 `use_delora` 路径匹配 |
| OOM on LLaMA-7B r=32 | 显存不够 | 脚本已开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；若还 OOM 加 `--micro_batch_size 8` |
| 某个 task 的 `all_results.json` 已存在但内容为空 | 上次 run 被 kill | `rm` 该文件后重跑，grid 脚本会自动跳过已完成的 |
| `peft >= 0.19` 与 CR PEFT fork 冲突 | 两套 peft 同时安装 | CR fork 安装为 editable (`pip install -e CR_MR/peft/`)，用 `python -c "import peft; print(peft.__file__)"` 确认加载的是 fork |

---

## 9. 打包结果回传

所有实验结束后，打包需要回传的文件：

```bash
tar -czf delora_results.tar.gz \
    NLU/output/glue/base_delora_*/model/all_results.json \
    CR_MR/output/llama-7b_delora_*/*.txt \
    delora_nlu_tb_logs.tar.gz \
    delora_cr_tb_logs.tar.gz \
    logs/delora_nlu_failures.log \
    logs/cr_failures.log

# 下载到本机
scp USER@H100_HOST:~/Subspace-Tuning/delora_results.tar.gz ./
```

回传后在本机更新 `RESULTS_LOCAL.md` 和 `summarize_nlu.py` 的论文对比，再填写 Table 2/4。
