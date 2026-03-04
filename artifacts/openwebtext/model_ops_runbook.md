# OpenWebText 模型运行清单（训练 / 推理 / 报告路径）

## 1. 当前文件路径

### 1.1 模型日志与指标（Git 内归档，推荐看这里）
- Dense
  - 日志: `/usr/local/src/assignment1-basics/artifacts/openwebtext/train_dense_baseline_log.txt`
  - 指标: `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_baseline_metrics.json`
- MoE（对比中使用的是 hetero5_shared1_small0p5）
  - 日志: `/usr/local/src/assignment1-basics/artifacts/openwebtext/train_moe_hetero5_shared1_small0p5_log.txt`
  - 指标: `/usr/local/src/assignment1-basics/artifacts/openwebtext/moe_hetero5_shared1_small0p5_metrics.json`
- Muon
  - 日志: `/usr/local/src/assignment1-basics/artifacts/openwebtext/train_muon_baseline_log.txt`
  - 指标: `/usr/local/src/assignment1-basics/artifacts/openwebtext/muon_baseline_metrics.json`
- MLA
  - 日志: `/usr/local/src/assignment1-basics/artifacts/openwebtext/train_mla_baseline_log.txt`
  - 指标: `/usr/local/src/assignment1-basics/artifacts/openwebtext/mla_baseline_metrics.json`
- MLA+MoE
  - 日志: `/usr/local/src/assignment1-basics/artifacts/openwebtext/train_mla_moe_hetero5_log.txt`
  - 指标: `/usr/local/src/assignment1-basics/artifacts/openwebtext/mla_moe_hetero5_metrics.json`

### 1.2 对比报告路径（Git 内归档）
- 四模型总报告（Dense/MoE/Muon/MLA）
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/four_model_comparison_report.md`
- Dense vs MoE(hetero5)
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_moe_hetero5_shared1_small0p5_comparison.json`
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_moe_hetero5_shared1_small0p5_comparison.txt`
- Dense vs Muon
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_muon_baseline_comparison.json`
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_muon_baseline_comparison.txt`
- Dense vs MLA
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_mla_baseline_comparison.json`
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_mla_baseline_comparison.txt`
- Dense/MoE/Muon/MLA 四模型汇总
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_moe_hetero5_muon_mla_comparison.json`
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_moe_hetero5_muon_mla_comparison.txt`
- Dense vs MLA+MoE
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_mla_moe_hetero5_comparison.json`
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_vs_mla_moe_hetero5_comparison.txt`
- 五模型汇总（Dense/MoE/Muon/MLA/MLA+MoE）
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_moe_hetero5_muon_mla_mla_moe_comparison.json`
  - `/usr/local/src/assignment1-basics/artifacts/openwebtext/dense_moe_hetero5_muon_mla_mla_moe_comparison.txt`

### 1.3 当前可直接推理的 checkpoint（系统盘）
- Muon 最终权重: `/usr/local/src/muon_checkpoint_final.pt`
- MLA+MoE 最终权重: `/usr/local/src/mla_moe_checkpoint_final.pt`

---

## 2. 统一环境初始化命令

```bash
cd /usr/local/src/assignment1-basics
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 3. 重新训练命令（含日志路径）

### 3.1 Dense
```bash
nohup python -u -m cs336_basics.train --config owt \
  --checkpoint-dir /root/autodl-tmp/data/openwebtext/checkpoints/dense_baseline \
  --metrics-json /root/autodl-tmp/data/openwebtext/metrics/dense_baseline_metrics.json \
  --checkpoint-interval 2000 --max-step-checkpoints 2 \
  > /root/autodl-tmp/data/openwebtext/train_dense_baseline.log 2>&1 &
```

### 3.2 MoE（hetero5_shared1_small0p5，与现有对比口径一致）
```bash
nohup python -u -m cs336_basics.train --config owt_moe \
  --checkpoint-dir /root/autodl-tmp/data/openwebtext/checkpoints/moe_hetero5_shared1_small0p5 \
  --metrics-json /root/autodl-tmp/data/openwebtext/metrics/moe_hetero5_shared1_small0p5_metrics.json \
  --moe-top-k 2 --moe-fixed-expert-idx 0 \
  --moe-expert-d-ffs 2048,1024,1024,1024,1024 \
  --batch-size 32 --micro-batch-size 16 \
  --checkpoint-interval 2000 --max-step-checkpoints 2 \
  > /root/autodl-tmp/data/openwebtext/train_moe_hetero5_shared1_small0p5.log 2>&1 &
```

### 3.3 Muon
```bash
nohup python -u -m cs336_basics.train --config owt_muon \
  --checkpoint-dir /root/autodl-tmp/data/openwebtext/checkpoints/muon_baseline \
  --metrics-json /root/autodl-tmp/data/openwebtext/metrics/muon_baseline_metrics.json \
  --checkpoint-interval 2000 --max-step-checkpoints 2 \
  > /root/autodl-tmp/data/openwebtext/train_muon_baseline.log 2>&1 &
```

### 3.4 MLA
```bash
nohup python -u -m cs336_basics.train --config owt_mla \
  --checkpoint-dir /root/autodl-tmp/data/openwebtext/checkpoints/mla_baseline \
  --metrics-json /root/autodl-tmp/data/openwebtext/metrics/mla_baseline_metrics.json \
  --checkpoint-interval 2000 --max-step-checkpoints 2 \
  > /root/autodl-tmp/data/openwebtext/train_mla_baseline.log 2>&1 &
```

### 3.5 MLA+MoE
```bash
nohup python -u -m cs336_basics.train --config owt_mla_moe \
  --checkpoint-dir /root/autodl-tmp/data/openwebtext/checkpoints/mla_moe_hetero5 \
  --metrics-json /root/autodl-tmp/data/openwebtext/metrics/mla_moe_hetero5_metrics.json \
  --checkpoint-interval 2000 --max-step-checkpoints 2 \
  > /root/autodl-tmp/data/openwebtext/train_mla_moe_hetero5.log 2>&1 &
```

### 3.6 断点续训通用写法
```bash
python -u -m cs336_basics.train --config <配置名> \
  --resume-checkpoint /root/autodl-tmp/data/openwebtext/checkpoints/<模型目录>/checkpoint_step_<step>.pt
```

---

## 4. 推理命令（统一脚本）

脚本路径：`/usr/local/src/assignment1-basics/cs336_basics/infer_openwebtext.py`

### 4.1 当前可直接运行（已有最终权重）
```bash
cd /usr/local/src/assignment1-basics
source .venv/bin/activate

python cs336_basics/infer_openwebtext.py \
  --model muon \
  --checkpoint /usr/local/src/muon_checkpoint_final.pt \
  --prompt "long long ago" \
  --seed 42 --temperature 0.9 --top-p 0.9 --max-new-tokens 120

python cs336_basics/infer_openwebtext.py \
  --model mla_moe \
  --checkpoint /usr/local/src/mla_moe_checkpoint_final.pt \
  --prompt "long long ago" \
  --seed 42 --temperature 0.9 --top-p 0.9 --max-new-tokens 120
```

### 4.2 其余模型（重训后）
```bash
python cs336_basics/infer_openwebtext.py \
  --model dense \
  --checkpoint /root/autodl-tmp/data/openwebtext/checkpoints/dense_baseline/checkpoint_final.pt \
  --prompt "long long ago"

python cs336_basics/infer_openwebtext.py \
  --model moe \
  --checkpoint /root/autodl-tmp/data/openwebtext/checkpoints/moe_hetero5_shared1_small0p5/checkpoint_final.pt \
  --prompt "long long ago"

python cs336_basics/infer_openwebtext.py \
  --model mla \
  --checkpoint /root/autodl-tmp/data/openwebtext/checkpoints/mla_baseline/checkpoint_final.pt \
  --prompt "long long ago"
```

---

## 5. 对比报告重新生成命令

### 5.1 Dense vs MLA+MoE
```bash
python cs336_basics/compare_metrics.py \
  --baseline /root/autodl-tmp/data/openwebtext/metrics/dense_baseline_metrics.json \
  --candidate /root/autodl-tmp/data/openwebtext/metrics/mla_moe_hetero5_metrics.json \
  --baseline-name dense_baseline --candidate-name mla_moe_hetero5 \
  --output-json /root/autodl-tmp/data/openwebtext/metrics/dense_vs_mla_moe_hetero5_comparison.json \
  --output-txt /root/autodl-tmp/data/openwebtext/metrics/dense_vs_mla_moe_hetero5_comparison.txt
```

### 5.2 五模型总对比
```bash
python cs336_basics/compare_five_metrics.py \
  --dense /root/autodl-tmp/data/openwebtext/metrics/dense_baseline_metrics.json \
  --moe /root/autodl-tmp/data/openwebtext/metrics/moe_hetero5_shared1_small0p5_metrics.json \
  --muon /root/autodl-tmp/data/openwebtext/metrics/muon_baseline_metrics.json \
  --mla /root/autodl-tmp/data/openwebtext/metrics/mla_baseline_metrics.json \
  --mla-moe /root/autodl-tmp/data/openwebtext/metrics/mla_moe_hetero5_metrics.json \
  --output-json /root/autodl-tmp/data/openwebtext/metrics/dense_moe_hetero5_muon_mla_mla_moe_comparison.json \
  --output-txt /root/autodl-tmp/data/openwebtext/metrics/dense_moe_hetero5_muon_mla_mla_moe_comparison.txt
```

---

## 6. 常用查看命令

```bash
# 看训练日志
 tail -f /root/autodl-tmp/data/openwebtext/train_mla_moe_hetero5.log

# 看磁盘
 df -h /root/autodl-tmp

# 看 checkpoint 文件
 find /root/autodl-tmp/data/openwebtext/checkpoints -type f -name '*.pt' | sort
```
