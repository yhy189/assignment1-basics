# OpenWebText 四模型对比报告（Dense / MoE / Muon / MLA）

## 1. 模型结构与训练策略说明

### 1.1 Dense Baseline（手写 Transformer）

- 结构：标准 Decoder-only Transformer
- 注意力：全维度多头自注意力（MHA）
- FFN：SwiGLU（`d_ff=2048`）
- 优化器：AdamW
- 训练策略：FP32，未启用 AMP，未启用激活检查点，`label_smoothing=0.0`

### 1.2 MoE（异构 5 专家）

- 结构：MoE Transformer（在 FFN 位置替换为 MoE）
- 专家配置：5 个专家，`expert_d_ffs=[2048, 1024, 1024, 1024, 1024]`
- 路由策略：固定激活 1 个共享大专家（index=0）+ 从其余 4 个小专家中选 1 个（`top_k=2`）
- 优化器：AdamW
- 训练策略：FP32，未启用 AMP，未启用激活检查点，`label_smoothing=0.0`

### 1.3 Muon（优化器路线）

- 结构：模型结构仍是 Dense Transformer（不是新架构）
- 优化器：Muon（矩阵参数走 Muon，其他参数回退 AdamW）
- 关键策略：
  - AMP FP16
  - 激活检查点（Activation Checkpointing）
  - `label_smoothing=0.05`
  - `micro_batch_size=8`（全局 batch 通过梯度累积保持 32）

### 1.4 MLA（Latent Attention）

- 结构：MLA Transformer
- 核心机制：注意力前先降到潜空间，再做注意力，再投影回模型维度
  - `d_model=512`
  - `mla_d_model=256`
- 优化器：AdamW
- 训练策略：AMP FP16，未启用激活检查点，`label_smoothing=0.0`

---

## 2. 实验设置对齐说明

为保证可比性，四组训练都使用 OpenWebText 同一套 token 数据，且主要规模参数一致：

- `total_tokens=100,000,000`
- `context_length=256`
- `batch_size=32`
- `num_layers=4`
- `d_model=512`
- `num_heads=8`

注：不同模型在优化器与内存策略上有差异（如 Muon/MLA 启用 AMP），因此该对比是“工程真实表现对比”，不是严格单变量消融。

---

## 3. 全指标对比表

### 3.1 结果总表

| 模型 | Final Train Loss | Final Valid Loss | Best Valid Loss | Best Step | Final Valid PPL | Best Valid PPL | Avg Tok/s (Tail200) | Max Mem (GB) | Train Time (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 4.3557 | 4.3349 | 4.2974 | 10360 | 76.31 | 73.51 | 10599.69 | 9.85 | 2.629 |
| MoE (hetero5) | 4.2863 | 4.2969 | 4.2318 | 11740 | 73.47 | 68.84 | 8837.32 | 9.61 | 3.135 |
| Muon | 6.0656 | 5.8439 | 5.6673 | 12140 | 345.13 | 289.27 | 17146.29 | 2.39 | 1.624 |
| MLA | 4.4664 | 4.4194 | 4.3790 | 12180 | 83.04 | 79.76 | 25849.31 | 7.40 | 1.076 |

### 3.2 相对 Dense 的对比（核心工程指标）

| 模型 | Final Valid Loss Δ vs Dense | Final Valid Loss 变化 | 速度倍率 vs Dense | 显存倍率 vs Dense | 训练时长倍率 vs Dense |
|---|---:|---:|---:|---:|---:|
| MoE (hetero5) | -0.0379 | -0.88%（更好） | 0.834x（更慢） | 0.976x（略省） | 1.192x（更久） |
| Muon | +1.5090 | +34.81%（更差） | 1.618x（更快） | 0.242x（显存最低） | 0.618x（更短） |
| MLA | +0.0845 | +1.95%（略差） | 2.439x（最快） | 0.752x（明显下降） | 0.409x（最短） |

### 3.3 排名（按关键维度）

| 维度 | 排名（从优到劣） |
|---|---|
| 收敛质量（Final Valid Loss 低） | MoE > Dense > MLA > Muon |
| 吞吐速度（Avg Tok/s 高） | MLA > Muon > Dense > MoE |
| 显存占用（Max Mem 低） | Muon > MLA > MoE > Dense |

---

## 4. 结论摘要

| 模型 | 一句话结论 |
|---|---|
| Dense | 作为基线稳定可靠，综合表现均衡。 |
| MoE | 质量最佳（最低 valid loss），但最慢、训练耗时最长。 |
| Muon | 显存与速度优势显著，但本次配置下收敛质量明显不足。 |
| MLA | 在质量接近 Dense 的前提下，速度和时长优势最大，工程性价比较高。 |

---

## 5. 数据来源

- `/root/autodl-tmp/data/openwebtext/metrics/dense_baseline_metrics.json`
- `/root/autodl-tmp/data/openwebtext/metrics/moe_hetero5_shared1_small0p5_metrics.json`
- `/root/autodl-tmp/data/openwebtext/metrics/muon_baseline_metrics.json`
- `/root/autodl-tmp/data/openwebtext/metrics/mla_baseline_metrics.json`
- `/root/autodl-tmp/data/openwebtext/metrics/dense_moe_hetero5_muon_mla_comparison.json`

