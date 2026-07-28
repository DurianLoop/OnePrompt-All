# OCSAM / AdaPromptSet — Multi-Reference Visual Prompting Research Suite

> **Language**: [English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

**OCSAM / AdaPromptSet** is a comprehensive research suite for multi-reference visual prompting and selective prediction with frozen foundation models (SAM, SAM2, DINOv2). The project spans 15+ research phases, systematically investigating reference-induced failure modes, risk estimation, and selective abstention in few-shot medical image segmentation.

### Key Research Contributions

#### 1. Reference-Instability Benchmark (Phase R/S/T)

We establish that reference choice is a structural, quantifiable source of error in two-stage visual prompting pipelines (DINOv2 + SAM). On MoNuSAC, swapping the reference image can change Dice from 0.82 to 0.02, and over 92% of queries exhibit best-worst gaps ≥ 0.05.

**Key findings (Phase T Holdout, 40 queries, 10/10 pre-registered gates passed):**

| Metric | Value |
|---|---|
| Queries with best-worst gap ≥ 0.05 | 92.5% |
| Mean within-query reference std (k=1) | 0.0742 |
| k=6 consensus Dice gain | +0.0266 (95% CI: +0.0140) |
| Prediction agreement (k=1 → k=6) | 0.649 → 0.719 |
| Catastrophic failure rate (single ref) | 6.33% |

**Cross-dataset**: MoNuSeg replicates (gain +0.0149), IDRID fails (baseline Dice ≈ 0.10).

#### 2. Prediction Consistency for Selective Abstention (Phase AA/AB)

A simple, training-free risk signal: `R_PC(q) = 1 - mean pairwise IoU` of segmentations from m=6 references. Coverage-based abstention dramatically reduces catastrophic failures.

**Primary pipeline (DINOv2-L + SAM-B, Development Cohort):**

| Metric | Prediction Consistency | Random Baseline |
|---|---|---|
| AURC | 0.0158 | 0.1295 |
| CFR@80 | 0.0156 | 0.1536 |
| CFR Reduction | **88.6%** | — |
| Dice@80 | 0.501 | 0.446 |

**Cross-pipeline (5 pipelines, all directional support):**

| Pipeline | AURC | CFR Reduction |
|---|---|---|
| DINOv2-L + SAM-B | 0.0158 | 88.6% |
| DINOv2-B + SAM-B | 0.0145 | 88.6% |
| DINOv2-L + SAM-L | 0.1002 | 23.1% |
| DINOv2-L + SAM2-Tiny | 0.0910 | 9.1% |
| DINOv2-L + SAM2-Small | 0.1339 | -1.6% |

#### 3. Action-Space Oracle Audit (Phase X/Y)

Systematic diagnosis of why prior policy learning failed: small discrete prompt banks (12 actions) systematically underestimate the upper bound of promptable segmentation. K+ scaling from 5 to 100 improves Dice by +0.097, and search oracle achieves 0.726 vs. 0.612 for best fixed action.

#### 4. Promptability Failure Taxonomy (Phase 2/3/4, DDR)

Negative results from GlaS automatic prompting: candidate point precision ≈ 0.97 but component coverage ≈ 0.42 (Gate D failure). Region proposals fail due to area priors. DDR stratified matcher reveals that 15/16 image-intent units have zero candidate hits.

### What Has Been Retracted

After rigorous VV4 forensic audit (92 claims extracted, 8 contradicted by later phases):

- **RAMP (Reference-Adaptive Multi-Prompt Fusion)**: Full weighting/fusion/consensus showed zero added value in Phase W. Prohibited from main method claims.
- **Anti-correlation as universal law**: Only significant in SAM-B pipelines (r=-0.51); not in SAM-L, SAM2-Tiny, or SAM2-Small. Removed from main text.
- **RRN (Reference Reliability Network)**: Grouped CV AUROC dropped to 0.573. Moved to appendix only.
- **SAM-Med2D**: Checkpoint unverifiable. Permanently removed from main tables.
- **External baseline incompleteness**: CycleSAM, H-SAM, SAM-Med2D have no reproducible predictions.

### Directory Structure

```
adapromptset/
├── src/adapromptset/          # Core package (matchers, segmenters, evaluation, statistics)
│   └── scripts/               # Package-level scripts (Phase U-W, VV4 forensics)
├── scripts/                   # Main working scripts (all phases)
├── configs/                   # YAML/JSON configs (actions, data, matchers, segmenters, frozen protocols)
├── docs/                      # Research protocols, claim registries, decision logs
├── reports/                   # All phase results, gate statuses, program statuses
├── paper/                     # Paper drafts (Phase U, V, T, AB), figures, tables, LaTeX
├── VV4/                       # V4 paper forensic audit package (evidence forensics)
├── manifests/                 # Episode/split manifests with SHA256 hashes
├── tests/                     # Unit tests
├── external_baselines/        # GF-SAM and MAUP reference implementations
└── .gitignore                 # Excludes checkpoints, datasets, external repos
```

### Quick Start

```powershell
# Clone
git clone https://github.com/DurianLoop/OCSAM.git
cd OCSAM/adapromptset

# Install
pip install -e .

# Run Phase AB primary experiment
$env:PYTHONPATH = "D:\SAM\adapromptset\src"
python scripts/run_phaseAB_wp3_wp4_closure.py
```

### Research Phase Map

| Phase | Branch | Focus | Status |
|---|---|---|---|
| Phase R | `phaseR-*` | Reference stability feasibility | Not feasible (deterministic) |
| Phase S | `phaseS-*` | Multi-track research (A/B/C/D) | Track A: benchmark valid |
| Phase T | `phaseT-*` | Holdout validation + cross-dataset | 10/12 gates passed |
| Phase U | — | RAMP, RRN, theory, baselines | Experimental exploration |
| Phase V | — | Repair, anti-correlation controls, grouped CV | Corrections applied |
| Phase W | `phaseW-*` | Experimental closure, null model fix | Full RAMP: no added value |
| Phase AA | `phaseAA-*` | Simplified selective prediction | 12/14 gates |
| Phase AB | `phaseAB-*` | Integrity closure, paper assembly | 12/15 gates, paper candidate |
| Phase X | `phaseX-*` | Action-space oracle audit | Reopen policy learning |
| Phase Y | `phaseY-*` | Policy feasibility | Route closed |
| Phase 2/3/4 | — | GlaS automatic prompting | All failed |
| Phase 4+ DDR | — | Promptability benchmark | Early exploration |
| VV4 | — | V4 paper forensic audit | Partially reconstructable |

### Publication Status

- **Current paper**: Phase AB — "When References Disagree: Prediction Consistency for Selective Visual Prompting" (12/15 gates, workshop/benchmark ready)
- **V4 paper**: "Reference Reliability Estimation for Visual Prompting Pipelines" — partially reconstructable, core method claims retracted
- **Recommended venue**: Workshop / Benchmark track

### Key References

- Kirillov et al., "Segment Anything", ICCV 2023
- Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024
- Oquab et al., "DINOv2: Learning Robust Visual Features", 2023
- Zhang et al., "GF-SAM: Graph-based Few-shot Segmentation", NeurIPS 2024
- Zhu et al., "MAUP: Multi-Anchor Uncertainty-Aware Prompting", MICCAI 2025

---

<a name="chinese"></a>
## 中文

### 概述

**OCSAM / AdaPromptSet** 是一个面向多参考视觉提示与选择性预测的综合研究套件，基于冻结基础模型（SAM、SAM2、DINOv2）。项目跨越 15+ 个研究阶段，系统性地研究了少样本医学图像分割中参考诱导的失败模式、风险估计和选择性弃权。

### 核心研究贡献

#### 1. 参考不稳定性基准 (Phase R/S/T)

我们确立了参考选择是两阶段视觉提示管线（DINOv2 + SAM）中一个结构性、可量化的误差来源。在 MoNuSAC 上，替换参考图像可使 Dice 从 0.82 变为 0.02，超过 92% 的查询存在 best-worst gap ≥ 0.05。

**Phase T 留出集关键结果（40 查询，10/10 预注册门控通过）：**

| 指标 | 数值 |
|---|---|
| best-worst gap ≥ 0.05 的查询比例 | 92.5% |
| 平均 within-query 参考标准差 (k=1) | 0.0742 |
| k=6 共识 Dice 增益 | +0.0266 (95% CI: +0.0140) |
| 预测一致性 (k=1 → k=6) | 0.649 → 0.719 |
| 灾难性失败率 (单参考) | 6.33% |

**跨数据集**：MoNuSeg 复现（增益 +0.0149），IDRID 失败（基线 Dice ≈ 0.10）。

#### 2. 预测一致性选择性弃权 (Phase AA/AB)

一种简单、无需训练的风险信号：`R_PC(q) = 1 - mean pairwise IoU`（m=6 个参考的分割结果）。基于覆盖度的弃权策略大幅降低灾难性失败。

**主管线 (DINOv2-L + SAM-B, Development Cohort)：**

| 指标 | Prediction Consistency | Random 基线 |
|---|---|---|
| AURC | 0.0158 | 0.1295 |
| CFR@80 | 0.0156 | 0.1536 |
| CFR 降低 | **88.6%** | — |
| Dice@80 | 0.501 | 0.446 |

**跨管线（5 个管线，全部方向性支持）：**

| 管线 | AURC | CFR 降低 |
|---|---|---|
| DINOv2-L + SAM-B | 0.0158 | 88.6% |
| DINOv2-B + SAM-B | 0.0145 | 88.6% |
| DINOv2-L + SAM-L | 0.1002 | 23.1% |
| DINOv2-L + SAM2-Tiny | 0.0910 | 9.1% |
| DINOv2-L + SAM2-Small | 0.1339 | -1.6% |

#### 3. 动作空间神谕审计 (Phase X/Y)

系统诊断了为什么先前的策略学习失败：小离散 prompt bank（12 个动作）系统性低估了可提示分割的上限。K+ 从 5 扩展到 100 提升 Dice +0.097，search oracle 达到 0.726 vs. 最佳固定动作 0.612。

#### 4. 可提示性失败分类 (Phase 2/3/4, DDR)

GlaS 自动提示的负结果：候选点精度 ≈ 0.97 但组件覆盖 ≈ 0.42（Gate D 失败）。区域提议因面积先验而失败。DDR 分层匹配器揭示 15/16 个 image-intent 单元零候选命中。

### 已被撤回的内容

经过严格的 VV4 取证审计（92 条声明提取，8 条被后续阶段推翻）：

- **RAMP**：完整 weighting/fusion/consensus 在 Phase W 中显示零增量价值。禁止作为主要方法声明。
- **反相关作为普适定律**：仅在 SAM-B 管线中显著 (r=-0.51)；SAM-L、SAM2-Tiny、SAM2-Small 不显著。已从主文移除。
- **RRN**：分组 CV AUROC 降至 0.573。仅放入附录。
- **SAM-Med2D**：checkpoint 不可验证。永久从主表移除。
- **外部基线不完整**：CycleSAM、H-SAM、SAM-Med2D 无可复现预测。

### 研究阶段总览

| 阶段 | 分支 | 焦点 | 状态 |
|---|---|---|---|
| Phase R | `phaseR-*` | 参考稳定性可行性 | 不可行（确定性方法） |
| Phase S | `phaseS-*` | 多轨研究 (A/B/C/D) | Track A: benchmark 有效 |
| Phase T | `phaseT-*` | 留出验证 + 跨数据集 | 10/12 门控通过 |
| Phase U | — | RAMP, RRN, 理论, 基线 | 实验探索 |
| Phase V | — | 修复、反相关控制、分组 CV | 修正已应用 |
| Phase W | `phaseW-*` | 实验闭合、null model 修复 | Full RAMP: 无增量价值 |
| Phase AA | `phaseAA-*` | 简化选择性预测 | 12/14 门控 |
| Phase AB | `phaseAB-*` | 完整性闭环、论文组装 | 12/15 门控，论文候选 |
| Phase X | `phaseX-*` | 动作空间神谕审计 | 重开策略学习 |
| Phase Y | `phaseY-*` | 策略可行性 | 路线关闭 |
| Phase 2/3/4 | — | GlaS 自动提示 | 全部失败 |
| Phase 4+ DDR | — | 可提示性基准 | 早期探索 |
| VV4 | — | V4 论文取证审计 | 部分可重建 |

### 发表状态

- **当前论文**：Phase AB — "When References Disagree: Prediction Consistency for Selective Visual Prompting"（12/15 门控，workshop/benchmark 候选）
- **V4 论文**："Reference Reliability Estimation for Visual Prompting Pipelines" — 部分可重建，核心方法声明已撤回
- **推荐投稿方向**：Workshop / Benchmark track

### 完整汇总

详见：[D:\SAM\reports\2.md](reports/2.md) — 全阶段研究大汇总，包含所有实现方式、运行结果和结论。

---

### Environment Setup

```powershell
# Clone
git clone https://github.com/DurianLoop/OCSAM.git
cd OCSAM

# Create conda environment
conda create -n ocsam python=3.10 -y
conda activate ocsam

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project
cd adapromptset
pip install -e .

# Run tests
$env:PYTHONPATH = "D:\SAM\adapromptset\src"
python -m pytest tests/
```

### License

This project is for research purposes. See individual external baseline directories (GF-SAM, MAUP) for their respective licenses.