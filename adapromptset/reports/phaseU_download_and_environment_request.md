# Phase U Download & Environment Request: External Comparators

**Date**: 2026-07-24
**Purpose**: 下载 GF-SAM 和 MAUP 官方代码以进行外部比较实验

## 需要下载

| Method | Repository | License | Size | Purpose |
|--------|-----------|---------|------|---------|
| GF-SAM | https://github.com/ANDYZAQ/GF-SAM | MIT | <100MB | Graph-based point aggregation, training-free, 使用 SAM+DINOv2 |
| MAUP | https://github.com/YazhouZhu19/MAUP | Unknown | <100MB | Multi-center uncertainty-aware prompting, training-free, medical domain |

## 理由

- 两者都是 Tier 2 (官方代码可用，接口需适配)
- 两者都是 training-free，与我们的 frozen pipeline 协议兼容
- GF-SAM 使用相同的 SAM+DINOv2 技术栈，最低适配成本
- MAUP 是 MICCAI 2025 论文，针对医学图像，domain 最匹配
- 满足 Option A (2 external methods) 的 external comparator 要求

## 环境要求

- 两者都需要 Python 3.11+ (sam_gpu 环境)
- 不需要额外 GPU 或 CUDA 环境
- 不需要申请 gated checkpoint
- 总下载 <200MB

## 适配计划

1. Clone 仓库到 `D:\SAM\code\external\`
2. 适配数据加载器到 MoNuSAC reference-resampling 协议
3. 在 MoNuSAC 训练集上运行 Phase 1+2 pipeline
4. 计算 M2 指标 (instability AUROC, catastrophic AUPRC, CFR reduction)
5. 与我们的 M2 方法进行 head-to-head 对比

## 风险

- MAUP 的 license 未确认 (需检查)
- 适配工作量: GF-SAM ~2天, MAUP ~2-3天

---

**请确认是否批准下载以上两个外部仓库。**