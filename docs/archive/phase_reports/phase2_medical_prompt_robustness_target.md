# Phase 2 Target: Medical Prompt Robustness Benchmark

## 总目标

Phase 2 不再只是“把更多模型跑一遍”，而是建立一个面向医学图像的 promptable segmentation 诊断框架：

```text
同一数据 + 同一 prompt 协议 + 同一评价脚本 -> 找出模型在哪里失败，以及 prompt 应该如何改进。
```

这会服务后面的核心创新：结构化医学 prompt、SAM2 memory 选择和轻量 adapter。

## 研究问题

1. 通用 SAM、MedSAM、SAM2 在医学图像上的失败模式是否一致？
2. box、point、auto box 三类 prompt 哪一类最稳定？
3. IDRID 这类小目标多实例任务，与 MonuSeg 这类密集病理核任务，是否需要不同的 prompt 生成策略？
4. 我们是否能通过自动医学 prompt 生成，在不改主干模型的情况下提升准确率和交互效率？

## 当前 Phase2-lite 已完成

- 数据：IDRID 20 cases + MonuSeg 20 cases。
- Prompt：GT box、perturbed box、centroid point、grid point、auto box。
- 模型：SAM ViT-B、MedSAM ViT-B、SAM2 Tiny。
- 输出：manifest、prompt jsonl、predicted masks、metrics、失败图集。

## 阶段性观察

- `auto_box` 当前均值最高，但它既可能说明图像启发式有效，也可能说明 GT 合并任务与局部 prompt 的定义不完全一致，需要人工检查失败图集。
- IDRID 明显更难，后续创新优先围绕小病灶、多实例、低对比区域。
- MedSAM 是强 box baseline，但不应被强行纳入 point/text prompt 比较。
- SAM/SAM2 在点提示上的稳定性需要做扰动曲线，而不是只看一个 centroid/grid 点。

## Phase 2 主输出

1. `phase2_lite_analysis_report.md`：小基准结果和失败模式。
2. `summary_by_*.csv`：按模型、数据集、prompt、difficulty 的指标表。
3. `failure_cases.csv`：带失败标签的逐样本表。
4. `galleries/*.png`：最佳、最差、模型分歧、auto-box 成功/失败图集。
5. `auto_medical_prompt_v1`：下一阶段方法原型。

## Phase 3 的连接点

Phase 2 最终要给 Phase 3 一个明确结论：

```text
优先改哪种 prompt？
在哪个数据集上改？
用什么失败类型证明这个改进是必要的？
```

推荐的 Phase 3 起点是：

```text
Auto Medical Prompt Generator v1
```

它不训练大模型，先用可解释的医学图像启发式和候选选择策略生成更好的 box/point prompt，再喂给 SAM-family 模型。这个路线成本低、实验闭环短，也容易写成论文中的方法模块。
