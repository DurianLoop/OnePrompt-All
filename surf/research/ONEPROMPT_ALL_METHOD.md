# OnePrompt-All：单提示同图多实例分割

## 问题定义

给定单张图像 \(I\)，用户只对某个目标实例提供一次粗提示 \(p\)：一个前景点击或一个可能包含背景的粗框。系统不使用参考图、类别名称、实例数量或精确掩码，而是输出图中全部与该目标同类的独立实例掩码：

\[
\mathcal{Y}=\{M_1,\ldots,M_K\}=\mathrm{OnePrompt\text{-}All}(I,p).
\]

这里的 \(K\) 由候选集合选择过程自然确定，而非由用户指定。

## 核心流程

```text
one coarse prompt
       │
       ▼
SAM3 seed + same-image instance proposals
       │
       ├── select the proposal containing the prompt → online seed mask
       │
       ▼
DINOv2 token features → masked seed prototype
       │
       ▼
soft proposal scoring + area filtering + diversity NMS
       │
       ▼
all same-class instance masks
```

### 1. 单提示种子解析（SAM3）

以唯一用户提示运行 SAM3，得到候选掩码及其 SAM3 置信度 \(\{(m_i,s_i)\}\)。对于点提示，优先选择包含点击位置、且 SAM3 置信度最高的候选作为种子 \(m_*\)；对于粗框，则从框提示响应中选择最高置信候选。这个种子只用于构造在线 exemplar，不要求用户提供精确掩码。

### 2. 特征净化与在线 exemplar（DINOv2-B）

提取 DINOv2-B 的 37×37 密集 token 特征 \(f(x)\)，并仅在种子掩码覆盖的 token 内聚合，形成单位化原型：

\[
z=\mathrm{Norm}\left(\frac{1}{|\Omega(m_*)|}\sum_{x\in\Omega(m_*)} f(x)\right).
\]

相比直接取点击点特征，掩码约束的聚合降低了粗点击位置、局部高光和纹理背景对 exemplar 的干扰。

### 3. 同图候选的软视觉证据

对每一个 SAM3 候选 \(m_i\)，计算其掩码内 token 与种子原型的平均余弦相似度：

\[
v_i=\frac{1}{|\Omega(m_i)|}\sum_{x\in\Omega(m_i)} f(x)^\top z.
\]

OnePrompt-All 不把 DINO 当作硬分类器，也不以相似度阈值直接删除候选；DINO 只作为 **软证据** 与 SAM3 置信度融合，用于打破重叠候选之间的排序：

\[
r_i=0.8\cdot\widetilde{s_i}+0.2\cdot\widetilde{v_i}.
\]

这样保留了外观变化实例的召回能力，同时抑制同一区域的低质量重复 proposal。

### 4. 面向实例集合的选择

候选先经过最小面积过滤（全量实验中为图像面积的 0.2%），再按融合分数执行 IoU=0.5 的 mask NMS。结果是相互去重的一组实例 mask，而不是单一相似度热图或一个粗连通区域。

## 负点的设计原则

系统提供“DINO 对比驱动的自适应负点”作为可选研究模块：只在种子外环与种子原型具有足够语义差异时，才从低相似度区域选取 0 到 \(N\) 个内部背景点。密集同类场景中，邻近区域很可能是真实目标实例；因此全量 benchmark 的默认配置采用 **0 个自动负点**，避免错误负点损害 Recall。

这体现了负点机制是条件性纠错工具，而不是固定数量的交互轮次；用户侧始终只有一次提示。

## 与跨图 matcher 的区别

| 维度 | 传统跨图 Matcher | OnePrompt-All |
|---|---|---|
| 参考信息 | 另一张参考图及精确 mask | 当前图中的一次粗提示 |
| exemplar | 离线给定 | 从 SAM3 种子在线构造 |
| 特征传播范围 | 跨图 | 同图实例候选集合 |
| DINO 的作用 | 通常作为匹配/阈值裁决 | 原型相似度的软排序证据 |
| 输出 | query 内匹配区域或 mask | 全部独立同类实例 mask |

## 已完成的全量实验配置

| Item | Value |
|---|---|
| Method | OnePrompt-All |
| Instance proposer | SAM3 |
| Online feature encoder | DINOv2-B/14 |
| User interaction | 1 random foreground click |
| Automatic negative points | 0 (default) |
| DINO fusion weight | 0.2 |
| Minimum candidate area | 0.2% of image area |
| Mask NMS IoU | 0.5 |
| Benchmark | Integrated OnePrompt-All Benchmark |
| Images / classes / trials | 2,220 / 232 / 11,100 |
| F1 / Precision / Recall / union IoU | 0.667 / 0.687 / 0.712 / 0.601 |

