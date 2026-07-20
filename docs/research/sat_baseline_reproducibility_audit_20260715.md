# Segment Any Tissue（SAT）基线复现审计

更新：2026-07-15

## 1. 为什么 SAT 是最直接先例

SAT（Medical Image Analysis, 2025, DOI: `10.1016/j.media.2025.103550`）已经包含以下核心链条：

1. 输入一张带像素标注的 reference image-mask；
2. 用预训练视觉特征做 reference-query patch matching；
3. 同时构造 positive、negative 与 hard prompts；
4. 在 feature/physical 双空间中优化 prompt 分布；
5. 把优化后的 prompts 输入冻结 SAM；
6. 用 cyclic feedback 与 prompt-scheme score 为每张 target 动态选择超参数；
7. 支持多 reference ensemble。

因此，本项目不能再把“one-shot medical reference matching”“自动点提示”“动态点数/参数”或“多 reference”单独写成创新。

## 2. 官方发布物状态

2026-07-15 克隆：

- 仓库：`external/Segment-Any-Tissue/`
- commit：`16eaf1ba3f733901bef5f0176fba801a4886183f`
- 内容：README 与展示图片；README 的 TODO 仍为 `Release code`
- 结论：当前无法从该仓库运行论文完整 SAT pipeline，也无法验证 cyclic feedback、prompt score 和 hyperparameter bank 的精确实现。

这意味着论文中不能写“已复现官方 SAT”。

## 3. 可运行前身 GBMSeg

SAT README 指向其 MICCAI 2024 前身 GBMSeg：

- 仓库：`external/GBMSeg/`
- commit：`d3e41786ba603bcc85f58a9b267d23de56aa2c87`
- 包含 DINOv2 matching、automatic prompt engineering 与 SAM 推理代码。

GBMSeg 代码公开了以下流程：

- reference foreground/background patch 分别做双向最近邻 matching；
- 删除不满足 cycle consistency 的匹配点；
- 根据 feature/physical distance 构造 hard negative；
- 通过固定半径对 positive/negative 点做空间去冗余；
- positive/negative joint prompts 输入 SAM。

但其默认 DINOv2 ViT-G 权重、本地数据路径和部分边界处理并不直接适配当前 MoNuSeg 环境。真实 reference 审计还发现：原始“14×14 patch 全部为前景”的规则在当前微小细胞核 mask 上得到 0 个前景 patch；适配版固定使用至少 50% foreground occupancy，得到 223 个前景 patch，并保留 202 个纯背景 patch。若改用本地 DINOv2-L、该占比规则与相同 SAM ViT-B，必须命名为 **GBMSeg-style adapted baseline**，并逐项披露改动。

## 4. 论文中的公平呈现

基线分三层：

1. **SAT reported method**：只在 related work 中描述，不填入本地 MoNuSeg 数值主表；
2. **GBMSeg-style adapted**：按公开前身代码实现同协议 MoNuSeg 基线，明确不是 SAT 官方复现；
3. **Our fixed fg-bg matcher**：冻结 K/lambda，不含 policy；用于隔离 foreground/background prototype 的机制收益。

只有当 SAT 作者后续释出完整代码，才补充真正的 official SAT reproduction。

## 5. 剩余可发表差异

本项目必须把贡献收缩为 SAT 没有直接回答的部分：

1. train-only policy：动作选择监督只来自 train，validation 选 threshold，official test 不做 prompt-scheme 自评分选参；
2. 可审计有限动作：显式联合 K、foreground-background suppression 与 candidate budget，并分解 prefilter/selector regret；
3. dense all-nuclei 机制：同时报告 point precision、component coverage、object recall、merge 与面积扩张；
4. reference 聚合统计：多 reference 是训练 episode 增强，测试按 query 聚合而非把相关 episode 当独立样本；
5. frozen segmenter response：证明同一 dense joint prompt 在 SAM ViT-B/L/H/SAM2.1 上响应不同，并据此限制 model-agnostic 主张。

如果 learned policy 最终不能稳定优于 validation-best fixed action，以上差异不足以支持强 prompt-policy 论文，应转为 foreground/background matcher 机制论文或扩容后重训。

## 6. 原始资料

- SAT: https://doi.org/10.1016/j.media.2025.103550
- SAT repository: https://github.com/SnowRain510/Segment-Any-Tissue
- GBMSeg repository: https://github.com/SnowRain510/GBMSeg
