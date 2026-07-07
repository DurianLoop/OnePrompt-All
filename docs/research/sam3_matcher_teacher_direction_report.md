# SAM3 + Matcher 老师方向逐项汇总

日期：2026-06-24  
工作目录：`D:\SAM`

## 0. 结论先行

老师截图里的方向可以整理成一条完整技术路线：

1. 先在自然场景里证明现象：SAM3 的点/框等视觉 prompt 对自然物体分割可靠，但 SAM3 直接文本 prompt 只适合短名词短语，对动作、关系、复杂指代、某些人物属性表达容易失效。
2. Stage 1 用 DINOv2/Matcher 做粗定位，输出一批候选点。若原图大于 DINO 固定输入尺寸且不能降采样，应采用滑动窗口/切块推理，并把窗口坐标映射回原图。
3. Stage 2 从候选点中反复采样多组 point prompts，送入 SAM/SAM3 产生多组 mask proposals。
4. Stage 3 对每个 proposal 做 instance-level matching，超过阈值保留，低于阈值丢弃；多目标场景下要先把候选点聚成 K 个簇，再按簇或实例分别匹配，避免所有目标混成一个 mask。
5. 改进方向是给 Stage 1 加 feedback：从 100 个点里人工选少量正确点和错误点，用它们更新 target feature/匹配向量，再重新匹配，预期可提升点的 precision 和 coverage。

推荐自然场景依据：

| 用途 | 推荐对象 | 为什么符合老师的 a/b |
|---|---|---|
| SAM3 自然场景主依据 | SAM3 + SA-Co benchmark | SAM3 官方定义 promptable concept segmentation，支持短文本概念、视觉 prompts，并发布 SA-Co 图像/视频 benchmark，标注 noun phrase、instance masks、unique IDs、negative prompts。可作为自然场景数据集基础。 |
| 文本 prompt 失效主依据 | SAM3-I 论文 | SAM3-I 明确指出 SAM3 原生文本主要是 short noun phrase，现实中的属性、空间关系、功能、动作、状态、隐式推理会超出原生 SAM3 文本能力。`execute` 属于动作/功能类，不是可直接分割的名词短语。 |
| 多目标/复杂自然图像 | RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO | 这些是自然照片中的 referring expression segmentation 数据集，包含属性、关系和多目标指代表达；适合构造 “girl in white / person executing action” 这类文本困难样例。 |
| Matcher 流程依据 | Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching | Matcher 原文就是 DINOv2 特征匹配 + SAM proposal + instance-level matching，且支持 parts、whole、multiple instances。 |

外部依据链接：

- SAM3 官方仓库：https://github.com/facebookresearch/sam3
- SAM3 论文：https://arxiv.org/abs/2511.16719
- SAM3-I 论文：https://arxiv.org/abs/2512.04585
- Matcher 官方仓库：https://github.com/aim-uofa/Matcher
- Matcher 论文：https://arxiv.org/abs/2305.13310

## 1. 老师最上方 a/b：自然场景里找符合现象的文章或数据集

### 1.1 a：SAM3 基于 point/foreground/background 的效果较好

可以用 SAM3 官方视觉 prompt 能力作为依据。SAM3 README 说明它可用 text 或 visual prompts，如 points、boxes、masks，进行图像/视频分割；SAM3-I 论文也把 SAM1/2 的点、框、mask 交互式分割称为 Promptable Visual Segmentation，强调每个视觉 prompt 可分割单目标。

建议实验样例：

1. 从 SA-Co/Gold 或 COCO/LVIS 自然图像中选一张多人、多物体图片。
2. 对目标物体给 foreground point，邻近干扰区域给 background point。
3. 记录 SAM3/SAM 输出 mask，并用 IoU、Dice 或人工可视化说明“点到物体后可分出来”。

如果本机 SAM3 checkpoint 受 Hugging Face 权限影响无法跑完整 SAM3，可以先用当前 SAM/SAM2/Matcher pipeline 做视觉 prompt 复现，文档中注明 SAM3 结论来自官方能力说明，实测部分使用本地可运行替代。

### 1.2 b：SAM3 文本提示失效

这不是偶然现象，而是 SAM3 文本接口的设计边界。SAM3 官方论文把 concept prompt 定义为 short noun phrases 或 image exemplars；SAM3-I 进一步指出真实使用里常见的属性、空间关系、功能、动作、状态、隐式推理超出短名词短语表达能力。

对截图里的两个词可这样解释：

| 文本 | 问题类型 | 建议处理 |
|---|---|---|
| `white girl` | 人物属性表达容易歧义；若目标是“穿白色衣服的女孩”，更合理的文本是 `girl` 或 `person`，再由视觉/人工筛选白衣目标。SAM3 agent 规则也建议人物目标用整体人类名词，如 `person`、`girl`，避免用识别性局部或复杂属性直接做核心 prompt。 |
| `execute` | 动作/动词，不是物体类别或名词短语。SAM3 原生文本 prompt 不适合直接用动作词分割；应改成可见对象名词，如 `person`、`tool`、`machine`，再通过后处理或外部模型判断动作。 |

推荐自然场景数据组合：

1. SA-Co：用于验证 SAM3 短名词短语、多实例、negative prompt、unique ID。
2. RefCOCO/gRefCOCO：用于收集复杂自然语言指代，比如“穿白衣的人”“正在执行某动作的人”。这些正好可证明 SAM3 原生 text prompt 的边界。
3. COCO/LVIS：用于多实例自然场景，和 Matcher 的多目标实验一致。

## 2. Stage 1 问题 1：图像大于 DINO 规定大小且不允许降采样，怎么办？

结论：不能把整图 resize 到 518 x 518，否则小目标和局部结构会丢失；应使用滑动窗口。

本地 Matcher 默认使用 DINOv2，配置里 `img_size=518`、`patch_size=14`，所以一张 518 x 518 图对应 37 x 37 个 patch。如果原图是 2048 x 2048，但不允许降采样，应执行：

1. 将原图按 518 x 518 切块，stride 建议 392 或 448，保留 15% 到 25% overlap。
2. 边缘不足 518 的窗口用 padding，不缩放图像内容。
3. 每个窗口独立跑 DINOv2，得到窗口内 matched points。
4. 将窗口内点 `(x_win, y_win)` 映射回全图坐标：`x = x0 + x_win - pad_left`，`y = y0 + y_win - pad_top`。
5. 对重叠窗口产生的重复点做去重，可用 DBSCAN、NMS 或按 patch 网格合并。
6. 后续 SAM/SAM3 分割使用原图坐标。若 SAM/SAM3 本身也受显存限制，则对每个点所在区域裁 crop + halo，分割后再 paste 回原图。

推荐窗口参数：

| 参数 | 建议 |
|---|---|
| window_size | 518 |
| stride | 392，目标很密时用 336 |
| overlap | 25% 左右 |
| halo for SAM crop | 64 到 128 像素 |
| point dedup distance | 7 到 14 像素，约 0.5 到 1 个 DINO patch |

这样满足“不降采样”的要求，同时仍兼容 DINO 固定输入大小。

## 3. Stage 1 问题 2：matching 的过程究竟怎么完成？

Matcher 原文流程是：

1. 输入 reference image `x_r`、reference mask `m_r`、target image `x_t`。
2. 用 DINOv2 提取 reference/target 的 patch-level features：`z_r`、`z_t`。
3. 计算 correspondence matrix：`S = sim(z_r, z_t)`，即 patch 间 cosine similarity。
4. Forward matching：reference mask 内的点/patch 去 target 上找最相似 patch，得到 target matched points。
5. Reverse matching：target matched points 反向匹配回 reference。
6. Mask filtering：反向匹配回来的点若不在 reference mask 内，则认为是 outlier，丢弃。
7. 得到过滤后的 matched points。
8. 对 matched points 做 k-means++，形成 K 个位置簇。
9. Robust Prompt Sampler 采样三类 prompt：
   - part-level：每个 cluster 内采样；
   - instance-level：所有 matched points 内采样；
   - global：cluster centers 采样。
10. 将多组 point/box prompts 送入 SAM，得到 mask proposals。
11. 对 proposals 做 instance-level matching，用 EMD、purity、coverage 评分。
12. 高分 proposal 保留，低分 proposal 丢弃，最后 merge top-k masks。

本地代码对应关系：

| 位置 | 作用 |
|---|---|
| `code\matcher\Matcher.py` | 当前主改造版，包含 MatchingMetrics、top-k matched points、可视化、RPS、mask scoring。 |
| `code\matcher\Matcher.py::patch_level_matching` | 本地 Stage 1：reference mask feature 加权平均，然后和 target features 做 similarity，取 top-k，再按 threshold 过滤。 |
| `code\matcher\Matcher.py::clustering` | k-means++ 聚类。 |
| `code\matcher\Matcher.py::mask_generation` | Stage 2 + Stage 3：采样 points，调用 SAM automatic mask generator，计算 purity/coverage/EMD，过滤并合并 mask。 |
| `code\matcher\k_means.py` | k-means++ 实现。 |
| `code\matcher\Matcher_SAM.py` | 另一个 box/point 更偏实验版，当前用 top-k points 生成多个小 box。 |

注意：本地 `patch_level_matching` 和论文原版不是完全一致。论文原版强调 forward + reverse bidirectional matching；本地当前主线更像“reference foreground feature prototype 与 target patches 相似度匹配”。这对加入 feedback 机制反而更方便，因为 target/reference prototype 可以被正确点和错误点更新。

## 4. Stage 2+3：sample N% points，重复 m 次，再 instance-level threshold

老师描述的流程可以落地为：

1. Stage 1 输出 `P = {p_1, ..., p_100}`。
2. 每次随机采样 `N%` 个点作为一次 prompt。例：100 points，N=5%，则每次采 5 点。
3. 重复 `m` 次，例如 m=30，则得到 30 组 point prompts。
4. 每组 point prompts 送入 SAM/SAM3，得到一个或多个 mask proposals。
5. 对每个 proposal 做 instance-level matching，计算 score。
6. score 高于 threshold 的 proposal 保留，低于 threshold 丢弃。
7. 最后对保留 proposal 做去重/合并，输出实例级 masks。

本地代码已经有接近实现：

- `RobustPromptSampler.sample_points()` 当前不是按 N% 写死，而是 `sample_range=(4, 6)`，点数多于 8 时随机采样 `max_sample_iterations=30` 次。
- 这等价于在 100 个点里每次采 4 到 6 个点，重复 30 次，约为 4% 到 6%。
- `mask_generation()` 已经对每个 SAM proposal 计算 purity、coverage、EMD，并用阈值/score 过滤。

建议把文档化公式写成：

```text
sample_size = max(1, round(N / 100 * len(points)))
for t in range(m):
    prompt_points_t = random_sample(points, sample_size)
    masks_t = SAM(image, prompt_points_t)
    for mask in masks_t:
        score = alpha * (1 - emd) + beta * purity * coverage^lambda
        keep mask if score >= threshold
```

其中论文公式是 `score = alpha * (1 - emd) + beta * purity * coverage^lambda`。本地代码里的 `emd` 变量已经是 `1 - emd_cost` 后的 score 形式，所以实现时要避免重复取反。

## 5. 多目标很多时，sample 还可行吗？

结论：直接从所有点里全局随机 sample 不稳定；先聚类再按实例/簇采样是可行的。

老师担心是对的：Matcher 原文很多任务是一类目标的一个或少数实例，虽然也做 COCO/LVIS multiple instances，但当我们场景里目标非常多时，全局随机采样可能出现：

1. 一组 points 混到多个物体，SAM 输出大块合并 mask。
2. 某些小物体没有被采样到，coverage 低。
3. instance-level matching 只按 reference mask vs proposal 匹配，可能不知道 target 里有 K 个对象。

改法：

1. Stage 1 输出所有 matched points。
2. 用 k-means++、DBSCAN 或 connected component 将 points 分成 K 个簇。
3. 每个簇视为一个候选实例或候选实例组。
4. Stage 2 不再全局 sample，而是每个簇内 sample：
   - 小簇：全部点或采 1 到 3 点；
   - 大簇：采 `N%` 或固定 4 到 6 点；
   - 每个簇重复 m 次。
5. Stage 3 对每个簇内 proposals 单独评分，避免不同实例互相竞争。
6. 最后跨簇做 mask NMS，去掉重复实例。

因此回答老师“sample 可行吗”：可行，但前提是从 global sampling 改成 cluster-aware sampling。多目标越多，越不能只依赖全局随机采样。

## 6. Instance-level matching：如果分割出 K 个物体，代码怎样知道 K 个物体存在？

原始 Matcher 不是显式先预测 K，而是通过 matched points 的空间聚类和 proposal selection 间接得到多个实例。老师希望代码“明白有 K 个物体存在”，建议改成显式 K-aware pipeline：

1. Stage 1 得到 points。
2. 聚类得到 `clusters = {C_1, ..., C_K}`。
3. 每个 cluster 创建一个 `instance_candidate_id`。
4. 每个 cluster 独立生成 proposals。
5. 每个 proposal 绑定来源 cluster id。
6. 对每个 cluster 保留 top proposal 或超过阈值的 proposals。
7. 输出时记录：
   - `K_points_clusters`
   - `K_pred_instances_before_nms`
   - `K_pred_instances_after_nms`
   - 每个实例的 matched score、coverage、purity、EMD

建议数据结构：

```python
InstanceCandidate = {
    "cluster_id": int,
    "points": np.ndarray,        # points in this cluster
    "center": tuple[int, int],
    "proposals": list[Mask],
    "best_score": float,
    "best_mask": np.ndarray,
}
```

这样答辩时可以说：K 不是由 SAM 自动告诉我们的，而是由 Stage 1 matched points 的空间聚类估计出来，再由 Stage 3 的 proposal matching/NMS 验证。

## 7. 个人记忆：100 个 points 后用 k-means 告诉代码有 K 个物体

这个记忆与 Matcher 原文一致。Matcher 原文明确先把 matched points 按位置用 k-means++ 分成 K 个 clusters，再从 cluster 内、全局点和 cluster centers 采样 prompts。

本地也已经有 `code\matcher\k_means.py` 和 `Matcher.clustering()`。当前风险是：

1. `num_centers` 是手动超参数，不一定等于真实物体数。
2. k-means++ 适合大致球形簇；如果物体形状细长、密集或大小差异大，DBSCAN/connected components 可能更稳。
3. 如果某个真实物体上只有 1 到 2 个点，k-means 可能把它吞进邻近簇。

建议方案：

| 方案 | 用途 |
|---|---|
| k-means++ with sweep K | 试 K=2..20，用 silhouette、簇内距离、簇间距离选 K。 |
| DBSCAN | 物体数量未知且点云分布明显分离时更适合。 |
| connected components on patch grid | 点来自 DINO patch 网格时最自然，可先把 matched patches 二值化，再连通域分析。 |
| SAM proposal feedback | 用 Stage 2 的 masks 反推 K，若一个 cluster 产生多个互不重叠高分 mask，则拆分。 |

推荐优先级：connected components 或 DBSCAN 先估 K，再用 k-means++ 细化 center。

## 8. 改变 1：Stage 1 加 feedback 机制，手动选择正确点/错误点更新 target feature

老师建议可以落地，而且和本地当前代码最匹配。当前 `patch_level_matching()` 已经把 reference mask features 加权平均成一个 prototype：

```text
ref_mask_feat_avg = weighted_average(ref_mask_feats)
similarity_vector = ref_mask_feat_avg @ target_features
topk + threshold -> matched points
```

Feedback 机制可以把人工选的点转成 target feature，并更新 prototype：

```text
positive_features = target_features[manual_positive_points]
negative_features = target_features[manual_negative_points]

positive_proto = mean(positive_features)
negative_proto = mean(negative_features)

updated_proto = normalize(
    (1 - gamma) * ref_proto
    + gamma * positive_proto
    - eta * negative_proto
)

similarity_vector = updated_proto @ target_features
```

推荐参数：

| 参数 | 建议初值 |
|---|---|
| `gamma` 正反馈权重 | 0.3 |
| `eta` 负反馈权重 | 0.2 |
| 正确点数 | 1 到 2 个 |
| 错误点数 | 1 到 2 个 |
| top-k | 100 或 200 |
| threshold | 从 0.80/0.85 开始扫 |

更稳的打分方式是对 negative 单独惩罚：

```text
score(p) = cos(target_feat_p, updated_positive_proto)
         - eta * max_j cos(target_feat_p, negative_feat_j)
```

这样错误点附近的背景或干扰物会被显式压低。

### 8.1 Feedback 后如何判断 matching 是否更好？

用本地已有 `MatchingMetrics` 思路即可：

1. Precision：matched points 中有多少落在真值 mask 内。
2. Coverage：真值 mask 的 patch 有多少被 matched points 覆盖。
3. Foreground-background separation：前景 feature 与背景 feature 的相似度是否下降。
4. SAM proposal quality：Stage 2 后保留下来的 masks 的 IoU/Dice 是否提升。

实验记录表：

| 实验 | 正反馈点 | 负反馈点 | Precision | Coverage | 最终 Dice/IoU | 备注 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0 | 0 |  |  |  | 原始 Matcher |
| feedback A | 1 | 1 |  |  |  | 最小交互 |
| feedback B | 2 | 2 |  |  |  | 老师建议设置 |
| feedback C | 2 | 0 |  |  |  | 只加正点 |
| feedback D | 0 | 2 |  |  |  | 只加负点 |

## 9. 推荐最终 pipeline

```text
Input:
  reference image + reference mask
  target image, possibly high resolution

Stage 1: windowed Matcher point mining
  1. If target > DINO size and cannot downsample:
       tile target into 518 x 518 windows with overlap.
  2. Extract DINOv2 features per window.
  3. Compute prototype similarity or bidirectional matching.
  4. Collect top candidate points in original-image coordinates.
  5. Optional manual feedback:
       select a positive correct points and b negative wrong points.
       update prototype and re-match.
  6. Cluster matched points to estimate K instances.

Stage 2: cluster-aware prompt sampling
  1. For each cluster C_k:
       sample N% points, repeat m times.
  2. Send each sampled prompt group to SAM/SAM3.
  3. Collect mask proposals with source cluster id.

Stage 3: instance-level matching and filtering
  1. Compute EMD / purity / coverage / score for each proposal.
  2. Keep proposals above threshold.
  3. Within each cluster, keep top proposals.
  4. Across clusters, apply mask NMS.
  5. Output K' instance masks and matching report.
```

## 10. 需要做的代码改动清单

| 优先级 | 改动 | 文件建议 |
|---|---|---|
| P0 | 把 Stage 1 点输出固定保存为 CSV/JSON，包含原图坐标、window id、similarity、cluster id | `code\matcher\Matcher.py` |
| P0 | 加 sliding-window target feature extraction，不降采样 | 新建 `code\matcher\windowed_matching.py` 或并入 `Matcher.py` |
| P0 | 加 feedback prototype update | `code\matcher\Matcher.py::patch_level_matching` |
| P1 | 把 `sample_range` 扩展为 `sample_percent` + `m` | `RobustPromptSampler` |
| P1 | 从 global sampling 改成 cluster-aware sampling | `mask_generation()` |
| P1 | 显式输出 K 个 instance candidates | 新建 `code\matcher\instance_candidates.py` |
| P2 | 增加 DBSCAN/connected component 估计 K | 新建 `code\matcher\point_clustering.py` |
| P2 | 生成实验报告表格和可视化 | `code\benchmarks\...` |

## 11. 可以直接回复老师的问题

1. 如果图像大于 DINO 输入且不能降采样，用滑动窗口。每个窗口按 DINO 的 518 x 518 跑，点坐标映射回原图，重叠区域做去重。SAM/SAM3 后续用原图坐标或局部 crop 分割。
2. Matching 是 DINOv2 patch 特征匹配。Matcher 原版是 reference mask patch 到 target patch 的 forward matching，再 reverse matching 过滤 outlier，之后 k-means++ 聚类和 prompt sampling。
3. Stage 2 的 sample 是可行的，但多目标很多时不能全局随机采，要先把 points 聚成 K 个簇，再每个簇独立采样。
4. Instance-level matching 不能只输出一个整体 mask，应显式维护 K 个 instance candidates。K 可由 matched points 的聚类/连通域估计，再由 SAM proposals 和 threshold 验证。
5. 反馈机制可行。人工选 1 到 2 个正确点和 1 到 2 个错误点后，用 target feature 更新匹配 prototype：正点拉近，错点推远，再重新计算 similarity/top-k，比较 precision、coverage、最终 Dice/IoU 是否提升。

## 12. 最小实验计划

1. 选 10 张自然场景图：COCO/LVIS/SA-Co/RefCOCO 均可，优先多实例且有属性/动作描述。
2. 对每张图记录两类 prompt：
   - noun prompt：`person`、`girl`、`soccer player`、`dog`。
   - hard text prompt：`girl in white`、`person executing ...`、带空间关系或动作的指代表达。
3. 跑三组方法：
   - SAM3/SAM visual point prompt。
   - SAM3 text prompt。
   - Matcher + feedback + cluster-aware sampling。
4. 记录：
   - text 是否返回目标 mask；
   - point 是否能分出目标；
   - Stage 1 points precision/coverage；
   - Stage 3 mask Dice/IoU；
   - K 估计是否接近真实实例数。

这个实验能完整覆盖老师截图中的 a/b、Stage 1、Stage 2+3、多目标、K 个物体和 feedback 改动。
