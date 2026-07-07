# SAM3 老师猜想最小验证闭环报告

日期：2026-06-24

本文档是给老师方向做的最小验证闭环，不是完整 benchmark。目标不是一次性证明方法有效，而是先正面回答老师提出的几个关键判断：SAM3 point 是否比 text 更适合当前问题、`white girl` / `execute` 这类文本提示是否失败、累计点和正负点 feedback 是否能改善 mask，以及老师提到的 Matcher Stage 1/2/3 后续应该如何验证。

## 0. 文档和结果位置

| 内容 | 路径 |
| --- | --- |
| 本报告 | `docs/reports/sam3/sam3_teacher_minimal_validation_report.md` |
| 最小验证脚本 | `code/benchmarks/sam3_teacher_minimal_validation.py` |
| 数值结果 CSV | `runs/teacher_minimal_validation/results.csv` |
| 完整结果 JSON | `runs/teacher_minimal_validation/results.json` |
| 可视化 overlay | `runs/teacher_minimal_validation/overlays/` |

复现命令：

```powershell
cd D:\SAM
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sam3_teacher_minimal_validation.py --out-dir runs\teacher_minimal_validation
```

使用的 SAM3 权重：

```text
assets/checkpoints/sam3_modelscope/sam3.pt
```

使用的图像：

| 图像 | 用途 |
| --- | --- |
| `code/gradio_demo/images/elephant1.png` | 两只大象的自然场景，用来比较 text、单点、累计点、正负点。 |
| `sam3/outputs/sam3_test_image_person.png` | 多人自然场景，用来测试 `girl`、`white girl`、`execute` 以及人物 point prompt。 |

## 1. 对老师内容的逐条正面回应

### 1.1 老师说：SAM3 基于 point 效果较好，点到物体 foreground/background 后应该能分割出来

本轮验证结论：部分支持，但需要更精确表述。

在大象图上，point prompt 的确能通过累计正点和背景负点把左侧目标大象分出来：

| 设置 | score | union area | best box | 现象 |
| --- | ---: | ---: | --- | --- |
| 单个 foreground 点 `(120,220,+)` | 0.652 | 11.57% | `[75,195,198,421]` | 只得到局部/不完整 mask。 |
| 累计两个 foreground 点 `(120,220,+);(160,260,+)` | 0.676 | 54.43% | `[0,55,297,422]` | mask 扩展到左侧大象主体。 |
| 两正点 + 一个 background 点 `(395,220,-)` | 0.738 | 54.41% | `[0,55,297,422]` | score 继续提高，并保持左侧大象目标。 |

所以可以正面回应老师：在自然物体场景中，SAM3 point prompt 确实能通过 foreground/background 交互得到更可控的实例级 mask。尤其当文本 `elephant` 会同时返回两只大象时，point 更适合指定“我要哪一只”。

但也要谨慎：这不能写成“point 一定全面优于 text”。因为 `elephant` 这个简单名词 text prompt 本身效果很好，返回了 2 个实例，score 0.953，union area 54.36%。更准确的结论是：

```text
SAM3 text 对简单名词有效；SAM3 point 对指定单个实例和交互式修正更可控。
```

### 1.2 老师说：SAM3 文本提示失效了，例如 `white girl`、`execute`

本轮验证结论：支持。

在多人自然图上：

| prompt | instances | score | 结论 |
| --- | ---: | ---: | --- |
| `girl` | 3 | 0.906 | 简单名词可以成功返回多个 girl 实例。 |
| `white girl` | 0 | - | 复合属性 + 名词失败。 |
| `execute` | 0 | - | 动作/动词类 prompt 失败。 |
| `person` | 0 | - | 泛化名词在该图也失败，说明 text 行为不稳定。 |

关键点是：同一张图里 `girl` 可以成功，而 `white girl` 返回 0，所以不能简单解释成“图中没有 girl”。这正面支持老师的判断：SAM3 的 text prompt 对复杂属性描述、动作词、非标准类别词不稳定，`white girl` 和 `execute` 是可复现的失败例子。

在大象图上：

| prompt | instances | 结论 |
| --- | ---: | --- |
| `elephant` | 2 | 简单物体名词成功。 |
| `execute` | 0 | 动作词失败。 |

因此当前可以给老师的回应是：

```text
我们复现到了 text prompt 的失败：`execute` 在两张自然图上均返回 0；`white girl` 在人物图上返回 0，而 `girl` 能返回 3 个实例。这说明问题不是没有目标，而是 SAM3 对复合属性和动作类文本提示不稳定。
```

### 1.3 老师说：网页 demo 里 SAM3 point 只按最新点分割，不会累计前几个 prompt，希望设置里能调整

本轮实现和验证结论：已修。

我在 `code/demos/medical_sam_click_app.py` 加了两个设置：

| 设置 | 含义 |
| --- | --- |
| `Point Run Mode` | `auto` 或 `manual`。SAM3 默认 manual，避免每点一次就立刻覆盖结果。 |
| `Point Scope` | `cumulative` 或 `latest`。可以选择累计所有点，或只用最新一个点。 |

并加了静态回归检查：

```text
code/benchmarks/check_demo_prompt_controls.py
```

验证命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_demo_prompt_controls.py
```

验证结果：

```text
Demo prompt control checks passed.
```

所以这个交互问题已经正面解决：现在 demo 可以用累计点，也可以切回只用最新点。

### 1.4 老师说：加入 feedback 机制，手动选择正确点和错误点，更新 target feature，matching 是否会更好

本轮验证结论：SAM3 mask 层面的正负点 feedback 有改善；Matcher feature 层面的 feedback 还没有完成。

已经完成的部分：

| 图像 | 对比 | 现象 |
| --- | --- | --- |
| 大象图 | 单点 → 累计两正点 → 两正点一负点 | score 从 0.652 到 0.676，再到 0.738；mask 从局部扩展为更完整的左侧大象。 |
| 人物图 | 单点 → 正点 + 负点 | score 从 0.684 到 0.766；mask area 从 2.01% 降到 0.47%，说明负点抑制了邻近人物，但最终 mask 仍偏局部。 |

这说明 foreground/background feedback 对 SAM3 interactive mask 是有用的。

还没完成的部分：老师说的“更新 target feature，让 matching 变好”指的是 Matcher / DINO feature matching 层面的反馈，而不是单纯 SAM3 interactive prompt。本轮最小闭环还没有改 Matcher 的 feature 更新公式，所以不能声称已经验证了 Matcher feedback。

当前正面回应应该写成：

```text
我们已经验证了正负点 feedback 对 SAM3 mask 生成有改善；但老师提到的基于 correct / wrong point 更新 target feature 的 Matcher feedback，还需要在 Stage 1 matching 里单独实现和验证。
```

### 1.5 老师问：如果图像大于 DINO 规定大小，又不允许降采样怎么办

本轮没有实验验证这一点，但已经有明确回应方案。

如果不能降采样，应该用滑动窗口/tiling，而不是 resize 整图。基本闭环是：

1. 按 DINO/SAM 支持的输入尺寸切 tile。
2. 每个 tile 内独立提取 DINO feature 和候选 points。
3. 将 tile 内坐标映射回原图坐标。
4. 对重叠区域的 points / masks 做 NMS 或按 score 合并。
5. 最终在原图尺度上输出 mask。

这部分下一步可以做一个最小脚本，不需要一开始做完整 benchmark：

```text
large image -> tiles -> per-tile matching points -> coordinate restore -> SAM3/SAM mask -> merge
```

### 1.6 老师问：matching 的过程究竟怎么完成

当前代码和文献方向可以这样回应：

Matcher 的 Stage 1 本质上是 feature-level retrieval：

1. 用 DINOv2 / backbone 对 support image 和 target image 提取 patch feature。
2. 在 support mask 或 support box 内取目标区域 feature，形成 reference / prototype。
3. 在 target feature map 上计算 similarity。
4. 取 top-N 或 top N% 高相似位置作为候选 point prompts。
5. 把这些 points 送进 SAM，得到 proposal masks。

老师问的核心不是“能不能匹配”，而是：

```text
top-N points 可能来自多个物体，也可能有 outlier，怎么知道有 K 个实例，以及怎么把它们送进 SAM 后筛掉错误 mask。
```

这就对应 Stage 2 和 Stage 3。

### 1.7 老师说：Stage 2 里 sample N% 匹配 points，送进 SAM，重复 m 次得到 m 个分割

本轮没有完整实现 Stage 2，但本轮 SAM3 point 结果支持这个方向的必要性：

- 单点容易得到局部 mask。
- 多点会改变 mask 范围。
- 正负点会改变 score 和 specificity。

因此 Stage 2 的合理实验闭环是：

1. Stage 1 输出 top N% matched points。
2. 按 sampling 策略重复采样 m 次。
3. 每次采样得到一组 point prompts。
4. 每组 points 送入 SAM/SAM3，得到一个 mask proposal。
5. 保存每次 proposal 的 score、area、box、overlay。

本轮结论对 Stage 2 的回应是：

```text
sampling points 是可行方向，但不能只靠单点；至少需要累计点或正负点策略，否则容易得到局部 mask。
```

### 1.8 老师问：我们有很多物体，但 Matcher 文章里物体只有一个或较少，sample 是否可行

本轮没有在多实例大规模图上跑 Matcher sampling，但从大象图可以看出问题存在：

- `elephant` text 一次返回两个实例。
- 指定左侧大象需要 point 来消歧。
- 如果 Stage 1 的 top points 同时落在多个目标上，直接随机 sample 可能把不同实例混在一起。

所以对老师的正面回应是：

```text
在多物体场景中，直接 sample top-N points 风险较大。需要先对 matched points 做聚类或 instance-aware grouping，再按 cluster 采样送入 SAM。
```

最小实现可以用：

```text
top matched points -> K-means / DBSCAN -> 每个 cluster 单独 sample -> 每个 cluster 生成 proposal -> instance-level matching 过滤
```

### 1.9 老师说：如果分割出 K 个物体，代码应该知道有 K 个物体，然后一一和 target match

本轮最小 SAM3 实验已经输出了 `instances`、`boxes` 和 `scores`，说明在 SAM3 text 输出层面可以知道返回了几个 instances。例如：

| prompt | instances |
| --- | ---: |
| `elephant` | 2 |
| `girl` | 3 |
| `white girl` | 0 |

但对于 Stage 2 sampling 得到的 m 个 masks，K 的估计还需要额外逻辑：

1. 对 masks 做 mask NMS，去掉重复 proposal。
2. 对保留 masks 提取 instance feature。
3. 和 target/reference feature 做 instance-level similarity。
4. 高于 threshold 的保留，低于 threshold 的丢弃。
5. 保留下来的数量就是估计 K。

所以当前回应是：

```text
SAM3 text 的输出天然有 instance 数；但 Matcher + sampling 的 K 需要通过 clustering/NMS + instance-level threshold 才能稳定估计。
```

### 1.10 老师提到：可能用 K-means 基于距离 clustering，告诉代码有 K 个物体

这个方向合理，但需要注意 K-means 要先知道 K。更稳的最小验证建议是：

| 方法 | 优点 | 风险 |
| --- | --- | --- |
| K-means | 简单，便于控制 K | 需要预设 K，不适合未知物体数量。 |
| DBSCAN | 不需要预设 K，可以按空间密度成簇 | 对 eps/min_samples 敏感。 |
| mask NMS 后反推 K | 更贴近最终实例输出 | 依赖 SAM proposal 质量。 |

建议下一步先做 DBSCAN 或“多 K 尝试 + mask NMS”，不要一开始固定 K。

## 2. 本轮最小实验结果总表

| case | prompt / points | instances | best score | union area | 解释 |
| --- | --- | ---: | ---: | ---: | --- |
| `elephant_text_elephant` | `elephant` | 2 | 0.953 | 54.36% | 简单名词 text 有效，找到两只大象。 |
| `elephant_text_execute` | `execute` | 0 | - | 0.00% | 动作词 text 失败。 |
| `elephant_point_single` | `(120,220,+)` | 3 | 0.652 | 11.57% | 单点偏局部。 |
| `elephant_point_cumulative_fg` | `(120,220,+);(160,260,+)` | 2 | 0.676 | 54.43% | 累计正点改善 mask 完整性。 |
| `elephant_point_posneg` | `(120,220,+);(160,260,+);(395,220,-)` | 2 | 0.738 | 54.41% | 正负点进一步提高 score。 |
| `person_text_person` | `person` | 0 | - | 0.00% | 泛化人物 text 失败，说明 text 不稳定。 |
| `person_text_girl` | `girl` | 3 | 0.906 | 5.76% | 简单名词 girl 成功。 |
| `person_text_white_girl` | `white girl` | 0 | - | 0.00% | 复合属性 text 失败。 |
| `person_text_execute` | `execute` | 0 | - | 0.00% | 动作词 text 失败。 |
| `person_point_girl_single` | `(540,470,+)` | 4 | 0.684 | 2.01% | 拥挤人物场景 point 不稳定。 |
| `person_point_girl_posneg` | `(540,470,+);(630,470,-);(430,460,-)` | 2 | 0.766 | 0.47% | 负点抑制邻近人物，但 mask 仍偏局部。 |

## 3. 当前闭环结论

已经被本轮实验支持：

1. `white girl` 和 `execute` 这类文本提示确实会失败。
2. 简单 noun prompt 不一定失败，例如 `elephant` 和 `girl` 可以成功。
3. point prompt 对指定实例更可控，尤其适合在多实例场景中选择某一个目标。
4. 累计点和正负点 feedback 能改善 SAM3 interactive mask，至少在大象图上表现明确。
5. 网页 demo 之前只按最新点分割的问题已经修复，现在可以选择 cumulative 或 latest。

还不能声称已经完成：

1. 还没有验证大图不降采样时的 sliding-window matching。
2. 还没有实现 Matcher Stage 1 的 correct/wrong point feature feedback。
3. 还没有实现 Stage 2 的 top N% matched points 重复采样 m 次。
4. 还没有实现 Stage 3 的 instance-level matching threshold。
5. 还没有解决多物体很多时 K 如何自动估计的问题。

## 4. 给老师的简短回应版本

可以这样汇报：

```text
老师，我们先做了一个最小闭环验证。结果显示：

1. SAM3 的 text prompt 对简单名词还可以，例如 elephant 返回 2 个实例，girl 返回 3 个实例；但 white girl 和 execute 都返回 0，说明您说的文本提示失败是可复现的，尤其是复合属性和动作词。

2. SAM3 point prompt 在指定单个实例上更可控。以两只大象图为例，text elephant 会同时分出两只，而 point 加 foreground/background 可以指定左侧大象。

3. 累计点和正负点 feedback 有改善：单点 score 是 0.652，两个正点变成 0.676，再加一个负点变成 0.738，mask 也从局部变成更完整的左侧大象。

4. 网页 demo 之前只使用最新点的问题已经改了，现在可以在设置里选择 cumulative points 或 latest point。

5. 但是 Matcher 层面的 feedback、top N% points sampling、instance-level threshold、K-means/DBSCAN 多实例聚类还没有做完。下一步应该先做 Stage 1 输出 matched points，然后加 correct/wrong point feedback，看 matching score 和点分布是否改善。
```

## 5. 下一步最小实验计划

不建议马上扩成大 benchmark。下一步做三个小闭环即可：

### 5.1 Matcher Stage 1 feedback 小闭环

目标：验证老师说的“选 a 个正确点、b 个错误点后更新 target feature，matching 是否更好”。

最小实现：

1. 让 `code/matcher/` 输出 top matched points 和 similarity。
2. 手动标记正点和负点。
3. 用正点拉近 target prototype，用负点惩罚错误区域。
4. 比较 feedback 前后的 top points precision、空间分布、similarity heatmap。

输出：

```text
runs/teacher_feedback_stage1/
```

### 5.2 Stage 2 point sampling 小闭环

目标：验证 top N% points sample 多次送入 SAM 是否可行。

最小实现：

1. 从 Stage 1 top matched points 中取 top 5%、10%、20%。
2. 每个比例 sample m 次。
3. 每次送入 SAM3/SAM 得到 mask。
4. 记录 mask area、score、box、重复率。

重点观察：单点是否偏局部，多点是否更完整，随机采样是否稳定。

### 5.3 多实例 K 估计小闭环

目标：回应老师说的“代码要知道 K 个物体存在”。

最小实现：

1. 对 matched points 做 DBSCAN 或多 K 的 K-means。
2. 每个 cluster 单独 sample points。
3. 每个 cluster 生成 mask proposal。
4. mask NMS + instance-level threshold 后得到估计 K。

输出指标：

| 指标 | 用途 |
| --- | --- |
| cluster 数 | 初步估计 K。 |
| proposal 数 | SAM 生成了多少候选实例。 |
| NMS 后实例数 | 去重后的 K。 |
| instance similarity | 判断是否和 target match。 |
| overlay | 给老师直接看效果。 |

## 6. 当前结论一句话

本轮最小验证已经支持老师关于 SAM3 text 失败、point 交互更可控、累计点/正负点 feedback 有改善的判断；但 Matcher feedback、sampling、instance-level threshold 和 K 估计还需要作为下一步实验单独闭环验证。
