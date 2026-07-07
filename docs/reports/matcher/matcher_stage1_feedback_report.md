# Matcher Stage 1 正负点反馈最小验证报告

日期：2026-06-24

## 1. 实验目的

本实验正面回应老师提出的猜想：

> 如果 Stage 1 给出一批 matched points，人工选择少量正确点和错误点后，更新或修正 target feature / matching 分数，matching 结果是否会更好？

当前先做最小闭环版本，不直接修改 DINO feature，也不接完整 Stage 2/3。做法是在 Matcher Stage 1 的相似度图上加入正负点反馈场：

- 正点附近加权，提高这些区域的候选点排名；
- 负点附近降权，压制明显错误区域；
- 重新排序 top-k matched points；
- 比较 feedback 前后的点分布、阈值内点数、到正负点的距离指标和可视化。

这个版本对应后续方法中的 V1 reranking。它不是最终论文方法，但可以快速判断“feedback 能不能改变 matching 分布，并朝目标实例集中”。

## 2. 实验输入

使用上一阶段导出的 elephant 自然场景匹配结果：

```text
runs/research_direction/stage1_matching/elephant_pair/
```

Stage 1 导出配置：

```text
reference image: code/gradio_demo/images/elephant1.png
target image:    code/gradio_demo/images/elephant2.png
reference box:   0,60,295,425
top-k:           100
threshold:       0.55
```

反馈点配置：

```text
positive points: 268,331
negative points: 30,430;455,420
sigma:           42
positive weight: 0.18
negative weight: 0.22
metric radius:   70 px
```

正点位于 Stage 1 最佳匹配点附近，即目标 elephant 身体区域；负点放在水面/背景等明显错误区域。

## 3. 可复现实验命令

先导出 Stage 1 matched points 和原始相似度图：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant2.png --reference-box 0,60,295,425 --top-k 100 --threshold 0.55 --out-dir runs\research_direction\stage1_matching\elephant_pair
```

再运行正负点 feedback reranking：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\apply_stage1_feedback.py --stage1-dir runs\research_direction\stage1_matching\elephant_pair --positive-points "268,331" --negative-points "30,430;455,420" --sigma 42 --positive-weight 0.18 --negative-weight 0.22 --top-k 100 --threshold 0.55 --metric-radius 70 --out-dir runs\research_direction\stage1_feedback\elephant_pair
```

检查脚本：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage1_exporter.py
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage1_feedback.py
```

## 4. 实验结果

### 4.1 Stage 1 原始匹配

Stage 1 原始结果：

```text
top-k: 100
threshold: 0.55
selected count: 9
best point: (268.108, 330.811)
best similarity: 0.605155
```

说明：不加 feedback 时，最高分点已经落在目标 elephant 附近，但 top-k 中仍有明显背景和水面 outliers。

### 4.2 加入 feedback 后

Feedback 后结果：

```text
before selected count: 9
after selected count: 43
before best similarity: 0.605155
after best similarity: 0.780131
```

top-k 级别指标：

| 指标 | Feedback 前 | Feedback 后 | 变化 |
| --- | ---: | ---: | --- |
| top-k 点数 | 100 | 100 | 固定 |
| 到正点平均距离 | 128.379 | 83.822 | 降低 |
| 到正点中位距离 | 113.660 | 75.855 | 降低 |
| 70px 正点邻域内点数 | 22 | 43 | 增加 |
| 70px 负点邻域内点数 | 16 | 0 | 降低 |

阈值内 selected 点指标：

| 指标 | Feedback 前 | Feedback 后 | 变化 |
| --- | ---: | ---: | --- |
| selected 点数 | 9 | 43 | 增加 |
| 到正点平均距离 | 58.322 | 54.515 | 略降 |
| 到正点中位距离 | 52.081 | 52.285 | 基本持平 |
| 70px 正点邻域内 selected 点数 | 5 | 31 | 增加 |
| 70px 负点邻域内 selected 点数 | 0 | 0 | 保持为 0 |

## 5. 可视化产物

输出目录：

```text
runs/research_direction/stage1_feedback/elephant_pair/
```

关键文件：

| 文件 | 说明 |
| --- | --- |
| `feedback_points.csv` | feedback 后 top-k 点表 |
| `feedback_points.json` | 完整参数、before/after 指标、点表 |
| `feedback_overlay_before_after.png` | feedback 前后点分布对比 |
| `feedback_heatmap_before_after.png` | feedback 前后相似度热力图对比 |
| `similarity_after.npy` | feedback 后相似度矩阵 |
| `summary.md` | 本次实验摘要 |

可视化观察：

- feedback 后，红色高分点明显向正点附近聚集；
- 负点附近的候选点被压制，top-k 中 70px 负点邻域候选从 16 降到 0；
- 仍存在部分低位或边界附近候选点，说明简单空间 reranking 只能证明现象，不能作为最终方法。

## 6. 对老师猜想的正面回答

结论：最小闭环结果支持老师的猜想。

具体回答如下：

1. “Stage 1 给 100 个 points，手动选择正确点和错误点后，matching 是否会更好？”

   初步会更好。反馈后 top-k 到正点的平均距离从 128.379 降到 83.822，正点邻域内候选从 22 增到 43，负点邻域候选从 16 降到 0。

2. “正负点能否告诉代码哪些地方是目标、哪些地方不是目标？”

   可以。当前 V1 通过正负点构造局部反馈场，已经能改变相似度排序。后续 V2 应进一步改成 feature/prototype update，而不是只做空间加权。

3. “这能不能支撑后续 Stage 2/3？”

   可以作为 Stage 2 的输入。feedback 后的候选点更集中，适合继续做 clustering 和 prompt sampling；同时 outlier 减少，有利于 Stage 3 instance-level filtering。

4. “当前结果是否已经足够作为最终方法？”

   还不够。当前只是最小验证：用空间距离近似用户反馈影响，没有真正更新 target/reference feature prototype，也没有接 SAM proposal 和 instance matching。它的价值是证明方向值得继续推进。

## 7. 下一步

优先继续 P0-3：

1. 对 feedback 后 top-k 点做 DBSCAN 或 K-means 聚类；
2. 估计候选实例数 K；
3. 每个 cluster 内采样 1/3/5 个 point prompts；
4. 把采样点送入 SAM3/SAM，生成多组 mask proposals；
5. 比较 no feedback、positive-only、positive+negative 三种设置的 proposal 质量。

然后进入 P0-4：

1. 对 proposals 做 mask NMS；
2. 提取 proposal-level DINO/SAM feature；
3. 计算 proposal 与 reference/target prototype 的 instance-level similarity；
4. 设 threshold 保留高匹配 mask，输出 K_pred。
