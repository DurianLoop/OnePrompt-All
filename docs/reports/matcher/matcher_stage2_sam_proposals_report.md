# Matcher Stage 2 SAM Proposals 最小验证报告

日期：2026-06-24

## 1. 实验目的

本实验继续完成老师提到的 Stage 2 后半段：

> 从 matched points 中 sample N% point prompts，然后把每次 sample 出来的 points 送进 SAM，得到一次分割，重复 m 次得到 m 个分割。

上一阶段 P0-3 已经完成：

```text
feedback 后 matched points -> DBSCAN 聚类 -> 生成 12 组 1/3/5 点 prompt groups
```

本阶段把这 12 组 prompt groups 真实送入本地 SAM ViT-B，验证是否能得到 12 个 mask proposals。

说明：这里使用本地稳定可运行的 SAM ViT-B 作为视觉 prompt backend，而不是强行使用 SAM3 原生 Windows 路径。原因是本阶段要验证 Matcher Stage 2 的采样逻辑，SAM/SAM3 都属于后端分割器；SAM3 后续可以替换进同一接口。

## 2. 实验输入

输入目录：

```text
runs/research_direction/stage2_sampling/elephant_pair/
```

关键输入文件：

| 文件 | 说明 |
| --- | --- |
| `prompt_groups.json` | 12 组 sampled point prompts |
| `sample_overlay.png` | prompt groups 可视化 |
| `target_image` | `code/gradio_demo/images/elephant2.png` |

SAM checkpoint：

```text
assets/checkpoints/sam_vit_b.pth
```

## 3. 可复现实验命令

检查脚本：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage2_proposals.py
```

真实生成 SAM proposals：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\generate_stage2_sam_proposals.py --stage2-dir runs\research_direction\stage2_sampling\elephant_pair --checkpoint assets\checkpoints\sam_vit_b.pth --model-type vit_b --device auto --max-groups 12 --out-dir runs\research_direction\stage2_proposals\elephant_pair
```

本次实际运行设备：

```text
cuda:0
```

## 4. Proposal 生成方式

对每个 prompt group：

1. 读取 group 中的点坐标；
2. 所有点都作为 foreground point，`point_labels=1`；
3. 调用 `SamPredictor.predict(..., multimask_output=True)`；
4. 从 SAM 返回的多个 masks 中选择 score 最高的 mask；
5. 保存 mask、overlay、score、面积和 bbox。

核心调用形式：

```python
masks, scores, _ = predictor.predict(
    point_coords=np.asarray(points, dtype=np.float32),
    point_labels=np.ones(len(points), dtype=np.int32),
    multimask_output=True,
)
best = int(np.argmax(scores))
```

## 5. 实验结果

输出目录：

```text
runs/research_direction/stage2_proposals/elephant_pair/
```

总体结果：

| 指标 | 结果 |
| --- | ---: |
| proposal count | 12 |
| SAM score mean | 0.905409 |
| SAM score min | 0.709140 |
| SAM score max | 0.987186 |
| mask area mean | 33.3608% |
| mask area min | 1.0176% |
| mask area max | 70.9880% |

按 sample size 分组：

| sample size | proposal 数 | score mean | area mean | area min | area max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.831625 | 21.3485% | 1.0176% | 36.6146% |
| 3 | 4 | 0.907203 | 24.5443% | 12.1611% | 36.6364% |
| 5 | 4 | 0.977397 | 54.1897% | 36.1188% | 70.9880% |

## 6. 可视化产物

| 文件或目录 | 说明 |
| --- | --- |
| `proposal_masks/` | 每个 group 的二值 mask |
| `proposal_overlays/` | 每个 group 的 overlay，可人工检查 |
| `proposal_scores.csv` | 每个 proposal 的 score、面积、bbox |
| `proposal_scores.json` | 完整结果和参数 |
| `proposal_grid.png` | 12 个 proposals 的网格总览 |
| `summary.md` | 本次运行摘要 |

视觉观察：

- 12 组 prompts 都成功生成了 mask proposals；
- 单点 prompts 的结果不稳定，有的落到水面或只覆盖局部；
- 3 点 prompts 的面积更适中，部分 proposal 更接近 elephant 主体；
- 5 点 prompts 的 SAM score 最高，但明显容易过分扩张到背景，面积均值达到 54.1897%；
- 因此不能只用 SAM score 选 proposal，必须进入 Stage 3 做 instance-level matching / area / NMS 过滤。

## 7. 对老师问题的正面回答

1. “sample N% points 送进 SAM，重复 m 次得到 m 个分割，可行吗？”

   可行。本次从 feedback 后的 matched points 中采样出 12 组 point prompts，并成功生成 12 个 SAM mask proposals。

2. “多点 prompt 是否一定比单点好？”

   不一定。3 点 prompts 在本例中更稳，平均 score 从单点的 0.831625 提升到 0.907203，面积也更合理；但 5 点 prompts 虽然 score 最高，却经常把背景也包含进去。

3. “SAM 内部 score 能否直接作为最终选择标准？”

   不能。5 点 prompts 的 score 最高，但视觉上有明显过分扩张。因此 Stage 3 需要额外的 instance-level matching、面积约束、NMS 或 reference similarity。

4. “当前是否完成老师说的 Stage 2？”

   基本完成 Stage 2 最小闭环：已经从 matched points 采样出多组 prompts，并真实得到 m 个分割。下一步是 Stage 3 筛选和 K 估计。

## 8. 下一步

下一步建议做 P0-4：instance-level filtering。

最小版本：

1. 读取 `proposal_scores.json` 和 `proposal_masks/`；
2. 对 masks 做 NMS，去掉高度重复 proposal；
3. 加入面积过滤，去掉过小和过大的 mask；
4. 用 Stage 1/2 的 cluster 点覆盖率计算 mask 与 matched points 的一致性；
5. 输出 retained masks、`K_pred` 和 `filtered_overlay.png`。

更接近 Matcher 论文的版本：

1. 对每个 proposal 内部提取 DINO feature；
2. 与 reference/target prototype 做 instance-level similarity；
3. 用 purity / coverage / EMD 组合打分；
4. threshold 过滤后输出最终实例。
