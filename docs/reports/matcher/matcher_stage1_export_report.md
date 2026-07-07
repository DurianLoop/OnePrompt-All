# Matcher Stage 1 Matched Points 导出报告

日期：2026-06-24

本报告对应研究方案中的 P0-1：先把 Matcher / DINO 的 Stage 1 matched points、similarity heatmap 和 top-k overlay 导出来。这个阶段不做 SAM mask proposal，也不做最终 Dice/IoU；目标是把 feature matching 的中间状态打开，供后续 feedback、sampling、clustering 和 instance-level filtering 使用。

## 1. 代码和输出位置

| 内容 | 路径 |
| --- | --- |
| 导出脚本 | `code/benchmarks/export_matcher_stage1_points.py` |
| 静态检查 | `code/benchmarks/check_stage1_exporter.py` |
| 实验输出目录 | `runs/research_direction/stage1_matching/elephant_pair/` |
| points CSV | `runs/research_direction/stage1_matching/elephant_pair/points.csv` |
| points JSON | `runs/research_direction/stage1_matching/elephant_pair/points.json` |
| top-k overlay | `runs/research_direction/stage1_matching/elephant_pair/topk_overlay.png` |
| similarity heatmap | `runs/research_direction/stage1_matching/elephant_pair/similarity_heatmap.png` |

## 2. 复现命令

```powershell
cd D:\SAM
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant2.png --reference-box 0,60,295,425 --top-k 100 --threshold 0.55 --out-dir runs\research_direction\stage1_matching\elephant_pair
```

验证命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage1_exporter.py
D:\SAM\conda_envs\sam_gpu\python.exe -B -c "from pathlib import Path; p=Path('code/benchmarks/export_matcher_stage1_points.py'); compile(p.read_text(encoding='utf-8'), str(p), 'exec'); print('stage1 exporter compiles')"
```

验证结果：

```text
Stage 1 exporter static checks passed.
stage1 exporter compiles
```

真实运行结果：

```text
Exported 100 top points; selected_by_threshold=9
Wrote D:\SAM\runs\research_direction\stage1_matching\elephant_pair
```

运行时出现 `xFormers not available`，这是当前 DINOv2 本地环境的提示，不影响本次普通推理路径和文件输出。

## 3. 实验设置

| 项目 | 设置 |
| --- | --- |
| reference image | `code/gradio_demo/images/elephant1.png` |
| target image | `code/gradio_demo/images/elephant2.png` |
| reference prompt | box `0,60,295,425`，大致覆盖 reference 图左侧大象 |
| DINO input size | `518 x 518` |
| DINO patch size | `14` |
| feature map size | `37 x 37` |
| top-k | `100` |
| threshold | `0.55` |
| selected points | `9` |

脚本输出两套坐标：

| 字段 | 含义 |
| --- | --- |
| `x_518`, `y_518` | DINO 输入尺度上的点坐标。 |
| `x_original`, `y_original` | 映射回 target 原图尺度的坐标。 |

## 4. 初步观察

### 4.1 Stage 1 已能产生可解释 matched points

`points.csv` 中最高相似度点如下：

| rank | x_original | y_original | similarity |
| ---: | ---: | ---: | ---: |
| 1 | 268.108 | 330.811 | 0.605155 |
| 2 | 320.000 | 278.919 | 0.591564 |
| 3 | 302.703 | 304.865 | 0.567244 |
| 4 | 268.108 | 317.838 | 0.566635 |
| 5 | 302.703 | 343.784 | 0.565555 |

这些点主要落在 target 图中的大象主体、水面反射和相似纹理区域。说明 DINO prototype matching 已经找到与 reference 大象相似的视觉区域，但也包含明显 outlier。

### 4.2 outlier 问题非常明显，正好支撑 P0-2 feedback

从 `topk_overlay.png` 和 `similarity_heatmap.png` 看：

1. 大象身体附近有高响应，说明 Stage 1 不是完全失效。
2. 水面、岩石、阴影等区域也有高响应，说明仅靠 feature similarity 会引入 outlier。
3. threshold 过高时会没有 selected points；threshold 设为 0.55 时能选出 9 个高相似点，但仍不能保证全部是目标区域。

这正好对应老师的担心：

```text
Stage 1 给很多 points，但里面有正确点也有错误点；需要用户选 a 个正确点、b 个错误点，让 matching 变好。
```

## 5. 对研究路线的意义

P0-1 已经打通：

```text
reference image/box -> DINO feature -> prototype similarity -> top-k matched points -> overlay/heatmap/CSV/JSON
```

因此下一步 P0-2 可以直接在 `points.csv` / `points.json` 上做：

1. 手动或脚本指定 positive points 和 negative points；
2. 对 similarity map 进行重排或 feature prototype update；
3. 输出 feedback 前后的 top-k overlay；
4. 比较 top-k 是否远离水面/岩石等错误区域，并更集中在目标大象上。

## 6. 下一步

建议立刻推进 P0-2：

```text
Feedback-guided Stage 1 reranking
```

最小实验：

1. 选一个正点：target 大象身体上的高相似点。
2. 选一个负点：水面或岩石上的高相似点。
3. 在 similarity map 上对正点邻域加权、负点邻域降权。
4. 导出 before/after overlay 和 heatmap。

成功标准：

```text
feedback 后 top-k points 更集中在目标大象区域，水面/岩石 outlier 减少。
```
