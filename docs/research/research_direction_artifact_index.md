# 研究方向实验产物索引

日期：2026-06-24

本文档用于快速定位当前 SAM3/Matcher 教师方向的脚本、报告和运行产物。

## 1. 先看哪些文档

| 文档 | 用途 |
| --- | --- |
| `docs/research/teacher_direction_closed_loop_summary.md` | 当前中文总报告，适合直接给老师看或继续改成汇报稿。 |
| `docs/reports/sam3/sam3_teacher_minimal_validation_report.md` | SAM3 text/point、`white girl`、`execute`、累计点和正负点的最小验证。 |
| `docs/research/research_direction_priority_plan.md` | 研究方向优先级和论文路线。 |
| `docs/reports/matcher/matcher_stage1_export_report.md` | Stage 1 matched points 导出。 |
| `docs/reports/matcher/matcher_stage1_feedback_report.md` | Stage 1 正负点 feedback。 |
| `docs/reports/matcher/matcher_stage2_sampling_report.md` | Stage 2 聚类和 prompt sampling。 |
| `docs/reports/matcher/matcher_stage2_sam_proposals_report.md` | Stage 2 prompts 送入 SAM 得到 proposals。 |
| `docs/reports/matcher/matcher_stage3_filtering_report.md` | Stage 3 mask-level filtering。 |
| `docs/reports/matcher/matcher_stage3_feature_matching_report.md` | Stage 3 DINO feature-level instance matching。 |
| `docs/reports/matcher/matcher_multi_instance_horse_validation_report.md` | horse-head 多实例 `K_pred=2` 验证。 |

## 2. 主线脚本入口

| 阶段 | 脚本 | 作用 |
| --- | --- | --- |
| SAM3 最小验证 | `code/benchmarks/sam3_teacher_minimal_validation.py` | 验证 text/point、`white girl`、`execute`、累计点和正负点。 |
| Demo 控制检查 | `code/benchmarks/check_demo_prompt_controls.py` | 检查 demo 已支持 cumulative/latest point 设置。 |
| Stage 1 | `code/benchmarks/export_matcher_stage1_points.py` | 导出 DINO/Matcher top-k matched points 和 similarity heatmap。 |
| Stage 1 feedback | `code/benchmarks/apply_stage1_feedback.py` | 用正负点对 similarity map 做 reranking。 |
| Stage 2 sampling | `code/benchmarks/sample_stage2_prompts.py` | 对 points 做 DBSCAN/K-means 风格聚类并生成 prompt groups。 |
| Stage 2 proposals | `code/benchmarks/generate_stage2_sam_proposals.py` | 把 prompt groups 送入 SAM，生成 mask proposals。 |
| Stage 3 mask filtering | `code/benchmarks/filter_stage3_proposals.py` | 用 score、area、bbox、coverage、NMS 做 mask-level filtering。 |
| Stage 3 feature matching | `code/benchmarks/feature_match_stage3_proposals.py` | 用 DINO proposal prototype 与 reference prototype 做 instance-level matching。 |

## 3. 静态检查脚本

| 脚本 | 检查内容 |
| --- | --- |
| `code/benchmarks/check_stage1_exporter.py` | Stage 1 exporter 关键结构。 |
| `code/benchmarks/check_stage1_feedback.py` | Stage 1 feedback 参数和输出结构。 |
| `code/benchmarks/check_stage2_sampler.py` | Stage 2 sampler 参数和输出结构。 |
| `code/benchmarks/check_stage2_proposals.py` | Stage 2 proposal generation 结构。 |
| `code/benchmarks/check_stage3_filtering.py` | Stage 3 mask filtering 结构。 |
| `code/benchmarks/check_stage3_feature_matching.py` | Stage 3 feature matching 结构，包含 `--max-retained-per-cluster`。 |

## 4. 关键运行产物

| 实验 | 输出目录 | 关键文件 |
| --- | --- | --- |
| SAM3 teacher minimal validation | `runs/teacher_minimal_validation/` | `results.csv`、`results.json`、`overlays/` |
| Elephant Stage 1 | `runs/research_direction/stage1_matching/elephant_pair/` | `points.csv`、`points.json`、`topk_overlay.png`、`similarity_heatmap.png` |
| Elephant Stage 1 feedback | `runs/research_direction/stage1_feedback/elephant_pair/` | `feedback_points.csv`、`feedback_points.json`、`feedback_overlay_before_after.png` |
| Elephant Stage 2 sampling | `runs/research_direction/stage2_sampling/elephant_pair/` | `cluster_points.csv`、`prompt_groups.csv`、`prompt_groups.json`、`cluster_overlay.png` |
| Elephant Stage 2 proposals | `runs/research_direction/stage2_proposals/elephant_pair/` | `proposal_scores.csv`、`proposal_scores.json`、`proposal_grid.png`、`proposal_masks/`、`proposal_overlays/` |
| Elephant Stage 3 mask filtering | `runs/research_direction/stage3_filtering/elephant_pair/` | `filtered_scores.csv`、`filtered_scores.json`、`retained_overlay.png` |
| Elephant Stage 3 feature matching | `runs/research_direction/stage3_feature_matching/elephant_pair/` | `feature_scores.csv`、`feature_scores.json`、`feature_retained_overlay.png` |
| Horse-head Stage 1 | `runs/research_direction/multi_instance_horse_head/stage1_matching/` | `points.csv`、`points.json`、`topk_overlay.png` |
| Horse-head Stage 2 sampling | `runs/research_direction/multi_instance_horse_head/stage2_sampling/` | `cluster_points.csv`、`prompt_groups.json`、`cluster_overlay.png` |
| Horse-head Stage 2 proposals | `runs/research_direction/multi_instance_horse_head/stage2_proposals/` | `proposal_scores.csv`、`proposal_grid.png`、`proposal_masks/` |
| Horse-head Stage 3 feature matching | `runs/research_direction/multi_instance_horse_head/stage3_feature_matching/` | `feature_scores.csv`、`feature_scores.json`、`feature_retained_overlay.png` |

## 5. 可复现实验命令

### 5.1 SAM3 text/point 最小验证

```powershell
cd D:\SAM
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sam3_teacher_minimal_validation.py --out-dir runs\teacher_minimal_validation
```

### 5.2 Elephant Stage 1 matched points

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant2.png --reference-box 0,60,295,425 --top-k 100 --threshold 0.55 --out-dir runs\research_direction\stage1_matching\elephant_pair
```

### 5.3 Elephant Stage 1 feedback

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\apply_stage1_feedback.py --stage1-dir runs\research_direction\stage1_matching\elephant_pair --positive-points "268,331" --negative-points "30,430;455,420" --sigma 42 --positive-weight 0.18 --negative-weight 0.22 --top-k 100 --threshold 0.55 --metric-radius 70 --out-dir runs\research_direction\stage1_feedback\elephant_pair
```

### 5.4 Elephant Stage 2 sampling

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sample_stage2_prompts.py --feedback-dir runs\research_direction\stage1_feedback\elephant_pair --selected-only --cluster-method dbscan --eps 55 --min-samples 3 --sample-sizes 1,3,5 --samples-per-cluster 4 --max-anchor-distance 95 --out-dir runs\research_direction\stage2_sampling\elephant_pair
```

### 5.5 Elephant Stage 2 proposals

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\generate_stage2_sam_proposals.py --stage2-dir runs\research_direction\stage2_sampling\elephant_pair --checkpoint assets\checkpoints\sam_vit_b.pth --model-type vit_b --device auto --max-groups 12 --out-dir runs\research_direction\stage2_proposals\elephant_pair
```

### 5.6 Elephant Stage 3 feature matching

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\feature_match_stage3_proposals.py --proposals-dir runs\research_direction\stage2_proposals\elephant_pair --stage1-dir runs\research_direction\stage1_matching\elephant_pair --dinov2-weights assets\checkpoints\dinov2_vitl14_pretrain.pth --min-instance-similarity 0.45 --min-score 0.80 --min-area-pct 5 --max-area-pct 45 --max-bbox-area-pct 45 --min-cluster-coverage 0.30 --nms-iou 0.75 --out-dir runs\research_direction\stage3_feature_matching\elephant_pair
```

### 5.7 Horse-head 多实例完整链路

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\horse2.png --target-image code\gradio_demo\images\horse3.png --reference-box 120,35,390,330 --top-k 160 --threshold 0.55 --out-dir runs\research_direction\multi_instance_horse_head\stage1_matching

D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\apply_stage1_feedback.py --stage1-dir runs\research_direction\multi_instance_horse_head\stage1_matching --positive-points " " --negative-points " " --sigma 42 --positive-weight 0 --negative-weight 0 --top-k 160 --threshold 0.55 --metric-radius 70 --out-dir runs\research_direction\multi_instance_horse_head\stage1_no_feedback

D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sample_stage2_prompts.py --feedback-dir runs\research_direction\multi_instance_horse_head\stage1_no_feedback --selected-only --cluster-method dbscan --eps 65 --min-samples 3 --sample-sizes 1,3 --samples-per-cluster 4 --max-anchor-distance 110 --out-dir runs\research_direction\multi_instance_horse_head\stage2_sampling

D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\generate_stage2_sam_proposals.py --stage2-dir runs\research_direction\multi_instance_horse_head\stage2_sampling --checkpoint assets\checkpoints\sam_vit_b.pth --model-type vit_b --device auto --max-groups 16 --out-dir runs\research_direction\multi_instance_horse_head\stage2_proposals

D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\feature_match_stage3_proposals.py --proposals-dir runs\research_direction\multi_instance_horse_head\stage2_proposals --stage1-dir runs\research_direction\multi_instance_horse_head\stage1_matching --dinov2-weights assets\checkpoints\dinov2_vitl14_pretrain.pth --min-instance-similarity 0.45 --min-score 0.80 --min-area-pct 0.5 --max-area-pct 45 --max-bbox-area-pct 50 --min-cluster-coverage 0.30 --nms-iou 0.75 --max-retained-per-cluster 1 --out-dir runs\research_direction\multi_instance_horse_head\stage3_feature_matching
```

## 6. 当前建议的阅读顺序

1. 先看 `docs/research/teacher_direction_closed_loop_summary.md`。
2. 想看具体数值，打开对应阶段报告。
3. 想复现实验，用本文档第 5 节命令。
4. 想看可视化，优先打开这些图片：

```text
runs/teacher_minimal_validation/overlays/
runs/research_direction/stage1_feedback/elephant_pair/feedback_overlay_before_after.png
runs/research_direction/stage2_sampling/elephant_pair/cluster_overlay.png
runs/research_direction/stage3_feature_matching/elephant_pair/feature_retained_overlay.png
runs/research_direction/multi_instance_horse_head/stage2_sampling/cluster_overlay.png
runs/research_direction/multi_instance_horse_head/stage3_feature_matching/feature_retained_overlay.png
```
