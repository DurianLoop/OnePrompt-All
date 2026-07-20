# OCSAM 论文实验执行 Checklist

日期：2026-07-14

对应详细设计：`docs/track/monuseg_sam123_medsam_paper_experiment_design.md`

当前总阶段：

```text
数据与权重准备基本完成
-> 已完成 DINOv2-B/L 与 SAM ViT-B/L/H 的 official MoNuSeg 横向筛选
-> SAM2.1 Large 的 DINOv2-L 完整交叉筛选正在运行
-> 下一步生成跨模型证据表，再决定是否进入 learning module
```

## 0. 数据与 Protocol

| 状态 | 任务 | 当前结果 |
| --- | --- | --- |
| [x] | 下载 official MoNuSeg training package | 已下载到 `code/datasets/MonuSeg/official/downloads/` |
| [x] | 下载 official MoNuSeg testing package | 已下载到 `code/datasets/MonuSeg/official/downloads/` |
| [x] | 下载 organ information / XML-to-mask 参考代码 | 已下载 |
| [x] | 解压官方 tif+xml | `code/datasets/MonuSeg/official/raw/` |
| [x] | XML annotation 转 binary mask | train/test masks 已生成 |
| [x] | 生成 official manifest | `runs/research_direction/dataset_audit/monuseg_official_manifest.csv` |
| [x] | 验证 mask 非空 | 51 masks，empty=0 |
| [x] | 生成 first screening manifest | 1 train support -> 14 test queries |
| [ ] | 确认严格 30/14 official subset | 当前先用 37/14，论文前必须确认 |

## 1. 模型与权重确认

| 状态 | 任务 | 当前结果 |
| --- | --- | --- |
| [x] | SAM ViT-B 权重 | 已接入，并跑完 official 14-query screening，当前 SAM1 最好 |
| [x] | SAM ViT-L 权重实验 | DINOv2-B 和 DINOv2-L 均已跑；DINOv2-L 完整网格 best Dice=0.3634，预测面积约95.6%，过分割 |
| [x] | SAM ViT-H 权重实验 | DINOv2-B 和 DINOv2-L 均已跑；DINOv2-L 完整网格 best Dice=0.3620，预测面积约96.0%，过分割 |
| [~] | SAM2.1 Large 实验 | DINOv2-B 14-query 已完成；DINOv2-L 完整 K/lambda grid 正在运行，尚未生成正式 metadata |
| [x] | SAM-Med2D 实验 | 已接入并复用 DINOv2-L matcher points 跑完 official 14-query；mean Dice 约 0.361，过分割明显 |
| [x] | MedSAM 实验 | 已按 box baseline 跑完 official 14-query |
| [ ] | SAM3 实验 | 暂不列 P0，先不作为主线依赖 |
| [x] | DINOv2-B backbone | 已用于 official 14-query screening |
| [x] | DINOv2-L backbone | 已跑 official 14-query，对 SAM ViT-B 有明显提升 |
| [~] | BioMedCLIP backbone | 权重和 config 已有；当前环境缺 `open_clip` / `timm` / `transformers`，待补依赖后实现 patch feature extraction |

## 2. 已有实验结果

| 状态 | 实验 | 当前结果 |
| --- | --- | --- |
| [x] | 历史 SAM ViT-B dense point 20-case | 已有结果，可作为历史探索 |
| [x] | 历史 SAM3 dense point 重测 | 已有结果，只作为 model response 旁证 |
| [x] | official DINOv2-B + SAM ViT-B 14-query screening | 154 candidates，已生成 summary/report |
| [x] | official DINOv2-B + SAM ViT-L 14-query screening | 154 candidates，mean Dice 约 0.362 |
| [x] | official DINOv2-B + SAM ViT-H 14-query screening | 154 candidates，mean Dice 约 0.362 |
| [x] | official DINOv2-B + SAM2.1 Large 14-query screening | 154 candidates，mean Dice 约 0.358 |
| [x] | official DINOv2-L + SAM ViT-B 14-query screening | 粗筛 154 candidates，mean Dice 约 0.507 |
| [x] | official DINOv2-L + SAM ViT-B K/lambda 交叉筛 | 840 candidates，best K=40、lambda=0.75，mean Dice 约 0.518，当前最好 |
| [x] | official DINOv2-L + SAM ViT-L K/lambda 交叉筛 | 840 candidates，best K=20、lambda=0.25，best layout=boundary_mix，mean Dice=0.3634 |
| [x] | official DINOv2-L + SAM ViT-H K/lambda 交叉筛 | 840 candidates，best K=20、lambda=0.5，best layout=best_topk，mean Dice=0.3620 |
| [~] | official DINOv2-L + SAM2.1 Large K/lambda 交叉筛 | 同一 840-candidate 协议正在运行；当前只记录运行状态，不提前写指标 |
| [x] | prompt baseline + MedSAM box baseline | 已跑 14-query，普通 box/point baseline 约 0.13-0.36 |
| [x] | K 初筛 | 粗筛 K=50 较好；交叉筛后 best K=40 |
| [x] | background suppression 初筛 | lambda=0.5 优于 lambda=0；交叉筛后 best lambda=0.75 |
| [x] | layout 初筛 | local_maxima 当前最好 |
| [x] | clustering 初筛 | DBSCAN 暂无明显收益 |

## 3. 主实验 Baseline

| 状态 | Baseline | 下一步 |
| --- | --- | --- |
| [x] | center point | 已跑 |
| [x] | grid point | 已跑 |
| [x] | random point | 已跑 3 seeds |
| [x] | full-image / coarse box | 已跑 |
| [x] | GT box oracle | 已跑，只作为 upper bound |
| [x] | fg-only matcher | 已在 official screening 中有 lambda=0 结果 |
| [x] | fg-bg matcher | 已在 official screening 中有 lambda=0.5 结果 |
| [x] | fg-bg local_maxima | 已在 official screening 中有结果 |
| [ ] | learned policy | 暂缓，等 baseline 和 model zoo screening 完成 |
| [ ] | nnU-Net fully supervised upper bound | 待决定是否纳入主表 |

## 4. 辅助证明实验

| 状态 | 实验 | 当前进度 |
| --- | --- | --- |
| [x] | point precision@K | 已在代表 case gallery 中统计 P@10/20/40/100 |
| [x] | component recall@K | 已在代表 case gallery 中统计 component recall@10/20/40/100 |
| [ ] | score-map AP / AUC | 待实现 downsampled GT 评估 |
| [x] | Dense K 实验 | 已测 K=1/2/3/5/8/10/15/20/30/40/50/75/100；核心交叉筛覆盖 K=20/30/40/50/75/100 |
| [x] | background lambda 实验 | 已测 lambda=0/0.1/0.25/0.5/0.75/1/1.25/1.5/2；核心交叉筛覆盖 0 到 2 |
| [~] | model response analysis | 已完成 DINOv2-L -> SAM ViT-B/L/H：仅 ViT-B 获得强提升，ViT-L/H 均维持约96%预测面积；SAM2.1 Large 完整 grid 运行中 |

## 5. 消融实验

| 状态 | 消融项 | 下一步 |
| --- | --- | --- |
| [x] | w/o background prototype | 已有 lambda=0 对比；best fg-bg 明显高于 fg-only |
| [x] | fixed K vs selected K | 已完成 K grid 和 K/lambda 交叉筛；K=40、lambda=0.75 当前最好 |
| [ ] | fixed layout vs local_maxima / learned layout | 待 layout 表扩展 |
| [ ] | no reference / random reference | 待补 |
| [ ] | no selector vs learned selector | 等 learning module 开始后再做 |

## 6. Learning Module

| 状态 | 模块 | 当前处理 |
| --- | --- | --- |
| [ ] | proposal selector | 暂缓；先补 baseline 和 model zoo |
| [ ] | K/lambda/layout policy | 暂缓；需要训练集 oracle labels |
| [ ] | score re-weighting | 暂缓；需要 feature-map supervision |
| [ ] | train/val/test split | 待严格定义，不能只靠 5-fold |

## 7. 跨模型公平对比

| 状态 | 表格 | 下一步 |
| --- | --- | --- |
| [~] | 同一 prompt policy 下比较 SAM/SAM2/MedSAM | SAM1 B/L/H 已完成同一 DINOv2-L + 840-candidate protocol；SAM2.1 Large grid 运行中；MedSAM 保留 box baseline |
| [ ] | 各模型最佳合理 prompt 比较 | 待在跨模型 evidence table 后补齐，MedSAM 单列 box protocol |
| [~] | candidates/case 统计 | point-model cross-grid 统一为 840 candidates / 14 cases；待 SAM2.1 完成后写入总表 |
| [ ] | runtime / memory 统计 | 待记录 |

## 8. 报告与展示

| 状态 | 产物 | 当前结果 |
| --- | --- | --- |
| [x] | official data manifest | 已生成 |
| [x] | first screening report | `runs/research_direction/monuseg_official_screening/dinov2b_samvitb_test14/report/` |
| [x] | SAM1 weight screening summary | `runs/research_direction/monuseg_official_screening/sam1_weight_screening_summary.csv` |
| [x] | model weight screening summary | `runs/research_direction/monuseg_official_screening/model_weight_screening_summary.csv` |
| [x] | screening leaderboard | `runs/research_direction/monuseg_official_screening/screening_leaderboard.csv` |
| [x] | DINOv2-L + SAM ViT-B K/lambda report | `runs/research_direction/monuseg_official_screening/dinov2l_samvitb_kxlambda_test14/report/index.html` |
| [x] | DINOv2-L + SAM ViT-B 定性 gallery | `runs/research_direction/monuseg_official_screening/dinov2l_samvitb_kxlambda_test14/qualitative_gallery/index.html` |
| [x] | SAM-Med2D matched-points report | `runs/research_direction/monuseg_official_screening/sammed2d_from_dinov2l_points_test14/report/index.html` |
| [~] | 主实验总表 | 已有 DINOv2-L × SAM ViT-B/L/H 三组完整结果；待 SAM2.1 完成后生成 `cross_model_generalization_summary.csv` |
| [x] | 定性图网页 | 已生成 5 个代表 case，可逐图放大查看 |
| [ ] | 论文故事线图 | 待实验结果稳定后画 |

## 9. 当前最靠前的执行顺序

1. 等 DINOv2-L + SAM2.1 Large 完整 grid 自然结束并核验 14 cases / 840 candidates。
2. 生成跨模型 evidence table：同一 matcher protocol 下报告 DINOv2-B/L、SAM ViT-B/L/H、SAM2.1 Large 的 Dice、IoU、预测面积和相对 baseline 增益。
3. 用当前证据明确论文表述：方法在 SAM ViT-B 上强提升；ViT-L/H 的 dense-point 响应出现过分割，是 model-dependent 负例，不宣称通用提升。
4. 补 BioMedCLIP 或其他 feature backbone 前，先完成同一 protocol 的统一汇总；之后再决定是否进入 learned policy。
