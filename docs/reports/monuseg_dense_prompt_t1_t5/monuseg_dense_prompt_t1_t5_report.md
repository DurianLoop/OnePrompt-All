# MonuSeg 单域 Dense Point Prompt T1-T5 闭环实验报告

## 结论摘要

- MonuSeg cases：20
- SAM proposals：340
- Round1 最优 K：20
- Round2 最优 lambda：0.25
- Round3 最优策略：local_maxima
- 旧 Stage3 mean Dice：0.0517
- 旧候选池 oracle mean Dice：0.3425
- 新最优单策略 mean Dice：0.3861
- 新候选池 oracle mean Dice：0.4464

> 本实验先按潜在空白方向推进，但不在没有系统查新的情况下声称“前人完全没人做过”。

## T3 点数预算

| K | cases | Dice | IoU | FG recall | Obj recall | Merge | Pred area % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 20.0000 | 0.0347 | 0.0181 | 0.1216 | 0.1349 | 0.1333 | 10.4867 |
| 3.0000 | 20.0000 | 0.1905 | 0.1178 | 0.4722 | 0.4897 | 0.0547 | 40.9908 |
| 5.0000 | 20.0000 | 0.2413 | 0.1517 | 0.7241 | 0.7267 | 0.0372 | 61.4828 |
| 10.0000 | 20.0000 | 0.3007 | 0.1991 | 0.8160 | 0.8241 | 0.0167 | 65.1803 |
| 20.0000 | 20.0000 | 0.3301 | 0.2258 | 0.8361 | 0.8081 | 0.0205 | 59.0408 |
| 50.0000 | 20.0000 | 0.3263 | 0.2250 | 0.9215 | 0.8849 | 0.1239 | 70.5408 |
| 100.0000 | 20.0000 | 0.3013 | 0.2050 | 0.9367 | 0.9014 | 0.3152 | 77.7476 |

## T4 前景-背景对比

| lambda | cases | Dice | IoU | FG precision | FG recall | Obj recall | Pred area % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 20.0000 | 0.3301 | 0.2258 | 0.2804 | 0.8361 | 0.8081 | 59.0408 |
| 0.2500 | 20.0000 | 0.3840 | 0.2746 | 0.3778 | 0.8014 | 0.7544 | 51.2477 |
| 0.5000 | 20.0000 | 0.3833 | 0.2720 | 0.4132 | 0.7352 | 0.7006 | 44.8986 |
| 1.0000 | 20.0000 | 0.3693 | 0.2578 | 0.3715 | 0.7678 | 0.7324 | 48.2621 |

## T5 点布局与聚类对照

| method | layout | clustering | Dice | IoU | FG recall | Obj recall | Merge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| best_topk | topk | none | 0.3840 | 0.2746 | 0.8014 | 0.7544 | 0.0325 |
| boundary_mix | boundary_mix | none | 0.3294 | 0.2242 | 0.8444 | 0.8149 | 0.0680 |
| dbscan_medoid | cluster_medoid | dbscan | 0.3709 | 0.2691 | 0.8709 | 0.8065 | 0.0360 |
| diverse_topk | diverse_topk | none | 0.3756 | 0.2739 | 0.8643 | 0.8055 | 0.1054 |
| heatmap_cc | heatmap_cc | connected_components | 0.3599 | 0.2593 | 0.8837 | 0.8328 | 0.1717 |
| local_maxima | local_maxima | none | 0.3861 | 0.2795 | 0.8142 | 0.7604 | 0.0419 |

## 报告网页

`D:\SAM\docs\reports\monuseg_dense_prompt_t1_t5\index.html`

## 原始产物

`D:\SAM\runs\research_direction\monuseg_dense_prompt_t1_t5`
