# Feedback-Learned Matcher 医学主线推进记录

日期：2026-07-07

## 1. 本轮目标

本轮围绕论文主线 `Feedback-Learned Matcher for Promptable Medical Segmentation` 推进四件事：

1. 修正 `review_gallery.html`，确保每个 proposal overlay 都能显示，并支持点击放大。
2. 对 20 例、240 个 proposals 生成 accept/reject 标签。
3. 重新跑 proposal learning：overall、IDRID-only、MonuSeg-only 和 leave-one-case-out。
4. 扩展到 50 个医学 case，形成论文主表。

## 2. Gallery 修正

原问题是 casebook 行中只有 `mask_path`，而 gallery 生成器读取的是 `overlay_path`，导致 HTML 中图片 `src` 为空。

本轮修改：

- `build_matcher_medical_casebook.py` 写出 `overlay_path`。
- `run_matcher_medical_one_click.py` 的 gallery 对旧 CSV 也能从 `mask_path` 推导 `proposal_overlays/group_xxx.png`。
- gallery 卡片支持 proposal overlay 显示、筛选、导出 review JSON。
- 增加点击图片放大 lightbox，方便逐张检查 proposal。

验证命令：

```powershell
python -B code\benchmarks\check_matcher_medical_gallery.py --out-dir runs\research_direction\matcher_medical_expanded
python -B code\benchmarks\check_matcher_medical_gallery.py --out-dir runs\research_direction\matcher_medical_expanded_50
```

两条检查均通过。

## 3. 20 例 accept/reject 标签

新增脚本：

```text
code/benchmarks/build_matcher_medical_manual_labels.py
```

输出：

```text
runs/research_direction/feedback_learned_matcher_mainline/manual_labels_20.json
```

说明：当前标签是 GT-assisted accept/reject 审查种子标签，规则为 `Dice >= 0.30` 标为 accept，否则标为 reject。它不是最终人工主观标签，但比 Stage3 retained/rejected 伪标签更接近 proposal 质量，可以作为后续人工修订入口。

标签统计：

| Cases | Proposals | Positive | Negative |
| ---: | ---: | ---: | ---: |
| 20 | 240 | 39 | 201 |

## 4. Proposal learning 结果

输出目录：

```text
runs/research_direction/feedback_learned_matcher_mainline/
```

论文主表：

```text
runs/research_direction/feedback_learned_matcher_mainline/paper_main_table.csv
```

| Setting | Cases | Proposals | Pos/Neg | Best feature | Stage3 F1 | Learned F1 | LOCO F1 | Stage3 Dice | Learned Dice | Oracle Dice |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall_20 | 20 | 240 | 39/201 | feedback_learning_score | 0.214286 | 0.677419 | 0.655738 | 0.205933 | 0.230958 | 0.343932 |
| idrid_20 | 12 | 144 | 23/121 | feedback_learning_score | 0.300000 | 0.978723 | 0.956522 | 0.343222 | 0.342880 | 0.348032 |
| monuseg_20 | 8 | 96 | 16/80 | cluster_coverage | 0.000000 | 0.578947 | 0.200000 | 0.000000 | 0.215647 | 0.337780 |
| overall_50 | 50 | 600 | 96/504 | feedback_learning_score | 0.165517 | 0.534247 | 0.496644 | 0.199561 | 0.204100 | 0.323295 |

## 5. 当前结论

1. Stage3 手写规则在医学域仍然偏保守，尤其 MonuSeg 上经常把好 proposal 全部过滤掉。
2. overall_20 中 learned policy 的 F1 从 Stage3 的 `0.214286` 提升到 `0.677419`，LOCO F1 为 `0.655738`。
3. IDRID-only 的学习信号最强，LOCO F1 达到 `0.956522`。
4. MonuSeg-only 显示单独分域很有必要：最优 feature 从 `feedback_learning_score` 变为 `cluster_coverage`，说明不同医学域不能强行共用同一个阈值。
5. 50 例扩展后仍然保持 learned policy 优于 Stage3，但 learned Dice 距离 oracle Dice 还有差距，说明候选池里存在更好 mask，后续主要问题仍是 proposal selection。

## 6. 可复现命令

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_feedback_learned_matcher_mainline.py
python -B code\benchmarks\check_feedback_learned_matcher_mainline.py
```

如果只重建 gallery：

```powershell
python -B code\benchmarks\run_matcher_medical_one_click.py --out-dir runs\research_direction\matcher_medical_expanded --build-gallery-only
python -B code\benchmarks\run_matcher_medical_one_click.py --out-dir runs\research_direction\matcher_medical_expanded_50 --build-gallery-only
```

## 7. 下一步

1. 用修好的 gallery 对 20 例或 50 例 labels 做真正人工修订。
2. 继续按 domain 学习 policy：IDRID、MonuSeg 不共用阈值。
3. 增加 bootstrap confidence interval，避免只报单点 F1。
4. 论文实验表建议拆成：proposal F1 主表、per-case Dice 表、domain ablation、feature ablation、failure case gallery。
