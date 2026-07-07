# 精进 Matcher：Feedback 与 Learning 拓展

日期：2026-07-01

## 1. 对老师任务的正面回应

老师给出的任务可以拆成两条线：

```text
方向1（易）：把 feedback 信息纳入 Matcher。
方向2（难）：把 learning 纳入 Matcher，让匹配方式、threshold 选择、proposal 保留不完全靠手调。
```

本次推进完成的是一个最小可复现版本：

```text
Feedback-Guided Matcher + Lightweight Learning Policy
```

也就是说，先不急着训练大模型，而是把正负点 feedback 和可解释 threshold learning 接入现有 Stage1-3 闭环。

## 2. 方向1：feedback 如何纳入 Matcher

原始 Matcher 的 Stage1 可以理解为：

```text
reference box/mask
-> 提取 reference 前景 DINO prototype
-> 与 target feature map 做 cosine similarity
-> 取 top-k matched points
```

现在加入正负点后，Matcher 会多获得两类信息：

```text
positive point: 这里更像目标
negative point: 这里更像背景/干扰物
```

对应公式是：

```text
updated_proto = normalize(reference_proto
                         + positive_weight * positive_proto
                         - negative_weight * negative_proto)
```

这一步的意义很关键：negative point 不再只是 SAM 的 background prompt，而是直接影响 Matcher 的相似度图。也就是说，后续 top-k matched points、聚类、prompt sampling 都会避开负点附近的背景特征。

已新增核心文件：

```text
code/benchmarks/matcher_feedback_learning.py
```

其中 `apply_feature_feedback` 负责完成这个 prototype update。

## 3. 方向2：learning 如何纳入 Matcher

这次没有直接做大模型 fine-tuning，而是先做更稳的轻量学习：

```text
学 threshold
学 proposal keep/reject
学一个可解释 ranking score
```

当前实现包含三件事：

### 3.1 Stage1 自适应 threshold

原来 threshold 通常手动设为 `0.55`。现在可以根据相似度分布和点数预算自动选择：

```text
select_threshold_by_budget(scores, min_selected, max_selected)
```

目标是让 selected points 不会太少，也不会太多。

### 3.2 Stage3 proposal 阈值学习

给少量人工标签：

```json
{
  "positive_group_ids": [1, 2],
  "negative_group_ids": [3, 4]
}
```

脚本会学习一个 instance similarity threshold：

```text
proposal instance_similarity >= learned_threshold -> keep
proposal instance_similarity < learned_threshold  -> reject
```

这正面回应了老师说的：

```text
threshold 是不是也可以是学习过程？
```

当前答案是：可以，先从小样本可解释阈值学习开始。

### 3.3 可解释 proposal score

新增的 proposal score 使用：

```text
instance_similarity
SAM score
cluster coverage
prompt coverage
mask area penalty
bbox area penalty
```

它不是黑盒分数，每一项都能解释：

- instance similarity：像不像 reference 目标；
- SAM score：SAM 自己对 mask 的信心；
- cluster coverage：是否覆盖同一 cluster 的 matched points；
- prompt coverage：是否包含输入 prompt points；
- area/bbox penalty：惩罚过大的背景粘连 mask。

## 4. 新增文件

| 文件 | 作用 |
| --- | --- |
| `code/benchmarks/matcher_feedback_learning.py` | Feedback prototype update 与轻量 learning 核心函数 |
| `code/benchmarks/run_matcher_feedback_learning_policy.py` | 从 Stage3 proposals 和少量标签学习阈值策略 |
| `code/benchmarks/check_matcher_feedback_learning.py` | 核心函数检查 |
| `code/benchmarks/check_matcher_feedback_learning_policy_cli.py` | CLI 检查 |
| `code/benchmarks/run_research_direction_case.py` | 已加入可选 `--labels-json`，Stage3 后自动生成 learning policy |

## 5. 可复现命令

核心函数检查：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_matcher_feedback_learning.py
```

输出：

```text
Matcher feedback learning checks passed.
```

CLI 检查：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_matcher_feedback_learning_policy_cli.py
```

输出：

```text
Matcher feedback learning policy CLI checks passed.
```

runner 检查：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_research_direction_runner.py
```

输出：

```text
Research direction runner static checks passed.
```

## 6. 在完整 Stage1-3 runner 中使用

原有 runner 可以继续不带 learning：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_research_direction_case.py --case-name elephant_pair --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant2.png --reference-box 0,60,295,425 --positive-points "268,331" --negative-points "30,430;455,420"
```

如果已经有人工 proposal 标签，则加上：

```powershell
--labels-json runs\research_direction\labels\elephant_pair_labels.json
```

runner 会自动追加：

```text
stage3_feature -> learning_policy
```

并输出：

```text
learning_policy/matcher_learning_policy.json
learning_policy/summary.md
```

## 7. 当前结论

1. 方向1已经有可运行实现：正负点 feedback 可以进入 Matcher prototype，并改变后续 similarity/matched points 的依据。
2. 方向2已经有最小 learning 版本：threshold 和 proposal keep/reject 可以通过少量标签学习，而不是完全手调。
3. 当前还不是大规模 benchmark，也不是 adapter/fine-tuning；它的价值是先证明“Matcher 的交互反馈与阈值选择可以被机制化、可解释化”。

## 8. 下一步建议

1. 用 elephant、horse、person/animal 多实例各做一份 labeled proposals。
2. 比较三组：

```text
no feedback
spatial feedback
feature/prototype feedback
```

3. 每组记录：

```text
selected points
cluster count / K_pred
proposal count
retained masks
instance similarity
人工 accept/reject
```

4. 当 3-5 个 case 稳定后，再把 learning policy 从单阈值升级为 logistic regression 或小 MLP。

## 9. 继续推进：跨 case 的 Matcher learning casebook

为了进一步靠近老师说的“如何匹配、如何选取 threshold 是不是一个学习过程”，本次继续新增了跨 case 的学习脚本：

```text
code/benchmarks/run_matcher_learning_casebook.py
```

它做的事情是：

```text
多个 case 的 Stage3 feature_scores.json
-> 读取 proposal metrics
-> 读取人工标签或使用 retained/rejected 伪标签
-> 计算 feedback_learning_score
-> 在多个候选 feature 上学习 keep/reject threshold
-> 选择 F1 最好的策略
```

候选 feature 包括：

```text
feedback_learning_score
instance_similarity
sam_score
cluster_coverage
```

其中 `feedback_learning_score` 是组合分数，包含：

```text
instance similarity
SAM score
cluster coverage
prompt coverage
mask area penalty
bbox area penalty
```

### 9.1 新增检查

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_matcher_learning_casebook.py
```

输出：

```text
Matcher learning casebook checks passed.
```

### 9.2 真实伪标签 casebook 实验

输入文件：

```text
runs/research_direction/matcher_learning_casebook/teacher_direction_cases.json
```

包含 3 个已有 case：

```text
elephant_pair
horse_head
elephant3_eps28
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_learning_casebook.py --cases-json runs\research_direction\matcher_learning_casebook\teacher_direction_cases.json --use-pseudo-labels --out-dir runs\research_direction\matcher_learning_casebook\teacher_direction_pseudo
```

输出：

```text
runs/research_direction/matcher_learning_casebook/teacher_direction_pseudo/casebook_policy.json
runs/research_direction/matcher_learning_casebook/teacher_direction_pseudo/casebook_rows.csv
runs/research_direction/matcher_learning_casebook/teacher_direction_pseudo/summary.md
```

关键结果：

| 指标 | 结果 |
| --- | ---: |
| case 数 | 3 |
| labeled proposal 数 | 38 |
| positive / negative | 9 / 29 |
| 最优 feature | feedback_learning_score |
| 最优 threshold | 0.80152 |
| F1 | 0.600000 |
| precision | 0.428571 |
| recall | 1.000000 |
| accuracy | 0.684211 |
| selected rows | 21 |

对比不同 feature：

| feature | F1 | precision | recall | accuracy |
| --- | ---: | ---: | ---: | ---: |
| feedback_learning_score | 0.600000 | 0.428571 | 1.000000 | 0.684211 |
| instance_similarity | 0.470588 | 0.320000 | 0.888889 | 0.526316 |
| sam_score | 0.518519 | 0.388889 | 0.777778 | 0.657895 |
| cluster_coverage | 0.521739 | 0.428571 | 0.666667 | 0.710526 |

### 9.3 阶段性解释

这个结果不能被写成最终性能结论，因为当前标签是 Stage3 retained/rejected 伪标签，不是人工 GT。但是它说明了一个重要趋势：

```text
单看 instance_similarity 不够；
单看 SAM score 也不够；
把 Matcher feature、SAM score、cluster coverage、prompt coverage、面积惩罚组合起来，能得到更强的 keep/reject 学习信号。
```

因此，老师方向2可以继续发展为：

```text
learning-to-rank / learning-to-threshold for Matcher proposals
```

下一步最值得做的是把伪标签替换成人工 accept/reject 标签。每个 case 只需要标注十几个 proposal，就能形成一个小但干净的 learning dataset。

## 10. 人工 accept/reject 标签版 casebook

在上一步伪标签结果之后，已经对当前 3 个 case 的 proposal 直接做了人工 accept/reject 标注。

人工标签位置：

```text
runs/research_direction/matcher_learning_casebook/manual_labels/elephant_pair_labels.json
runs/research_direction/matcher_learning_casebook/manual_labels/horse_head_labels.json
runs/research_direction/matcher_learning_casebook/manual_labels/elephant3_eps28_labels.json
```

标注原则：

```text
positive = proposal 可以作为目标实例或目标候选 mask 使用
negative = 明显背景、过大粘连、局部碎片、或与当前 case 的实例目标不一致
```

### 10.1 各 case 标注

#### elephant_pair

```json
{
  "positive_group_ids": [4, 5, 8],
  "negative_group_ids": [1, 2, 3, 6, 7, 9, 10, 11, 12]
}
```

解释：`4/5/8` 较好覆盖大象主体；其它 proposal 主要是水面、瀑布、背景粘连或过小局部。

#### horse_head

```json
{
  "positive_group_ids": [3, 4, 6, 7, 8, 11, 12, 13, 14],
  "negative_group_ids": [1, 2, 5, 9, 10, 15, 16]
}
```

解释：该 case 按完整马实例评估，因此完整棕马/白马 proposal 标正；头部小块、局部框、非完整实例标负。

#### elephant3_eps28

```json
{
  "positive_group_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24],
  "negative_group_ids": [11, 18]
}
```

解释：这是多实例小目标场景。紧凑落在可见大象上的小 proposal 标正；`11/18` 是明显河岸/树林/大背景粘连，标负。

### 10.2 人工标签 casebook 结果

输入：

```text
runs/research_direction/matcher_learning_casebook/teacher_direction_manual_cases.json
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_learning_casebook.py --cases-json runs\research_direction\matcher_learning_casebook\teacher_direction_manual_cases.json --out-dir runs\research_direction\matcher_learning_casebook\teacher_direction_manual
```

输出：

```text
runs/research_direction/matcher_learning_casebook/teacher_direction_manual/casebook_policy.json
runs/research_direction/matcher_learning_casebook/teacher_direction_manual/casebook_rows.csv
runs/research_direction/matcher_learning_casebook/teacher_direction_manual/summary.md
```

关键结果：

| 指标 | 结果 |
| --- | ---: |
| case 数 | 3 |
| labeled proposal 数 | 52 |
| positive / negative | 34 / 18 |
| 最优 feature | feedback_learning_score |
| 最优 threshold | 0.781671 |
| F1 | 0.931507 |
| precision | 0.871795 |
| recall | 1.000000 |
| accuracy | 0.903846 |
| selected rows | 39 |

不同 feature 对比：

| feature | F1 | precision | recall | accuracy |
| --- | ---: | ---: | ---: | ---: |
| feedback_learning_score | 0.931507 | 0.871795 | 1.000000 | 0.903846 |
| instance_similarity | 0.845070 | 0.810811 | 0.882353 | 0.788462 |
| sam_score | 0.800000 | 0.666667 | 1.000000 | 0.673077 |
| cluster_coverage | 0.819277 | 0.693878 | 1.000000 | 0.711538 |

### 10.3 对老师方向2的直接回应

人工标签结果比伪标签更能说明问题：

```text
feedback_learning_score > instance_similarity > cluster_coverage > sam_score
```

也就是说，Matcher 的 proposal 保留策略确实不应该只看单一相似度或 SAM 分数。把 Matcher feature、SAM 置信度、cluster/prompt 覆盖和面积惩罚组合起来之后，可以学出更可靠的 threshold。

这一步已经从“手动调阈值”推进到：

```text
用少量人工标注 proposal 学一个可解释的 Matcher keep/reject 策略
```

下一步可以继续做 leave-one-case-out 验证：用两个 case 学 threshold，在第三个 case 上测试，避免只报告训练集效果。

## 11. P0-P4 全部推进结果

本节对应后续拓展任务：从 P0 到 P4 全部做成可复现脚本和产物。

### 11.1 P0：Leave-one-case-out 验证

新增脚本：

```text
code/benchmarks/run_matcher_learning_loco.py
code/benchmarks/check_matcher_learning_loco.py
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_learning_loco.py --cases-json runs\research_direction\matcher_learning_casebook\teacher_direction_manual_cases.json --out-dir runs\research_direction\matcher_learning_loco\teacher_direction_manual_loco
```

输出：

```text
runs/research_direction/matcher_learning_loco/teacher_direction_manual_loco/loco_results.json
runs/research_direction/matcher_learning_loco/teacher_direction_manual_loco/loco_predictions.csv
runs/research_direction/matcher_learning_loco/teacher_direction_manual_loco/summary.md
```

结果：

| Model | Macro F1 |
| --- | ---: |
| feedback_threshold | 0.911663 |
| instance_threshold | 0.644222 |
| logistic_regression | 0.451629 |
| decision_stump | 0.717660 |

解释：在 leave-one-case-out 下，`feedback_learning_score` 的单阈值策略仍然明显强于只看 `instance_similarity`。这说明前面训练集上的结果不是完全偶然；不过由于 case 数仍少，暂时只能作为小样本机制证据。

### 11.2 P1：轻量分类器 baseline

LOCO 脚本同时实现了两个轻量学习 baseline：

```text
logistic_regression
decision_stump
```

当前结果中：

```text
decision_stump F1 = 0.717660
logistic_regression F1 = 0.451629
```

解释：小样本下，复杂一点的模型不一定更好。当前最稳定的仍是可解释的 `feedback_learning_score + threshold`。这对论文叙事反而有帮助：先把手工设计的组合 score 做扎实，再逐步增加学习复杂度。

### 11.3 P2：连续 feedback 闭环

新增脚本：

```text
code/benchmarks/run_matcher_feedback_loop.py
code/benchmarks/check_matcher_feedback_loop.py
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_feedback_loop.py --cases-json runs\research_direction\matcher_learning_casebook\teacher_direction_manual_cases.json --max-feedback-rounds 5 --out-dir runs\research_direction\matcher_feedback_loop\teacher_direction_manual_feedback_loop
```

输出：

```text
runs/research_direction/matcher_feedback_loop/teacher_direction_manual_feedback_loop/feedback_loop_results.json
runs/research_direction/matcher_feedback_loop/teacher_direction_manual_feedback_loop/feedback_loop_curves.csv
runs/research_direction/matcher_feedback_loop/teacher_direction_manual_feedback_loop/feedback_events.csv
runs/research_direction/matcher_feedback_loop/teacher_direction_manual_feedback_loop/summary.md
```

结果：

| Round | Mean F1 | Mean Precision | Mean Recall | Mean labeled feedback |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.911663 | 0.880952 | 0.969697 | 0.000 |
| 1 | 0.911663 | 0.880952 | 0.969697 | 1.000 |
| 2 | 0.919784 | 0.880952 | 0.984848 | 2.000 |
| 3 | 0.927536 | 0.880952 | 1.000000 | 3.000 |
| 4 | 0.927536 | 0.880952 | 1.000000 | 4.000 |
| 5 | 0.927536 | 0.880952 | 1.000000 | 5.000 |

解释：这一步把 feedback 从“一次性修正”推进到了迭代闭环。每轮选择最不确定 proposal，请人工反馈标签，然后更新 threshold。当前小样本下 F1 从 0.911663 提升到 0.927536，说明交互式反馈有继续做的价值。

### 11.4 P3：扩展 casebook

新增脚本：

```text
code/benchmarks/build_matcher_casebook_manifest.py
code/benchmarks/check_matcher_casebook_expansion.py
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\build_matcher_casebook_manifest.py --search-root runs\research_direction --out-dir runs\research_direction\matcher_learning_casebook\expanded
```

扩展后共找到 7 个 Stage3 case：

| Case | Proposals | Retained | K_pred |
| --- | ---: | ---: | ---: |
| multi_instance_horse_head | 16 | 2 | 2 |
| p0_p3_closure_elephant3_tuning_stage3_eps20_relaxed | 32 | 7 | 7 |
| p0_p3_closure_elephant3_tuning_stage3_eps28_relaxed | 24 | 6 | 6 |
| p0_p3_closure_elephant3_tuning_stage3_eps35_relaxed | 20 | 5 | 5 |
| p0_p3_closure_elephant3_tuning_stage3_eps45_relaxed | 12 | 3 | 3 |
| p0_p3_closure_p0_runner_elephant3_multi_actual | 2 | 0 | 0 |
| elephant_pair | 12 | 1 | 1 |

使用伪标签运行 expanded casebook：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_learning_casebook.py --cases-json runs\research_direction\matcher_learning_casebook\expanded\expanded_cases.json --use-pseudo-labels --out-dir runs\research_direction\matcher_learning_casebook\expanded_pseudo_policy
```

结果：

| 指标 | 结果 |
| --- | ---: |
| Cases | 7 |
| Labeled rows | 80 |
| Positive / negative | 24 / 56 |
| Best feature | feedback_learning_score |
| Best threshold | 0.889215 |
| F1 | 0.636364 |
| Selected rows | 20 |

解释：expanded 版使用的是 retained/rejected 伪标签，因此结果低于人工标签版。它的作用是扩大候选池和发现 failure mode，不应该替代人工标签结论。

### 11.5 P4：医学迁移 gate/report

新增脚本：

```text
code/benchmarks/run_matcher_medical_transfer_plan.py
code/benchmarks/check_matcher_medical_transfer.py
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_medical_transfer_plan.py --out-dir runs\research_direction\matcher_medical_transfer
```

输出：

```text
runs/research_direction/matcher_medical_transfer/medical_transfer_plan.json
runs/research_direction/matcher_medical_transfer/summary.md
```

结果：

| 指标 | 结果 |
| --- | ---: |
| medical image candidates | 5000 |
| medical mask candidates | 256 |
| existing prediction pairs | 1 |
| existing Dice mean | 0.020698 |
| data with masks available | True |
| prediction pairs available | True |

解释：本地有医学图像和 mask，因此可以做医学迁移闭环；但现有 prediction pair 的 Dice 很低，只能作为 gate 结果，不能作为 Matcher 医学有效性的证明。P4 下一步应该选 5 个医学 support/query case，真实跑 Matcher Stage1/2/3，并像自然图一样标 proposal accept/reject。

## 12. 阶段性总判断

当前从 P0 到 P4 的结论可以写成：

```text
1. feedback_learning_score 在人工标签 casebook 和 leave-one-case-out 中都优于单独 instance similarity。
2. 轻量学习模型在当前小样本下不一定优于可解释 threshold；这说明现阶段更适合走 interpretable learning。
3. 迭代 feedback 可以进一步提升 F1，说明“用户反馈 -> 更新 Matcher 决策”的方向可行。
4. expanded casebook 已经能自动扫描已有 Stage3 结果，但伪标签质量不足，后续要继续补人工标签。
5. 医学迁移具备数据条件，但还没有完成真实 Matcher 医学闭环，不能过度宣称医学效果。
```

推荐下一步是：

```text
先补 2-3 个 expanded case 的人工标签，再做 leave-one-case-out；
同时选 5 个医学 case 跑 Matcher Stage1/2/3，开始建立 medical proposal labels。
```
## 13. 下一步已推进：医学 support/query pilot manifest

这一步把 P4 中“医学迁移只完成 gate，还没有真实闭环”的问题往前推进了一格：我没有直接宣称医学 Matcher 已经有效，而是先把能复现实验的医学 support/query 清单建出来。它的作用是让后续 Stage1/2/3 可以在同一批医学图像上重复运行、记录反馈、再形成 medical casebook。

新增脚本：
```text
code/benchmarks/build_matcher_medical_pilot_manifest.py
code/benchmarks/check_matcher_medical_pilot_manifest.py
```

运行命令：
```powershell
python -B code\benchmarks\build_matcher_medical_pilot_manifest.py --out-dir runs\research_direction\matcher_medical_pilot --idrid-cases 3 --monuseg-cases 2
python -B code\benchmarks\check_matcher_medical_pilot_manifest.py
```

输出文件：
```text
runs/research_direction/matcher_medical_pilot/medical_matcher_pilot_manifest.csv
runs/research_direction/matcher_medical_pilot/medical_matcher_pilot_manifest.json
runs/research_direction/matcher_medical_pilot/summary.md
```

当前 pilot 规模：
| 数据集 | 数量 | 目标 |
| --- | ---: | --- |
| IDRID | 3 | 视网膜病灶 foreground |
| MonuSeg | 2 | 细胞核 foreground |
| 合计 | 5 | 医学 support/query 最小闭环输入 |

每一行都包含：
| 字段 | 含义 |
| --- | --- |
| `support_image_path` / `support_mask_path` | 参考图和参考 mask，用来定义“我要找的医学目标” |
| `support_box_xyxy` | 从 support mask 外接得到的 reference box，可直接用于老师截图里提到的 reference box/mask prototype |
| `query_image_path` / `query_mask_path` | 目标图和真值 mask，用来验证匹配点、SAM 分割和 Stage3 筛选 |
| `target_hint` | 记录目标语义，方便报告解释，不把它当作唯一模型输入 |

对老师问题的正面回应：
1. 老师问“医学/自然场景能否做闭环”：现在已有 5 个医学 support/query case，可以开始真实跑 Stage1 matching、Stage2 SAM point segmentation、Stage3 instance-level 筛选。
2. 老师问“reference box/mask 是什么”：这里的 `support_box_xyxy` 就是从 support mask 的前景像素外接出来的框；后续 Matcher 会从这个框或 mask 内提取 DINO feature prototype。
3. 老师问“feedback 怎么纳入 matcher”：这个 pilot 的 query mask 允许我们把每个 proposal 标为 accept/reject，再复用自然图 casebook 中的 `feedback_learning_score`、threshold 和 leave-one-case-out 评估方式。
4. 老师问“learning 怎么纳入 matcher”：医学 pilot 的第一步不是直接训练复杂模型，而是先收集医学 proposal labels；等 label 足够后，再比较 interpretable threshold、decision stump、logistic regression 或更强学习器。

阶段结论：
```text
医学方向现在从“只有数据 gate”推进到“已有可复现 support/query 输入清单”。
这还不是医学有效性结论，但已经是后续医学 Matcher 闭环实验的入口。
下一步应当对这 5 个 case 运行 Stage1/2/3，并手动标注 proposal accept/reject。
```

## 14. 医学 pilot Stage1/2/3 已跑通：结论与问题

在第 13 节的 manifest 基础上，我继续把 5 个医学 support/query case 跑到了 Stage1 matching、Stage2 point sampling、SAM proposal generation 和 Stage3 filtering，并加入了 query mask Dice/IoU 评估。

新增脚本：
```text
code/benchmarks/run_matcher_medical_pilot.py
code/benchmarks/check_matcher_medical_pilot_runner.py
code/benchmarks/evaluate_matcher_medical_pilot.py
code/benchmarks/check_matcher_medical_pilot_eval.py
```

运行命令：
```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_medical_pilot.py --manifest runs\research_direction\matcher_medical_pilot\medical_matcher_pilot_manifest.csv --out-dir runs\research_direction\matcher_medical_pilot\run --run-stage1
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_medical_pilot.py --manifest runs\research_direction\matcher_medical_pilot\medical_matcher_pilot_manifest.csv --out-dir runs\research_direction\matcher_medical_pilot\run --run-stage2-sampling --cluster-eps 120 --min-samples 2 --sample-sizes 1,2,3 --samples-per-cluster 4
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\evaluate_matcher_medical_pilot.py --manifest runs\research_direction\matcher_medical_pilot\medical_matcher_pilot_manifest.csv --run-dir runs\research_direction\matcher_medical_pilot\run --out-dir runs\research_direction\matcher_medical_pilot\eval
```

输出位置：
```text
runs/research_direction/matcher_medical_pilot/run/
runs/research_direction/matcher_medical_pilot/eval/
```

关键结果：
| Case | Dataset | Proposals | Stage3 retained | Best Dice | Best retained Dice |
| --- | --- | ---: | ---: | ---: | ---: |
| idrid_01 | IDRID | 12 | 2 | 0.691088 | 0.689427 |
| idrid_02 | IDRID | 12 | 1 | 0.158815 | 0.158518 |
| idrid_03 | IDRID | 12 | 0 | 0.050765 | 0.0 |
| monuseg_01 | MonuSeg | 12 | 0 | 0.612698 | 0.0 |
| monuseg_02 | MonuSeg | 12 | 0 | 0.177491 | 0.0 |

整体指标：
| 指标 | 数值 |
| --- | ---: |
| Cases | 5 |
| Proposals | 60 |
| Stage3 retained | 3 |
| Best Dice mean | 0.338171 |
| Best IoU mean | 0.235865 |
| Best retained Dice mean | 0.169589 |
| Best retained IoU mean | 0.122426 |

对老师猜想的回应：
1. **reference mask/box 可以迁移到医学图像。** Stage1 已经能从 support mask 提取 prototype，并在 query 图上输出 matched points、heatmap 和 top-k overlay。
2. **聚类参数必须改。** 自然图里 `eps=28` 对 IDRID 大图太小，导致 IDRID 的 prompt group 为 0；调到 `eps=120` 后，IDRID 和 MonuSeg 都能形成 prompt groups。这正面回应了“简单调参数是否可能改善”的问题：可以改善 Stage2 输入，但不保证最终分割一定好。
3. **SAM proposal 不是主要瓶颈。** 每个 case 都成功生成 12 个 proposal，且部分 case 的 best Dice 不低，例如 `idrid_01=0.691088`、`monuseg_01=0.612698`。
4. **当前最大问题是 Stage3 医学域筛选规则。** Stage3 只保留了 3/60 个 proposal，MonuSeg 两例虽然存在 Dice 较好的 proposal，却被全过滤掉。这说明自然图规则迁移到医学域时过于保守，需要学习型/反馈型筛选。
5. **下一步不是盲目扩大 benchmark，而是先修 Stage3。** 现在已有证据表明“候选里有好 mask，但 retained 没选出来”，因此最优先方向是把 query mask/manual accept-reject label 纳入 learning matcher，学习医学域的 threshold 和 proposal ranking。

阶段结论：
```text
医学最小闭环已经从数据清单推进到 Stage1/2/3 + Dice/IoU 评估。
结果支持“Matcher 能产生医学候选”，但不支持“当前 Stage3 已能稳定筛好候选”。
后续 P0 应该做 medical proposal label casebook：把 60 个 proposal 标成 accept/reject，再训练/比较医学域 threshold、decision stump、logistic regression 和 feedback_learning_score。
```

## 15. 医学 proposal casebook：learning 方向的第一版验证

在第 14 节中已经看到：医学候选里存在较好的 proposal，但 Stage3 规则没有稳定选出来。因此这一节继续推进老师提出的“将 learning 加入 matcher 框架”：把 60 个医学 proposal 按 query mask 的 Dice 自动标成 accept/reject，然后学习一个医学域 proposal policy。

新增脚本：
```text
code/benchmarks/build_matcher_medical_casebook.py
code/benchmarks/check_matcher_medical_casebook.py
```

主实验命令：
```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\build_matcher_medical_casebook.py --manifest runs\research_direction\matcher_medical_pilot\medical_matcher_pilot_manifest.csv --run-dir runs\research_direction\matcher_medical_pilot\run --out-dir runs\research_direction\matcher_medical_pilot\casebook --positive-dice 0.30
```

输出：
```text
runs/research_direction/matcher_medical_pilot/casebook/medical_casebook_rows.csv
runs/research_direction/matcher_medical_pilot/casebook/medical_casebook_policy.json
runs/research_direction/matcher_medical_pilot/casebook/summary.md
```

### 15.1 主结果：Dice >= 0.30 作为 accept

| 指标 | 数值 |
| --- | ---: |
| Cases | 5 |
| Proposal rows | 60 |
| Positive / negative | 13 / 47 |
| Best policy | scalar threshold on `feedback_learning_score` |
| Policy threshold | 1.278738 |
| Stage3 F1 | 0.125000 |
| Learned policy F1 | 0.869565 |
| Stage3 retained Dice mean | 0.169589 |
| Learned best Dice mean | 0.213440 |
| Oracle best Dice mean | 0.338171 |
| Leave-one-case-out F1 | 0.000000 |

解释：
1. **训练集上 learning 明显优于原 Stage3。** Stage3 只保留 3 个 proposal，且只有 1 个是真正 positive，F1 只有 0.125；学习到的 `feedback_learning_score >= 1.278738` 能保留 10 个 proposal，precision 1.0、recall 0.769231、F1 0.869565。
2. **per-case Dice 也有提升，但还远不到 oracle。** Stage3 retained best Dice mean 是 0.169589，learned best Dice mean 提到 0.213440；而 oracle best proposal Dice mean 是 0.338171，说明“候选池里还有更好的 mask 没被学出来”。
3. **LOCO 为 0 是最重要的警告。** 这说明当前 5 个 case 太少，正样本集中在少数 case 上；同一批数据上学到的阈值不能直接当作泛化结论。

### 15.2 label 阈值敏感性

为了确认结论不是由单一 Dice 阈值造成的，我又跑了 `positive-dice=0.15/0.30/0.50` 三组：

| Positive rule | Pos / Neg | Stage3 F1 | Learned F1 | LOCO F1 | Stage3 Dice mean | Learned Dice mean | Oracle Dice mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dice >= 0.15 | 27 / 33 | 0.133333 | 0.761905 | 0.289855 | 0.169589 | 0.213440 | 0.338171 |
| Dice >= 0.30 | 13 / 47 | 0.125000 | 0.869565 | 0.000000 | 0.169589 | 0.213440 | 0.338171 |
| Dice >= 0.50 | 13 / 47 | 0.125000 | 0.869565 | 0.000000 | 0.169589 | 0.213440 | 0.338171 |

解释：
1. `Dice >= 0.15` 时正样本更多，LOCO F1 有 0.289855，说明有一点跨 case 信号。
2. `Dice >= 0.30/0.50` 时正样本更少且集中，LOCO F1 变成 0，说明当前 case 数不足以支持强泛化判断。
3. 三个阈值下 Stage3 F1 都很低，说明“现有 Stage3 规则对医学域不够好”这个结论比较稳定。

### 15.3 对老师任务的直接回应

**方向 1：feedback 信息如何纳入 matcher。**  
现在已经可以把每个医学 proposal 标成 accept/reject。accept 可以视为 positive feedback，reject 可以视为 negative feedback；下一轮 matcher 可以用这些标签更新 prototype 或 proposal ranking，而不是只靠初始 DINO similarity。

**方向 2：learning 如何纳入 matcher。**  
这一步已经验证了最小 learning 入口：用 proposal 特征学习 accept/reject policy。当前最有效的是 `feedback_learning_score` 的可解释阈值，但 LOCO 失败说明还不能声称模型泛化。下一步应该扩大医学 casebook，并加入人工标注或更稳定的弱标签。

阶段结论：
```text
learning 方向值得继续：它在当前 60 个医学 proposal 上明显优于手写 Stage3 retained 规则。
但这还不是最终方法：当前 case 太少，LOCO 泛化不足，因此下一步重点不是换复杂模型，而是扩充医学 casebook、补人工 accept/reject 标签、按 domain 分开学习 threshold。
```

## 16. 扩充并一键执行：20 例医学闭环

为了回应上一节“5 例太少、LOCO 泛化不足”的问题，我新增了一键脚本，把医学闭环从 5 例扩到 20 例，并完整跑通：

```text
manifest -> Stage1 matching -> Stage2 point sampling -> SAM proposals -> Stage3 filtering -> Dice/IoU eval -> medical casebook -> review gallery
```

新增脚本：
```text
code/benchmarks/run_matcher_medical_one_click.py
code/benchmarks/check_matcher_medical_one_click.py
```

一键命令：
```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_matcher_medical_one_click.py --out-dir runs\research_direction\matcher_medical_expanded --idrid-cases 12 --monuseg-cases 8
```

快速检查命令：
```powershell
python -B code\benchmarks\check_matcher_medical_one_click.py
```

输出位置：
```text
runs/research_direction/matcher_medical_expanded/
runs/research_direction/matcher_medical_expanded/review_gallery.html
runs/research_direction/matcher_medical_expanded/eval/summary.md
runs/research_direction/matcher_medical_expanded/casebook/summary.md
```

### 16.1 扩充规模

| 数据集 | Cases | Proposals per case | Total proposals |
| --- | ---: | ---: | ---: |
| IDRID | 12 | 12 | 144 |
| MonuSeg | 8 | 12 | 96 |
| 合计 | 20 | 12 | 240 |

### 16.2 分割候选质量

| 指标 | 结果 |
| --- | ---: |
| Cases | 20 |
| OK cases | 20 |
| Proposals | 240 |
| Stage3 retained | 17 |
| Best Dice mean | 0.343932 |
| Best IoU mean | 0.242159 |
| Best retained Dice mean | 0.205933 |
| Best retained IoU mean | 0.152458 |

解释：
1. 20 例后，候选池质量和 5 例时相近，说明扩充没有把 pipeline 跑崩。
2. Oracle best Dice mean 是 0.343932，而 Stage3 retained Dice mean 只有 0.205933，仍然说明“好候选存在，但 Stage3 规则选得不够好”。
3. MonuSeg 的好候选仍经常被 Stage3 全过滤，说明医学域需要 domain-specific threshold 或学习型筛选。

### 16.3 Learning 结果

主标签规则仍然使用 `Dice >= 0.30` 作为 accept：

| 指标 | 结果 |
| --- | ---: |
| Labeled rows | 240 |
| Positive / negative | 39 / 201 |
| Best policy | `feedback_learning_score >= 1.233281` |
| Stage3 F1 | 0.214286 |
| Learned policy F1 | 0.677419 |
| Leave-one-case-out F1 | 0.655738 |
| Stage3 retained Dice mean | 0.205933 |
| Learned best Dice mean | 0.230958 |
| Oracle best Dice mean | 0.343932 |

这是相比 5 例最关键的进展：5 例时 `Dice >= 0.30` 的 LOCO F1 是 0；扩到 20 例后 LOCO F1 变成 0.655738。也就是说，learning 不只是训练集上看起来好，已经开始出现跨 case 泛化信号。

### 16.4 Dice 阈值敏感性

| Positive rule | Pos / Neg | Stage3 F1 | Learned F1 | LOCO F1 | Stage3 Dice mean | Learned Dice mean | Oracle Dice mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dice >= 0.15 | 71 / 169 | 0.204545 | 0.560440 | 0.424581 | 0.205933 | 0.170470 | 0.343932 |
| Dice >= 0.30 | 39 / 201 | 0.214286 | 0.677419 | 0.655738 | 0.205933 | 0.230958 | 0.343932 |
| Dice >= 0.50 | 26 / 214 | 0.232558 | 0.857143 | 0.833333 | 0.205933 | 0.230958 | 0.343932 |

解释：
1. 三个阈值下，learned F1 都高于 Stage3 F1。
2. `Dice >= 0.30/0.50` 的 LOCO 已经明显可用，说明扩充 case 后泛化问题被缓解。
3. `Dice >= 0.15` 的 learned best Dice mean 反而低，是因为过宽松标签把较弱 proposal 也当成 positive，会削弱“选最好 mask”的目标。

### 16.5 Review gallery

新增静态交互网页：
```text
runs/research_direction/matcher_medical_expanded/review_gallery.html
```

它包含：
1. 每个 case 的 12 个 proposal overlay。
2. 每个 proposal 的 Dice、IoU、pseudo label、Stage3 keep/drop。
3. 按 pseudo accept/reject 和 Stage3 kept/dropped 筛选。
4. 人工 review 下拉框，并可导出 JSON。

这个 gallery 的目的不是替代实验指标，而是让下一步人工标注变得可执行：我们可以重点看“Dice 高但 Stage3 drop”的 proposal，把它们转成明确的 positive feedback。

阶段结论：
```text
20 例医学闭环已经完整跑通。扩充后，learning 的 LOCO F1 从 5 例时的不稳定/为 0，提升到 0.655738（Dice>=0.30）和 0.833333（Dice>=0.50）。
这说明老师提出的“把 learning 加入 matcher 框架”已经有正向证据。
下一步应优先做人审 review_gallery，把伪标签改成人工 accept/reject 标签；然后按 IDRID/MonuSeg 分域学习 threshold，解决 MonuSeg 被 Stage3 过度过滤的问题。
```
