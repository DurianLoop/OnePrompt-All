# AdaPromptSet / RefShift 发表推进执行指令
## Phase T — Publication Completion for Reference-Induced Instability Benchmark and Risk-Aware Mitigation

> **用途**：将本文档完整交给 Codex（下文简称 CX），开启 `/goal` 模式连续执行。  
> **最终目的**：把 Phase S 已建立的“参考图像诱发不稳定性”核心发现，推进成一项具备确认性证据、跨数据集复现、强基线对比、实用缓解方法和完整论文材料的可投稿研究。  
> **基本原则**：优先完成一篇证据扎实的 Benchmark / failure-boundary / risk-aware mitigation 论文，而不是继续无边界寻找更复杂模型。不得承诺一定录用，不得用 formal test 或 holdout 反复调参制造正结果。

---

# 0. 当前科学状态

项目目录：

```text
D:\SAM\adapromptset
```

Phase S 当前分支：

```text
phaseS-multitrack-reference-robust-research
```

Phase S 当前 HEAD：

```text
3e63733
```

建议新分支：

```text
phaseT-publication-completion
```

已有核心证据：

1. **Reference instability 成立**
   - corrected `V_ref_frac` 约 0.1376–0.3312；
   - 80.5%–90.2% queries 的 best-worst reference gap ≥ 0.05；
   - test-time-safe features 对不稳定性具有可预测信号。

2. **Mean consensus 是当前最强自动方法**
   - B1 random single Dice ≈ 0.4297；
   - B2 mean consensus Dice ≈ 0.4756；
   - 提升约 +0.046；
   - catastrophic failure 明显下降；
   - 仍然只需要一次 SAM segmentation call，但需要多个 reference matching passes。

3. **Learned weighting 未超过 uniform mean**
   - 最佳 GBM reference utility predictor 的 validation Spearman 约 0.478；
   - weighted consensus Dice ≈ 0.4732；
   - 比 uniform mean 低约 0.0024。

4. **Set-conditioned heuristic 未成功**
   - agreement gate、disagreement penalty、stable candidate scoring 均未超过 mean consensus；
   - 27 个组合中最优为所有附加系数为 0。

5. **现阶段不是方法论文完成状态**
   - holdout=40 尚未使用；
   - 只有 MoNuSAC；
   - 没有统一协议下的外部相关方法对比；
   - formal test 仍锁定。

现阶段正确状态应写为：

```text
PHASES_BENCHMARK_CORE_FINDING_ESTABLISHED
```

不得在新报告中直接写：

```text
SUBMISSION_READY
PUBLICATION_COMPLETED
SOTA
```

---

# 1. 本轮论文定位

本轮优先完成：

# Reference-Induced Instability Benchmark
## with Risk-Aware Multi-Reference Mitigation

推荐论文问题：

> How much does the choice of a valid reference affect visual-reference segmentation, how should this instability be measured, and can a simple risk-aware strategy mitigate it without retraining the frozen matcher and segmenter?

中文：

> 合法参考图像的选择会在多大程度上影响视觉参考分割？应该如何正确测量这种不稳定性？能否在不更新冻结 matcher 与 segmenter 的条件下，通过简单的风险感知策略缓解该问题？

论文应包含三层贡献：

1. **现象与指标**
   - reference-induced instability；
   - correct variance decomposition；
   - cross-reference / reference-resampling protocol；
   - tail-risk metrics。

2. **系统 Benchmark**
   - 多数据集；
   - 多 reference counts；
   - 多 prompting / matching baselines；
   - 成本与失败边界。

3. **实用缓解方法**
   - mean consensus 强基线；
   - dynamic reference budget；
   - instability-aware selective prediction / abstention；
   - 必要时 lightweight reliability model，但不得再以 weighting 为唯一方法。

---

# 2. 本轮注册的四个工作包

```text
WP1 — Confirmatory Benchmark
WP2 — Cross-Dataset and Cross-Method Replication
WP3 — Risk-Aware Mitigation
WP4 — Paper Construction and Mock Review
```

执行优先级：

1. WP1 必须先完成；
2. WP1 通过后执行 WP2；
3. WP2 至少得到一个额外数据集支持后执行 WP3；
4. WP1 或 WP2 失败时，不得访问 formal test；
5. WP3 失败时，仍可保留 Benchmark / failure-boundary 论文路线；
6. 只有 WP1、WP2、WP4 全部达标，才能写 `PAPER_CANDIDATE_READY`。

---

# 3. 开始前必须创建的控制文件

在任何新推理前创建：

```text
docs/phaseT_goal_read_receipt.md
docs/phaseT_publication_charter.md
docs/phaseT_claim_registry.md
docs/phaseT_dataset_registry.md
docs/phaseT_reference_protocol.md
docs/phaseT_metric_protocol.md
docs/phaseT_external_baseline_protocol.md
docs/phaseT_risk_mitigation_protocol.md
docs/phaseT_holdout_protocol.md
docs/phaseT_cross_dataset_protocol.md
docs/phaseT_paper_outline.md
docs/phaseT_test_lock_notice.md

reports/phaseT_program_status.md
reports/phaseT_program_status.json
reports/phaseT_compute_ledger.csv
reports/phaseT_protocol_snapshot.json
reports/phaseT_input_hashes.json
reports/phaseT_publication_readiness.md
reports/phaseT_risk_register.md
```

更新但不得覆盖历史：

```text
docs/protocol_history.md
docs/decision_log.md
docs/data_exposure_log.md
checklist.md
```

创建 preregistration commit：

```text
phaseT: preregister publication completion protocol
```

该 commit 前禁止读取 holdout、正式 test 或新增结果驱动参数。

---

# 4. 数据角色

## 4.1 MoNuSAC

沿用：

```text
Train = 80
Validation = 41
Internal Holdout = 40
Formal Test = locked
```

角色：

- Train：模型训练、阈值选择、reference-budget policy；
- Validation：方法选择、Gate；
- Internal Holdout：确认 Benchmark 现象和最终方法；
- Formal Test：本轮仍不访问，除非所有论文准备 Gate 完成并暂停请求用户批准。

历史 Phase X 10 queries 不进入主要统计。

## 4.2 第二数据集优先级

优先按本地数据完整性和历史暴露情况选择：

```text
1. GlaS official train split
2. MoNuSeg official data
3. DDR train/validation 中一个冻结 lesion intent
```

要求：

- 先生成数据可用性与暴露审计；
- 不访问 GlaS testA/testB；
- 不访问 DDR test；
- 不将历史已观察 validation 当全新 holdout；
- 明确标记：
  ```text
  development replication
  secondary diagnostic
  confirmatory holdout
  formal test
  ```

## 4.3 第三数据集

Benchmark 论文理想状态为至少 3 个数据集。

第三数据集优先使用已有本地数据。若本地没有合适数据，生成：

```text
reports/phaseT_missing_dataset_proposal.md
```

列出：

- 推荐数据集；
- 任务类型；
- license；
- 下载规模；
- 与现有数据的互补性；
- 是否有 reference construction protocol；
- 是否需要用户授权联网。

未经用户批准不得下载。

---

# 5. WP1 — Confirmatory Benchmark

## 5.1 目标

用未观察的 MoNuSAC holdout=40，确认以下核心现象：

1. reference-induced instability；
2. mean consensus 优于 random single；
3. mean consensus 改善 tail risk；
4. instability 可由 test-time-safe disagreement features 预测。

## 5.2 Holdout 前冻结

必须冻结：

```text
reference pool
reference subset sizes
resampling seeds
prompt config
matching normalization
candidate selection
SAM config
metric definitions
catastrophic threshold
bootstrap procedure
qualitative sampling rules
cost accounting
```

创建：

```text
docs/phaseT_monusac_holdout_freeze.md
configs/frozen/phaseT_monusac_benchmark.yaml
reports/phaseT_monusac_holdout_hashes.json
```

确认 hash 后，MoNuSAC holdout 只能运行一次。

## 5.3 Reference resampling

每个 holdout query：

```text
reference subset sizes k ∈ {1, 3, 6}
fixed resampling seeds = 10
```

要求：

- reference 与 query patient/case 不重叠；
- 每个方法共享完全相同的 subsets；
- query 是统计单位；
- reference episode 不能视为独立样本。

## 5.4 冻结方法

至少评估：

```text
B1 Random Single
B2 Mean Consensus
B3 Median Consensus
B4 Trimmed Mean
B5 Disagreement-Penalized
B6 Quantile Consensus
B7 Phase S Best Learned Weighting
```

其中 B7 必须使用 Phase S 冻结模型，不得在 holdout 重训。

Prompt configs 至少包括：

```text
P20-N1
P30-N0
P50-N0
```

主配置必须在 validation 前已冻结。

## 5.5 核心指标

### Mean performance

- Dice；
- IoU；
- AJI；
- component recall；
- precision / recall；
- boundary metric，如现有实现可靠。

### Reference instability

\[
V_{\mathrm{ref}}
=
\frac{1}{N}
\sum_q
\operatorname{Var}_S[D(q,S)]
\]

\[
MAD_{\mathrm{ref}}
=
\frac{1}{N}
\sum_q
\mathbb E_{S_i,S_j}
|D(q,S_i)-D(q,S_j)|
\]

\[
J_{\mathrm{pred}}
=
\mathbb E_{S_i,S_j}
IoU(\hat M(q,S_i),\hat M(q,S_j))
\]

### Tail risk

- worst-subset Dice；
- CVaR\(_{0.1}\)；
- CVaR\(_{0.2}\)；
- catastrophic failure rate；
- best-worst reference gap。

### Cost

- reference count；
- DINOv2 matching passes；
- SAM calls；
- runtime；
- memory；
- cached / uncached cost。

## 5.6 WP1 Gate

必须同时满足：

1. 至少 50% holdout queries 的 best-worst gap ≥ 0.05；
2. mean within-query reference std ≥ 0.02；
3. `V_ref_frac ≥ 0.10`；
4. Mean consensus 比 random single：
   \[
   \Delta Dice \ge 0.010
   \]
5. paired bootstrap 95% CI 下界 > 0；
6. catastrophic failure 不增加；
7. CVaR\(_{0.1}\) 不下降；
8. prediction consistency 提高；
9. disagreement feature 与 instability：
   ```text
   Spearman |rho| >= 0.25
   ```
10. 无数据泄漏和代码轴混淆。

通过：

```text
PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED
```

失败：

```text
PHASET_WP1_MONUSAC_HOLDOUT_FAIL
```

WP1 失败后停止投稿推进，不访问 formal test。

---

# 6. WP2 — Cross-Dataset Replication

## 6.1 目标

证明 reference instability 不是 MoNuSAC 特例。

至少在两个额外数据角色上复现以下现象：

- best-worst reference gap；
- `V_ref`；
- mean consensus gain；
- tail-risk behavior；
- reference-count scaling。

## 6.2 数据审计

生成：

```text
reports/phaseT_cross_dataset_audit/
├── glas_availability.md
├── monuseg_availability.md
├── ddr_availability.md
├── exposure_matrix.csv
├── dataset_role_decision.md
└── dataset_manifest_hashes.json
```

## 6.3 统一协议

所有数据集尽可能共享：

```text
matcher = DINOv2-L
segmenter = SAM-B
SAM calls = 1
reference counts k ∈ {1, 3, 6}
aggregation = single / mean / median / best learned frozen
metrics = V_ref / MAD / J_pred / CVaR / CFR / Dice
```

允许 dataset-specific prompt config，但必须：

- 仅在该数据集 development split 冻结；
- 不能使用 formal test；
- 在报告中明确区分 universal settings 与 dataset-specific settings。

## 6.4 跨数据集 Gate

每个额外数据集单独判定。

### Directional support

满足：

1. ≥30% queries best-worst gap ≥0.05；
2. mean reference std ≥0.015；
3. mean consensus Dice 不低于 random single；
4. catastrophic failure 不增加；
5. prediction consistency不下降。

状态：

```text
PHASET_DATASET_DIRECTIONAL_SUPPORT
```

### Strong support

进一步满足：

1. mean consensus Dice ≥ random single +0.01；
2. bootstrap point estimate 为正；
3. CVaR 改善；
4. instability feature correlation ≥0.20。

状态：

```text
PHASET_DATASET_STRONG_SUPPORT
```

Benchmark 投稿最低要求：

```text
MoNuSAC holdout confirmed
+ 至少 2 个额外数据集 directional support
```

---

# 7. WP2b — Related Method Comparison

## 7.1 目标

不只比较自己设计的 B1–B7，还要验证其他 reference-guided prompting 方法是否同样存在 reference instability。

优先相关方法：

```text
GF-SAM
SynPo
ProSAM
MAUP
SAMIX
AlignSAM
Point Prompt Defender
```

不要求全部成功复现，但必须建立可审计的 comparator matrix。

## 7.2 Comparator tiers

### Tier 1：官方代码和权重可直接复用

在同一 reference-resampling protocol 下运行。

### Tier 2：官方代码存在但接口不兼容

做最小必要适配，记录改动。

### Tier 3：无可用代码或权重

只做概念比较，不得伪造数值。

### Tier 4：方法不适用于当前任务

解释不适用原因。

生成：

```text
reports/phaseT_external_baselines/
├── comparator_availability.csv
├── implementation_notes.md
├── protocol_alignment.md
├── reproduced_results.csv
├── reference_instability_results.csv
└── unsupported_comparators.md
```

## 7.3 最低外部对比要求

投稿前至少满足之一：

```text
A. 复现 2 个外部方法
B. 复现 1 个外部方法 + 3 个强内部方法家族
C. 无法复现外部方法，但提供公开协议、代码接口和清晰限制；此时投稿定位降低
```

不得因缺少复现而宣称全面优于相关工作。

---

# 8. WP3 — Risk-Aware Mitigation

现有 learned weighting 失败，因此本轮不再以“预测最优 reference 权重”为唯一方法。

WP3 注册两条更直接的实用路线：

```text
M1 — Dynamic Reference Budget
M2 — Instability-Aware Selective Prediction
```

---

# 9. M1 — Dynamic Reference Budget

## 9.1 动机

Phase S 已表明：

- instability 可以被 test-time-safe features 预测；
- k=6 mean consensus 比 k=1 更稳定、更准确；
- 但 k=6 需要更多 matching passes。

因此研究：

> 能否只在高风险 query 上使用更多 references，在低风险 query 上使用更少 references，从而达到更好的性能—成本折中？

## 9.2 动作

Policy 只选择：

```text
k ∈ {1, 3, 6}
```

每个 k 内固定使用 mean consensus。

不选择：

- 具体 reference ID；
- point subset；
- SAM calls；
- grouping；
- K+ / K-；
- segmentation model。

## 9.3 Test-time-safe features

- matching-map disagreement；
- prototype separation；
- score entropy；
- candidate dispersion；
- reference-set embedding spread；
- leave-one-reference-out map deviation；
- predicted foreground area variance；
- prompt coordinate shift；
- query/reference global similarity。

## 9.4 模型阶梯

```text
M1-0 Always k=1
M1-1 Always k=3
M1-2 Always k=6
M1-3 Handcrafted threshold
M1-4 Logistic / ordinal regression
M1-5 Gradient boosting
M1-6 Small MLP
```

禁止大型 set transformer，除非轻量模型已证明有稳定正向信号。

## 9.5 训练目标

风险—成本效用：

\[
U(q,k)
=
Dice(q,k)
-
\lambda_c C(k)
-
\lambda_r Risk(q,k)
\]

其中：

- \(C(k)\)：matching cost；
- \(Risk\)：预测 instability、catastrophic probability 或 CVaR surrogate。

预注册：

```text
λc ∈ {0, 0.01, 0.025, 0.05}
λr ∈ {0, 0.25, 0.5, 1.0}
```

只能在 Train / inner CV 选择。

## 9.6 M1 Gate

相对 `Always k=6`，满足以下一种发表价值模式：

### Cost-saving mode

1. Mean Dice 不低于 k=6 超过 0.003；
2. CVaR 不下降超过 0.005；
3. catastrophic failure 不增加；
4. 平均 reference count 至少降低 25%。

### Quality-cost mode

1. 在与 k=3 相近的平均 reference count 下：
   \[
   \Delta Dice \ge 0.0075
   \]
2. CVaR 提高；
3. catastrophic failure 不增加。

状态：

```text
PHASET_M1_DYNAMIC_BUDGET_PASS
PHASET_M1_DYNAMIC_BUDGET_NO_GAIN
```

---

# 10. M2 — Instability-Aware Selective Prediction

## 10.1 动机

即使无法提升所有 query 的 Dice，也可能可靠识别高风险病例。

模型预测：

```text
stable
unstable
catastrophic-risk
```

可采取：

- 输出正常结果；
- 使用更多 references；
- 标记低置信度；
- abstain；
- 请求人工检查。

## 10.2 标签

只在 Train 上构造：

### Instability label

```text
best-worst gap >= 0.05
```

### Catastrophic label

```text
Dice < 0.1
```

### Tail-risk label

```text
bottom 10% performance
```

## 10.3 模型

```text
M2-0 Entropy threshold
M2-1 Disagreement threshold
M2-2 Logistic regression
M2-3 Gradient boosting
M2-4 Small MLP
```

## 10.4 评价

- AUROC；
- AUPRC；
- calibration；
- ECE；
- risk-coverage curve；
- selective Dice；
- selective CFR；
- abstention rate；
- cost。

## 10.5 M2 Gate

必须满足：

1. instability AUROC ≥0.70；
2. catastrophic-risk AUPRC 明显高于 prevalence；
3. ECE ≤0.10 或经过 validation calibration 后 ≤0.08；
4. 在 80% coverage 下，CFR relative reduction ≥30%；
5. 在 80% coverage 下，mean Dice 不下降；
6. holdout 方向一致。

状态：

```text
PHASET_M2_SELECTIVE_PREDICTION_PASS
PHASET_M2_SELECTIVE_PREDICTION_NO_GAIN
```

---

# 11. WP3 路线选择

优先执行：

1. M2 failure prediction；
2. M1 dynamic budget；
3. 只有 M1/M2 均失败，才考虑新的 learned fusion。

原因：

- Phase S 已否证简单 weighting；
- instability 可预测；
- selective prediction 和动态预算更贴合现有证据；
- 方法主张更容易与 SAMIX 的 reference selection、ProSAM 的 prompt uncertainty 区分。

---

# 12. 最终方法 Holdout

M1 或 M2 在 validation 通过后：

1. 冻结模型；
2. 冻结阈值；
3. 冻结 calibration；
4. 冻结 cost definition；
5. 冻结 risk-coverage reporting；
6. 在 MoNuSAC holdout 上一次性运行。

不得根据 holdout 改模型。

---

# 13. 论文贡献最低组合

## Benchmark-only 投稿候选

必须满足：

- WP1 holdout confirmed；
- 至少 2 个额外数据集 directional support；
- 至少 1 个外部方法或充分的强内部方法比较；
- corrected metric suite；
- reference-count scaling；
- failure taxonomy；
- 公开 manifests / evaluation code；
- 完整 limitation。

## Hybrid benchmark + method 投稿候选

在上述基础上，再满足：

- M1 或 M2 holdout pass；
- 明确的 cost/risk gain；
- 跨数据集至少 directional support；
- ablation；
- calibration；
- runtime。

---

# 14. 论文核心表格

至少生成：

1. Dataset statistics；
2. Reference protocol；
3. Correct variance decomposition；
4. Single-reference instability；
5. Reference-count scaling；
6. Mean / median / robust consensus；
7. Tail-risk metrics；
8. Prediction consistency；
9. External method instability；
10. Dynamic budget；
11. Selective prediction；
12. Cost-quality Pareto；
13. Holdout confirmation；
14. Cross-dataset replication；
15. Ablations；
16. Failure taxonomy；
17. Runtime；
18. Claim registry。

---

# 15. 论文核心图

至少生成：

1. Benchmark overview；
2. Reference resampling protocol；
3. Variance decomposition；
4. Best-worst gap distribution；
5. Reference-count scaling；
6. Dice vs `V_ref`；
7. CVaR and CFR；
8. Prediction consistency；
9. Method-wise instability comparison；
10. Risk-coverage curve；
11. Dynamic budget policy；
12. Cost-quality Pareto；
13. Cross-dataset summary；
14. High-instability examples；
15. Mean-consensus success cases；
16. Mean-consensus failure cases；
17. Selective prediction failures；
18. Final method diagram。

定性图按预注册规则选，不得手工挑最好结果。

---

# 16. 统计要求

统计单位：

```text
query / case
```

禁止把 reference episode 当独立样本。

必须报告：

- mean；
- median；
- paired difference；
- paired bootstrap 95% CI；
- win/tie/loss；
- effect size；
- within-query reference std；
- `V_ref`；
- MAD；
- CVaR；
- CFR；
- risk-coverage；
- seed variability；
- runtime；
- matching passes；
- SAM calls。

多个方法比较使用 Holm correction。

---

# 17. Novelty Audit

更新：

```text
reports/phaseT_novelty_audit.md
```

必须明确与以下工作的差异：

- GF-SAM：graph grouping；
- SynPo：positive/negative point construction；
- AlignSAM：RL iterative prompting；
- Point Prompt Defender：point filtering；
- ProSAM：probabilistic prompt robustness；
- MAUP：uncertainty-aware prompt quantity；
- SAMIX：reference selection；
- few-shot segmentation；
- selective prediction；
- set aggregation。

核心差异候选：

> 本研究不只选择 reference 或生成更强 prompt，而是系统测量 reference resampling 下的预测分布、尾部风险与失败概率，并研究动态 reference budget 与 selective prediction。

---

# 18. Mock Review

生成：

```text
reports/phaseT_mock_review.md
reports/phaseT_rebuttal_plan.md
```

至少模拟：

- 一位支持审稿人；
- 一位质疑 novelty 的审稿人；
- 一位质疑数据规模和统计的审稿人；
- 一位质疑与 ProSAM/MAUP/SAMIX 区别的审稿人；
- 一位质疑只用医学数据的审稿人。

每条 criticism 必须对应：

- 当前已有证据；
- 需要补的实验；
- 无法解决的 limitation；
- 是否影响投稿。

---

# 19. 论文材料

最终生成：

```text
paper/phaseT/
├── title_candidates.md
├── abstract.md
├── contributions.md
├── introduction.md
├── related_work.md
├── method.md
├── benchmark_protocol.md
├── experiments.md
├── results.md
├── limitations.md
├── conclusion.md
├── appendix.md
├── figure_plan.md
├── table_plan.md
├── reproducibility_checklist.md
└── full_draft.md
```

论文标题候选至少包含：

```text
Reference-Induced Instability in Visual-Reference Segmentation:
A Cross-Reference Benchmark and Failure-Boundary Analysis

Beyond a Single Reference:
Measuring and Mitigating Reference-Induced Risk in Prompted Segmentation

RefShift:
A Benchmark for Reference Resampling, Tail Risk, and Selective Prompting
```

---

# 20. Publication Readiness Gate

只有以下全部满足，才允许状态：

```text
PHASET_PAPER_CANDIDATE_READY
```

条件：

1. MoNuSAC holdout confirmed；
2. 至少 2 个额外数据集 directional support；
3. 至少 1 个外部方法被统一协议复现，或提供充分理由和替代强基线；
4. benchmark metrics 实现通过测试；
5. mean consensus gain 跨数据集方向稳定；
6. failure taxonomy 完成；
7. M1 或 M2 至少一个有 validation 正信号；Benchmark-only 路线可不要求方法 pass，但必须降低投稿定位；
8. 所有主表、主图完成；
9. full paper draft 完成；
10. mock review 无致命 unresolved issue；
11. formal test 仍锁定；
12. claim registry 无违规主张。

否则使用：

```text
PHASET_BENCHMARK_NEEDS_MORE_EVIDENCE
PHASET_METHOD_NOT_READY_BENCHMARK_CONTINUES
PHASET_RESEARCH_DIRECTION_NOT_SUPPORTED
```

---

# 21. Formal Test Gate

即使 `PHASET_PAPER_CANDIDATE_READY`，也不得自动访问 formal test。

必须生成：

```text
docs/phaseU_formal_test_freeze.md
configs/frozen/phaseU_final_benchmark.yaml
reports/phaseU_test_hashes.json
reports/phaseU_statistical_plan.md
reports/phaseU_qualitative_plan.md
```

然后暂停请求用户批准。

---

# 22. 计算预算

建议上限：

| 工作包 | GPU-hours |
|---|---:|
| WP1 MoNuSAC holdout | 15 |
| WP2 GlaS/MoNuSeg/DDR | 45 |
| External baselines | 60 |
| M1 Dynamic Budget | 25 |
| M2 Selective Prediction | 20 |
| Ablations | 30 |
| Paper figures / statistics | CPU mostly |
| 总上限 | 195 |

预算节点：

```text
40 GPU-hours  → WP1/WP2 interim
90 GPU-hours  → Cross-dataset review
140 GPU-hours → Method and comparator review
195 GPU-hours → stop and request approval
```

优先复用：

- DINOv2 features；
- matching maps；
- candidate pools；
- SAM caches；
- Phase R/S manifests；
- reference subsets。

---

# 23. 必须暂停用户确认的情况

1. 需要联网下载代码、数据或权重；
2. 需要访问 formal test；
3. 需要超过 195 GPU-hours；
4. 发现 patient/case leakage；
5. holdout 已被意外读取；
6. 需要改变 preregistered holdout metric；
7. 需要加入全新主任务；
8. 外部方法 license 不允许；
9. 本地第二/第三数据集不完整；
10. 需要以结果为依据重新划分 split。

普通代码问题自行修复并记录。

---

# 24. 最终总结报告

生成：

```text
reports/phaseT_final_publication_report.md
reports/phaseT_final_publication_report.json
```

必须回答：

1. MoNuSAC holdout 是否确认 reference instability？
2. mean consensus 是否确认优于 random single？
3. corrected `V_ref`、MAD、CVaR、CFR 是多少？
4. reference count 1/3/6 的性能—成本曲线是什么？
5. 哪些额外数据集支持现象？
6. 哪些额外数据集不支持？
7. 哪些外部方法成功复现？
8. 外部方法是否同样存在 reference instability？
9. dynamic reference budget 是否有效？
10. selective prediction 是否有效？
11. 是否有 holdout-confirmed mitigation？
12. 最强自动方法是什么？
13. 最强 deterministic baseline 是什么？
14. 是否具备 Benchmark-only 投稿条件？
15. 是否具备 Hybrid paper 投稿条件？
16. 仍缺哪些实验？
17. 最适合的投稿类型是什么？
18. 推荐标题是什么？
19. 核心贡献是什么？
20. 禁止哪些主张？
21. formal test 是否值得？
22. 所有关键 artifacts 在哪里？
23. one-command reproduction 是什么？
24. 最终唯一状态是什么？

---

# 25. 最终终端回复格式

最终终端只回复：

1. Phase T final state；
2. WP1/WP2/WP3/WP4 状态；
3. MoNuSAC holdout结果；
4. cross-dataset结果；
5. external baseline结果；
6. strongest deterministic baseline；
7. strongest automatic method；
8. dynamic budget结果；
9. selective prediction结果；
10. benchmark readiness；
11. method-paper readiness；
12. recommended paper title；
13. formal test状态；
14. key artifacts；
15. one-command reproduction；
16. unresolved blockers。

---

# 26. 现在开始执行

严格顺序：

1. 阅读 Phase X/Y/R/S 全部 artifacts；
2. 创建 `phaseT-publication-completion` 分支；
3. 创建 publication charter 与 claim registry；
4. 冻结 MoNuSAC holdout benchmark protocol；
5. 运行 WP1；
6. WP1 失败则停止；
7. 审计 GlaS/MoNuSeg/DDR；
8. 运行至少两个额外数据集；
9. 建立 external comparator matrix；
10. 复现可用外部方法；
11. 训练 M2 selective prediction；
12. 训练 M1 dynamic reference budget；
13. validation pass 后冻结最终 mitigation；
14. 在合法 holdout 上一次性确认；
15. 完成跨数据集结果；
16. 生成全部统计、表格和图；
17. 更新 novelty audit；
18. 完成 mock review；
19. 生成 full paper draft；
20. 执行 Publication Readiness Gate；
21. 不访问 formal test；
22. 输出唯一最终状态。

不要再以“训练出更复杂模型”为成功标准。

真正成功标准是：

> 用未观察 holdout 和多个真实数据集确认 reference-induced instability，建立可复现的评估协议，证明简单 mean consensus 的强基线价值，并至少提供一种风险感知的实用缓解方式或可靠 failure detection，使整项研究形成清晰、可信、可投稿的 Benchmark 或 Hybrid 论文。
