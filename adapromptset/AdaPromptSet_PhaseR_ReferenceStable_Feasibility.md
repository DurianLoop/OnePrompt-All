# AdaPromptSet / RefShift 下一阶段执行指令
## Phase R — Real-Data Feasibility Study for Reference-Stable Prompting

> 用途：将本文档完整交给 Codex（下文简称 CX），开启 `/goal` 模式连续执行。
> 本轮性质：真实数据可行性验证，不是正式论文终稿，不是 formal test，不允许为了得到正结果反复调参。
> 本轮核心问题：能否通过多参考共识、参考可靠性估计和稳定性约束，降低 reference-induced prompt instability，并在真实数据上提高跨参考表现、最坏参考表现与灾难性失败鲁棒性？

---

# 0. 项目上下文与本轮起点

工作目录：

```text
D:\SAM
```

项目目录：

```text
D:\SAM\adapromptset
```

建议新分支：

```text
phaseR-reference-stable-feasibility
```

已有关键历史结论：

1. 旧 12-action bank 严重不足；
2. 扩展 action bank 在同一 reference 条件下显示较大 raw oracle gap；
3. 但 Phase Y 证明：
   - raw oracle gap = 0.0669；
   - stable cross-reference oracle gap = 0.0061；
   - raw search Dice = 0.5947；
   - cross-reference search Dice = 0.4105；
   - Search Oracle 对 reference 明显过拟合；
4. 当前主导瓶颈不是 action-bank size，而是：
   ```text
   REFERENCE_INSTABILITY
   ```
5. 当前最终状态：
   ```text
   PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY
   ```

本轮不得重新训练旧 Bank Policy，也不得把 Search Oracle 当作自动方法。

---

# 1. 本轮唯一主张候选

本轮只验证以下命题：

> Reference-stable aggregation and stability-aware prompt generation can reduce performance variance across valid references and improve cross-reference segmentation reliability under the same SAM-call budget.

中文：

> 在相同 SAM 调用预算下，多参考稳定聚合与稳定性感知提示生成，能否降低不同合法参考图像带来的性能波动，并提高跨参考分割可靠性。

本轮不验证：

- 最终顶会 SOTA；
- 跨模型泛化；
- SAM2 / MedSAM；
- 强化学习；
- point-subset policy；
- formal test；
- universal promptability；
- 端到端大模型。

---

# 2. 必须先回答的三个科学问题

## RQ1：Reference instability 是否稳定存在

对于同一 query、不同合法 references：

```text
same query
+ Reference Set A
→ prediction A

same query
+ Reference Set B
→ prediction B
```

本轮先确认：

- Dice 差异；
- action rank 差异；
- candidate overlap；
- matching-map disagreement；
- catastrophic failure 差异；
- 最优 K+ / K- 差异。

如果 reference instability 在新数据上不稳定存在，则本路线停止。

## RQ2：简单 consensus 是否已足够

在不训练新模型的前提下，比较：

- single reference；
- mean consensus；
- median consensus；
- trimmed mean；
- minimum-risk consensus；
- disagreement-penalized consensus。

如果一个简单确定性聚合已经显著改善稳定性，则说明该方向可行，而且方法不必立刻复杂化。

## RQ3：是否存在可学习的 reference reliability signal

只有简单 consensus 通过后，才训练轻量 reference reliability estimator，回答：

> 是否可以仅根据 test-time-safe reference-query evidence，预测哪张 reference 更可靠或如何加权多个 references？

---

# 3. 本轮数据角色

## 3.1 MoNuSAC：主真实数据可行性集

沿用 Phase Y fresh split：

```text
Policy Train Queries      = 80
Policy Validation Queries = 41
Confirmatory Holdout      = 40
```

要求：

- 历史 Phase X 10 queries 不进入主统计；
- Phase Y holdout 仍保持未观察；
- 本轮开发只使用 train=80 和 validation=41；
- 只有 Gate R3 通过后才允许冻结并一次性运行 holdout=40；
- 不访问 formal test。

MoNuSAC 是本轮主数据。

## 3.2 GlaS：次级形态跨域诊断

只使用：

```text
official train 85
```

已有 split：

```text
55 development
30 strict validation
```

用途：

- 作为 pathology morphology shift；
- 检查 reference-stable aggregation 是否在 gland morphology 上方向一致；
- 不训练主模型；
- 不访问 testA/testB；
- 由于 GlaS validation 已在历史阶段观察，只能作为 secondary diagnostic，不作为 confirmatory evidence。

## 3.3 DDR：可选第三域，仅做 readiness

本轮默认不运行 DDR。

只有当 MoNuSAC Gate R2 通过，并且现有 DDR train/valid protocol 可直接复用时，才允许：

- 运行一个 lesion intent；
- 只做 fixed consensus feasibility；
- 不做完整方法训练；
- 不访问 DDR test。

---

# 4. 开始前必须创建的文档

运行任何新推理前，创建：

```text
docs/phaseR_goal_read_receipt.md
docs/phaseR_problem_statement.md
docs/phaseR_data_roles.md
docs/phaseR_reference_resampling_protocol.md
docs/phaseR_consensus_methods.md
docs/phaseR_reference_reliability_protocol.md
docs/phaseR_statistics_protocol.md
docs/phaseR_claim_registry.md
docs/phaseR_test_lock_notice.md
reports/phaseR_gate_status.md
reports/phaseR_compute_ledger.csv
reports/phaseR_protocol_snapshot.json
reports/phaseR_input_hashes.json
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
phaseR: preregister reference-stable prompting feasibility
```

在此 commit 前禁止新推理。

---

# 5. Reference Resampling Protocol

## 5.1 每个 query 的 references

每个 MoNuSAC query 使用：

```text
6 个 train-only references
```

拆成：

```text
Reference Set A = 3
Reference Set B = 3
```

要求：

- A/B 完全不重叠；
- reference 与 query 不同 patient/case；
- reference 只来自允许 train reference pool；
- 固定 seed；
- 不按结果挑 reference；
- 所有方法共享相同 references；
- query 是统计单位；
- reference episode 不是独立样本。

## 5.2 额外 resampling

为了估计 reference distribution，不只使用一组 A/B。

在 Policy Train Queries 上，预注册：

```text
5 个不同 reference resampling seeds
```

每个 seed 都产生 6 references，并拆成 A/B。

用途：

- 估计 instability；
- 训练 reference reliability；
- 不用于 validation 模型选择。

Validation Queries 只使用一个固定、预注册的 A/B split，防止多次观察。

---

# 6. 冻结的基础 pipeline

本轮固定：

```text
matcher = DINOv2-L
segmenter = SAM-B
SAM calls = 1
prompt group = 1
multimask rule = frozen
resize = frozen
mask threshold = frozen
postprocess = frozen
```

主 prompt baseline 固定为：

```text
K+ ∈ {20, 30, 50}
K- ∈ {0, 1}
positive selector = TOPK
negative selector = NONE or BOTTOMK
grouping = ONE_GROUP
calls = 1
```

不重新开放：

- KMeans；
- graph grouping；
- multi-call；
- one-point-per-group；
- mask fusion；
- Search Oracle；
- point swap。

本轮只研究 reference aggregation 对固定强 dense prompting 的影响。

---

# 7. 必须建立的 Reference-Stability Baselines

## B0. Single Reference

分别使用每一张 reference 单独生成 matching map 和 prompt。

报告：

- mean over references；
- best reference；
- worst reference；
- reference std；
- catastrophic failure rate。

`best reference` 只能作为 privileged diagnostic，不可部署。

## B1. Random Single Reference

从 6 个 references 中按固定 seed 随机选 1 张。

作为可部署但高方差基线。

## B2. Mean Consensus

对每个 reference 的 contrast map 做相同归一化后：

\[
C_{mean}(x)=\frac{1}{m}\sum_{r=1}^{m}C_r(x)
\]

## B3. Median Consensus

逐位置中位数：

\[
C_{median}(x)=\operatorname{median}_r C_r(x)
\]

## B4. Trimmed Mean

每个像素删除最高和最低 reference score 后求均值：

\[
C_{trim}(x)=\operatorname{mean}
\left(
C_r(x)\setminus
\{\max C_r(x),\min C_r(x)\}
\right)
\]

仅在 references ≥ 5 时使用。

## B5. Disagreement-Penalized Consensus

计算：

\[
\mu(x)=\frac{1}{m}\sum_r C_r(x)
\]

\[
\sigma(x)=
\sqrt{
\frac{1}{m}\sum_r(C_r(x)-\mu(x))^2
}
\]

稳定图：

\[
C_{stable}(x)=\mu(x)-\beta\sigma(x)
\]

预注册：

```text
β ∈ {0.25, 0.5, 1.0}
```

β 只能在 train 上选择。

## B6. Quantile Consensus

要求至少一定比例 references 支持该位置：

\[
C_q(x)=Q_\tau(\{C_r(x)\})
\]

预注册：

```text
τ ∈ {0.50, 0.67, 0.83}
```

## B7. Reliability-Weighted Mean

为每个 reference 计算 train-time-safe reliability score：

\[
w_r=
\frac{\exp(g_r/T)}
{\sum_j\exp(g_j/T)}
\]

融合：

\[
C_{weighted}(x)=\sum_rw_rC_r(x)
\]

B7 只有 Gate R2 通过后才能训练。

---

# 8. Reference Reliability Features

只允许使用 test-time-safe 特征。

## 8.1 Reference-only features

- reference foreground area ratio；
- foreground component count；
- mask compactness；
- mask fragmentation；
- reference embedding mean/std；
- foreground/background prototype separation；
- reference stain/intensity statistics。

## 8.2 Reference-query pair features

- global embedding cosine；
- foreground prototype max/mean score；
- background prototype max/mean score；
- contrast mean/std/entropy；
- top-score decay；
- peak count；
- candidate dispersion；
- estimated foreground ratio；
- map smoothness；
- high-score connected components；
- cross-reference disagreement；
- candidate overlap with other references。

## 8.3 禁止特征

- query GT；
- query component count GT；
- query Dice；
- oracle reference ID；
- best-reference label from validation；
- formal-test information；
- search trajectory。

---

# 9. 阶段 R0：只读历史证据校正

不运行新 SAM。

读取 Phase Y：

```text
reports/phaseY_*
manifests/phaseY/*
configs/actions/*
```

输出：

```text
reports/phaseR_historical_audit/
├── phaseY_reference_instability_summary.csv
├── raw_vs_cross_reference_decomposition.csv
├── symmetric_fixed_baseline.csv
├── symmetric_stable_oracle.csv
├── action_tie_analysis.csv
├── reference_pair_difficulty.csv
├── phaseY_claim_correction.md
└── phaseR_hypothesis_summary.md
```

必须确认：

1. stable fixed baseline 是否对称；
2. stable oracle gap 的精确定义；
3. diff-from-fixed rate 的定义；
4. Search cross-reference 是坐标复用还是 rank 复用；
5. compact bank 是否实际缺失；
6. 哪些历史结论只适用于 full 210-action bank。

---

# 10. Gate R1：Reference Instability Replication

在 MoNuSAC train=80 上，使用 5 个 reference resampling seeds。

固定 prompt 配置先用：

```text
P20-N1
P30-N0
P50-N0
```

每个 query、每个 seed、每个 reference 分别运行。

必须报告：

- per-reference Dice；
- reference std；
- worst-reference Dice；
- best-reference Dice；
- best-worst gap；
- catastrophic failure；
- matching-map disagreement；
- candidate Jaccard；
- prompt-coordinate overlap；
- prediction-mask Jaccard；
- reference ranking stability。

## Gate R1 通过条件

必须同时满足：

1. 至少 30% train queries 的 best-worst Dice gap ≥ 0.05；
2. mean reference std ≥ 0.02；
3. 不同 resampling seeds 下方向一致；
4. 至少一种 prompt 配置表现出稳定的 reference sensitivity；
5. instability 不是由代码错误、坐标错误或 reference-query leakage 导致。

通过：

```text
PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED
```

失败：

```text
PHASER_GATE_R1_REFERENCE_INSTABILITY_NOT_REPLICATED
```

Gate R1 失败后停止本路线。

---

# 11. Gate R2：Deterministic Consensus Feasibility

只在 MoNuSAC train=80 上选择 β、τ 和 prompt config。

然后冻结配置，在 validation=41 上评估：

```text
B1 Random Single
B2 Mean
B3 Median
B4 Trimmed Mean
B5 Disagreement-Penalized
B6 Quantile
```

## 11.1 主要比较

主比较：

```text
Best Deterministic Consensus
vs
Random Single Reference
```

次比较：

```text
Best Deterministic Consensus
vs
Mean Consensus
```

## 11.2 主要指标

### Mean Dice

跨 query 平均 Dice。

### Worst-Reference Dice

固定 consensus 方法不依赖单一 reference，和单-reference 方法的最坏 reference 比较。

### Reference Variance Reduction

\[
VR
=
1-
\frac{
\operatorname{Var}(D_{consensus})
}{
\operatorname{Var}(D_{single})
}
\]

### Catastrophic Failure Reduction

\[
CFR
=
1-
\frac{
F_{consensus}
}{
F_{single}
}
\]

### Cross-Set Stability

用 A 生成 consensus，用 B 生成 consensus，比较：

- Dice difference；
- prompt overlap；
- prediction overlap。

## Gate R2 通过条件

至少一个 deterministic consensus 同时满足：

1. 相对 Random Single：
   \[
   \Delta Dice \ge 0.010
   \]
2. paired bootstrap 95% CI 下界 > 0；
3. reference variance 至少下降 25%；
4. catastrophic failure 不增加；
5. A/B prediction-mask overlap 提高；
6. 仍为 1 SAM call；
7. 不使用 query GT；
8. 3 个 prompt configs 中至少 2 个方向一致。

通过：

```text
PHASER_GATE_R2_DETERMINISTIC_CONSENSUS_PASS
```

失败：

```text
PHASER_GATE_R2_CONSENSUS_NO_GAIN
```

---

# 12. Gate R3：Reference Reliability Learnability

只有 Gate R2 通过才训练。

## 12.1 训练标签

在 train queries 上，为每张 reference 构造：

```text
reference utility target
```

可使用：

\[
y_r=Dice(q,r)
\]

但必须只在 train 上生成。

Validation 不生成训练标签。

## 12.2 模型阶梯

### M0. Uniform weights

等权平均。

### M1. Handcrafted reliability score

例如：

\[
g_r=
\text{prototype separation}
-
\lambda\text{map entropy}
\]

### M2. Linear regression

预测 reference utility。

### M3. Gradient boosting

轻量非线性模型。

### M4. Small MLP

只在 M2/M3 不足时运行。

禁止：

- 大型 transformer；
- RL reference selector；
- SAMIX-style sequential selection；
- validation GT supervision。

## 12.3 训练目标

主要目标：

\[
L_{utility}=(\hat y_r-y_r)^2
\]

稳定性正则：

\[
L_{rank}
=
\max(0,1-(\hat y_i-\hat y_j)\operatorname{sign}(y_i-y_j))
\]

最终：

\[
L=L_{utility}+\lambda L_{rank}
\]

## 12.4 Validation 比较

```text
Reliability-Weighted Consensus
vs
Best Deterministic Consensus
vs
Mean Consensus
vs
Random Single
```

## Gate R3 通过条件

必须同时满足：

1. Weighted Consensus 比 best deterministic：
   \[
   \Delta Dice \ge 0.005
   \]
2. 比 Random Single：
   \[
   \Delta Dice \ge 0.015
   \]
3. paired bootstrap CI 点估计为正；
4. reference utility prediction：
   ```text
   Spearman rho >= 0.30
   ```
5. worst-reference risk 降低；
6. catastrophic failure 不增加；
7. 3 seeds 方向一致；
8. 权重不塌缩到永远选择同一 reference position；
9. 不使用 query GT。

通过：

```text
PHASER_GATE_R3_REFERENCE_WEIGHTING_PASS
```

失败：

```text
PHASER_GATE_R3_REFERENCE_WEIGHTING_NOT_LEARNABLE
```

---

# 13. Gate R4：Confirmatory Holdout

只有 R2 或 R3 通过，才冻结最终方法。

冻结：

```text
reference count
reference sampling
aggregation rule
β / τ
reliability model
prompt config
feature normalization
SAM config
statistics plan
qualitative selection
```

创建：

```text
docs/phaseR_holdout_freeze.md
configs/frozen/phaseR_refstable_method.yaml
reports/phaseR_holdout_hashes.json
```

然后一次性运行 holdout=40。

## Holdout 通过条件

相对 Random Single：

1. Dice 提升 ≥ 0.0075；
2. paired bootstrap point estimate > 0；
3. reference variance 至少下降 20%；
4. catastrophic failure 不增加；
5. worst-query tail 提升；
6. 1 SAM call；
7. 不允许 holdout 后调参。

通过：

```text
PHASER_REFERENCE_STABILITY_CONFIRMED
```

失败：

```text
PHASER_HOLDOUT_CONFIRMATION_FAIL
```

---

# 14. GlaS Secondary Diagnostic

只有 MoNuSAC R2 通过后运行。

在 GlaS 55/30 上：

- 55 只用于选择同一 β/τ 中的一个冻结配置；
- 30 只评价；
- 不训练 reliability model；
- 不访问 testA/testB。

比较：

```text
Random Single
Mean
Median
Disagreement-Penalized
Quantile
```

只报告方向：

- Dice；
- reference std；
- worst-reference Dice；
- catastrophic failure；
- cross-reference prompt overlap。

GlaS 不决定主 Gate。

---

# 15. 必须生成的表格

1. Dataset roles；
2. Reference resampling protocol；
3. Historical Phase Y correction；
4. Single-reference instability；
5. Matching-map disagreement；
6. Candidate overlap；
7. Prompt-coordinate overlap；
8. Prediction overlap；
9. Deterministic consensus comparison；
10. Variance reduction；
11. Catastrophic failure；
12. Reference utility prediction；
13. Weighted consensus；
14. Holdout confirmation；
15. GlaS diagnostic；
16. Calls/runtime；
17. Gate decisions；
18. Claim boundaries。

---

# 16. 必须生成的图

1. per-query best-worst reference gap；
2. reference std distribution；
3. matching-map disagreement；
4. candidate Jaccard；
5. single vs mean consensus paired scatter；
6. single vs stable consensus paired scatter；
7. variance reduction；
8. worst-reference improvement；
9. catastrophic failure reduction；
10. β sweep；
11. quantile sweep；
12. reliability prediction calibration；
13. predicted vs actual reference utility；
14. reference weights distribution；
15. cross-set prompt overlap；
16. cross-set prediction overlap；
17. MoNuSAC success/failure cases；
18. GlaS secondary diagnostic；
19. gate flow；
20. final method diagram。

定性案例按预注册规则选择：

- 最大 best-worst gap；
- 最大 consensus gain；
- 最大 consensus loss；
- catastrophic failure；
- high disagreement；
- low disagreement。

---

# 17. 计算预算

先做 dry-run estimate。

建议上限：

| Stage | 上限 |
|---|---:|
| Historical audit | CPU only |
| R1 resampling | 25 GPU-hours |
| R2 deterministic consensus | 20 GPU-hours |
| R3 reliability models | 20 GPU-hours |
| Holdout | 10 GPU-hours |
| GlaS diagnostic | 10 GPU-hours |
| 统计与图表 | CPU mostly |
| 总上限 | 85 GPU-hours |

预算节点：

```text
25 GPU-hours → R1 interim
50 GPU-hours → R2/R3 review
70 GPU-hours → holdout readiness
85 GPU-hours → stop and request approval
```

优先复用：

- DINOv2 features；
- per-reference matching maps；
- candidate pools；
- identical SAM prompt cache；
- Phase Y query/reference manifests。

---

# 18. 自动状态机

本轮只能使用：

```text
PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED
PHASER_GATE_R1_REFERENCE_INSTABILITY_NOT_REPLICATED
PHASER_GATE_R2_DETERMINISTIC_CONSENSUS_PASS
PHASER_GATE_R2_CONSENSUS_NO_GAIN
PHASER_GATE_R3_REFERENCE_WEIGHTING_PASS
PHASER_GATE_R3_REFERENCE_WEIGHTING_NOT_LEARNABLE
PHASER_REFERENCE_STABILITY_CONFIRMED
PHASER_HOLDOUT_CONFIRMATION_FAIL
PHASER_GLAS_DIRECTIONAL_SUPPORT
PHASER_GLAS_NO_TRANSFER
PHASER_BLOCKED_INSUFFICIENT_REFERENCES
PHASER_BLOCKED_TECHNICAL
```

---

# 19. 必须暂停询问用户的情况

只有以下情况暂停：

1. reference pool 不足以为每 query 提供 6 个不重叠 references；
2. 需要访问 formal test；
3. 需要联网下载数据或模型；
4. 需要修改 DINOv2-L；
5. 需要修改 SAM-B；
6. 需要超过 85 GPU-hours；
7. 发现 patient/case leakage；
8. 需要新增未预注册 aggregation family；
9. 需要改变 Gate threshold；
10. holdout 已被意外读取。

普通代码问题自行修复并记录。

---

# 20. 最终允许的结论

## 只有 R1 通过

允许：

> Valid references induce substantial segmentation variability under the current frozen matching-and-prompting pipeline.

## R2 通过

允许：

> Deterministic multi-reference consensus reduces reference-induced variance and improves cross-reference reliability under one SAM call.

## R3 通过

允许：

> Test-time-safe reference reliability estimation further improves over uniform consensus.

## Holdout 通过

允许：

> Reference-stable prompting confirms on a fresh internal holdout and is ready for a separately frozen formal-test protocol.

## 永久禁止

- “reference instability solved”；
- “universal across datasets”；
- “SOTA”；
- “formal test success”；
- “ProSAM/MAUP/SAMIX fully outperformed”；
- “reference weighting is model agnostic”；
- “mean Dice alone proves robustness”。

---

# 21. 最终报告

生成：

```text
reports/phaseR_final_report.md
reports/phaseR_final_report.json
```

必须回答：

1. reference instability 是否在 train resampling 中复现？
2. best-worst gap 是多少？
3. reference std 是多少？
4. 哪种 prompt config 最稳定？
5. mean / median / trimmed / penalized / quantile 谁最好？
6. Dice 提升是多少？
7. variance reduction 是多少？
8. catastrophic failure 是否下降？
9. A/B prompt overlap 是否提高？
10. A/B prediction overlap 是否提高？
11. reference utility 是否可预测？
12. weighted consensus 是否优于 deterministic？
13. holdout 是否确认？
14. GlaS 是否方向一致？
15. 方法是否仍为 1 SAM call？
16. formal test 是否继续锁定？
17. 最终科学状态是什么？
18. 下一步是否值得进入完整方法论文？

---

# 22. 最终终端输出格式

CX 最终终端只回复：

1. Phase R final state；
2. MoNuSAC R1/R2/R3/R4 状态；
3. strongest single-reference baseline；
4. strongest deterministic consensus；
5. strongest weighted consensus；
6. Dice gain；
7. variance reduction；
8. catastrophic failure change；
9. holdout result；
10. GlaS diagnostic；
11. formal test status；
12. key artifacts；
13. one-command reproduction；
14. unresolved blockers。

---

# 23. 现在开始执行

严格顺序：

1. 阅读 Phase Y 全部 artifacts；
2. 创建 `phaseR-reference-stable-feasibility` 分支；
3. 完成 Phase Y 只读校正；
4. 审计 reference pool；
5. 创建 train resampling 和 validation A/B manifests；
6. 冻结 prompt configs；
7. 冻结 consensus methods；
8. 冻结 β、τ grid；
9. 冻结 statistics 和 Gate；
10. 生成 hashes；
11. 运行 focused tests；
12. 提交 preregistration commit；
13. 运行 Gate R1；
14. R1 通过后运行 deterministic consensus；
15. 执行 Gate R2；
16. R2 通过后训练 reliability estimator；
17. 执行 Gate R3；
18. R2 或 R3 通过后冻结 holdout；
19. 一次性运行 holdout；
20. 仅在 MoNuSAC R2 通过后运行 GlaS secondary diagnostic；
21. 生成全部表格、图和最终报告；
22. 不访问 formal test；
23. 写出唯一 Phase R final state。

本轮的关键不是立刻训练一个复杂模型。

本轮的关键是先用真实数据证明：

> 参考图像不稳定性是一个可重复、可量化、可被简单共识方法缓解的真实问题，而不是 Phase Y 中偶然出现的统计现象。
