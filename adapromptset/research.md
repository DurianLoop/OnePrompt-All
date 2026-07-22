# AdaPromptSet / RefShift 多方向科研探索执行指令
## Phase S — Multi-Track Research Exploration for Publishable Reference-Robust Segmentation

> **用途**：将本文档完整交给 Codex（下文简称 CX），开启 `/goal` 模式连续执行。  
> **本轮目标**：不再只沿单一路线机械推进，而是在严格控制数据泄漏、验证集过拟合和事后调参的前提下，对多个潜在方向进行系统探索、假设验证、模型训练、门控裁决和论文价值评估，最终形成完整科研总结报告。  
> **原则**：尽可能发现可发表的真实科学问题、稳定现象或有效方法，但不得承诺一定得到正结果，也不得通过降低门槛、反复观察 holdout 或扩大主张来制造“成功”。

---

# 0. 项目位置与当前科学状态

工作目录：

```text
D:\SAM
```

项目目录：

```text
D:\SAM\adapromptset
```

建议从 Phase R 最新稳定 commit 创建：

```text
phaseS-multitrack-reference-robust-research
```

必须保留的历史事实：

- Phase X 证明旧 12-action bank 过窄，但其高 oracle 仅是探索性 in-reference 上界；
- Phase Y fresh split 为 train=80 / validation=41 / holdout=40，stable cross-reference oracle gap 仅约 0.0061，Search Oracle 对 reference 明显过拟合；
- Phase R 确认 reference instability：88.75% queries 的 best-worst gap ≥0.05，mean reference std≈0.0737；
- Phase R 的 mean consensus 将 Dice 从约0.4297提高至0.4756，并降低 catastrophic failure，但预注册 variance gate 未通过；
- 当前解释应为：

```text
Reference instability confirmed
+ deterministic consensus utility gain confirmed
+ stability reduction unresolved
+ learned reliability scientifically motivated
```

不得把 Phase R 简化成“方向彻底失败”。

---

# 1. 本轮总体科研目标

本轮要回答：

> 在 reference-guided segmentation 中，参考图像选择、参考集合组成和匹配证据不稳定，是否可以通过正确的稳定性指标、多参考建模、可靠性估计、风险优化或直接分割得到缓解，并形成具有独立研究价值、可复现、可投稿的技术或基准贡献？

英文：

> Can reference-induced instability in visual-reference segmentation be measured correctly, predicted from test-time-safe evidence, and mitigated through robust aggregation, reliability modeling, risk-aware prompting, or direct reference-conditioned segmentation?

允许的最终成果：

1. 方法论文；
2. Benchmark / diagnostic paper；
3. Benchmark + mitigation method 的混合论文；
4. 具有一般化发现的 failure-boundary paper。

---

# 2. 本轮禁止事项

禁止：

- 只看 mean Dice，不看 within-query reference variance、tail risk 和 failure；
- 将 cross-query variance 误写成 reference-induced variance；
- 将 Search Oracle 当作可部署方法；
- 将 GT-guided search labels 直接训练 policy；
- 继续无限扩大 K、grouping 或 action bank；
- 在同一 validation 上反复增加模块；
- holdout 被观察后继续调参；
- 只用 MoNuSAC 得出“普适”结论；
- 使用 formal test 决定路线；
- 将生成 PDF 当作投稿完成。

---

# 3. 四条注册路线

```text
Track A — Metric Correction and Stability Benchmark
Track B — Learned Reference Reliability and Robust Aggregation
Track C — Set-Conditioned Robust Prompting
Track D — Direct Reference-Conditioned Segmentation
```

执行优先级：

1. Track A 必须先完成；
2. Track B 为首选方法路线；
3. Track C 仅在 B 有可学习信号或达到平台期后进行；
4. Track D 是 prompt interface 受限时的 fallback；
5. 任一路线通过强 Gate 后，可以暂停较弱路线，集中形成论文；
6. 不得四条路线无限展开。

---

# 4. 开始前必须创建的控制文件

运行新实验前创建：

```text
docs/phaseS_goal_read_receipt.md
docs/phaseS_research_charter.md
docs/phaseS_track_registry.md
docs/phaseS_hypothesis_registry.md
docs/phaseS_data_roles.md
docs/phaseS_exposure_map.md
docs/phaseS_metric_definitions.md
docs/phaseS_model_registry.md
docs/phaseS_claim_registry.md
docs/phaseS_publication_readiness.md
docs/phaseS_test_lock_notice.md

reports/phaseS_program_status.md
reports/phaseS_program_status.json
reports/phaseS_compute_ledger.csv
reports/phaseS_decision_tree.md
reports/phaseS_risk_register.md
reports/phaseS_protocol_snapshot.json
reports/phaseS_input_hashes.json
```

更新但不得覆盖：

```text
docs/protocol_history.md
docs/decision_log.md
docs/data_exposure_log.md
checklist.md
```

创建 preregistration commit：

```text
phaseS: preregister multitrack reference-robust research program
```

该 commit 前不得新训练或推理。


# 5. 数据角色与冻结边界

## 5.1 MoNuSAC

沿用：

```text
Train = 80
Validation = 41
Holdout = 40
Formal test = locked
```

- Train：训练、参数选择、内部交叉验证；
- Validation：模型选择与 Gate；
- Holdout：仅最终冻结方法使用一次；
- Formal test：本轮不访问；
- 历史 Phase X 10 queries 不进入主要统计。

## 5.2 GlaS

只使用：

```text
official train 55 / strict validation 30
```

作为 secondary pathology morphology diagnostic；testA/testB 锁定。

## 5.3 DDR

如现有 train/valid 可直接复用：

- 只使用 train / valid；
- 选择一个 lesion intent 做跨域 feasibility；
- test 锁定；
- 不得根据 DDR valid 改写 MoNuSAC 方法。

## 5.4 额外数据

仅当主路线已在 MoNuSAC validation 或 holdout 上成立，且跨域验证成为必要条件时，生成：

```text
missing_data_proposal.md
```

未经用户批准不得联网下载。

---

# 6. Track A — Metric Correction and Stability Benchmark

## 6.1 目标

确认 Phase R 的 variance 指标是否测量了正确对象，并建立可复用的 reference-instability metric suite。

## 6.2 必须区分的指标

### Cross-query variance

\[
V_{query}=\operatorname{Var}_{q}[D(q)]
\]

衡量病例难度差异，不等于 reference instability。

### Within-query reference variance

\[
V_{ref}=rac{1}{N}\sum_q\operatorname{Var}_{S}[D(q,S)]
\]

其中 \(S\) 是不同 reference subsets。

### Pairwise reference deviation

\[
MAD_{ref}=rac{1}{N}\sum_q\mathbb E_{S_i,S_j}|D(q,S_i)-D(q,S_j)|
\]

### Prediction consistency

\[
J_{pred}=\mathbb E_{S_i,S_j}IoU(\hat M(q,S_i),\hat M(q,S_j))
\]

### Tail risk

- worst-subset Dice；
- 10% CVaR；
- catastrophic failure；
- worst-reference gap。

## 6.3 Phase R 只读审计

读取：

```text
reports/phaseR_*
configs/frozen/phaseR_*
manifests/phaseY/*
manifests/phaseR/*
```

生成：

```text
reports/phaseS_trackA_metric_audit/
├── variance_code_trace.md
├── phaseR_var_single_definition.md
├── phaseR_var_consensus_definition.md
├── cross_query_vs_within_query_variance.csv
├── corrected_reference_variance.csv
├── corrected_pairwise_deviation.csv
├── prediction_consistency.csv
├── cvar_tail_risk.csv
├── catastrophic_failure_decomposition.csv
├── method_cost_decomposition.csv
└── phaseR_interpretation_addendum.md
```

必须回答：

1. `var_single` 和 `var_consensus` 的计算轴；
2. 是否把 query variability 当作 reference variability；
3. B1/B2 reference count 是否不同；
4. matcher cost 是否被忽略；
5. B2 mean 的 beta/tau 字段是否为 exporter 错误；
6. Phase R 科学结论需要怎样修正。

## 6.4 Reference Resampling Benchmark

MoNuSAC Train：

```text
reference subset sizes k ∈ {1, 3, 6}
resampling seeds = 10
```

Validation 使用固定、预注册的10个 subset seeds。

比较：

- random single；
- mean；
- median；
- trimmed mean；
- disagreement-penalized；
- quantile。

输出：

```text
reports/phaseS_trackA_benchmark/
├── per_query_per_subset_metrics.csv
├── within_query_reference_variance.csv
├── pairwise_reference_deviation.csv
├── worst_subset_metrics.csv
├── cvar_metrics.csv
├── prediction_consistency.csv
├── reference_count_scaling.csv
├── cost_scaling.csv
└── benchmark_summary.md
```

## 6.5 Track A Gate

必须满足：

1. corrected metrics 下 reference instability 仍显著；
2. 至少30% validation queries 的 best-worst gap ≥0.05；
3. mean within-query reference std ≥0.02；
4. metric definitions 与代码一致；
5. reference subset size 趋势可解释；
6. 至少一个稳定性 metric 与 catastrophic failure 有关联；
7. 至少一个 test-time-safe disagreement feature 能预测不稳定性，Spearman |rho|≥0.25。

状态：

```text
PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID
PHASES_TRACK_A_METRIC_OR_PHENOMENON_NOT_STABLE
```

Track A 失败时，停止 reference-stability 主线并评估 Track D。


# 7. Track B — Learned Reference Reliability and Robust Aggregation

## 7.1 目标

学习每张 reference 对当前 query 的可靠性：

\[
\hat u(q,R_r)
\]

最终融合：

\[
C_{weighted}(x)=\sum_r w_rC_r(x)
\]

## 7.2 Train-only labels

只在 Train=80 生成：

### Utility label

\[
u(q,R_r)=Dice(\hat M(q,R_r),M_q)
\]

### Pairwise label

\[
y_{ij}=\mathbf 1[u_i>u_j]
\]

### Harmfulness label

\[
h_r=\mathbf 1[u(q,R_r)<u_{mean}(q)-\delta]
\]

```text
δ ∈ {0.02, 0.05}
```

Validation 不生成训练标签。

## 7.3 Test-time-safe features

### Reference-only

- reference foreground ratio；
- component count；
- compactness；
- fragmentation；
- foreground/background prototype separation；
- stain/intensity statistics；
- embedding statistics。

### Reference-query

- global embedding similarity；
- foreground/background matching statistics；
- contrast entropy；
- top-score decay；
- peak density；
- candidate dispersion；
- candidate agreement；
- predicted foreground ratio；
- map smoothness；
- reference-set disagreement。

### Leave-one-reference-out features

不使用 GT：

- 去掉某 reference 后 consensus map 的变化；
- candidate overlap change；
- prompt coordinate shift；
- matching-map proxy 的变化。

## 7.4 模型阶梯

按顺序训练：

1. Uniform Mean；
2. Handcrafted Score；
3. Linear / Ridge Regression；
4. Pairwise Ranking；
5. Gradient Boosting；
6. Small MLP，仅当前述有正向信号时。

禁止大型 Transformer、RL reference selector、validation GT、test-time SAM search。

## 7.5 权重与风险目标

### Softmax weighting

\[
w_r=rac{\exp(\hat u_r/T)}{\sum_j\exp(\hat u_j/T)}
\]

设置 weight floor \(w_r\ge\epsilon\)，防止塌缩。

### Robust utility

\[
U_{robust}=\mathbb E_S[D(q,S)]-\lambda\operatorname{Std}_S[D(q,S)]
\]

### Optional CVaR

```text
λ ∈ {0, 0.25, 0.5, 1.0}
α ∈ {0.1, 0.2}
T ∈ {0.25, 0.5, 1.0, 2.0}
ε ∈ {0, 0.05}
```

全部只在 train / inner CV 选择。

## 7.6 Track B Validation Gate

相对 uniform mean，必须满足以下至少三项，并且第1或第2项成立：

1. Mean Dice +0.005；
2. Within-query reference std 降低≥10%；
3. Pairwise reference deviation 降低≥10%；
4. CVaR0.1 +0.01；
5. Catastrophic failure relative reduction≥15%；
6. Prediction consistency IoU +0.02。

同时要求：

- utility prediction Spearman≥0.30；
- 3 seeds 方向一致；
- 不使用 query GT；
- 最终 SAM calls=1；
- 权重不塌缩；
- validation bootstrap point estimate为正。

状态：

```text
PHASES_TRACK_B_RELIABILITY_WEIGHTING_PASS
PHASES_TRACK_B_UTILITY_ONLY_PASS
PHASES_TRACK_B_STABILITY_ONLY_PASS
PHASES_TRACK_B_NO_GAIN
```

---

# 8. Track C — Set-Conditioned Robust Prompting

## 8.1 启动条件

仅当：

- Track B 有稳定正向信号但达到平台；
- handcrafted / weighted mean 有用但不能利用 reference interactions；
- reference utility 可预测，Spearman≥0.30。

## 8.2 候选方法

### C1 DeepSets Reliability Encoder

\[
z_{\mathcal R}=
ho\left(\sum_r\phi(z_r)
ight)
\]

### C2 Lightweight Set Attention

仅1–2层 attention。

### C3 Pixel-wise Agreement Gate

学习低容量 \(w_r(x)\)。

### C4 Stable Candidate Scoring

\[
s_i=\mu_i-eta\sigma_i+\gamma a_i
\]

## 8.3 复杂度限制

- 参数优先<5M；
- 不更新 DINOv2-L；
- 不更新 SAM-B；
- 不训练大型 decoder；
- 最终只调用1次 SAM；
- Train训练、Validation选择、Holdout不调参。

## 8.4 Track C Gate

相对 Track B 最强方法：

1. Mean Dice +0.005；
2. CVaR +0.01，或 reference std 再降≥10%；
3. catastrophic failure不增加；
4. 3 seeds一致；
5. reference count变化时方向稳定；
6. leave-one-reference-out不崩溃；
7. 参数和运行成本可接受；
8. 不只是高容量拟合train。

状态：

```text
PHASES_TRACK_C_SET_MODEL_PASS
PHASES_TRACK_C_SET_MODEL_NO_ADDED_VALUE
```


# 9. Track D — Direct Reference-Conditioned Segmentation

## 9.1 启动条件

如果：

- Track A 证明 prompt-interface instability 很强；
- Track B/C 无法改善稳定性；
- raw matching maps 有稳定信号，但转成 point prompt 后损失明显；
- SAM refinement 增益不稳定。

## 9.2 方法范围

禁止仅做：

```text
threshold + morphology
```

至少实现一个技术核心：

- Multi-Prototype Matching；
- Reference-Set Uncertainty Fusion；
- Lightweight Dense Refiner；
- Topology / Component-Aware Loss。

## 9.3 Baselines

- raw mean matching mask；
- disagreement-penalized mask；
- threshold + morphology；
- strongest fixed prompt；
- strongest Track B/C prompting；
- lightweight refiner；
- supervised nnU-Net，仅作不同输入条件上界。

## 9.4 Track D Gate

1. 比 raw matching mask Dice +0.03；
2. 比 strongest fixed prompt Dice +0.01；
3. MoNuSAC validation 成立；
4. GlaS 或 DDR 至少一个 secondary dataset 方向一致；
5. reference removal ablation 明显下降；
6. 参数远小于完整 supervised model；
7. uncertainty 与 failure 有关联；
8. holdout 前冻结。

状态：

```text
PHASES_TRACK_D_DIRECT_METHOD_PASS
PHASES_TRACK_D_DIRECT_METHOD_NO_GAIN
```

---

# 10. Track Selection 与自动路由

Track A 完成后创建：

```text
reports/phaseS_track_selection.md
reports/phaseS_track_selection.json
```

评分：

| 维度 | 权重 |
|---|---:|
| 科学现象稳定性 | 20 |
| 方法新颖性 | 20 |
| validation effect size | 15 |
| 可预测性 | 10 |
| 跨数据集潜力 | 10 |
| 与已有工作的区分度 | 10 |
| 计算成本 | 5 |
| 论文叙事完整性 | 10 |

自动规则：

- Track A 强 + reliability 可预测 → B；
- B 有增益 → 优先完成 B；
- B 平台且 set interaction 有信号 → C；
- B/C 失败但 raw matching 有价值 → D；
- A 足以形成 benchmark 且方法增益有限 → Benchmark Track；
- 所有路线失败 → Negative / boundary paper assessment。

---

# 11. Confirmatory Holdout Gate

只有一条路线达到 validation pass 后，才冻结并使用 holdout=40。

冻结：

```text
code commit
data manifest
reference sampling
feature extractor
aggregation rule
model checkpoint
prompt config
thresholds
statistics
qualitative selection
cost accounting
```

创建：

```text
docs/phaseS_holdout_freeze.md
configs/frozen/phaseS_final_method.yaml
reports/phaseS_holdout_hashes.json
reports/phaseS_holdout_statistical_plan.md
```

Holdout只能运行一次。

## B/C Prompting 通过标准

相对 uniform mean 或 strongest fixed：

- Dice +0.0075；
- paired bootstrap point estimate>0；
- reference std 或 pairwise deviation下降；
- catastrophic failure不增加；
- 1 SAM call。

## D Direct Method 通过标准

相对 strongest fixed prompt：

- Dice +0.01；
- failure不增加；
- reference robustness指标方向一致。

状态：

```text
PHASES_HOLDOUT_CONFIRMATION_PASS
PHASES_HOLDOUT_CONFIRMATION_FAIL
```

Holdout失败后不得再次调参并重用。

---

# 12. 跨数据集验证

只有 holdout 通过后运行。

优先顺序：

1. GlaS 55/30；
2. DDR train/valid 中一个 lesion intent；
3. 其他数据需用户批准。

必须区分：

```text
zero-shot frozen transfer
same-protocol retraining
dataset-specific tuning
```

---

# 13. Novelty 与相关工作审计

相关工作至少包括：

- GF-SAM；
- SynPo；
- AlignSAM；
- Point Prompt Defender；
- ProSAM；
- MAUP；
- SAMIX；
- few-shot segmentation；
- visual reference segmentation；
- selective prediction / abstention；
- robust aggregation；
- set learning。

创建：

```text
reports/phaseS_novelty_audit.md
```

回答：

1. 本工作是否只是 reference selection？
2. 是否只是 uncertainty weighting？
3. 是否只是 mean consensus？
4. 是否只是更复杂 point selector？
5. 是否真正解决 cross-reference stability？
6. 是否提出新 metric/protocol？
7. 是否有稳定、可证伪的主张？

---

# 14. 必须生成的表格

1. Historical phase correction；
2. Metric definitions；
3. Reference resampling setup；
4. Instability benchmark；
5. Within-query variance；
6. Pairwise deviation；
7. CVaR / tail risk；
8. Prediction consistency；
9. Reference count scaling；
10. Deterministic baselines；
11. Reliability prediction；
12. Weighted aggregation；
13. Set model comparison；
14. Direct method comparison；
15. Ablations；
16. Cost/runtime；
17. Holdout；
18. Cross-dataset；
19. Failure taxonomy；
20. Final gate decisions。

---

# 15. 必须生成的图

1. Reference instability distribution；
2. Corrected variance decomposition；
3. Mean Dice vs reference count；
4. Within-query std vs reference count；
5. Worst-subset Dice；
6. CVaR；
7. Candidate overlap；
8. Prediction consistency；
9. Utility prediction calibration；
10. Reference weight distribution；
11. Reliability score vs true utility；
12. Uniform vs weighted consensus；
13. Set model gain；
14. Direct method gain；
15. Cost-quality Pareto；
16. Catastrophic failure；
17. High-instability cases；
18. Low-instability cases；
19. Method success cases；
20. Method failure cases；
21. Cross-dataset cases；
22. Gate state flow；
23. Final method diagram；
24. Benchmark protocol diagram。

定性案例按预注册规则选择：最大 best-worst gap、最大方法增益、最大方法损失、catastrophic failure、high/low disagreement、high-confidence failure、cross-domain failure。

---

# 16. 消融要求

方法论文至少包括：

- reference count；
- mean vs median vs weighted；
- disagreement feature；
- prototype separation；
- entropy；
- leave-one-reference-out feature；
- weight temperature；
- weight floor；
- risk objective；
- set encoder；
- uncertainty；
- prompt config；
- direct fallback；
- frozen matcher/segmenter；
- reference removal。

不得只做累加式消融。


# 17. 统计要求

统计单位：

```text
query / case
```

禁止将 reference episode、point、patch 当独立样本。

必须报告：

- mean；
- median；
- paired difference；
- bootstrap 95% CI；
- win/tie/loss；
- effect size；
- seed mean/std；
- within-query variance；
- CVaR；
- worst-reference；
- failure rate；
- runtime；
- cost。

多模型比较使用 Holm correction。

---

# 18. 计算预算

建议总上限：

| 阶段 | GPU-hours |
|---|---:|
| Track A metric benchmark | 20 |
| Track B reliability models | 35 |
| Track C set models | 45 |
| Track D direct method | 50 |
| Holdout | 10 |
| Cross-dataset | 20 |
| Ablations | 30 |
| 总上限 | 210 |

预算节点：

```text
50 GPU-hours  → Track A/B interim review
100 GPU-hours → Primary track decision
160 GPU-hours → Holdout readiness review
210 GPU-hours → stop and request user approval
```

优先复用 DINOv2 features、matching maps、candidate pools、SAM output cache、reference manifests 和 Phase R artifacts。

---

# 19. 允许自动执行与必须暂停

允许自动执行：

- 创建分支和协议；
- 只读审计；
- 训练轻量模型；
- validation；
- Gate；
- 方向切换；
- 生成表图；
- mock review；
- 总结报告。

必须暂停：

1. 需要 formal test；
2. 需要联网；
3. 需要下载数据或模型；
4. 需要超过210 GPU-hours；
5. 发现 patient/case leakage；
6. holdout已意外访问；
7. 需要改变 frozen threshold；
8. 需要新增未注册主方向；
9. 需要破坏历史 artifacts；
10. 所有路线均缺乏足够 fresh data。

---

# 20. 最终状态机

```text
PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID
PHASES_TRACK_A_METRIC_OR_PHENOMENON_NOT_STABLE

PHASES_TRACK_B_RELIABILITY_WEIGHTING_PASS
PHASES_TRACK_B_UTILITY_ONLY_PASS
PHASES_TRACK_B_STABILITY_ONLY_PASS
PHASES_TRACK_B_NO_GAIN

PHASES_TRACK_C_SET_MODEL_PASS
PHASES_TRACK_C_SET_MODEL_NO_ADDED_VALUE

PHASES_TRACK_D_DIRECT_METHOD_PASS
PHASES_TRACK_D_DIRECT_METHOD_NO_GAIN

PHASES_BENCHMARK_PAPER_READY
PHASES_METHOD_PAPER_READY
PHASES_HYBRID_PAPER_READY
PHASES_HOLDOUT_CONFIRMATION_PASS
PHASES_HOLDOUT_CONFIRMATION_FAIL
PHASES_CROSS_DATASET_SUPPORT
PHASES_CROSS_DATASET_NO_TRANSFER
PHASES_RESEARCH_PROGRAM_EXHAUSTED
PHASES_BLOCKED_TECHNICAL
PHASES_BLOCKED_USER_APPROVAL
```

---

# 21. 投稿价值评估

## 方法论文候选

必须满足：

- 新方法；
- validation + holdout；
- 至少一个跨数据集方向；
- 强 baseline；
- stability metric；
- failure analysis；
- ablation；
- cost；
- novelty audit。

## Benchmark paper 候选

必须满足：

- reference instability 现象稳定；
- 新 metric/protocol；
- 至少2–3 datasets；
- 多种 prompt/matching 方法；
- failure taxonomy；
- 至少一个预测 baseline；
- 可复现 manifests。

## 混合论文候选

最优形态：

> 新 reference-instability benchmark + deterministic baseline + learned reliability method + cross-dataset analysis。

---

# 22. 最终总结报告

无论成功、部分成功还是失败，都必须生成：

```text
reports/phaseS_final_research_summary.md
reports/phaseS_final_research_summary.json
```

报告必须包含：

1. 历史研究问题如何演化；
2. Phase R metric 是否有定义问题；
3. corrected instability metrics；
4. Track A/B/C/D 结果；
5. 每条路线的假设与 Gate；
6. 支持与否证的假设；
7. holdout 与跨数据集状态；
8. 最强自动方法、deterministic baseline、upper bound；
9. mean Dice、reference variance、CVaR、catastrophic failure；
10. cost/runtime；
11. ablations；
12. failure taxonomy；
13. novelty audit；
14. 与相关工作的差异；
15. 最适合的投稿方向；
16. 推荐论文标题和核心主张；
17. 不允许的主张；
18. formal test 是否值得；
19. artifact 路径与 git commits；
20. one-command reproduction；
21. unresolved blockers；
22. 最终唯一状态。

---

# 23. 最终终端输出格式

最终终端只回复：

1. Phase S final state；
2. Track A/B/C/D 状态；
3. 主研究发现；
4. strongest automatic method；
5. strongest deterministic baseline；
6. Dice / variance / CVaR / failure；
7. holdout result；
8. cross-dataset result；
9. publication track；
10. suggested paper title；
11. key novelty；
12. formal test status；
13. key artifacts；
14. one-command reproduction；
15. unresolved blockers。

---

# 24. 现在开始

严格顺序：

1. 阅读 Phase X/Y/R 全部 artifacts；
2. 创建 Phase S 分支；
3. 创建多方向 registry；
4. 完成 metric correction audit；
5. 执行 Track A；
6. 根据 Track A 结果选择 B 或 D；
7. Track B 有信号时训练轻量 reliability models；
8. B 平台后再决定是否运行 C；
9. B/C 无法解决但 matching evidence 有价值时运行 D；
10. validation pass 后冻结 holdout；
11. holdout只运行一次；
12. holdout pass后才做跨数据集；
13. 生成表格、图、消融和失败分析；
14. 进行 novelty audit；
15. 进行 mock review；
16. 生成完整最终科研总结；
17. 输出唯一状态。

不要因为某一个 Gate 失败就停止全部研究。

也不要因为某个模型有小幅正结果就立即宣称可发表。

本轮真正目标是：

> 在多个注册方向中，通过真实数据、可复现训练、稳定性指标、严格 holdout 和跨域验证，找到一个科学问题清楚、证据充分、与现有工作有区分度、能够形成论文的研究方向；若没有，则明确记录为什么没有。
