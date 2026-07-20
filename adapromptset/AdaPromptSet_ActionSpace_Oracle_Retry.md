# AdaPromptSet 重试指令：动作空间充分性、稠密点提示与搜索 Oracle 审计

> 本轮不是继续沿用旧的 12-action Oracle 结论，而是重新验证：此前的 Oracle 是否只是因为动作库过窄，遗漏了适合“目标数量多、空间分散”的稠密提示配置。

---

## 0. 本轮必须先承认并修正的历史表述

此前 MoNuSAC 实验使用的是一个包含 **12 个合法预注册动作**的有限动作库。

此前所谓的 Oracle 实际是：

> **Finite-Bank Oracle（有限动作库预言机）**：  
> 对每张 query，只在这 12 个动作中选择 Dice 最高的动作。

它不是：

> **Global Prompt Oracle（全局提示预言机）**，也不是对所有可能的正负点数量、点子集、分组方式和聚类算法的穷举最优。

因此，历史结果：

```text
Finite-bank Dice oracle gap ≈ 0.005
```

只能说明：

> 在当前 12 个动作内部，逐图选择动作相对当前动作库中最强固定动作的平均增益较小。

它不能说明：

- 更大的正点数量无效；
- 更多负点无效；
- 更密集的空间覆盖无效；
- Top-50、Top-100 等动作无效；
- 其他分组、聚类或多次调用策略无效；
- adaptive prompt allocation 在更广动作空间中无价值。

本轮必须将历史结论更名为：

```text
FINITE_BANK_ORACLE_GAP_SMALL
```

禁止继续简称为“Oracle gap 证明 adaptive prompting 无效”。

---

# 一、为什么必须重试

当前数据中的目标可能具有以下特点：

- 单张图中目标实例数量很多；
- 目标分散在整张图像的多个区域；
- 少量 Top-3、Top-5 正点无法覆盖完整目标集合；
- 少量点的 Dice 绝对值可能本身很低；
- 当前最强固定动作和有限 bank oracle 可能都处于低性能区域；
- 两个低性能结果之间差值小，不代表 prompt policy 没有潜力。

例如：

```text
Top-3 Dice = 0.30
12-action finite-bank oracle Dice = 0.305
gap = 0.005
```

这个结果不能解释为“Oracle 已接近最优”。

它更可能说明：

> 当前动作库中的所有动作都不适合该任务。

如果真正有效的动作是：

```text
Top-100 positives
+ 20 hard negatives
+ spatial grouping
+ 10–20 SAM calls
```

而旧动作库从未测试，那么旧 Oracle 根本看不到这个解。

因此，本轮必须同时回答：

1. **旧动作的绝对性能是否过低？**
2. **随着正点数扩展到 10、20、50、75、100，性能和 component coverage 是否继续增长？**
3. **更多点应该放在一个 group，还是拆成多个 groups？**
4. **负点数量是否也需要随目标复杂度增长？**
5. **有限动作库是否已经达到性能饱和？**
6. **搜索得到的近似 Oracle 是否显著高于旧的 12-action Oracle？**
7. **在相同成本预算下，逐图自适应选择是否具有超过 0.01 Dice 的价值？**

---

# 二、本轮任务名称

```text
Phase X — Action-Space Adequacy,
Dense Prompt Scaling,
and Search-Oracle Audit
```

工作目录：

```text
D:\SAM
```

项目目录：

```text
D:\SAM\adapromptset
```

从当前最新稳定 commit 创建新分支：

```text
phaseX-action-space-oracle-audit
```

本轮不得覆盖此前任何报告、动作库、PDF 或 Gate 记录。

---

# 三、本轮唯一核心科学问题

给定冻结的：

- matcher；
- candidate generator；
- segmenter；
- reference-query protocol；

旧的 12-action bank 是否足以近似 prompt search space？

更具体地说：

> 对于目标数量多、空间分散的 dense segmentation query，随着正点数量、负点数量、空间分组和 SAM-call budget 扩大，最优 prompt performance 是否仍在显著提高？

以及：

> 在一个已经基本饱和的动作空间内，query-specific Oracle 相对 strongest fixed action 是否仍有足够大的增益，从而支持 policy learning？

---

# 四、开始前必须创建的文档

运行任何新推理前，创建：

```text
docs/phaseX_goal_read_receipt.md
docs/action_space_problem_statement.md
docs/historical_oracle_correction.md
docs/dense_prompt_scaling_protocol.md
docs/search_oracle_protocol.md
docs/action_space_saturation_protocol.md
docs/formal_test_lock_notice.md
reports/phaseX_gate_status.md
reports/phaseX_compute_ledger.csv
reports/phaseX_protocol_snapshot.json
reports/phaseX_input_hashes.json
```

必须更新但不能覆盖历史：

```text
docs/protocol_history.md
docs/decision_log.md
docs/data_exposure_log.md
checklist.md
```

然后提交：

```text
phaseX: preregister action-space adequacy and search-oracle audit
```

在该 commit 前不得运行新实验。

---

# 五、必须先做的历史动作库审计

读取旧的：

```text
configs/actions/prompt_set_bank_v1.*
configs/actions/prompt_set_bank_v2.*
reports/action_bank/*
reports/oracle_gap_pilot/*
reports/oracle_gap_pilot_v2/*
reports/gates/gate_f_oracle_gap.*
reports/decision_checkpoint/*
```

生成：

```text
reports/phaseX_historical_bank_audit/
├── action_bank_exact_definitions.csv
├── action_bank_exact_definitions.md
├── fixed_action_absolute_metrics.csv
├── fixed_action_rankings.csv
├── old_bank_oracle_query_metrics.csv
├── old_bank_oracle_summary.md
├── cost_distribution.csv
└── historical_claim_correction.md
```

必须明确回答：

1. 旧的 12 个合法动作分别是什么？
2. 每个动作的：
   - 正点数；
   - 负点数；
   - 正点选择规则；
   - 负点选择规则；
   - grouping；
   - cluster 数；
   - SAM calls；
   - fusion；
   - runtime；
   - Dice；
   - IoU；
   - AJI；
   - component recall；
   - merge rate。
3. 旧 strongest fixed action 的精确配置是什么？
4. 它的绝对 Dice 是多少？
5. Top-1、Top-3、Top-5 的绝对 Dice 分别是多少？
6. 旧 finite-bank oracle 的绝对 Dice 是多少？
7. 0.005 的 gap 是在什么绝对性能水平上产生的？
8. strongest fixed action 和 oracle 是否都处于低性能区域？
9. 旧动作库的最大正点数和最大负点数是多少？
10. 是否曾测试过 20、50、75、100 个正点？
11. 是否曾测试过高密度点拆成多个 groups？
12. 是否曾测试过多预算下的 fixed/oracle 曲线？

不得只报告差值。

主报告必须同时展示：

```text
absolute fixed performance
absolute oracle performance
oracle-fixed difference
```

---

# 六、数据与测试边界

本轮优先在此前已用于开发的 MoNuSAC development/validation 范围内进行。

不得：

- 使用正式 test 选择动作；
- 使用 GlaS testA/testB；
- 使用 IDRID 调参；
- 将 observed test 重新包装成 validation；
- 根据 formal test 结果增加 Top-K 候选；
- 根据 formal test 改变 cluster 数或预算。

本轮是：

> development-stage action-space audit

不是正式 test paper result。

所有 query、reference 和 split 必须使用已冻结 manifest。

如果历史 manifest 不完整或无法复现：

```text
PHASEX_BLOCKED_MANIFEST_INCONSISTENCY
```

并停止。

---

# 七、第一层实验：Candidate Coverage Scaling

在运行 SAM 前，先检查 candidate pool 本身是否支持稠密目标覆盖。

正点候选预算：

```text
K+ ∈ {1, 3, 5, 10, 20, 30, 50, 75, 100}
```

如果候选池不足 100：

- 使用实际最大候选数；
- 明确记录；
- 不复制重复点；
- 不补虚假点。

负点候选预算：

```text
K- ∈ {0, 1, 3, 5, 10, 20, 30, 50}
```

对每个 K+ 报告：

- point precision；
- component coverage；
- foreground area coverage proxy；
- spatial dispersion；
- nearest-neighbor distance；
- redundancy；
- candidate entropy；
- missed component count；
- coverage saturation curve。

对每个 K- 报告：

- background precision；
- hard-background rate；
- boundary proximity；
- easy-background fraction；
- false-negative risk；
- spatial dispersion。

生成：

```text
reports/phaseX_candidate_scaling/
├── positive_scaling_query_metrics.csv
├── negative_scaling_query_metrics.csv
├── positive_coverage_curve.csv
├── negative_quality_curve.csv
├── component_coverage_by_k.csv
├── spatial_dispersion_by_k.csv
├── scaling_summary.md
└── scaling_summary.json
```

必须生成图：

```text
paper_phaseX/figures/candidate_scaling/
├── component_coverage_vs_positive_k.png
├── point_precision_vs_positive_k.png
├── dispersion_vs_positive_k.png
├── negative_quality_vs_negative_k.png
└── coverage_precision_tradeoff.png
```

## Candidate Scaling Gate

如果：

- K+=5 时 coverage 很低；
- K+=50 或 100 时 coverage 明显继续提升；

则必须记录：

```text
OLD_SMALL_K_BANK_INADEQUATE
```

如果：

- 即使 K+=100，component coverage 仍然很低；

则记录：

```text
CANDIDATE_UNIVERSE_INADEQUATE
```

此时不应把问题归因于 policy，而应归因于 matcher/candidate generator。

---

# 八、第二层实验：Dense Prompt Scaling

## 8.1 正点选择策略

至少测试：

```text
P_TOPK
P_LOCAL_PEAKS
P_SPATIAL_NMS
P_SPREAD
P_COMPONENT_BALANCED
P_DIVERSITY_GREEDY
```

其中：

- `P_TOPK`：按 contrast score 取前 K；
- `P_LOCAL_PEAKS`：取局部峰；
- `P_SPATIAL_NMS`：空间非极大抑制；
- `P_SPREAD`：在高分候选中最大化空间分散；
- `P_COMPONENT_BALANCED`：按候选空间组件均衡取点；
- `P_DIVERSITY_GREEDY`：逐步选择与已选点最不冗余的高分点。

不得使用 query GT 进行自动点选择。

## 8.2 负点选择策略

至少测试：

```text
N_NONE
N_BOTTOMK
N_HIGH_BACKGROUND
N_BOUNDARY_HARD
N_LOCAL_CONFUSION
N_DIVERSITY
```

## 8.3 正点数量

```text
K+ ∈ {1, 3, 5, 10, 20, 30, 50, 75, 100}
```

## 8.4 负点数量

```text
K- ∈ {0, 1, 3, 5, 10, 20, 30, 50}
```

## 8.5 Grouping 方式

至少测试：

```text
G_ONE_GROUP
G_ONE_POINT_PER_GROUP
G_FIXED_CHUNK_5
G_FIXED_CHUNK_10
G_FIXED_CHUNK_20
G_KMEANS_3
G_KMEANS_5
G_KMEANS_10
G_KMEANS_20
G_SPATIAL_GRID
G_GRAPH
G_CANDIDATE_COMPONENT
```

定义：

- `G_ONE_GROUP`：所有点一次输入 SAM；
- `G_ONE_POINT_PER_GROUP`：每点单独一次调用；
- `G_FIXED_CHUNK_n`：每 n 个点一组；
- `G_KMEANS_k`：按空间坐标聚为 k 组；
- `G_SPATIAL_GRID`：按固定网格分组；
- `G_GRAPH`：按图关系分组；
- `G_CANDIDATE_COMPONENT`：按候选空间连通组件分组。

## 8.6 SAM-call budgets

```text
B_call ∈ {1, 3, 5, 10, 20}
```

## 8.7 总点预算

```text
B_point ∈ {5, 10, 20, 50, 100, 150}
```

不同方法必须在：

- 相同 call budget；
- 相同 point budget；
- 或同一明确 Pareto 曲线；

下公平比较。

---

# 九、不能直接做完整笛卡尔积

上述空间可能非常大。

因此必须采用分阶段搜索，而不是无脑全跑。

## Stage S1：单变量 Scaling

固定：

```text
selection = Top-K
negative = none
grouping = one group
calls = 1
```

只扩大 K+。

目的：

> 检查单纯增加点是否能显著提高绝对 Dice 和 component recall。

## Stage S2：Grouping Scaling

选择 S1 中有代表性的：

```text
K+ ∈ {10, 30, 50, 100}
```

比较 grouping。

目的：

> 判断大量散点应放在一个 group，还是拆为多个空间 groups。

## Stage S3：Negative Scaling

在 S2 的最强正点/group 配置上，测试 K- 和负点策略。

## Stage S4：Joint Coarse Grid

组合最有潜力的：

- 3 个 positive selectors；
- 3 个 K+；
- 3 个 negative settings；
- 4 个 groupings；
- 3 个 call budgets。

## Stage S5：Successive Halving

使用：

> **Successive Halving（逐步淘汰）**

规则：

1. 所有动作先在小型 train-only audit subset 上运行；
2. 淘汰绝对 Dice、component recall 和成本明显劣势的动作；
3. 保留前若干动作进入完整 validation；
4. 不得根据 formal test 淘汰或保留。

不得只根据 Dice 淘汰。

至少联合考虑：

```text
Dice
component recall
merge rate
SAM calls
runtime
```

---

# 十、建立嵌套动作库

构建：

```text
A_small ⊂ A_medium ⊂ A_large ⊂ A_search_augmented
```

## A_small

对应旧动作规模：

```text
K+ ≤ 5
K- ≤ 4
calls ≤ 3
```

## A_medium

```text
K+ ≤ 20
K- ≤ 10
calls ≤ 5
```

## A_large

```text
K+ ≤ 100
K- ≤ 30
calls ≤ 20
包含 dense grouping
```

## A_search_augmented

```text
A_large
+ greedy/beam search 产生的高效新动作
```

每个动作必须有唯一：

```text
action_id
action_hash
selection_rule
K+
K-
grouping
cluster_count
call_budget
point_budget
fusion
```

不得根据 query GT 把某个搜索结果直接当作 automatic action。

GT-guided search 的结果只能用于：

- Oracle；
- action-space adequacy analysis；
- 发现动作家族。

若要将某个搜索发现转为自动动作，必须：

1. 在 train-only search 中发现；
2. 抽象成 query-GT-free deterministic rule；
3. 在 validation 前冻结；
4. 作为一次允许的 action-bank redesign 注册。

---

# 十一、第三层实验：Fixed Baseline Saturation

对于每个嵌套动作库 \(\mathcal A_j\)，计算：

\[
D_{\text{fixed}}(\mathcal A_j)
=
\max_{a\in\mathcal A_j}
\frac{1}{N}\sum_q D(q,a)
\]

必须输出：

- strongest fixed action；
- exact config；
- mean Dice；
- median Dice；
- IoU；
- AJI；
- component recall；
- calls；
- runtime；
- win/tie/loss。

并绘制：

```text
best fixed Dice vs action-bank size
best fixed Dice vs max positive K
best fixed Dice vs point budget
best fixed Dice vs call budget
best fixed component recall vs max positive K
```

必须回答：

> 当 K+ 从 5 扩展到 100 时，最佳固定动作是否还在显著提升？

如果最佳固定动作从低 K 转为高 K，说明旧 baseline 不充分。

---

# 十二、第四层实验：Finite-Bank Oracle Saturation

对每个动作库计算：

\[
D_{\text{bank-oracle}}(\mathcal A_j)
=
\frac{1}{N}
\sum_q
\max_{a\in\mathcal A_j}D(q,a)
\]

以及：

\[
\Delta_{\text{bank}}(\mathcal A_j)
=
D_{\text{bank-oracle}}(\mathcal A_j)
-
D_{\text{fixed}}(\mathcal A_j)
\]

必须绘制：

```text
bank oracle Dice vs action-bank size
fixed Dice vs action-bank size
oracle gap vs action-bank size
oracle gap vs point budget
oracle gap vs call budget
oracle action distribution vs target count
```

## 解释规则

### 情况 1

Fixed 和 Oracle 都随 bank 扩大继续上升：

```text
ACTION_SPACE_NOT_SATURATED
```

### 情况 2

Oracle 显著上升，Fixed 基本饱和：

```text
ADAPTIVE_HETEROGENEITY_EMERGES
```

### 情况 3

Fixed 与 Oracle 同步上升，但 gap 仍很小：

```text
STRONG_FIXED_DENSE_PROMPT_SUFFICIENT
```

### 情况 4

Fixed 与 Oracle 都不再上升，且 gap 小：

```text
ACTION_SPACE_SATURATED_ORACLE_GAP_SMALL
```

只有情况 4，才可以较强地否定当前预算下的 adaptive value。

---

# 十三、第五层实验：GT-Guided Search Oracle

旧的 finite-bank oracle 不足以验证动作库充分性。

因此在 train/validation 的合法 audit subset 上构建：

> **GT-Guided Search Oracle（GT 引导的搜索预言机）**

只用于诊断，不可部署。

## 13.1 Greedy Add Search

从安全 anchor 开始：

1. 尝试加入每个候选正点；
2. 选择 Dice 提升最大的点；
3. 重复直到无提升或达到预算；
4. 再搜索负点；
5. 再搜索 grouping 改变。

## 13.2 Greedy Add/Remove/Swap

允许：

- add positive；
- remove positive；
- swap positive；
- add negative；
- remove negative；
- swap negative；
- merge groups；
- split groups；
- change group assignment。

## 13.3 Beam Search

束宽：

```text
beam_width ∈ {4, 8}
```

最大深度：

```text
depth ∈ {10, 20, 50}
```

在预算允许时，保留多个高效候选状态。

## 13.4 Search Budgets

至少报告：

```text
B_call ∈ {1, 3, 5, 10, 20}
B_point ∈ {10, 20, 50, 100}
```

## 13.5 搜索结果

计算：

\[
D_{\text{search-oracle}}(B)
\]

比较：

\[
D_{\text{search-oracle}}(B)
-
D_{\text{bank-oracle}}(B)
\]

以及：

\[
D_{\text{search-oracle}}(B)
-
D_{\text{fixed}}(B)
\]

## Search Oracle 解释

如果：

```text
search oracle - bank oracle >= 0.01 Dice
```

说明：

```text
OLD_ACTION_BANK_INADEQUATE
```

如果：

```text
search oracle - fixed >= 0.01
```

且不同 query 的搜索最优配置不同，说明：

```text
POLICY_SPACE_POTENTIALLY_MEANINGFUL
```

如果 search oracle 本身绝对 Dice 仍然很低：

```text
PROMPT_INTERFACE_OR_CANDIDATE_LIMITATION
```

---

# 十四、必须区分三种 Oracle

最终所有报告中必须明确使用：

## 1. Finite-Bank Oracle

```text
在冻结动作库中逐图选最优动作
```

## 2. Search Oracle

```text
在冻结候选池和预算内，用 GT-guided greedy/beam search 找近似最优 prompt
```

## 3. Privileged Coordinate Oracle

```text
允许直接使用 GT component/centroid/box 构造 prompt 的上界
```

三者不能混写。

禁止再使用无修饰的：

```text
Oracle
```

---

# 十五、绝对性能 Gate

本轮不允许只看 gap。

必须先定义最低绝对可用性。

建议在运行前根据历史指标分布冻结：

```text
D_min_viable
```

如果暂时不设单一 Dice 门槛，也必须使用：

- 相对 strongest simple baseline；
- component recall；
- catastrophic failure；
- merge rate；

联合判断。

需要回答：

1. Top-3 的 Dice 是否极低？
2. Top-50/100 是否显著提高 Dice？
3. 提升来自 recall 还是 false-positive 扩张？
4. 大量点是否导致 mask 接近全图？
5. SAM calls 增加后是否只增加成本而不增加质量？
6. 高 K 是否改善 component recall 但损害 precision？
7. 哪个 K 开始达到收益饱和？

---

# 十六、成本公平性

Top-100 与 Top-3 不能只按 Dice 直接比较。

必须报告：

- requested positive count；
- actual valid positive count；
- requested negative count；
- actual valid negative count；
- number of groups；
- number of SAM calls；
- wall-clock runtime；
- VRAM；
- failed calls；
- fusion cost。

计算预算约束 Oracle：

\[
D_{\text{oracle}}(B)
=
\frac{1}{N}
\sum_q
\max_{a:C(a)\le B}D(q,a)
\]

固定 baseline：

\[
D_{\text{fixed}}(B)
=
\max_{a:C(a)\le B}
\frac{1}{N}\sum_qD(q,a)
\]

以及：

\[
\Delta(B)
=
D_{\text{oracle}}(B)-D_{\text{fixed}}(B)
\]

至少对以下预算分别报告：

```text
1 call
3 calls
5 calls
10 calls
20 calls
```

同时报告 point-budget 曲线。

---

# 十七、新 Gate 状态机

## Gate X1：Historical Baseline Audit

通过条件：

- 旧 12 actions 精确定义可恢复；
- strongest fixed exact config 可恢复；
- 所有绝对 Dice 可恢复；
- cost metadata 可恢复。

失败：

```text
BLOCKED_HISTORICAL_ARTIFACT_INCOMPLETE
```

## Gate X2：Candidate Scaling

通过条件：

- K+ 扩展到至少 50；
- 可生成 coverage/precision 曲线；
- 坐标合法；
- 无大规模重复点。

结果状态：

```text
CANDIDATE_SCALING_SUPPORTS_LARGE_K
CANDIDATE_SCALING_SATURATES_SMALL_K
CANDIDATE_UNIVERSE_INADEQUATE
```

## Gate X3：Dense Prompt Viability

通过条件：

- 至少一个高 K 配置在绝对 Dice、component recall 或 catastrophic failure 上显著优于旧小 K 动作；
- 提升不能完全由更多 calls 未计成本造成。

状态：

```text
DENSE_PROMPT_VIABILITY_ESTABLISHED
DENSE_PROMPT_NO_GAIN
DENSE_PROMPT_COST_ONLY_GAIN
```

## Gate X4：Action-Space Saturation

状态：

```text
ACTION_SPACE_NOT_SATURATED
ACTION_SPACE_SATURATED
OLD_ACTION_BANK_INADEQUATE
```

## Gate X5：Search-Oracle Adequacy

状态：

```text
SEARCH_ORACLE_EXCEEDS_BANK_ORACLE
SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE
SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW
```

## Gate X6：Adaptive Value

只有以下同时满足才继续 policy：

1. search oracle 相对 strongest fixed：
   \[
   \ge 0.010\ Dice
   \]
2. 至少 25% query 的最优结构不同于 strongest fixed；
3. 不是单纯更多 calls；
4. absolute performance 达到可用水平；
5. action-space saturation 已基本建立；
6. query-specific best config 与 morphology/candidate statistics存在可预测关系。

通过：

```text
REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE
```

否则：

```text
PIVOT_STRONG_FIXED_DENSE_PROMPT
PIVOT_PROMPT_INTERFACE_LIMITATION
PIVOT_CANDIDATE_LIMITATION
PIVOT_ORACLE_ANALYSIS_EXPANDED_SPACE
```

---

# 十八、如果重新开放 Policy

只有状态为：

```text
REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE
```

才允许训练 policy。

Policy 不直接预测 100 个点坐标。

优先分层预测：

1. point-budget bin：
   ```text
   {small, medium, large, very_large}
   ```
2. negative-budget bin；
3. grouping family；
4. call-budget；
5. candidate selector family。

例如动作层级：

```text
K+ bin:
1–5 / 10–20 / 30–50 / 75–100

K- bin:
0 / 1–5 / 10–20 / 30–50

group:
one / chunks / kmeans / graph / component

calls:
1 / 3 / 5 / 10 / 20
```

然后由确定性模块在选定 family 中构造实际点集。

先训练：

- global prior；
- multiclass classifier；
- utility regression；
- pairwise ranking；
- hierarchical router。

不使用 PPO/DQN。

---

# 十九、必须生成的表格

1. 旧 12-action exact definition；
2. 旧动作绝对性能；
3. Top-K scaling；
4. Positive K × negative K；
5. Grouping comparison；
6. Calls-budget comparison；
7. Candidate coverage scaling；
8. Best fixed saturation；
9. Finite-bank oracle saturation；
10. Search oracle comparison；
11. Bank oracle vs search oracle；
12. Oracle-fixed gap by budget；
13. Absolute Dice and component recall；
14. Runtime/VRAM/calls；
15. Query morphology vs best K；
16. Query target count vs best K；
17. Search-selected prompt statistics；
18. Final Gate decision table。

---

# 二十、必须生成的图

1. point precision vs K+；
2. component coverage vs K+；
3. Dice vs K+；
4. component recall vs K+；
5. Dice vs K-；
6. Dice vs groups；
7. Dice vs SAM calls；
8. fixed/oracle/search-oracle saturation curves；
9. oracle gap vs bank size；
10. search-oracle gain distribution；
11. best K distribution；
12. best grouping distribution；
13. target count vs optimal K；
14. morphology vs optimal prompt budget；
15. Pareto Dice–calls；
16. Pareto Dice–runtime；
17. small-K failure cases；
18. large-K success cases；
19. large-K oversegmentation failures；
20. old-bank vs expanded-bank qualitative comparisons。

定性图必须按预注册规则选取，不能只选最好案例。

---

# 二十一、计算预算

先生成 dry-run estimate。

建议新增上限：

```text
historical audit: CPU only
candidate scaling: 4 GPU-hours
single-variable K scaling: 12 GPU-hours
grouping scaling: 16 GPU-hours
negative scaling: 12 GPU-hours
coarse joint grid: 24 GPU-hours
successive halving validation: 24 GPU-hours
greedy/beam search oracle: 40 GPU-hours
statistics and figures: CPU mostly
total initial cap: 132 GPU-hours
```

达到：

```text
40 GPU-hours
```

生成 interim report。

达到：

```text
90 GPU-hours
```

重新评估 search value。

超过：

```text
132 GPU-hours
```

前必须暂停并请求用户批准。

优先复用：

- DINOv2 features；
- matching maps；
- candidate pools；
- identical prompt outputs；
- SAM mask caches；
- action utility matrix。

---

# 二十二、禁止事项

本轮禁止：

- 使用 formal test 选择 K；
- 使用 test 选择 cluster 数；
- 根据 test 新增 Top-100；
- 将 search oracle 写成 automatic method；
- 将 GT-guided search 写成可部署策略；
- 只报告 0.005 gap，不报告绝对 Dice；
- 将 finite-bank oracle 写成 global oracle；
- 将高 K 的更多 calls 隐藏；
- 对不同动作使用不同后处理；
- 删除 large-K 失败 case；
- 只选择有利 seed；
- 无限扩大 action grid；
- 在结果后降低 0.01 adaptive-value threshold；
- 在没有 saturation audit 时再次宣布 policy 无价值。

---

# 二十三、最终报告必须回答的问题

生成：

```text
reports/phaseX_final_report.md
reports/phaseX_final_report.json
```

必须明确回答：

1. 旧 strongest fixed action 到底是什么？
2. 旧 Top-3 Dice 到底是多少？
3. 旧 finite-bank oracle 的绝对 Dice 是多少？
4. 为什么旧 gap 小？
5. K+ 从 3 增长到 100 后，Dice 如何变化？
6. component coverage 如何变化？
7. component recall 如何变化？
8. 最优 K 是否和 query target count 相关？
9. 最优 grouping 是否和目标空间分散程度相关？
10. Top-100 是有效提示还是导致过分割？
11. 负点是否在 large-K 下变得更重要？
12. 最强固定 dense action 是什么？
13. expanded-bank oracle 是多少？
14. search oracle 是多少？
15. search oracle 与 old bank oracle 相差多少？
16. search oracle 与 strongest fixed 相差多少？
17. 动作空间是否饱和？
18. 0.01 adaptive value 是否成立？
19. 是否重新开放 policy？
20. 最终科学结论属于哪一种：
    - action bank inadequate；
    - candidate limitation；
    - prompt-interface limitation；
    - strong fixed dense prompt sufficient；
    - adaptive policy has meaningful space。

---

# 二十四、最终允许的结论

## 如果 expanded/search oracle 显著更高

允许：

> The previous 12-action finite-bank oracle substantially underestimated the attainable prompt performance. Dense prompt budgets and broader grouping choices reveal previously hidden action-space capacity.

## 如果 strongest fixed dense action 已接近 search oracle

允许：

> Dense prompting is beneficial, but a strong fixed high-budget action captures most of the available gain, leaving limited value for query-specific policy learning.

## 如果 search oracle 相对 fixed gap 超过 0.01

允许：

> After establishing action-space adequacy, query-specific prompt allocation exhibits a meaningful oracle advantage under matched budgets, justifying policy learning.

## 如果 search oracle 仍然很低

允许：

> The limitation lies upstream in candidate evidence or in the prompt interface, rather than in the policy architecture.

---

# 二十五、开始执行

现在开始，但不得直接运行大规模实验。

严格顺序：

1. 读取所有历史 action-bank 和 oracle artifacts；
2. 创建历史 Oracle 修正文档；
3. 导出旧 12-action exact definitions；
4. 导出每个旧动作绝对 Dice；
5. 明确旧 strongest fixed exact config；
6. 创建 candidate scaling protocol；
7. 创建 dense prompt scaling protocol；
8. 创建 search oracle protocol；
9. 冻结 K、negative、grouping、call grids；
10. 生成 config/input hashes；
11. 运行 focused tests；
12. 提交 preregistration commit；
13. 运行 candidate coverage scaling；
14. 运行 single-variable K+ scaling；
15. 判断是否支持 large-K；
16. 运行 grouping scaling；
17. 运行 negative scaling；
18. 运行 coarse joint grid；
19. 运行 successive halving；
20. 构建 nested action banks；
21. 计算 best-fixed saturation；
22. 计算 finite-bank oracle saturation；
23. 运行 GT-guided greedy/beam search oracle；
24. 比较 bank oracle 与 search oracle；
25. 计算 matched-budget oracle gap；
26. 判断 action-space adequacy；
27. 只有合法时重新开放 policy；
28. 生成完整表格、图和报告；
29. 更新所有历史 claim；
30. 输出唯一最终状态。

不要因为旧的 0.005 gap 再次提前停止。

本轮第一优先级不是训练 policy，而是判断：

> 旧的 12-action bank 是否错误地把一个未充分搜索的低性能空间，当成了“接近 Oracle”的空间。
