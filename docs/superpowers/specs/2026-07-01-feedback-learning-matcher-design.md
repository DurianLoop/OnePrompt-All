# Feedback- and Learning-Guided Matcher 设计说明

日期：2026-07-01

## 目标

本设计回应老师会后给出的两个方向：

1. 方向1（易）：将用户 feedback 纳入 Matcher。正点提供目标信息，负点提供背景信息，Matcher 不再只依赖 reference box/mask。
2. 方向2（难）：将 learning 纳入 Matcher。先不训练大模型，而是学习可解释的阈值和过滤策略，例如 Stage1 threshold、feedback 权重、Stage3 instance threshold。

当前策略是先做最小可解释闭环：

```text
reference prototype
+ positive feedback prototype
- negative feedback prototype
-> updated target similarity
-> matched points / clusters / SAM proposals
-> Stage3 proposal metrics
-> learned or adaptive threshold
```

## 方向1：feedback 如何进入 Matcher

原始 Matcher 的 Stage1 主要流程是：

```text
reference box/mask -> DINO feature prototype -> target feature similarity -> top-k matched points
```

拓展后增加一个 feedback prototype update：

```text
updated_proto = normalize(reference_proto
                         + positive_weight * positive_proto
                         - negative_weight * negative_proto)
```

直观解释：

- positive point 附近的 DINO patch 被看作“更像目标”的补充证据；
- negative point 附近的 DINO patch 被看作“背景/干扰物”的反证；
- 更新后的 prototype 重新和 target feature map 做 cosine similarity；
- 因此变化发生在 Matcher 自身，而不是只把正负点交给 SAM。

## 方向2：learning 如何进入 Matcher

本阶段不直接训练 DINO/SAM，也不直接做 LoRA/Adapter。先学习更小、更可解释的决策：

- Stage1：根据相似度分布自动选择 threshold，让 selected points 数量落在合理预算内；
- Stage3：根据少量人工标签学习 instance similarity 的 keep/reject 阈值；
- Proposal ranking：用 instance similarity、SAM score、cluster coverage、prompt coverage、mask area、bbox area 组合成可解释分数；
- 后续再扩展为 logistic regression、Bayesian search 或小 MLP。

最小学习输入：

```json
{
  "positive_group_ids": [1, 2],
  "negative_group_ids": [3, 4]
}
```

输出：

```text
learned instance similarity threshold
selected group ids
feedback learning score
```

## 文件边界

- `code/benchmarks/matcher_feedback_learning.py`
  - 纯函数核心：prototype feedback、预算阈值、标注阈值学习、proposal 打分。
- `code/benchmarks/run_matcher_feedback_learning_policy.py`
  - CLI：从 Stage3 `feature_scores.json` 和少量标签生成 `matcher_learning_policy.json`。
- `code/benchmarks/run_research_direction_case.py`
  - 在完整 Stage1-3 runner 中增加可选 `--labels-json`，自动追加 learning policy 步骤。
- `code/benchmarks/check_matcher_feedback_learning.py`
  - 核心函数检查。
- `code/benchmarks/check_matcher_feedback_learning_policy_cli.py`
  - CLI 检查。
- `code/benchmarks/check_research_direction_runner.py`
  - runner 静态检查。

## 成功标准

1. 正负 feedback 能扩大正点与背景点的相似度 margin。
2. 自适应 threshold 能控制 selected points 数量。
3. 少量 labeled proposals 能学习出正确的 keep/reject threshold。
4. 完整 runner 支持在 Stage3 后接入 learning policy。

## 当前限制

- 这不是大规模训练，只是 learning 的最小可解释版本。
- 需要更多自然图和医学图 case 才能证明泛化。
- 当前标签粒度是 proposal/group 级，后续可以扩展到 mask-level IoU 或 human accept/reject。
