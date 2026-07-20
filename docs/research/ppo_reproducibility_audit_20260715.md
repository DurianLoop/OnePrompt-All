# Plug-and-Play PPO 官方发布代码复现审计

更新：2026-07-15

## 1. 为什么审计 PPO

Plug-and-Play PPO（CVPR 2025）与本文最接近的部分，是它也尝试自动优化送入 SAM 的点提示。因此它应进入 related work 和 baseline 讨论，但必须先确认官方发布物在新的 MoNuSeg reference-query episode 上是否真的执行了已学习策略，而不是仅仅运行了一个随机点过程。

本次审计对象：

- 官方仓库：`external/PPO/`
- 固定版本：`4c906b5fc59214f38d29b562938edb56efccec72`
- 官方权重：`external/PPO/model/best_q_table.pkl`
- 可复现审计：`code/benchmarks/audit_ppo_release.py`
- 机器可读结果：`runs/research_direction/baselines/ppo_release_audit/audit.json`

## 2. 已确认的发布物

官方 Google Drive 文件已完整下载，大小为 143,162,336 bytes。它可以被 `torch.load(..., weights_only=False)` 正常加载，类型为 `collections.defaultdict`，不是 HTML 下载页或错误压缩包。

Q-table 统计如下：

| 项目 | 数值 |
| --- | ---: |
| 状态-动作项总数 | 1,861,809 |
| 非零 Q 值项 | 85,678 |
| 状态类型 | `networkx.MultiGraph` |
| Q 最小值 | -9.899728 |
| Q 最大值 | 32.364522 |
| Q 均值 | 0.218816 |

## 3. 关键复现问题一：状态键无法对新图泛化

官方 Q-table 的键为：

```text
(NetworkX MultiGraph object, (node_id, operation))
```

审计脚本取出一个真实训练状态，其动作 `(120, restore_pos)` 的 Q 值为 21.962705；随后复制出一个结构同构的新图。两个图在 NetworkX 意义下同构，但新图不是原来的 Python 对象，因此：

```text
原状态查询：命中，Q=21.962705
同构新状态查询：不命中，Q=0
```

这意味着新测试图产生的状态无法直接复用训练图对象对应的 Q 值。即使新状态和某个训练状态结构相同，发布表也没有一个与对象身份无关的状态编码可供检索。

## 4. 关键复现问题二：发布推理入口默认随机探索

AST 自动审计确认：

- `QLearningAgent.__init__` 的 `epsilon_start` 默认值为 `1.0`；
- `main_FM.py` 构造 agent 时没有传入新的 epsilon；
- 加载 Q-table 后也没有给 `agent.epsilon` 重新赋值。

按发布代码的 `get_action` 实现，`random.random() < epsilon` 时随机选动作。因此入口默认 epsilon=1.0 时，推理阶段始终进入随机分支。它与上面的状态键不命中问题叠加，使“加载 Q-table”不等于“使用了已学习策略”。

## 5. 论文中如何公平处理

不能静默修复发布代码后仍把结果标成“官方 PPO”。后续主表将明确分开：

1. **PPO released-code reproduction**：保留官方图环境、动作和入口行为，报告多随机种子均值/方差，并披露状态键和 epsilon 问题。
2. **PPO deterministic inference probe**：仅把 epsilon 固定为 0，检查新状态下 Q-table 命中率；如果命中率仍为 0，它只能作为诊断，不能作为有效学习基线。
3. **PPO-style repaired baseline**：若实现与对象身份无关的状态向量或哈希，必须标为我们的适配版本；使用相同 train/val/test 协议重新训练，不能使用 test mask。
4. **本文方法**：reference foreground/background matching 定义任务意图，学习离散的 K、背景抑制强度和候选预算，并由 anchored gate 约束相对固定策略的风险。

## 6. 当前结论边界

现在已经能够确认“官方权重存在且可加载”，也能够确认发布推理路径有实质性的泛化与随机性问题；但尚未生成 PPO 在 MoNuSeg 上的 Dice，因此不能写成“本文优于 PPO”。在完成同输入、同 SAM、同 query split、同评价指标的运行前，PPO 只能出现在复现审计和方法边界中。
