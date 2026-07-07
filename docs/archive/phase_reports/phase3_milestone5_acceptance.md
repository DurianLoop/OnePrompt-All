# Phase3 第五验收标准推进报告

## 1. 本次目标

按 `docs/phase3_topconf_upgrade_plan.md` 的近期执行路线推进到第五个验收标准：

1. Support Prompt v1
2. Perturbation Uncertainty v1
3. Cross-model Calibration
4. 扩数据准备
5. 强基线比较

这次没有伪造任何未运行模型的指标。前 3 项基于已有 Phase2-lite 预测结果做可复现的二次评估；第 5 项给出本地强基线接入审计，区分“已经可跑”和“仍需接入”。

## 2. 已生成文件

- `runs\phase3_milestone5\support_guided_prompt_v1.csv`
- `runs\phase3_milestone5\perturbation_uncertainty_proxy_v1.csv`
- `runs\phase3_milestone5\cross_model_calibrated_portfolio_v4.csv`
- `runs\phase3_milestone5\strong_baseline_registry.csv`
- `runs\phase3_milestone5\summary_by_policy.csv`
- `runs\phase3_milestone5\summary_by_dataset.csv`

## 3. 方法链条结果

| 方法 | rows | Mean Dice | Mean Delta |
| --- | ---: | ---: | ---: |
| fixed SAM ViT-B auto_box | 40 | 0.3170 | - |
| support_guided_prompt_v1 | 120 | 0.3089 | -0.0029 vs per-model auto_box |
| perturbation_uncertainty_proxy_v1 | 120 | 0.3116 | -0.0002 vs per-model auto_box |
| cross_model_calibrated_portfolio_v4 | 40 | 0.3555 | 0.0385 vs fixed SAM ViT-B auto_box |

portfolio_v4 paired bootstrap:

```text
mean delta = 0.0385
95% CI     = [-0.0093, 0.0903]
p-value    = 0.1344
```

## 4. 五个验收标准状态

| 验收项 | 状态 | 结论 |
| --- | --- | --- |
| Step 1 Support Prompt v1 | 已完成可复现评估 | 用同域 support case 的历史表现选择 query prompt，验证 support 信息是否能带来互补性。 |
| Step 2 Perturbation Uncertainty v1 | 已完成 proxy 版 | 当前基于 auto-family mask consensus 近似 prompt-induced uncertainty；真正 jitter 重跑版仍是下一步。 |
| Step 3 Cross-model Calibration | 已完成 v4 | 已从单模型 selector 扩展到 SAM / MedSAM / SAM2 的跨模型 portfolio 选择。 |
| Step 4 扩数据准备 | 部分完成 | 代码仍基于 IDRID + MonuSeg 40 cases；扩到 100-200 cases、3+ domains 仍未完成。 |
| Step 5 强基线比较 | 工程验收完成，科学验收未完成 | 已生成强基线 registry；目前只有 SAM ViT-B / MedSAM / SAM2 Tiny 有 Phase2-lite 指标，P0 强基线仍需下载/接入/运行。 |

## 5. 强基线接入审计

当前本地可直接进入比较或接入的条目数：5 / 11。

P0 仍缺失：

- MedSAM2 / Medical SAM2: Need repository, checkpoint, and dataset adapter.
- SAM-Med2D: High-priority strong baseline for 2D medical segmentation.
- Med-SA / Medical SAM Adapter: High-priority adapter baseline.
- nnU-Net: Requires training/evaluation split; strongest supervised reference.

## 6. 诚实判断

我们已经达到“第五个验收标准”的工程版本：知道强基线在哪里、哪些已经有结果、哪些必须补齐，并且有了从 support 到 uncertainty 再到 cross-model calibration 的可复现实验链条。

但还没有达到顶会论文所需的科学版本。现在还不能写“超过近几年强医学分割基线”。下一步真正要做的是：

1. 接入并运行 SAM-Med2D / Med-SA / MedSAM2 中至少两个 P0 基线。
2. 加入一个轻量 U-Net 或 nnU-Net supervised baseline。
3. 把当前 40 cases 扩到至少 100-200 cases、3 个医学域。
4. 把 uncertainty proxy 升级成真实 prompt jitter 重跑版。
5. 用同一 manifest、同一 prompt、同一 metric 输出最终主表。

## 7. 当前可用 claim

目前可以保守写：

```text
OCSAM-X 将 support-guided prompt selection、prompt-induced uncertainty proxy
和 cross-model reliability calibration 串成了一个可复现的医学 SAM 自动提示适配框架。
在 Phase2-lite 上，它已经形成了比固定自动 prompt 更完整的候选选择闭环，
但仍需要更大数据和强医学基线来支撑顶会级 SOTA claim。
```
