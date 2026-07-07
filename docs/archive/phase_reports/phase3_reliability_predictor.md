# Phase3 Reliability Predictor v1

目标：把上一版失败的简单 stability selector 升级为可训练的 reliability predictor。

## 方法

- 训练方式：leave-one-case-out ridge regression。
- 训练标签：训练病例上的 Dice。
- 测试选择：测试病例不使用 GT，只用模型名、数据集、prompt family、SAM score、预测面积、候选内部面积稳定性等可部署特征。
- 候选池：SAM ViT-B / SAM ViT-H / MedSAM / SAM2 Tiny 的基础 prompt 与真实 jitter prompt。

## 结果

| Scope | Mean Dice | Delta vs SAM ViT-B auto_box | Delta vs best auto-box oracle |
| --- | ---: | ---: | ---: |
| IDRID | 0.3693 | -0.1281 | -0.2262 |
| MonuSeg | 0.2034 | -0.1083 | -0.1284 |
| all | 0.2863 | -0.1182 | -0.1773 |

Paired bootstrap vs SAM ViT-B auto_box:

```text
mean delta = -0.1182
95% CI     = [-0.1707, -0.0661]
p-value    = 0.0000
```

## 判断

这个版本仍未超过固定 SAM ViT-B auto_box，说明特征或模型选择策略还需要继续增强。

注意：如果它没有超过 best auto-box oracle，不是问题；oracle 使用 GT 选模型，只是上限。真正需要超过的是固定单模型/固定 prompt 的可部署 baseline。

## v2/v3 保守强基线版本

v1 失败后，v2 把候选限制为各模型的 auto_box；v3 进一步默认使用最强固定基线 SAM ViT-H，只在 predictor 有把握时切换。

| Policy | Scope | Mean Dice | Delta vs SAM-B auto_box | Delta vs SAM-H auto_box | Switch rate |
| --- | --- | ---: | ---: | ---: | ---: |
| leave_one_case_out_auto_box_reliability_v2 | IDRID | 0.531 | 0.0336 | -0.0444 |  |
| leave_one_case_out_auto_box_reliability_v2 | MonuSeg | 0.3201 | 0.0085 | -0.0006 |  |
| leave_one_case_out_auto_box_reliability_v2 | all | 0.4256 | 0.021 | -0.0225 |  |
| conservative_auto_box_hybrid_v3 | IDRID | 0.5445 | 0.0471 | -0.0309 | 0.3 |
| conservative_auto_box_hybrid_v3 | MonuSeg | 0.3202 | 0.0085 | -0.0006 | 0.04 |
| conservative_auto_box_hybrid_v3 | all | 0.4323 | 0.0278 | -0.0157 | 0.17 |

v2 paired bootstrap vs SAM ViT-H auto_box:

```text
mean delta = -0.0225
95% CI     = [-0.0406, -0.0062]
p-value    = 0.0087
```

v3 paired bootstrap vs SAM ViT-H auto_box:

```text
mean delta = -0.0157
95% CI     = [-0.0329, -0.0001]
p-value    = 0.0619
```

结论：如果 v3 至少不低于 SAM ViT-H，说明我们可以用它作为安全部署策略；如果 v2/v3 仍低于 SAM ViT-H，则 Phase3 的真正创新点应转向更强的医学特征和 P0 baseline，而不是继续做浅层 selector。
