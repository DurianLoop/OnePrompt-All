# Phase V Gate V5: External Baseline Fairness Audit

## Comparator Registry

| Method | 原任务 | 是否Reference-Guided | 是否Training-Free | 默认Backbone | 是否官方代码 | 改动 | Shared Backbone可行 | 主比较/诊断 |
|:---|:---|:---:|:---:|:---|:---:|:---|:---:|:---|
| GF-SAM | One-shot SAM prompting | YES | YES | SAM-ViT-H | YES | 无 | NO (SAM-H vs SAM-B) | 主比较 |
| MAUP | Medical image prompting | YES | YES | SAM-ViT-H | YES | 无 | NO (SAM-H vs SAM-B) | 主比较 |
| CycleSAM | Surgical scene segmentation | YES | YES | SAM-B | YES | 无 | YES | 主比较 |
| H-SAM | Hierarchical text segmentation | NO | NO (fine-tuned) | SAM-L | YES | 任务适配 | NO (SAM-L vs SAM-B) | 诊断 |
| SAM-Med2D | Medical image adaptation | YES | NO (fine-tuned) | SAM-B | YES | Checkpoint格式问题 | YES | 诊断 |
| SynPo | Medical image prompting | YES | YES | SAM-B | YES | 无 | YES | 主比较 |
| Matcher | One-shot matching | YES | YES | SAM-B | YES | 无 | YES | 主比较 |
| PerSAM | Personalized SAM | YES | YES | SAM-B | YES | 无 | YES | 主比较 |

## 公平性分类

### Group A: 任务高度匹配 (主比较)
- GF-SAM: reference-guided one-shot SAM prompting
- MAUP: reference-guided medical prompting
- CycleSAM: reference-guided segmentation
- 问题: GF-SAM和MAUP使用SAM-H backbone, RAMP使用SAM-B

### Group B: 可适配但任务不同 (诊断)
- H-SAM: 文本分割, 非reference-guided
- SAM-Med2D: 医学fine-tuned, checkpoint格式问题
- 问题: H-SAM不是reference-guided方法; SAM-Med2D checkpoint blocked

### Group C: 不适合做主SOTA比较
- 当前无

## 关键问题

1. **Backbone不匹配**: GF-SAM/MAUP使用SAM-H (641M), RAMP使用SAM-B (94M)
   - 论文声称"identical conditions"但实际上backbone不同
   - 修正: 已改为"shared reference-resampling protocol"

2. **SAM-Med2D checkpoint问题**: Checkpoint包含optimizer state, 需要weights_only=False
   - 当前状态: BLOCKED, 无法复现
   - 论文引用的是arXiv'23版本, 但数据来自Phase U

3. **H-SAM任务不匹配**: 原任务是hierarchical text segmentation, 非reference-guided
   - 论文将其列为prompting method比较, 不够准确
   - 建议: 移至diagnostic comparison

4. **CycleSAM**: 原任务是surgical scene segmentation
   - 与核分割任务不同, 但方法是通用的reference-guided
   - 可以保留

## 修正建议

1. 主比较表应分两个子表:
   - Table A: Shared-backbone comparison (SAM-B methods: CycleSAM, SynPo, Matcher, PerSAM)
   - Table B: Method-default comparison (GF-SAM/SAM-H, MAUP/SAM-H, H-SAM/SAM-L)

2. 明确标注backbone差异

3. SAM-Med2D标记为"results from original paper, not reproduced in our environment"

4. H-SAM标记为"cross-task diagnostic"