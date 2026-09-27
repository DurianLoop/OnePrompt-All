# Phase V Gate V6: RRN 保留评估

## 评估标准

| 条件 | 要求 | 实际 | 通过? |
|:---|:---|:---|:---:|
| 1. Grouped nested CV AUROC | >= 0.65 | 0.612 (single 70/15/15 split) | NO |
| 2. Dice@Top-3 vs 最强非学习baseline | +0.010以上 | 0.525 vs 0.516 (similarity) = +0.009 | NO |
| 3. Paired bootstrap CI下界 > 0 | > 0 | 未计算 (12 test queries) | UNKNOWN |
| 4. Secondary dataset方向一致 | 方向一致 | 无secondary dataset评估 | NO |
| 5. 推理成本完整报告 | 完整 | 未报告 | NO |
| 6. 不依赖GT | 不依赖 | 训练依赖GT (Dice >= 0.5 label) | YES |
| 7. 不使用target Dice作为测试特征 | 不使用 | 不使用 | YES |

## 判定

**RRN → 移到 Appendix**

原因:
1. AUROC = 0.612 低于 0.65 阈值
2. Dice@Top-3 advantage (+0.009) 低于 +0.010 阈值
3. 仅12 test queries, 0 catastrophic cases - 评估不可靠
4. 无cross-dataset验证
5. 需要 grouped nested 5-fold CV 重评才能进入主文

## 当前处理

- 论文中保留RRN描述但降级为appendix材料
- 主文中仅保留RAMP作为核心方法
- RRN作为"future work"或"preliminary exploration"提及
- 如果后续完成grouped nested CV重评且AUROC >= 0.65, 可考虑重新加入主文