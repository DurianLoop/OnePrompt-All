# 论文证据索引

更新：2026-07-16  
原则：论文数字只从本索引列出的 CSV/JSON 读取；旧网页、截图和早期泄漏实验不作为最终证据。

## 1. 数据与协议

| 内容 | 证据 |
| --- | --- |
| MoNuSeg 官方 37 train + 14 test | `runs/research_direction/dataset_audit/monuseg_official_manifest.csv` |
| MoNuSeg 5-fold nested-CV 定义 | `runs/research_direction/prompt_policy_splits/monuseg_nested_cv5/protocol_metadata.json` |
| 每折 train/val/test episodes | `runs/research_direction/prompt_policy_splits/monuseg_nested_cv5/fold_*/episodes_{train,val,test}.csv` |
| 37 outer queries / 185 reference episodes 覆盖 | `runs/research_direction/nested_cv5_negative_budget/aggregate/coverage_audit.json` |
| MoNuSAC patient-disjoint 154/46/101 | `runs/research_direction/prompt_policy_splits/monusac_strict_patient_disjoint/protocol_metadata.json` |
| MoNuSAC patient/case assignments | `runs/research_direction/prompt_policy_splits/monusac_strict_patient_disjoint_ref1/case_assignments.csv` |

## 2. MoNuSeg 主结果

| 结论 | 数字 | 证据 |
| --- | --- | --- |
| Fold-specific anchor -> learned | 0.537458 -> 0.547555，delta=+0.010097，CI=[0.003870,0.018777] | `nested_cv5_negative_budget/aggregate/paired_summary.json` |
| 五折逐折结果 | f0 +.016459；f1 0；f2 +.009506；f3 +.015080；f4 +.009974 | `nested_cv5_negative_budget/aggregate/fold_summary.csv` |
| FG-only -> FG-BG | 0.473325 -> 0.538050，delta=+.064725，CI=[.046994,.084141] | `paper_artifacts_20260715/nested_cv_fg_vs_fgbg/paired_summary.json` |
| Global N5 -> learned | 0.538372 -> 0.547555，delta=+.009183，CI=[.000437,.019333] | `paper_artifacts_20260715/nested_cv_vs_global_n5/paired_summary.json` |
| Fixed budget curve | N0/N5/N10/N20 Dice=.538263/.538372/.528942/.517184 | `paper_artifacts_20260715/nested_cv_fixed_budget_summary.csv` |
| Per-query oracle | Dice=.552270；gate 关闭 66.1% fixed-to-oracle gap | `paper_artifacts_20260715/nested_cv_oracle_gap.csv` 与 `final_results_metadata.json` |
| Gate switch audit | 49 switches；平均增益 +.03812；42/7 switched win/loss | `nested_cv5_negative_budget/aggregate/switch_audit/` |

## 3. MoNuSAC 外部结果

| 结论 | 数字 | 证据 |
| --- | --- | --- |
| Fixed budgets | N0/N5/N10/N20=.528250/.544167/.539729/.537454 | `monusac_prompt_policy/negative_gate_ref1_grid_v2/summary.csv` |
| Frozen MoNuSeg gate, patch | N0 .528250 -> .536154，delta=.007904，CI=[.001635,.015518] | `negative_gate_ref1_frozen_v2/paired_analysis/paired_summary.json` |
| Frozen MoNuSeg gate, patient | .534951 -> .542568，delta=.007617，CI=[.001808,.014621] | `negative_gate_ref1_frozen_v2/patient_level_analysis/patient_summary.json` |
| Within-domain gate, patch | .528250 -> .534201，delta=.005951，CI=[-.000732,.013891] | `within_domain_negative_gate/policy/paired_analysis/paired_summary.json` |
| Within-domain gate, patient | .534951 -> .540510，delta=.005559，CI=[.000625,.011330] | `within_domain_negative_gate/policy/patient_level_analysis/patient_summary.json` |
| 外部边界 | zero-shot/domain gate 均低于 post-hoc global N5=.544167 | `paper_artifacts_20260715/final_performance_table.csv` |

## 4. 数值一致性与统计

| 内容 | 证据 |
| --- | --- |
| CPU/GPU coordinates 一致、max Dice diff=1.45e-4 | `nested_cv5_negative_budget/aggregate/device_consistency_audit.json` |
| 128 次 device 配对明细 | `nested_cv5_negative_budget/aggregate/device_consistency_pairs.csv` |
| SAM-Med2D learned gate 严格负结果 | `nested_cv5_sammed2d_negative_gate_n0n5/aggregate/paired_summary.json` |
| SAM ViT-B / SAM-Med2D 对照表 | `paper_artifacts_20260715/cross_segmenter_gate_table.csv` |
| Segmenter runtime/VRAM/参数量 | `paper_artifacts_20260715/runtime_table.csv` 与 `runtime_*.json` |
| 统一最终成对效应表 | `paper_artifacts_20260715/final_paired_effects.csv` |
| 统一性能表 | `paper_artifacts_20260715/final_performance_table.csv` |

## 5. 图像资产

| 图 | 位置 |
| --- | --- |
| 最终主结果与 forest plot | `runs/research_direction/paper_artifacts_20260715/final_main_results.png` |
| 五折 nested-CV | `.../nested_cv_forest.png` |
| MoNuSeg 成功/失败，含 reference mask | `.../monuseg_nested_gate_qualitative.png` |
| MoNuSAC 成功/失败，含 reference mask | `.../monusac_within_domain_gate_qualitative.png` |
| 定性图原始 N0/N5 overlays | `.../qualitative_inputs/{monuseg,monusac}/rendered/` |

## 6. 可复现代码

| 任务 | 脚本 |
| --- | --- |
| MoNuSeg nested splits | `code/benchmarks/build_monuseg_nested_cv_splits.py` |
| DINO fg/bg candidates | `code/benchmarks/run_monuseg_dense_prompt_experiment.py` |
| N0/N5/N10/N20 SAM grid | `code/benchmarks/evaluate_negative_prompt_ablation.py` |
| Negative-budget gate | `code/benchmarks/train_negative_budget_policy.py` |
| MoNuSeg 5-fold 一键执行 | `code/benchmarks/run_monuseg_nested_cv5_negative_budget.ps1` |
| SAM-Med2D N0/N5 5-fold | `code/benchmarks/run_monuseg_nested_cv5_sammed2d_negative_budget.ps1` |
| 五折聚合 | `code/benchmarks/aggregate_nested_cv_prompt_policy.py` |
| MoNuSAC strict splits | `code/benchmarks/build_monusac_strict_policy_splits.py` |
| MoNuSAC zero-shot | `code/benchmarks/run_monusac_negative_gate_zero_shot_ref1.ps1` |
| MoNuSAC within-domain | `code/benchmarks/run_monusac_within_domain_negative_gate.ps1` |
| Patient-level statistics | `code/benchmarks/analyze_monusac_patient_level.py` |
| 最终主表/主图 | `code/benchmarks/build_prompt_policy_final_assets.py` |
| Reference-aware 定性图 | `code/benchmarks/build_negative_gate_domain_qualitative.py` |
| 两候选推理成本 benchmark | `code/benchmarks/benchmark_negative_gate_runtime.py` |

## 7. Baseline 与创新边界

| 内容 | 证据 |
| --- | --- |
| FoB official code | `external/FoB_SAM/` commit `40d0e23...` |
| FoB 复现审计 | `docs/research/fob_baseline_reproducibility_audit_20260716.md` |
| PPO official release audit | `docs/research/ppo_reproducibility_audit_20260715.md` |
| SAT release audit | `docs/research/sat_baseline_reproducibility_audit_20260715.md` |
| GBMSeg-style adapted baseline | `runs/research_direction/baselines/gbmseg_style_adapted_test14/` |
| Fixed strategy cross-segmenter boundary | `runs/research_direction/cross_model/fixed_k40_lambda075/summary.json` |

当前不能写“优于 FoB/PPO/SAT 官方实现”。FoB 无病理核权重；PPO 状态键复现存在限制；SAT 尚无可运行官方实现。所有 adapted baseline 必须明确标注 adapted。

## 8. 写作文件

- 中文论文工作稿：`docs/research/prompt_policy_paper_draft_zh_20260715.md`
- 论文就绪判定：`docs/research/paper_readiness_decision_20260715.md`
- 发表清单：`docs/track/checklist_pub.md`
- FoB 强基线审计：`docs/research/fob_baseline_reproducibility_audit_20260716.md`

## 9. 验证命令

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\conda\python.exe -B -m pytest code\benchmarks -q -p no:cacheprovider
D:\conda\python.exe -B code\benchmarks\build_prompt_policy_final_assets.py
```

最终复核：2026-07-16，`112 passed`。
