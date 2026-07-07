# D:\SAM 文档阅读地图

日期：2026-06-25

这个目录已经按用途重新归类。现在建议把 `docs/README.md` 和 `docs/workspace_index.md` 当作入口，其他文档按需要打开。

## 1. 应该先看什么

### 第一遍：快速了解当前项目

1. `docs/research/teacher_direction_closed_loop_summary.md`
   - 先看这份。它是当前最重要的中文总报告，逐条回应老师的猜想，并总结已经完成的实验闭环。

2. `docs/research/research_direction_artifact_index.md`
   - 第二个看。它告诉你每个实验脚本在哪里、输出在哪里、如何复现。

3. `docs/research/research_direction_priority_plan.md`
   - 第三个看。它把后续研究方向按优先级排好了，适合决定下一步做什么。

### 第二遍：看实验细节

4. `docs/reports/sam3/sam3_teacher_minimal_validation_report.md`
   - 看 SAM3 text/point、`white girl`、`execute`、累计点、正负点 feedback 的最小验证。

5. `docs/reports/matcher/matcher_stage1_export_report.md`
   - 看 Matcher Stage 1 如何导出 matched points 和 heatmap。

6. `docs/reports/matcher/matcher_stage1_feedback_report.md`
   - 看正负点 feedback 是否改善 matching。

7. `docs/reports/matcher/matcher_stage2_sampling_report.md`
   - 看 top points 如何聚类和采样 prompt groups。

8. `docs/reports/matcher/matcher_stage2_sam_proposals_report.md`
   - 看 sampled prompts 如何送入 SAM 得到 mask proposals。

9. `docs/reports/matcher/matcher_stage3_feature_matching_report.md`
   - 看 instance-level feature matching 如何筛 proposal。

10. `docs/reports/matcher/matcher_multi_instance_horse_validation_report.md`
    - 看多实例自然图里如何得到 `K_pred=2`。

### 第三遍：准备继续写代码或做实验

11. `docs/ops/code_entrypoints.md`
    - 看当前应该从哪些脚本入口继续改。

12. `docs/ops/sam_workspace_cleanup_experiment_plan.md`
    - 看之前对工作区和实验整理的计划。

13. `docs/ops/cleanup_todo.md`
    - 看还剩哪些清理事项。

## 2. 目录分类

| 目录 | 内容 | 什么时候看 |
| --- | --- | --- |
| `docs/research/` | 当前研究主线、总报告、实验产物索引、方向优先级 | 想理解项目主线、给老师汇报、决定下一步 |
| `docs/reports/sam3/` | SAM3 text/point 最小验证报告 | 想看 SAM3 prompt 行为 |
| `docs/reports/matcher/` | Matcher Stage 1/2/3 和多实例实验报告 | 想看实验闭环细节 |
| `docs/ops/` | 代码入口、清理计划、工作区整理说明 | 想继续改代码或整理工程 |
| `docs/ui/` | workbench/UI 设计和图示 | 想改网页 demo 或界面 |
| `docs/workspace/` | 工作区环境和布局说明 | 想了解本机环境 |
| `docs/papers/`、`docs/papers2/`、`docs/papers3/` | 论文 PDF 和部分文本抽取 | 想补论文背景 |
| `docs/archive/` | 历史 phase 报告和旧 prompt selector 记录 | 需要追溯旧工作时再看 |
| `docs/superpowers/plans/` | 每次实现/整理计划 | 想看每一步怎么执行的 |
| `docs/setup_legacy/` | 旧依赖列表 | 只有复查旧环境时再看 |

## 3. 按任务选择阅读顺序

### 如果要给老师汇报

```text
docs/research/teacher_direction_closed_loop_summary.md
-> docs/research/research_direction_priority_plan.md
-> docs/reports/matcher/matcher_multi_instance_horse_validation_report.md
```

### 如果要复现实验

```text
docs/research/research_direction_artifact_index.md
-> docs/reports/sam3/sam3_teacher_minimal_validation_report.md
-> docs/reports/matcher/matcher_stage1_export_report.md
-> docs/reports/matcher/matcher_stage1_feedback_report.md
-> docs/reports/matcher/matcher_stage2_sampling_report.md
-> docs/reports/matcher/matcher_stage2_sam_proposals_report.md
-> docs/reports/matcher/matcher_stage3_feature_matching_report.md
```

### 如果要继续推进研究

```text
docs/research/research_direction_priority_plan.md
-> docs/research/teacher_direction_closed_loop_summary.md
-> docs/ops/code_entrypoints.md
-> docs/research/research_direction_artifact_index.md
```

### 如果要改 demo 或交互

```text
docs/ops/code_entrypoints.md
-> docs/ui/sam_workbench_ui_research_and_design.md
-> code/demos/medical_sam_click_app.py
```

### 如果要补论文背景

```text
docs/papers2/README.md
-> docs/papers2/text/03_SAM3_Segment_Anything_with_Concepts.txt
-> docs/papers2/text/13_Matcher_One_Shot_Feature_Matching.txt
-> docs/papers2/text/04_MedSAM_Segment_Anything_in_Medical_Images.txt
-> docs/papers2/text/14_One_Prompt_to_Segment_All_Medical_Images.txt
-> docs/papers2/text/15_UniverSeg_Universal_Medical_Image_Segmentation.txt
```

## 4. 最短阅读路线

如果只想花 15 分钟重新进入状态，按这个顺序：

```text
1. docs/research/teacher_direction_closed_loop_summary.md
2. docs/research/research_direction_artifact_index.md
3. docs/research/research_direction_priority_plan.md
```

如果只想看实验结果是否支撑老师猜想，按这个顺序：

```text
1. docs/reports/sam3/sam3_teacher_minimal_validation_report.md
2. docs/reports/matcher/matcher_stage1_feedback_report.md
3. docs/reports/matcher/matcher_stage3_feature_matching_report.md
4. docs/reports/matcher/matcher_multi_instance_horse_validation_report.md
```
