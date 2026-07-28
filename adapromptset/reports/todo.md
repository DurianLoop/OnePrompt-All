# AdaPromptSet 下一阶段总指令
## Phase VV4 — V4 证据取证、资产归档与 Claim-Safe 论文重建

> **用途**：将本文档完整交给 AI，开启 `/goal` 模式执行。  
> **项目根目录**：`D:\SAM\adapromptset`  
> **唯一归档目录**：`D:\SAM\adapromptset\VV4`
>
> **本阶段首要目标不是修改论文，也不是补新实验。**
>
> 首先必须回答：
>
> > `full_draft_v4` 中每一个数字、图表、方法组件、理论结论和贡献，究竟能否追溯到真实代码、真实预测、真实数据与真实统计产物？
>
> 只有完成证据取证后，才允许决定：
>
> 1. V4 哪些内容可以原样保留；
> 2. 哪些内容可以降级后保留；
> 3. 哪些内容已被后续 Phase W / AA / AB 推翻；
> 4. 哪些内容找不到任何实验来源，疑似占位、误写或 AI 自动补全；
> 5. 是否能以 V4 的叙事和版式为骨架，重建一篇证据真实的新论文。

---

# 0. 本阶段最终目标

将此前所有与 V4 论文有关的：

- 论文版本；
- LaTeX / Markdown / Word / PDF 源文件；
- 代码；
- 配置；
- 数据切分；
- reference manifests；
- per-query predictions；
- 中间特征；
- 风险分数；
- checkpoints；
- 指标 JSON / CSV；
- 表格；
- 图片；
- 日志；
- 命令；
- Git commit；
- 决策文档；
- Phase 报告；
- Claim Registry；
- 后续反证与修正文档；

统一整理到：

```text
D:\SAM\adapromptset\VV4
```

并建立一条完整的：

```text
V4正文句子
→ Claim
→ 实验定义
→ 代码入口
→ 配置
→ 输入数据
→ 原始预测
→ 统计脚本
→ 表格/图片
→ 后续审计结果
→ 最终裁决
```

证据链。

---

# 1. 不可违反的原则

## 1.1 只能复制，不能移动或删除

不得：

- 移动原项目文件；
- 删除旧结果；
- 覆盖旧预测；
- 修改旧 JSON / CSV；
- 修改旧 commit；
- 将新结果伪装成旧结果；
- 重命名原始文件后失去来源信息。

所有资产进入 `VV4` 时必须：

- 保留原始相对路径；
- 记录原始绝对路径；
- 记录 Git commit；
- 记录文件大小；
- 记录修改时间；
- 计算 SHA256；
- 标记是原始文件、复制文件、重建文件还是新审计文件。

---

## 1.2 审计阶段不得补实验掩盖缺口

如果 V4 某个数字找不到来源：

```text
NO_EVIDENCE_FOUND
```

不得先重跑一个相似实验，再声称找到了原始来源。

重新运行的结果只能标记为：

```text
REPRODUCED_AFTER_V4
```

不能标记为：

```text
ORIGINAL_V4_EVIDENCE
```

---

## 1.3 不得轻易使用“捏造”标签

只有满足以下条件之一，才允许标记：

```text
LIKELY_AI_FABRICATED
```

- 数字出现在论文中，但整个仓库、Git 历史、日志、报告和预测资产均无对应来源；
- 数字与代码实际输出不可能同时成立；
- 表格存在模板占位符或复制粘贴痕迹；
- 所有所谓不同实验实际引用同一结果文件；
- checkpoint 从未成功加载，却报告了正式实验结果；
- 图中数据与对应 CSV / JSON 不一致；
- 后续报告明确确认该数字来自 builder 占位或错误复用。

否则使用更中性的状态：

```text
UNVERIFIABLE
SOURCE_NOT_FOUND
CONTRADICTED_BY_LATER_AUDIT
PLACEHOLDER_SUSPECTED
```

---

## 1.4 Formal Test 继续锁定

不得访问：

- 任何标记为 Formal_Test 的数据；
- 未授权的新数据；
- 未授权 checkpoint；
- 未授权外部下载。

需要访问时必须暂停并请求用户批准。

---

# 2. 创建 VV4 目录结构

首先创建：

```text
D:\SAM\adapromptset\VV4
├── 00_README/
├── 01_V4_PAPER_ORIGINAL/
│   ├── pdf/
│   ├── latex/
│   ├── markdown/
│   ├── figures/
│   ├── tables/
│   ├── bibliography/
│   └── build_logs/
├── 02_CODE_ORIGINAL/
│   ├── exact_v4_code/
│   ├── related_pre_v4_code/
│   ├── related_post_v4_code/
│   └── git_patches/
├── 03_CONFIGS_AND_PROTOCOLS/
│   ├── frozen_configs/
│   ├── experimental_configs/
│   ├── split_manifests/
│   ├── reference_manifests/
│   └── environment/
├── 04_DATA_AND_PREDICTIONS/
│   ├── raw_data_manifests/
│   ├── data_links/
│   ├── query_splits/
│   ├── reference_assignments/
│   ├── features/
│   ├── predictions/
│   ├── risk_scores/
│   └── checkpoints/
├── 05_RESULTS_ORIGINAL/
│   ├── json/
│   ├── csv/
│   ├── tables/
│   ├── figures/
│   ├── logs/
│   └── qualitative_examples/
├── 06_REPORTS_AND_DECISIONS/
│   ├── pre_v4/
│   ├── v4_generation/
│   ├── phaseV/
│   ├── phaseW/
│   ├── phaseAA/
│   ├── phaseAB/
│   └── claim_registries/
├── 07_CLAIM_FORENSICS/
│   ├── extracted_claims/
│   ├── claim_to_artifact/
│   ├── number_provenance/
│   ├── table_forensics/
│   ├── figure_forensics/
│   └── hallucination_audit/
├── 08_REPRODUCTION/
│   ├── environments/
│   ├── commands/
│   ├── reproduced_outputs/
│   ├── diffs/
│   └── failures/
├── 09_V4_SALVAGE/
│   ├── keep/
│   ├── qualify/
│   ├── rewrite/
│   ├── remove/
│   └── needs_new_experiment/
├── 10_REBUILT_PAPER/
│   ├── outline/
│   ├── markdown/
│   ├── latex/
│   ├── figures/
│   ├── tables/
│   └── compiled/
└── 99_MANIFESTS/
    ├── all_files_manifest.csv
    ├── all_files_manifest.json
    ├── sha256_manifest.csv
    ├── git_commit_manifest.csv
    ├── missing_assets.csv
    └── collection_log.md
```

---

# 3. WP0 — 仓库与 Git 历史全量扫描

## 3.1 扫描整个项目

扫描：

```text
D:\SAM\adapromptset
```

包括：

- 当前工作树；
- 所有 Git branches；
- tags；
- stash；
- reflog 可访问记录；
- deleted-but-tracked files；
- untracked files；
- ignored 但存在的结果目录；
- zip / tar / backup；
- 临时 paper build 文件。

不得只看当前 branch。

---

## 3.2 关键词扫描

至少搜索以下关键词及大小写变体：

```text
full_draft_v4
Reference Reliability Estimation
Reference Matters
RAMP
RRN
Reference-Adaptive Multi-Prompt Fusion
Reference Reliability Network
anti-correlation
catastrophic failure
CFR
Prediction Consistency
MC Dropout
Deep Ensemble
SAM-Med2D
GF-SAM
MAUP
CycleSAM
H-SAM
0.446
0.388
0.279
0.412
0.405
0.438
56-84
55.9
37.5
84.4
0.612
0.525
0.82
0.02
six segmenters
four matchers
calibration-free
training-free
identical conditions
deployment envelope
Pascal-5i
COCO-20i
```

扫描：

- `.py`
- `.ipynb`
- `.json`
- `.csv`
- `.yaml`
- `.yml`
- `.toml`
- `.md`
- `.tex`
- `.log`
- `.txt`
- `.ps1`
- `.bat`
- `.pdf`
- `.zip`

---

## 3.3 Git 时间线

生成：

```text
VV4/06_REPORTS_AND_DECISIONS/v4_generation/v4_git_timeline.md
VV4/99_MANIFESTS/git_commit_manifest.csv
```

每个可能与 V4 有关的 commit 记录：

| Commit | Date | Branch | Author | Files | Message | V4 relevance | Evidence type |
|---|---|---|---|---|---|---|---|

还原 V4 形成过程：

```text
哪一个 commit 生成正文
→ 哪一个 commit 生成表格
→ 哪一个 commit 生成图片
→ 哪一个 commit 写入最终数字
```

---

# 4. WP1 — 收集所有 V4 相关资产

## 4.1 收集范围

不仅收集文件名包含 `v4` 的资产。

任何被 V4：

- 正文引用；
- 图表引用；
- 数字引用；
- 方法引用；
- 结论引用；
- bibliography 引用；
- 构建脚本引用；

的文件都必须进入 VV4。

---

## 4.2 文件复制规范

每个文件记录：

```text
source_absolute_path
source_relative_path
destination_path
file_size
mtime
sha256
git_commit
git_branch
tracked/untracked
original/rebuilt
collection_reason
```

使用复制而不是移动。

推荐 PowerShell / Windows 命令：

```powershell
Copy-Item -LiteralPath $src -Destination $dst -Force
Get-FileHash -Algorithm SHA256 $dst
```

目录较大时可使用：

```powershell
robocopy $src $dst /E /COPY:DAT /DCOPY:T /R:1 /W:1
```

不得使用 `/MOVE`、`/PURGE` 或会删除目标文件的参数。

---

## 4.3 原始大数据处理

对于 MoNuSAC、MoNuSeg、IDRID、GlaS、CPM17 等共享原始数据：

- 不要默认复制整个大型原始数据集；
- 必须复制 V4 使用的：
  - split manifest；
  - query IDs；
  - reference IDs；
  - patient IDs；
  - hashes；
  -预处理脚本；
  -处理后输入清单；
- 在 `04_DATA_AND_PREDICTIONS/data_links/` 创建只读路径说明或 Windows junction；
- 若用户要求物理复制且预计超过 50 GB，暂停请求批准；
- V4 产生的 predictions、features、metrics 必须物理复制。

---

# 5. WP2 — 锁定 V4 的准确文本版本

当前可能存在多个同名 V4 PDF。

必须比较：

```text
full_draft_v4.pdf
full_draft_v4(1).pdf
所有 Git 历史中的 V4 PDF
所有 LaTeX 编译源
```

生成：

```text
VV4/01_V4_PAPER_ORIGINAL/v4_version_registry.csv
VV4/01_V4_PAPER_ORIGINAL/v4_pdf_hashes.csv
VV4/01_V4_PAPER_ORIGINAL/v4_text_diff.md
VV4/01_V4_PAPER_ORIGINAL/canonical_v4_decision.md
```

选出一个：

```text
CANONICAL_V4
```

并说明：

- 为什么它是主审计对象；
- 与其他 V4 的差异；
- 哪个版本与用户所说的“效果最好”一致；
- 它对应的源代码 / LaTeX 是否存在。

如果只找到 PDF，找不到源文件，明确：

```text
V4_SOURCE_MISSING_PDF_ONLY
```

---

# 6. WP3 — V4 全文逐句 Claim 提取

必须解析 V4 全文，不仅检查摘要。

将所有可以被验证或反驳的句子提取为 Claim。

范围包括：

- Abstract；
- Contributions；
- Introduction；
- Related Work；
- Method；
- Lemmas / Theorem；
- Experimental Setup；
- 每个表格；
- 每个图；
- Results；
- Ablation；
- Cross-dataset；
- Discussion；
- Limitations；
- Conclusion。

---

## 6.1 Claim 分类

每个 Claim 赋予：

```text
C-V4-0001
C-V4-0002
...
```

生成：

```text
VV4/07_CLAIM_FORENSICS/extracted_claims/v4_claim_registry.csv
VV4/07_CLAIM_FORENSICS/extracted_claims/v4_claim_registry.md
```

字段：

| Claim ID | Page | Section | 原文 | Claim 类型 | 数值 | 数据集 | Pipeline | 统计单位 | 预期来源 |
|---|---:|---|---|---|---|---|---|---|---|

Claim 类型至少包括：

```text
Problem-definition
Novelty
Method
Mathematical
Numerical
Comparative
Statistical
Cross-pipeline
Cross-dataset
Efficiency
Calibration
Deployment
Reproducibility
Literature
```

---

# 7. WP4 — 每个 Claim 建立证据链

对每个 Claim 找出：

1. 原始代码；
2. 配置；
3. 输入 manifest；
4. 数据角色；
5. 原始预测；
6. 原始 JSON / CSV；
7. 统计代码；
8. 表格 builder；
9. 图片 builder；
10. Git commit；
11. 日志；
12. 后续 Phase 审计；
13. 是否能够复现。

生成：

```text
VV4/07_CLAIM_FORENSICS/claim_to_artifact/claim_artifact_matrix.csv
VV4/07_CLAIM_FORENSICS/claim_to_artifact/claim_evidence_graph.md
```

字段：

| Claim ID | Code | Config | Data | Prediction | Metrics | Figure/Table | Commit | Later audit | Status |
|---|---|---|---|---|---|---|---|---|---|

---

# 8. WP5 — 高风险 Claim 专项审计

以下 V4 主张必须逐项审计，不得遗漏。

---

## 8.1 “六个 segmenters、四个 matchers”

V4 声称：

```text
consistently anti-correlated across six segmenters and four matchers
```

检查：

- 实际运行了多少 segmenters；
- 实际运行了多少 matchers；
- 每个组合是否有 prediction files；
- 每个组合 query 数；
- 是否只是正文列名，没有真实数据；
- 是否把未运行模型当成已运行；
- 是否把 matcher/segmenter 单因素结果误写成全 factorial。

最终状态只能为：

```text
VERIFIED
PARTIALLY_VERIFIED
CONTRADICTED
UNVERIFIABLE
LIKELY_AI_FABRICATED
```

---

## 8.2 Anti-correlation 数字

V4 声称：

```text
Pearson r = -0.43 to -0.71, p < 0.001
```

检查：

- 每个 r 的原始 query-level数据；
- 是否使用 episode-level伪重复；
- 是否 5 seeds 被当成独立 query；
- Holm correction；
- partial correlation；
- null model；
- Phase V/W/AA/AB 修正；
- SAM2 是否不支持；
- mechanical coupling 是否解释主要相关性。

不得只确认数字存在，还要确认统计解释合法。

---

## 8.3 三个 Lemmas / Theorem

检查：

- LaTeX 原始定义；
- 是否有严格证明；
- entropy 是否能决定空间 dispersion；
- 是否存在相同 entropy、不同空间排列的反例；
- SAM 是否存在文中所说的有限 point receptive-field radius；
- theorem 是否真的从 lemmas 推出；
- 后续 Phase 是否已删除。

分类：

```text
MATHEMATICALLY_VALID
EMPIRICAL_HYPOTHESIS_ONLY
INVALID
UNPROVEN
```

---

## 8.4 RAMP 组件与推理顺序

V4 声称：

```text
risk-adaptive reference weighting
adaptive prompt configuration
risk-stratified consensus
```

检查：

- 实际代码是否实现；
- weighting 是否改变任何权重；
- fusion 是否改变任何 mask；
- prompt adaptation 是否真实进入 inference；
- 风险特征是否依赖已经产生的 masks；
- 是否存在循环依赖；
- component ablation；
- mask hash；
- Phase W 是否确认 weighting / fusion 无增益；
- Full RAMP 是否被简单 Prediction Consistency 超越。

---

## 8.5 “training-free”

检查：

- 是否训练 gradient boosting；
- 是否使用 labeled risk targets；
- 是否训练 RRN；
- 是否仅 foundation models frozen。

允许替代：

```text
no foundation-model fine-tuning
parameter-free risk score
```

---

## 8.6 “calibration-free”

检查：

- 是否输出概率；
- 是否使用阈值；
- IDRID ECE；
- 是否发生 calibration drift；
- coverage ranking 与概率 calibration 是否混淆。

---

## 8.7 “CFR 降低 56–84%”

检查：

- 所有 pipeline 的真实值；
- 是否存在 37.5%；
- 是 absolute 还是 relative；
- coverage；
- baseline；
- query count；
- CI；
- 是否选择性省略最差 pipeline；
- 后续 Phase 的 88.6% Prediction Consistency 是否属于不同方法，不能倒填给 RAMP。

---

## 8.8 RRN

V4 声称：

```text
AUROC = 0.612
Dice@Top-3 = 0.525
narrows the oracle gap
```

检查：

- 数据 split；
- 测试 query 数；
- catastrophic events；
- grouped CV；
- Phase W AUROC 0.573；
- 是否 appendix-only；
- 0.525 是否与不同 baseline / split 混用。

---

## 8.9 外部 Baseline 表

逐项检查：

```text
Top-K 0.446
GF-SAM 0.388
MAUP 0.279
CycleSAM 0.412
H-SAM 0.405
SAM-Med2D 0.438
MC Dropout 10.7%
Deep Ensemble 6.3%
```

每一行必须找到：

- 官方代码；
- commit；
- checkpoint；
-环境；
- inference log；
- per-query prediction；
- metric file；
-统计脚本；
- backbone；
- prompt budget；
- reference protocol。

重点：

- SAM-Med2D checkpoint 是否从未成功加载；
- GF-SAM / MAUP 是否使用 SAM-H；
- H-SAM 是否使用 SAM-L；
- 是否真的属于 identical conditions；
- MC Dropout / Deep Ensemble 是否真实运行，还是表格占位；
- 是否存在方法名称或 venue 误写。

---

## 8.10 Pascal-5i / COCO-20i

V4 声称自然图像跨域结果时，检查：

- 是否只有 fold-0；
- query 数；
- reference协议；
- prediction文件；
- anti-correlation数字；
- CFR reduction数字；
- 其他 folds 是否缺失；
- 是否被后续报告标为 pending。

---

## 8.11 Deployment envelope

检查：

```text
5%–60%
```

是否：

- 来自真实 coverage sweep；
- 来自理论推导；
- 来自经验观察；
- 有跨数据集支持；
- 后续 Phase 是否仅验证 50–100% coverage。

---

## 8.12 “First RRE / first benchmark”

必须：

- 查 V4 当时的 literature audit；
- 区分问题定义和方法先例；
- 检查 ProSAM、VRP-SAM、SAMIX 等相邻工作；
- 最终限定 scope。

如果需要最新文献，暂停并请求用户允许联网检索。

---

# 9. WP6 — 表格取证

对 V4 每张表建立：

```text
VV4/07_CLAIM_FORENSICS/table_forensics/table_<N>/
├── paper_table.csv
├── source_files.md
├── source_metrics.csv
├── reproduction_command.md
├── reproduced_table.csv
├── diff.csv
└── verdict.md
```

检查：

- 表格是否由代码生成；
- 是否手工填写；
- 是否包含 hard-coded 数字；
- 是否引用不存在的 JSON；
- 是否把不同 split 混在一起；
- 是否把不同 backbone 放进“identical conditions”；
- p 值是否合法；
- 统计单位是否 query-level；
- 是否有选择性汇报。

---

# 10. WP7 — 图片取证

对 V4 每张图片建立：

```text
VV4/07_CLAIM_FORENSICS/figure_forensics/figure_<N>/
├── paper_figure.png
├── generation_script.py
├── source_data.csv
├── reproduced_figure.png
├── pixel_or_data_diff.md
└── verdict.md
```

检查：

- 图是否由真实数据生成；
- 图上每个点的来源；
- 图例中模型是否真实运行；
- panel 与 caption 是否一致；
- 图中数字是否和表格一致；
- 是否存在模拟图、占位图或手绘数据；
- 是否存在截断坐标或误导性 scale。

---

# 11. WP8 — 原始结果复现

## 11.1 复现层级

每个关键结果分三层：

### Level 1：Artifact Replay

直接读取原始 predictions，重算 metric。

### Level 2：Pipeline Replay

使用原配置和缓存 features，重新生成 predictions。

### Level 3：Full Reproduction

从输入图像、reference mask、checkpoint 开始完整运行。

---

## 11.2 优先复现结果

优先顺序：

1. Top-K Dice = 0.446；
2. Reference variation 0.82 → 0.02；
3. Catastrophic prevalence >14%；
4. Prediction Consistency / disagreement；
5. V4 CFR reduction；
6. 外部 baseline 表；
7. RRN；
8. MC Dropout / Deep Ensemble；
9. Pascal / COCO。

---

## 11.3 复现输出

```text
VV4/08_REPRODUCTION/reproduction_registry.csv
```

字段：

| Result ID | Original value | Reproduced value | Difference | Level | Same split | Same code | Verdict |
|---|---:|---:|---:|---|---:|---:|---|

容差必须预先定义。

---

# 12. WP9 — V4 与后续 Phase 对照

必须将 V4 的每个主张与以下材料对照：

```text
Phase V Claim Registry
Phase W Final Experimental Closure
Phase AA Final Report
Phase AB Final Report
Phase T Final Publication Report
Phase S Final Research Summary
```

生成：

```text
VV4/06_REPORTS_AND_DECISIONS/claim_registries/v4_vs_later_phases.csv
```

字段：

| Claim ID | V4 claim | Phase V | Phase W | Phase AA | Phase AB | Final interpretation |
|---|---|---|---|---|---|---|

特别确认：

- V4 的 80-query “holdout”是否实际为 Development Cohort；
- Phase T 的 n=40 unseen holdout 与后来 80-query development 是否为不同 cohort；
- 不得将不同 Phase 的 holdout、validation、development 数字混用；
- 不得用 Phase AB 的 Prediction Consistency 88.6% 替换 V4 的 RAMP 数字；
- 不得用后来结果反向证明当时不存在的代码。

---

# 13. WP10 — AI 瞎写 / 占位 / 误写审计

生成：

```text
VV4/07_CLAIM_FORENSICS/hallucination_audit/hallucination_registry.csv
VV4/07_CLAIM_FORENSICS/hallucination_audit/hallucination_report.md
```

每项分类：

```text
A — VERIFIED_ORIGINAL
B — VERIFIED_WITH_QUALIFICATION
C — REPRODUCIBLE_BUT_NOT_INDEPENDENT
D — PARTIALLY_SUPPORTED
E — CONTRADICTED_BY_LATER_AUDIT
F — UNVERIFIABLE_NO_SOURCE
G — PLACEHOLDER_OR_RESULT_REUSE_SUSPECTED
H — LIKELY_AI_FABRICATED
I — REQUIRES_NEW_EXPERIMENT
J — LITERATURE_CLAIM_REQUIRES_WEB_VERIFICATION
```

`LIKELY_AI_FABRICATED` 必须提供具体依据，不能凭印象。

最终统计：

```text
V4 total claims
verified %
qualified %
contradicted %
unverifiable %
suspected fabricated %
requires new experiments %
```

---

# 14. WP11 — V4 Salvage Map

完成取证后，将 V4 每一节分类：

```text
KEEP
KEEP_WITH_MINOR_EDIT
KEEP_WITH_MAJOR_QUALIFICATION
REWRITE_FROM_PHASE_AB
MOVE_TO_APPENDIX
REMOVE
REQUIRES_NEW_EXPERIMENT
```

生成：

```text
VV4/09_V4_SALVAGE/v4_section_salvage_map.csv
VV4/09_V4_SALVAGE/v4_section_salvage_map.md
```

至少给出：

| V4 section | 原主题 | 证据状态 | 处理方式 | 新内容来源 |
|---|---|---|---|---|

推荐初始方向，但必须由证据裁决：

### 可能保留

- V4 的问题动机；
- reference choice 的部署场景；
- RRE problem formulation；
- related-work 的问题空白；
- reference-resampling protocol 的基本框架；
-部分 baseline 讨论。

### 预计重写

- Abstract；
- Contributions；
- Method；
- Main experiments；
- Cross-pipeline；
- Cross-dataset；
- Discussion；
- Conclusion。

### 预计删除或 Appendix

- universal anti-correlation；
- formal lemmas / theorem；
- Full RAMP 主线；
- weighting / fusion；
- RRN 主文；
- calibration-free；
- identical conditions；
- SAM-Med2D；
-未经验证自然图像结果；
- deployment envelope。

---

# 15. WP12 — 决定是否能以 V4 为骨架重建

最终只能选择一个状态：

## Route 1

```text
VV4_RECONSTRUCTABLE_FROM_V4_SKELETON
```

条件：

- V4 的 Introduction / RRE / protocol 可保留；
- 关键基础数字可复现；
- 主要问题只是 claim 过强；
- 可用 Phase AB 结果替换方法主线。

## Route 2

```text
VV4_PARTIALLY_RECONSTRUCTABLE
```

条件：

- 只有部分叙事和基础实验可靠；
- 大部分方法、理论和 baseline 需重建；
- V4 仅作为写作参考。

## Route 3

```text
VV4_SUBSTANTIALLY_UNSUPPORTED
```

条件：

- 大量主表、主图或方法数字无来源；
- 关键代码不存在；
- 多项结果疑似占位；
- 无法建立主要证据链。

## Route 4

```text
VV4_AUDIT_BLOCKED
```

条件：

- Git 历史缺失；
- 原始磁盘资产丢失；
- 关键文件无法读取；
- 需要用户提供外部备份。

---

# 16. WP13 — 新论文骨架

只有 Route 1 或 Route 2 才执行。

不要直接编辑 V4 原文件。

在：

```text
D:\SAM\adapromptset\VV4\10_REBUILT_PAPER
```

建立新稿。

推荐题目：

```text
When References Disagree:
Prediction Consistency for Selective Visual Prompting
```

推荐主线：

```text
V4 的问题动机和 RRE 框架
+
Phase AB 的 Prediction Consistency
+
coverage-based abstention
+
5 pipeline architecture-dependent boundary
+
跨数据集边界
```

最终三项贡献候选：

1. RRE 问题与 reference-resampling evaluation；
2. Prediction Consistency 这一简单、GT-free、无需额外学习的 query-level risk score；
3. 五个 pipeline 和多个医学数据集上的 coverage-risk 评估及适用边界。

不得写：

- universal；
- holdout-confirmed，除非存在真实 fresh cohort；
- across all models；
- cross-dataset generalization；
- training-free segmentation；
- calibration-free；
- SOTA；
- identical conditions。

---

# 17. 本阶段禁止直接补的新实验

在 V4 真实性审计完成前，不运行：

- SAM3；
- DINOv3；
- GlaS；
- CPM17；
- Formal Test；
- 新的 uncertainty baseline；
- 新的 RRN；
- 新的 RAMP；
- 新 action bank；
- 新 policy。

先回答“旧结果是真的吗”，再决定“还缺什么”。

---

# 18. 审计完成后的实验建议文件

只输出计划，不立即执行：

```text
VV4/09_V4_SALVAGE/required_new_experiments.md
```

每项新实验说明：

| Experiment | 修复哪个 Claim | 是否 P0 | 数据需求 | 模型需求 | 预计成本 | 是否需用户批准 |
|---|---|---:|---|---|---:|---:|

优先可能包括：

- Fresh confirmatory cohort；
- Prediction Consistency vs standard uncertainty；
- 一个现代 DINOv3 / SAM3 pipeline；
- 外部数据完整 multi-seed；
- qualitative high/low consistency cases；
- latency / reference-cost audit。

---

# 19. 最终必须生成的核心文件

```text
D:\SAM\adapromptset\VV4\00_README\VV4_AUDIT_README.md
D:\SAM\adapromptset\VV4\99_MANIFESTS\all_files_manifest.csv
D:\SAM\adapromptset\VV4\99_MANIFESTS\sha256_manifest.csv
D:\SAM\adapromptset\VV4\07_CLAIM_FORENSICS\extracted_claims\v4_claim_registry.csv
D:\SAM\adapromptset\VV4\07_CLAIM_FORENSICS\claim_to_artifact\claim_artifact_matrix.csv
D:\SAM\adapromptset\VV4\07_CLAIM_FORENSICS\hallucination_audit\hallucination_report.md
D:\SAM\adapromptset\VV4\09_V4_SALVAGE\v4_section_salvage_map.md
D:\SAM\adapromptset\VV4\08_REPRODUCTION\reproduction_registry.csv
D:\SAM\adapromptset\VV4\VV4_FINAL_AUDIT_REPORT.md
D:\SAM\adapromptset\VV4\VV4_FINAL_AUDIT_REPORT.json
```

---

# 20. 最终报告必须回答的问题

1. Canonical V4 是哪个文件？
2. V4 的 LaTeX / Markdown 源是否存在？
3. V4 共有多少个可验证 Claim？
4. 多少 Claim 有完整原始证据？
5. 多少 Claim 只有后来重跑结果？
6. 多少 Claim 被后续 Phase 推翻？
7. 多少数字找不到来源？
8. 是否发现 hard-coded / placeholder 数字？
9. 是否发现不同实验复用同一结果文件？
10. “六个 segmenters、四个 matchers”是否真实？
11. Anti-correlation 是否真实、统计是否合法？
12. Lemmas / Theorem 是否成立？
13. RAMP 的 weighting / fusion / prompt adaptation 是否真实执行？
14. RAMP CFR 数字是否可复现？
15. Full RAMP 是否优于简单风险评分？
16. RRN 0.612 / 0.525 是否可复现？
17. 外部 baseline 每一行是否有真实预测？
18. SAM-Med2D 0.438 来源是什么？
19. MC Dropout 10.7% 和 Deep Ensemble 6.3% 是否真实运行？
20. Pascal / COCO 结果是否真实且完整？
21. “training-free”是否准确？
22. “calibration-free”是否准确？
23. “identical conditions”是否准确？
24. 哪些 V4 内容可以直接保留？
25. 哪些必须降级？
26. 哪些必须删除？
27. 哪些需要新实验？
28. V4 是否存在 AI 瞎写或占位内容？
29. V4 能否作为最终论文骨架？
30. 推荐的下一篇论文主线是什么？

---

# 21. 最终终端输出格式

最终终端只回复：

1. VV4 final state；
2. Canonical V4；
3. 收集文件数量与总大小；
4. Claim 总数；
5. Verified 数量；
6. Qualified 数量；
7. Contradicted 数量；
8. Unverifiable 数量；
9. Suspected fabricated 数量；
10. V4 关键数字复现结果；
11. RAMP 裁决；
12. RRN 裁决；
13. Anti-correlation 裁决；
14. Theory 裁决；
15. External baseline 裁决；
16. V4 salvage ratio；
17. 推荐论文路线；
18. 需要的新实验；
19. Key artifacts；
20. Unresolved blockers。

---

# 22. 必须暂停用户确认的情况

1. 需要复制超过 50 GB 原始数据；
2. 需要下载外部 checkpoint；
3. 需要联网验证最新文献；
4. 需要访问 Formal Test；
5. 需要恢复已删除 Git objects；
6. 需要从外部硬盘或云盘读取备份；
7. 发现 V4 主表疑似来源于无记录的外部脚本；
8. 发现关键数据可能包含 patient leakage；
9. 发现原始预测文件已经被覆盖；
10. 准备运行任何新实验。

---

# 23. 严格执行顺序

1. 创建 `VV4` 目录；
2. 建立 preregistration 与只读规则；
3. 扫描工作树、Git branches、tags、reflog；
4. 锁定 Canonical V4；
5. 收集所有相关文件并计算 SHA256；
6. 提取 V4 全文 Claims；
7. 建立 Claim → Artifact 证据链；
8. 审计高风险主张；
9. 取证每张表；
10. 取证每张图；
11. 执行 Artifact Replay；
12. 在必要时执行 Pipeline Replay；
13. 与 Phase V/W/AA/AB 对照；
14. 输出 AI 瞎写 / 占位审计；
15. 生成 V4 Salvage Map；
16. 裁决 V4 是否可作为论文骨架；
17. 只在审计通过后创建新论文骨架；
18. 不运行新实验；
19. 输出完整最终报告；
20. 输出唯一最终状态。

本阶段最重要的原则：

> 不因为 V4 写得漂亮就相信它，也不因为后续路线改变就全部否定它。逐句、逐数、逐表、逐图追溯证据，保留真实的部分，删除或降级没有证据的部分，再决定是否以 V4 为骨架重建论文。
