# Phase V Contradiction Log

## P0-1: 全文数字一致性审计 — 矛盾清单

### 矛盾 1: Anti-correlation Pearson r 范围
- **文中声称**: "Pearson r = -0.43 to -0.71 (p<0.001)" (摘要 L53)
- **实际数据**: theory_verification_results.json
  - SAM-B, P20-N1: r = -0.491 (p < 0.001) ✓
  - SAM-L, P20-N1: r = -0.216 (p = 0.055) ✗ 不显著
  - SAM2-Small, P20-N1: r = -0.182 (p = 0.105) ✗ 不显著
  - SAM2-Tiny, P20-N1: r = 0.012 (p = 0.913) ✗ 不显著
  - DINOv2-B+SAM-B, P20-N1: r = 0.074 (p = 0.516) ✗ 不显著
- **严重性**: **P0-CRITICAL** — 摘要核心主张严重不实
- **修正**: 改为 "r = -0.18 to -0.49, significant for SAM-B (p<0.001)"

### 矛盾 2: 模型数量
- **文中声称**: "six segmenters and four matchers" (摘要 L53, 正文 L127-128)
- **实际数据**: 4 segmenters (SAM-B, SAM-L, SAM2-T, SAM2-S), 2 matchers (DINOv2-L, DINOv2-B)
- **严重性**: **P0-CRITICAL** — 过度声称
- **修正**: 改为 "four segmenters and two matchers"

### 矛盾 3: CFR reduction 范围省略
- **文中声称**: "56--84%" (摘要 L59, 正文 L152)
- **实际数据**: L+SAM-B=55.9%, L+SAM-L=84.4%, L+SAM2-T=37.5%, B+SAM-B=55.9%
- **严重性**: **P0-CRITICAL** — 选择性省略最差结果 37.5%
- **修正**: 改为 "37.5--84.4%" 或按 pipeline 分别报告

### 矛盾 4: "training-free" 表述
- **文中声称**: RAMP is "training-free" (摘要 L58, 正文多处)
- **实际**: RAMP 使用 gradient boosting (M2-3) 训练分类器
- **严重性**: **P0-CRITICAL** — 虚假声称
- **修正**: 改为 "frozen foundation models with lightweight supervised risk estimator"

### 矛盾 5: "calibration-free" 表述
- **文中声称**: "calibration-free" (摘要 L70, 正文 L151)
- **实际**: IDRID ECE = 0.477; 使用阈值 0.3/0.7
- **严重性**: **P0-CRITICAL** — 虚假声称
- **修正**: 改为 "coverage-based ranking does not require a fixed probability threshold"

### 矛盾 6: 参考选择策略数量
- **文中声称**: "six strategies" (摘要 L63)
- **实际**: 表中仅 5 种 (random, similarity, entropy, RRN, oracle)
- **严重性**: **P0-HIGH**
- **修正**: 改为 "five strategies" 或补充 CLIP-similarity 数据

### 矛盾 7: RAMP 算法伪代码循环依赖
- **文中声称**: Algorithm 1 先提取特征再运行 segmenter (L445-448)
- **实际**: f4 (mask consistency) 和 f5 (prediction uncertainty) 需要 segmenter 输出
- **严重性**: **P0-CRITICAL** — 算法不可执行
- **修正**: 重构为两阶段或 post-hoc selective prediction

### 矛盾 8: 统计单位混淆
- **文中声称**: 使用 "query" 作为统计单位 (L528)
- **实际**: baselines_summary.json 显示 n_episodes=1200 (400 episodes per config)
- **严重性**: **P0-HIGH**
- **修正**: 所有统计应以 query 为单位聚合

### 矛盾 9: Equivalence 声称
- **文中声称**: "Δ<0.001, p>0.05 (equivalence)" (Table 2 footnote)
- **实际**: p>0.05 不等于 equivalence
- **严重性**: **P0-HIGH**
- **修正**: 改为 "no statistically significant difference detected"

### 矛盾 10: RAMP 组件消融数据不可靠
- **文中声称**: 各消融变体有不同 CFR_Red 值 (Table 6)
- **实际**: ramp_results.json 中所有消融变体 dice 相同
- **严重性**: **P0-CRITICAL** — 消融实验数据造假
- **修正**: 重新运行消融实验或删除该表

### 矛盾 11: 外部方法比较中的 segmenter 差异
- **文中声称**: "under identical conditions" (L67)
- **实际**: GF-SAM 使用 SAM-H, MAUP 使用 SAM-H, H-SAM 使用 SAM-L, 而 RAMP 使用 SAM-B
- **严重性**: **P0-HIGH** — 不公平比较
- **修正**: 明确标注 backbone 差异，分两类表格报告

### 矛盾 12: RRN 测试集无 catastrophic failure
- **文中声称**: RRN "CFR_Red = 0.0%" (Table 5 footnote)
- **实际**: rrn_results.json 显示 test set CFR=0.0 (prevalence_catastrophic=0.0)
- **严重性**: **P0-MEDIUM** — 12 test queries 无 catastrophic 案例，评估不可靠
- **修正**: 需要 grouped nested 5-fold CV 重评

### 矛盾 13: Cross-dataset anti-correlation 数值不一致
- **文中声称**: MoNuSAC r=-0.67 (Table 9)
- **实际数据**: theory_verification 中 SAM-B r=-0.491
- **严重性**: **P0-HIGH** — 同一数据集不同位置数值矛盾
- **修正**: 统一数据来源

### 矛盾 14: Pascal-5i/COCO-20i 单 fold 问题
- **文中声称**: "cross-domain generality" (L171)
- **实际**: 仅 fold-0 数据
- **严重性**: **P0-HIGH** — 需要 4-fold 完整评估
- **修正**: 补充 4-fold 数据或降级为 "initial cross-domain evidence"

## Gate V0 状态: PHASEV_P0_NUMERICAL_CONSISTENCY_FAIL

原因:
- 14 个矛盾中 7 个为 P0-CRITICAL 级别
- 核心主张 (anti-correlation 范围, training-free, calibration-free, 模型数量) 均需修正
- 消融实验数据不可靠需要重跑
- 算法伪代码不可执行