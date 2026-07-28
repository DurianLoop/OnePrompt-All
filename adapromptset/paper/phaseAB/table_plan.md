# Table Plan

## Table 1: Main Results (Development Cohort)
- Columns: Method, AUROC, AURC, CFR@80, Dice@80, CFR Reduction@80
- Rows: Random, Entropy, Candidate Jaccard, Prompt Overlap, Prediction Consistency, M2-Instability
- Highlight: Prediction Consistency row

## Table 2: Cross-Pipeline
- Columns: Pipeline, Support, AURC(PC), AURC(Rand), CFR Red@80
- Rows: P1-P5
- Highlight: Primary pipeline (P1) row

## Table 3: Cross-Dataset
- Columns: Dataset, N, Base Dice, CFR, Risk Rho, AURC, CFR Red@80, Verdict
- Rows: MoNuSAC, MoNuSeg, IDRID, Validation41
- Note: Boundary interpretation for MoNuSeg and IDRID

## Table 4: Ablation - Reference Count
- Columns: m, AURC, CFR@80, Dice@80
- Rows: 2, 4, 6, 8
