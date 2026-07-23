# Phase T Paper Outline

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`
Related: directive §1 (positioning), §14 (tables), §15 (figures), §19 (materials)

## Title Candidates (directive §19)

1. **Reference-Induced Instability in Visual-Reference Segmentation: A
   Cross-Reference Benchmark and Failure-Boundary Analysis**
2. **Beyond a Single Reference: Measuring and Mitigating Reference-Induced
   Risk in Prompted Segmentation**
3. **RefShift: A Benchmark for Reference Resampling, Tail Risk, and
   Selective Prompting**

The final title is chosen at WP4 based on which contributions actually pass
their Gates. Title #1 is the default for the Benchmark-only route; titles
#2 / #3 are candidates for the Hybrid route (M1 or M2 passes).

## Paper Question (directive §1)

> How much does the choice of a valid reference affect visual-reference
> segmentation, how should this instability be measured, and can a simple
> risk-aware strategy mitigate it without retraining the frozen matcher and
> segmenter?

## Section Outline

### 1. Introduction

- Motivation: visual-reference segmentation depends on a reference image,
  but the choice of reference is rarely audited.
- Phenomenon: even valid references induce substantial prediction variance.
- Contribution preview: (i) phenomenon + metrics, (ii) system benchmark,
  (iii) practical risk-aware mitigation.
- Scope: frozen DINOv2-L + SAM-B; medical imaging focus; not a SOTA claim.

### 2. Related Work

- Visual-reference / few-shot segmentation.
- Prompt-robustness methods: GF-SAM, SynPo, AlignSAM, Point Prompt Defender.
- Reference-aware methods: ProSAM, MAUP, SAMIX.
- Selective prediction and abstention.
- Set aggregation and consensus.
- Boundary statements differentiating Phase T from each (per
  `reports/phaseT_novelty_audit.md`).

### 3. Benchmark Protocol

- Datasets and roles (MoNuSAC primary; GlaS / MoNuSeg / DDR secondary /
  tertiary).
- Reference construction protocol (train-only, patient-disjoint, 6
  references per query, k∈{1,3,6}, 10 seeds).
- Frozen matcher (DINOv2-L) and segmenter (SAM-B).
- Prompt configs (P20-N1, P30-N0, P50-N0).
- Cost accounting.

### 4. Metrics

- Variance decomposition: V_query, V_ref, V_ref_frac.
- Pairwise: MAD_ref, J_pred.
- Tail risk: CVaR_0.1, CVaR_0.2, CFR (Dice<0.1), worst-subset Dice,
  best-worst gap.
- Mean performance: Dice, IoU, AJI, component recall, precision / recall.
- Statistical unit = query / case; paired bootstrap 95% CI; Holm correction.

### 5. Results — MoNuSAC Holdout (WP1)

- Confirmatory holdout (n=40) results for B1–B7.
- Variance decomposition table.
- Best-worst gap distribution.
- Reference-count scaling (k=1/3/6).
- Tail-risk metrics.
- WP1 Gate condition evaluation.

### 6. Results — Cross-Dataset (WP2)

- GlaS, MoNuSeg, DDR directional / strong support.
- Per-dataset V_ref, CVaR, CFR.
- Cross-dataset synthesis (qualitative; no pooled significance).
- External comparator matrix (Tier 1 / 2 reproduced; Tier 3 / 4 documented).

### 7. Results — External Baselines (WP2b)

- Per-method reference-instability metrics.
- Head-to-head vs B2 mean under matched protocol.
- Honest reporting of reproduction limitations.

### 8. Risk-Aware Mitigation — M1 Dynamic Reference Budget (WP3)

- Policy ladder M1-0 to M1-6.
- Cost-quality utility U(q,k).
- Cost-saving mode and quality-cost mode results.
- M1 Gate evaluation.

### 9. Risk-Aware Mitigation — M2 Instability-Aware Selective Prediction (WP3)

- Three-class risk label construction (train-only).
- Model ladder M2-0 to M2-4.
- AUROC / AUPRC / ECE / risk-coverage curve.
- M2 Gate evaluation.
- Holdout confirmation.

### 10. Failure Taxonomy

- Failure modes: small sparse target, diffuse matching map, prompt-interface
  bottleneck, reference-induced catastrophic failure.
- Qualitative examples selected by pre-registered rules (worst-5, best-5,
  median-5 by V_ref).
- High-instability vs low-instability case studies.

### 11. Limitations

- Single primary dataset (MoNuSAC); secondary datasets are smaller.
- Frozen DINOv2-L + SAM-B; results may not transfer to other matchers /
  segmenters.
- No formal test access; claims are bounded by holdout evidence.
- M1 / M2 may not pass; Benchmark-only route is the fallback.
- External comparator reproduction is best-effort.
- Medical imaging focus; generalization to natural images not tested.

### 12. Conclusion

- Restate the phenomenon, the corrected measurement, and the strong-baseline
  value of mean consensus.
- State which mitigation route (M1 / M2 / none) passed holdout confirmation.
- State the final Phase T state and the recommended next step (Phase U
  formal-test freeze, if warranted).

## Core Tables (directive §14, minimum 18)

1. Dataset statistics.
2. Reference protocol.
3. Correct variance decomposition.
4. Single-reference instability.
5. Reference-count scaling.
6. Mean / median / robust consensus.
7. Tail-risk metrics.
8. Prediction consistency.
9. External method instability.
10. Dynamic budget (M1).
11. Selective prediction (M2).
12. Cost-quality Pareto.
13. Holdout confirmation.
14. Cross-dataset replication.
15. Ablations.
16. Failure taxonomy.
17. Runtime.
18. Claim registry.

## Core Figures (directive §15, minimum 18)

1. Benchmark overview.
2. Reference resampling protocol.
3. Variance decomposition.
4. Best-worst gap distribution.
5. Reference-count scaling.
6. Dice vs V_ref.
7. CVaR and CFR.
8. Prediction consistency.
9. Method-wise instability comparison.
10. Risk-coverage curve.
11. Dynamic budget policy.
12. Cost-quality Pareto.
13. Cross-dataset summary.
14. High-instability examples.
15. Mean-consensus success cases.
16. Mean-consensus failure cases.
17. Selective prediction failures.
18. Final method diagram.

Qualitative figures are selected by pre-registered rules; hand-picking
favorable cases is prohibited.

## Paper Materials Location (directive §19)

```
paper/phaseT/
├── title_candidates.md
├── abstract.md
├── contributions.md
├── introduction.md
├── related_work.md
├── method.md
├── benchmark_protocol.md
├── experiments.md
├── results.md
├── limitations.md
├── conclusion.md
├── appendix.md
├── figure_plan.md
├── table_plan.md
├── reproducibility_checklist.md
└── full_draft.md
```

## Submission Route Decision

The route is decided at WP4 based on which Gates passed:

- **Benchmark-only route**: WP1 pass + ≥2 datasets directional support +
  external comparator minimum + corrected metrics + failure taxonomy. No
  method holdout pass.
- **Hybrid route**: Benchmark-only requirements + M1 OR M2 holdout pass +
  cost/risk gain + cross-dataset directional support + ablation +
  calibration + runtime.

The Hybrid route is the target; the Benchmark-only route is the fallback.
The route decision is recorded in `reports/phaseT_final_publication_report.md`.
