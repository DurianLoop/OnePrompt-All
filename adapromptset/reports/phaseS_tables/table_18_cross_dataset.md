# Table 18 — Cross-Dataset

**Status:** BLOCKED. Directive §12 requires holdout PASS — holdout is BLOCKED.

## 1. Cross-Dataset Status

| Field | Value |
|---|---|
| Status | **BLOCKED** |
| Reason | Directive §12 requires holdout PASS; holdout is BLOCKED |
| Cross-dataset queries | not evaluated |
| SAM calls | 0 |
| GPU-hours | 0 |

## 2. Planned Cross-Dataset Protocol (NOT RUN — structure only)

If cross-dataset were unblocked, the protocol would be:

| Step | Description |
|---|---|
| 1 | Apply the chosen method config (B2_mean baseline + Track A benchmark) to a held-out dataset (e.g. a second modality or organ set) |
| 2 | Report V_ref_frac, mean_ref_std, best-worst gap, CVaR, CFR on the new dataset |
| 3 | Compare to MoNuSAC validation metrics (qualitative trend must match) |
| 4 | Confirm the Track A instability claim generalises (V_ref_frac ≥ 0.10 on the new dataset) |
| 5 | Confirm the Track B predictability claim generalises (val Spearman ≥ 0.3 on new dataset) |

## 3. Planned Cross-Dataset Metrics (BLOCKED — not run)

| Metric | MoNuSAC (validation) | Cross-dataset (NOT RUN) | Trend threshold |
|---|---|---|---|
| V_ref_frac | 0.2456 | BLOCKED | ≥ 0.10 |
| Mean ref std (k=1) | 0.0893 | BLOCKED | > 0.05 |
| Best-worst gap ≥0.05 fraction | 82.9% | BLOCKED | ≥ 50% |
| CVaR_0.1 | 0.0738 | BLOCKED | < mean Dice |
| Track B val Spearman | 0.4781 | BLOCKED | ≥ 0.3 |

## 4. Cross-Dataset Gate Conditions (NOT EVALUATED)

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| X1 | V_ref_frac on new dataset ≥ 0.10 | 0.10 | not evaluated | BLOCKED |
| X2 | Mean ref std on new dataset > 0.05 | 0.05 | not evaluated | BLOCKED |
| X3 | Best-worst gap ≥0.05 on ≥ 50% of new queries | 50% | not evaluated | BLOCKED |
| X4 | Track B val Spearman on new dataset ≥ 0.3 | 0.3 | not evaluated | BLOCKED |
| X5 | Qualitative trend matches MoNuSAC | matches | not evaluated | BLOCKED |

## 5. Implication for Paper

- The benchmark + failure-boundary paper is **limited to a single dataset (MoNuSAC)**.
- The lack of cross-dataset validation is documented as an unresolved blocker and a key weakness in the mock review (`reports/phaseS_mock_review.md`).
- Per directive §21, the paper is allowed as a Benchmark + failure-boundary paper **despite** single-dataset scope, but the formal test remains LOCKED_UNUSED until cross-dataset is run.

## 6. Planned Cross-Dataset Datasets (NOT USED)

| Dataset | Modality | Status |
|---|---|---|
| MoNuSAC (primary) | H&E histology, multi-organ | used (Phase R/S primary) |
| GLAS (secondary diagnostic, Phase R) | H&E histology, prostate | Phase R diagnostic only — not used as cross-dataset for Phase S |
| (held-out dataset) | TBD | NOT RUN |

## 7. Reference

- `docs/adapromptset/research.md` (directive §12)
- `reports/phaseS_gate_status.md`
- `reports/phaseR_final/table_15_glas_diagnostic.md`
