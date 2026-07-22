# Table 10 — Deterministic Baselines

**Goal:** Compare the 6 deterministic (no-learning) consensus baselines from Phase R. B2_mean is the primary baseline used by Track A/B/C.

## 1. Baseline Definitions

| Baseline | Symbol | Definition |
|---|---|---|
| B1_random | `B1_random` | Pick a single reference at random |
| B2_mean | `B2_mean` | Mean consensus across k references (primary) |
| B3_median | `B3_median` | Median consensus across k references |
| B4_trimmed | `B4_trimmed` | Trimmed-mean consensus (drop top/bottom 20%) |
| B5_disagreement | `B5_disagreement` | Disagreement-weighted consensus |
| B6_quantile | `B6_quantile` | Quantile-based consensus |

**Note on B2_mean artefact:** The Phase R exporter tagged B2_mean with `beta` and `tau` cosmetic fields. Phase S audit (Track A finding F5) confirmed these are exporter artefacts; B2_mean is plain reference-mean consensus with no parameters.

## 2. Headline Numbers (P30-N0, k=6, B2_mean)

| Metric | Value |
|---|---|
| Mean Dice (val) | **0.475608** |
| CVaR_0.1 | **0.073764** |
| CFR (Dice<0.1) | **0.0732** |
| V_ref | 0.0055 |
| mean_ref_std | 0.0732 |
| Mean pairwise dev | 0.1139 |

## 3. Baseline Comparison (Phase R values, k=6, P30-N0)

| Baseline | Mean Dice | CVaR_0.1 | CFR | mean_ref_std |
|---|---|---|---|---|
| B1_random | 0.4188 | 0.0583 | 0.1463 | 0.0893 |
| B2_mean | **0.4756** | 0.0738 | 0.0732 | 0.0732 |
| B3_median | 0.4712 | 0.0711 | 0.0756 | 0.0744 |
| B4_trimmed | 0.4733 | 0.0722 | 0.0744 | 0.0738 |
| B5_disagreement | 0.4701 | 0.0702 | 0.0768 | 0.0756 |
| B6_quantile | 0.4688 | 0.0691 | 0.0780 | 0.0768 |

(B1_random is k=1-equivalent — single reference. Others use k=6.)

## 4. Per-Config B2_mean Summary (k=6)

| Config | Mean Dice | CVaR_0.1 | CFR | mean_ref_std |
|---|---|---|---|---|
| P20-N1 | 0.4521 | 0.0644 | 0.1098 | 0.0855 |
| P30-N0 | **0.4756** | 0.0738 | 0.0732 | 0.0732 |
| P50-N0 | 0.4891 | 0.0855 | 0.0488 | 0.0551 |

**P50-N0 has the highest Dice** (0.4891) but the **lowest tail-risk metrics** (CVaR 0.0855, CFR 4.88%) — i.e. larger candidate pool → better and more stable.

## 5. Track B/C vs B2_mean

| Method | Mean Dice | Delta vs B2_mean | Status |
|---|---|---|---|
| B2_mean (P30-N0, k=6) | 0.4756 | — (baseline) | — |
| Track B weighted (M4_gbm) | 0.4732 | -0.0024 | FAILED cond1 (need +0.005) |
| Track C best (α=0,β=0,γ=0 = B2_mean) | 0.477861 (val) | +0.0047 vs Track B baseline | FAILED cond1 (need +0.005 vs Track B) |

**Both Track B and Track C failed to beat B2_mean.** The strongest automatic method is M4_gbm weighted (val Dice 0.4732) — but it is **worse** than plain B2_mean (0.4756).

## 6. Upper Bound Reference (Phase Y Search Oracle)

| Bound | Source | Cross-ref Dice |
|---|---|---|
| Search Oracle (Phase Y) | `reports/phaseY_final_report/tables/06_raw_cross_stable_oracle.md` | ~0.4338 (cross-reference) |

**Note:** The Phase Y Search Oracle is the per-query best-action upper bound, not a deployable method. Its cross-reference Dice (~0.4338) is **below** B2_mean's k=6 Dice (0.4756) — because Phase Y's oracle was computed under a different (cross-reference, action-bank) protocol. This is a known historical protocol shift documented in `reports/phaseR_historical_audit/phaseY_claim_correction.md`.

## 7. Why B2_mean Is the Primary Baseline

- Highest Dice among deterministic baselines.
- No tunable parameters (no overfitting risk).
- Reproducible across seeds.
- Used as the **Track B uniform baseline** and the **Track C α=0,β=0,γ=0 baseline**.

## 8. Reference

- `reports/phaseR_final/table_09_deterministic_consensus_comparison.md`
- `reports/phaseS_trackA_benchmark/benchmark_summary.md`
- `reports/phaseS_trackB/val_comparison.csv`
- `reports/phaseS_trackC/val_comparison.csv`
- `reports/phaseY_final_report/tables/06_raw_cross_stable_oracle.md`
- `reports/phaseR_historical_audit/phaseY_claim_correction.md`
