# Table 02 — Metric Definitions (Phase S Track A)

All metrics are defined on the **within-query reference axis** unless otherwise stated. Each query q has k references sampled; for each (q, r) we obtain a SAM-refined prediction mask C_{q,r}.

## 1. Variance Decomposition

| Metric | Symbol | Definition | Axis |
|---|---|---|---|
| Reference variance | `V_ref(q)` | `Var_r( C_{q,r} )` computed across the k references for query q (mask-pixel variance collapsed via mean over foreground pixels or per-pixel Jaccard variance) | within-query, reference |
| Query variance | `V_query(q)` | Variance attributable to query difficulty (estimated via cross-query scaling) | query |
| Total variance | `V_total(q)` | `V_query(q) + V_ref(q)` | query × reference |
| Reference variance fraction | `V_ref_frac(q)` | `V_ref(q) / V_total(q)` ∈ [0, 1] | fraction |
| Mean ref std | `mean_ref_std` | Mean over queries of within-query std of Dice across the k references | within-query, reference |

## 2. Pairwise Deviation

| Metric | Symbol | Definition |
|---|---|---|
| Mean Absolute Deviation of references | `MAD_ref(q)` | Mean over reference pairs (r_i, r_j) of `|Dice(C_{q,r_i}) - Dice(C_{q,r_j})|` |
| Mean pairwise deviation | averaged across queries | `mean_q MAD_ref(q)` |

## 3. Prediction Consistency

| Metric | Symbol | Definition |
|---|---|---|
| Prediction Jaccard | `J_pred` | Mean pairwise pixel-level Jaccard between SAM-refined predictions C_{q,r} across references, computed for single-reference (A set) vs consensus (B set) and A-vs-B |
| `prediction_jaccard_mean` | J_pred averaged over reference pairs | Strongest test-time-safe predictor (Spearman |ρ| = 0.8832 on train, P50-N0, std_dice) |

## 4. Tail-Risk Metrics

| Metric | Symbol | Definition |
|---|---|---|
| Conditional Value at Risk | `CVaR_0.1` | Mean of the worst 10% of (query × reference) Dice outcomes — quantifies tail risk |
| Worst Dice | `worst_dice` | Minimum Dice observed across (query × reference) pairs |
| 10th percentile | `p10` | 10th percentile of Dice distribution |
| Catastrophic Failure Rate | `CFR` | Fraction of (query × reference) pairs with Dice < 0.1 |

## 5. Best-Worst Subset Gap

| Metric | Symbol | Definition |
|---|---|---|
| Best-worst gap | `gap_{bw}(q)` | `max_r Dice(C_{q,r}) - min_r Dice(C_{q,r})` for query q |
| Best-worst gap fraction | fraction of queries with `gap_{bw} ≥ 0.05` | Per config: P20-N1 = 90.2%, P30-N0 = 82.9%, P50-N0 = 80.5% |

## 6. Reference-Count Scaling

| Metric | Symbol | Definition |
|---|---|---|
| V_ref(k) | `V_ref(k)` | Reference variance when k references are sampled; computed for k ∈ {1, 3, 6} |
| Dice(k) | `Dice(k)` | Mean Dice of consensus prediction with k references |
| Reference-count scaling | trend | How V_ref(k) and Dice(k) change as k grows |

## 7. Deterministic Baselines (Phase R family, retained for comparison)

| Baseline | Symbol | Definition |
|---|---|---|
| B1_random | `B1_random` | Random reference (single) |
| B2_mean | `B2_mean` | Mean consensus across k references (primary baseline) |
| B3_median | `B3_median` | Median consensus |
| B4_trimmed | `B4_trimmed` | Trimmed-mean consensus |
| B5_disagreement | `B5_disagreement` | Disagreement-weighted consensus |
| B6_quantile | `B6_quantile` | Quantile-based consensus |

Note: B2_mean `beta`/`tau` fields present in the Phase R exporter are a cosmetic artefact (audit finding F5) and are not used in metric computation.

## 8. Reliability Metrics (Track B)

| Metric | Symbol | Definition |
|---|---|---|
| Reliability weight | `w_r` | Weight assigned to reference r based on predicted utility |
| Inner CV Spearman | `ρ_train` | Spearman correlation between predicted and actual reference utility on inner CV (train) |
| Validation Spearman | `ρ_val` | Spearman on the validation set |
| Max weight | `w_max` | Maximum observed weight (collapse diagnostic) |

## 9. Set-Conditioned Metrics (Track C)

| Metric | Symbol | Definition |
|---|---|---|
| Pixel-wise Agreement Gate | `C3` | `w_r(x) = 1/(1+α·|C_r(x)-C_mean(x)|)`, `C_gated(x) = Σ_r w_r·C_r / Σ_r w_r` |
| Stable Candidate Scoring | `C4` | `s_i = μ_i - β·σ_i + γ·a_i` (μ=mean contrast, σ=std across refs, a=agreement fraction) |
| Train-val gap | — | `mean_dice_train - mean_dice_val` (must be < 0.05) |
| LOO drop | — | Drop in Dice when one reference is left out (must not exceed 0.05) |

## 10. Bootstrap Confidence Intervals

All delta comparisons (weighted vs uniform, Track C vs Track B) are reported with bootstrap 95% CI over the query set.

## Reference

- `reports/phaseS_trackA_benchmark/benchmark_summary.md`
- `reports/phaseS_trackA_metric_audit/` (corrected definitions)
- `docs/phaseS_claim_registry.md`
