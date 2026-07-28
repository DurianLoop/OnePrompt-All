# Figure Plan (18 figures, directive §15)

Qualitative figures are selected by pre-registered rules (worst-5,
best-5, median-5 by `V_ref`); hand-picking favorable cases is prohibited.
Every quantitative figure is generated from a frozen gate-summary CSV;
no figure is regenerated after holdout observation.

## F1. Benchmark overview
**Caption:** *Reference-resampling benchmark overview.* Each query
receives six patient-disjoint references drawn from the train-only pool,
resampled across ten seeds; k∈{1,3,6} subsets are evaluated under three
frozen prompt configs (P20-N1, P30-N0, P50-N0). The frozen DINOv2-L
matcher and SAM-B segmenter are reused unchanged; the only learned
components are the M2 logistic regression and the M1 regressors.

## F2. Reference resampling protocol
**Caption:** *Reference-resampling protocol.* Six references per query
are drawn from the train-only pool, patient-disjoint from the query.
k=1 uses the six individual references; k=3 uses mean consensus over the
first three; k=6 uses mean consensus over all six. All methods evaluated
on the same query share the exact same six-reference pool.

## F3. Variance decomposition
**Caption:** *Corrected variance decomposition on the MoNuSAC holdout.*
`V_total = V_query + V_ref`, reported per (config, k). The
reference-attributable fraction `V_ref_frac` is 0.061–0.263 across
configs and k, confirming that reference choice is a substantial
component of total variance. Bars: `V_query` (query-attributable) and
`V_ref` (reference-attributable); line: `V_ref_frac`.

## F4. Best-worst gap distribution
**Caption:** *Distribution of per-query best-worst Dice gap.* 92.5% of
holdout queries (37/40) have a mean best-worst gap ≥ 0.05 (dashed line,
gate threshold). The histogram is over the 40 holdout queries at k=1,
averaged across the 10 seeds; the vertical line marks the 0.05 gate
threshold.

## F5. Reference-count scaling
**Caption:** *Reference-count scaling (k=1, 3, 6).* Mean Dice, `CVaR_0.1`,
and CFR as functions of k, averaged across configs. Consensus (k=6)
improves mean Dice (+0.027), improves `CVaR_0.1`, and reduces CFR
relative to the random single-reference baseline (k=1).

## F6. Dice vs V_ref
**Caption:** *Per-query mean Dice against `V_ref`.* Each point is one
holdout query (k=1, P30-N0). High-instability queries (large `V_ref`)
span a wide Dice range; the worst-reference Dice for high-`V_ref`
queries is markedly lower than for low-`V_ref` queries.

## F7. CVaR and CFR
**Caption:** *Tail-risk metrics by (config, k).* `CVaR_0.1` (left) and
CFR (right) for the three prompt configs at k∈{1,3,6}. Consensus (k=6)
raises `CVaR_0.1` and lowers CFR across all configs.

## F8. Prediction consistency
**Caption:** *Prediction consistency `J_pred`.* Mean cross-seed `J_pred`
at k=1 (0.649) and k=6 (0.719) on the MoNuSAC holdout. Consensus
improves prediction consistency: predictions produced by different
six-reference consensus subsets overlap more than predictions produced
by different single-reference subsets.

## F9. Method-wise instability comparison
**Caption:** *Internal B1–B7 family instability comparison.* `V_ref_frac`,
`CVaR_0.1`, and CFR for B1 (random single), B2_mean, B3_median,
B4_trimmed, B5, B6, B7_learned. B2_mean is the strongest automatic
baseline on the holdout; learned weighting (B7) does not beat it.

## F10. Risk-coverage curve
**Caption:** *M2 risk-coverage curve, validation and holdout.* Selective
Dice (left axis) and CFR (right axis) as functions of coverage, for the
catastrophic-trained M2-2 logistic regression. At 80% coverage, the
holdout CFR reduction is 79.2% (CFR 0.050 → 0.010) and Dice change is
+0.009. The holdout curve is above the validation curve on CFR
reduction.

## F11. Dynamic budget policy
**Caption:** *M1 dynamic-budget policy behaviour.* Per-query selected k
under the best policy (M1-3 handcrafted threshold), validation split.
The policy almost always selects k=6 (avg_k 5.63); the cost-quality
optimum collapses to `λ_c=λ_r=0`, i.e. "always k=6", across all five
policy variants.

## F12. Cost-quality Pareto
**Caption:** *Cost-quality Pareto frontier, M1.* Mean Dice against
avg_k for the six M1 policy variants and the three static baselines
(k=1, 3, 6). M1-2 (always k=6) Pareto-dominates the learned policies;
no adaptive policy beats static mean consensus on the cost-quality
frontier.

## F13. Cross-dataset summary
**Caption:** *Cross-dataset replication summary.* Per-dataset
`V_ref_frac` (k=1), consensus Dice gain, and best Spearman |ρ| for
MoNuSAC (holdout), MoNuSeg, and IDRiD. MoNuSeg shows strong support
(9/9); IDRiD fails (consensus gain −0.012, CI lower −0.032).

## F14. High-instability examples
**Caption:** *High-instability qualitative examples (worst-5 by
`V_ref`, rule-selected).* For each of the five worst-`V_ref` holdout
queries, the best-reference and worst-reference predictions are shown
alongside the ground truth. Best-worst gaps exceed 0.20 in this subset.

## F15. Mean-consensus success cases
**Caption:** *Mean-consensus success cases (median-5 by `V_ref`,
rule-selected).* For each of five median-`V_ref` queries, the
single-reference (k=1) prediction, the mean-consensus (k=6) prediction,
and the ground truth. Consensus recovers boundary detail lost under the
worst single reference.

## F16. Mean-consensus failure cases
**Caption:** *Mean-consensus failure cases (worst-5 by mean Dice under
k=6, rule-selected).* Even mean consensus fails on these queries; the
failure mode is typically a small sparse target or a diffuse matching
map, not a single bad reference.

## F17. Selective prediction failures
**Caption:** *M2 residual failures.* Holdout episodes the
catastrophic-trained M2 model rates as low-risk (probability < 0.5) but
that end in catastrophic failure (Dice < 0.1). These are the hardest
cases — the residual 0.010 CFR at 80% coverage — and motivate a
human-in-the-loop policy for the abstained 20%.

## F18. Final method diagram
**Caption:** *Final Phase T method diagram.* The frozen DINOv2-L +
SAM-B pipeline (left); the reference-resampling benchmark that measures
`V_ref`, `CVaR`, CFR (centre); and the dual-model M2 selective
predictor that abstains on high-risk episodes (right). M1 (dynamic
budget) is shown as a documented negative result.
