# Phase T Metric Protocol

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Statistical Unit

The statistical unit is the **query / case**, NOT the reference episode.
Multiple references per query are repeated measurements and MUST NOT be
treated as independent samples. All aggregate statistics are computed by
first averaging within-query across references (or across resampling seeds),
then computing the cross-query statistic. Patient-cluster bootstrap is used
where patients contribute multiple queries (MoNuSAC).

## Mean-Performance Metrics

- **Dice** — standard Dice coefficient between predicted and ground-truth
  semantic-union mask.
- **IoU** — Jaccard index.
- **AJI** — Aggregated Jaccard Index (instance-aware).
- **Component recall** — fraction of ground-truth connected components
  intersected by the prediction.
- **Precision / Recall** — pixel-level, computed on the semantic union.
- **Boundary metric** — only if the existing implementation is reliable;
  otherwise reported as N/A with justification.

## Reference-Instability Metrics

### V_query

Within-query cross-seed variance of the per-reference Dice, averaged over
references within each (query, seed) and then over seeds. Captures the
component of variance attributable to query identity.

### V_ref

Within-query variance of per-reference Dice, averaged over queries. Captures
the component of variance attributable to reference choice:

```
V_ref = (1/N) Σ_q Var_S [ D(q, S) ]
```

where S indexes reference subsets and D(q, S) is the Dice of query q under
reference subset S.

### V_ref_frac

Fraction of total variance attributable to references:

```
V_ref_frac = V_ref / (V_query + V_ref)
```

Phase S established V_ref_frac ∈ [0.1376, 0.3312] across the three frozen
prompt configs. WP1 must reconfirm V_ref_frac ≥ 0.10 on holdout.

### MAD_ref

Mean absolute pairwise reference deviation:

```
MAD_ref = (1/N) Σ_q E_{S_i, S_j} | D(q, S_i) − D(q, S_j) |
```

### J_pred

Prediction-consistency Jaccard: expected IoU between predictions produced by
two different reference subsets for the same query:

```
J_pred = E_{S_i, S_j} IoU( M̂(q, S_i), M̂(q, S_j) )
```

## Tail-Risk Metrics

### CVaR_0.1

Conditional value-at-risk at the 10% tail: the mean Dice of the worst 10%
of (query, reference-subset) episodes. Lower CVaR_0.1 indicates worse
tail risk.

### CVaR_0.2

Conditional value-at-risk at the 20% tail: the mean Dice of the worst 20%
of episodes.

### CFR (Catastrophic Failure Rate)

Fraction of episodes with Dice < 0.1 (the frozen catastrophic threshold).
The catastrophic threshold (0.1) is frozen in
`docs/phaseT_holdout_protocol.md` and may not be retuned after holdout.

### Worst-Subset Dice

The minimum per-query Dice across all reference subsets for that query,
averaged over queries. Measures the worst-case reference for each query.

### Best-Worst Gap

For each query, the difference between the best and worst reference-subset
Dice. Reported as:

- Mean best-worst gap.
- Fraction of queries with best-worst gap ≥ 0.05 (WP1 Gate cond1:
  ≥ 50% on holdout).

## Cost Metrics

- Reference count (k).
- DINOv2 matching passes (cached / uncached).
- SAM calls.
- Runtime (wall-clock seconds).
- Peak memory.
- GPU-hours.

## Paired Bootstrap 95% CI Procedure

For paired method comparisons (e.g., B2 mean vs B1 random single):

1. For each query q, compute the paired difference
   `d_q = Dice_methodA(q) − Dice_methodB(q)`.
2. Resample queries with replacement B = 10,000 times (Phase T frozen
   bootstrap iterations = 10,000).
3. For MoNuSAC, resample at the patient-cluster level (patient is the
   independent unit when multiple queries share a patient).
4. For GlaS / MoNuSeg / DDR, resample at the query level (one query per
   patient in the strict-validation splits).
5. Compute the bootstrap mean difference for each resample.
6. The 95% CI is the 2.5th and 97.5th percentile of the bootstrap
   distribution.
7. A method is declared significantly better if the CI lower bound > 0.

## Win / Tie / Loss

For each query, method A wins if `d_q > +ε`, ties if `|d_q| ≤ ε`, loses if
`d_q < −ε`, where ε = 0.001 (frozen). Reported as counts and fractions.

## Effect Size

- Cohen's d (paired): mean difference divided by the standard deviation of
  the paired differences.
- Rank-biserial correlation for the paired sign test.

## Multiple-Comparison Correction

When comparing more than two methods simultaneously, the Holm step-down
correction is applied to the family of pairwise p-values. The familywise
error rate is controlled at α = 0.05.

## Reporting Requirements

Every Phase T results table must report:

- Mean, median, and paired difference.
- Paired bootstrap 95% CI.
- Win / tie / loss counts.
- Effect size (Cohen's d).
- Within-query reference std.
- V_ref, V_query, V_ref_frac.
- MAD_ref, J_pred.
- CVaR_0.1, CVaR_0.2, CFR.
- Risk-coverage curve (for M2).
- Seed variability (std across the 10 resampling seeds).
- Runtime, matching passes, SAM calls.

Reporting mean Dice without variance / tail-risk evidence is a prohibited
over-claim (see `docs/phaseT_claim_registry.md` prohibited claim #7).

## Prohibited Metric Operations

1. Treating reference episodes as independent samples.
2. Reporting mean Dice without V_ref / CVaR / CFR.
3. Retuning the catastrophic threshold after holdout observation.
4. Changing bootstrap iterations below 10,000 after Gate evaluation.
5. Computing significance on a subset of queries selected post-hoc.
