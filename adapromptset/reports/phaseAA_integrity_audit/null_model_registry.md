# Gate AA0 — Phase V/W Null Model Registry

**Date**: 2026-07-27
**Audit**: Phase AA Integrity Audit

## Null Model Definitions

### Null 1: Random Beta (Mean-Dice Preserving) — Phase V
- **Definition**: For each query, draw a random beta-distributed pseudo-CFR with the same mean Dice as the observed data. Compute Pearson r between observed instability and pseudo-CFR.
- **Phase V Result**: r_null = -0.583 (SAM-B pipeline)
- **Interpretation**: A mean-Dice preserving null — if the anti-correlation is purely mechanical (driven by mean Dice level), this null should reproduce the observed correlation.
- **Phase V Finding**: Null 1 reproduces ~94% of observed correlation (-0.583 vs -0.619), suggesting a substantial mechanical component.

### Null 2: Shuffled — Phase V
- **Definition**: Shuffle the CFR column, breaking all query-level associations. Compute Pearson r.
- **Phase V Result**: r_null = +0.643
- **Interpretation**: Positive correlation after shuffling confirms the observed negative correlation is not random.

### Null 3: Uniform — Phase V
- **Definition**: Draw uniform random pseudo-CFR. Compute Pearson r.
- **Phase V Result**: r_null = +0.381
- **Interpretation**: Weaker than shuffled but still positive, confirming the observed negative correlation.

### Null 0 (Trivial) — Phase W
- **Definition**: r_null = 0.0 (no computation)
- **Phase W Result**: r_null_mean = 0.0 for all pipelines
- **Status**: **BUG CONFIRMED** — The Phase W script did not actually compute a null model. The `compute_null_correlation` function was either not called or returned 0 for all pipelines.

## Resolution

### Corrected Null Model Values

| Pipeline | r_observed (Phase W) | r_null (Phase V Null 1) | Δr (corrected) | Δr (Phase W, buggy) |
|----------|---------------------|------------------------|----------------|---------------------|
| L+SAM-B | -0.512 | -0.583 | **+0.071** | -0.512 |
| B+SAM-B | -0.549 | -0.583 | **+0.034** | -0.549 |
| L+SAM-L | -0.221 | (not computed) | TBD | -0.221 |
| L+SAM2-T | -0.111 | (not computed) | TBD | -0.111 |
| L+SAM2-S | -0.179 | (not computed) | TBD | -0.179 |

**Note**: Phase V Null 1 was only computed for the SAM-B pipeline. For other pipelines, the null model needs to be recomputed.

### Impact on Anti-Correlation Claim

The Phase W conclusion that "Δr = r_observed" (i.e., the null is zero) is **incorrect**. When the proper Null 1 (random beta, mean-Dice preserving) is used, the residual Δr is much smaller:
- L+SAM-B: Δr = -0.512 - (-0.583) = **+0.071** (positive! The residual is in the WRONG direction)
- B+SAM-B: Δr = -0.549 - (-0.583) = **+0.034** (positive! The residual is in the WRONG direction)

This means the observed anti-correlation is **entirely explained** by the mechanical relationship between mean Dice and CFR, and the "structural" residual is actually **positive** (anti-correlation is weaker than the null would predict).

This is a **P0-CRITICAL** finding: the anti-correlation claim in Phase W is based on a buggy null model. The corrected null model shows the anti-correlation is mechanical, not structural.