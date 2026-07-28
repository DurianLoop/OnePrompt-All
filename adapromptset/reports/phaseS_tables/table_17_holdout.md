# Table 17 — Holdout

**Status:** BLOCKED. Directive §11 requires Track B/C/D PASS for holdout — none passed.

## 1. Holdout Status

| Field | Value |
|---|---|
| Status | **BLOCKED** |
| Reason | Directive §11 requires Track B/C/D PASS; all method tracks NO_GAIN or NOT_STARTED |
| Holdout queries | not evaluated |
| SAM calls | 0 |
| GPU-hours | 0 |

## 2. Holdout Protocol (NOT RUN — structure only)

If holdout were unblocked, the protocol would be:

| Step | Description |
|---|---|
| 1 | Lock the chosen method config (Track B/C best, or B2_mean baseline) |
| 2 | Apply to the holdout query split (committed in preregistration `2de8b4d`) |
| 3 | Report mean Dice, CVaR_0.1, CFR, mean_ref_std on holdout |
| 4 | Compare to validation metrics (delta must be < 0.02 to claim generalisation) |
| 5 | Compute bootstrap 95% CI on holdout Dice |

## 3. Planned Holdout Metrics (BLOCKED — not run)

| Metric | Validation (P30-N0, k=6, B2_mean) | Holdout (NOT RUN) | Delta threshold |
|---|---|---|---|
| Mean Dice | 0.4756 | BLOCKED | <0.02 |
| CVaR_0.1 | 0.0738 | BLOCKED | <0.02 |
| CFR | 0.0732 | BLOCKED | <0.02 |
| Mean ref std | 0.0732 | BLOCKED | <0.02 |

## 4. Holdout Gate Conditions (NOT EVALUATED)

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| H1 | Holdout Dice ≥ val Dice - 0.02 | -0.02 | not evaluated | BLOCKED |
| H2 | Holdout CFR ≤ val CFR + 0.02 | +0.02 | not evaluated | BLOCKED |
| H3 | Bootstrap CI on holdout Dice excludes 0 | excludes 0 | not evaluated | BLOCKED |

## 5. Implication for Paper

- The benchmark + failure-boundary paper **does not claim holdout validation**.
- All reported metrics are on the validation split only.
- The lack of holdout is documented as an unresolved blocker in the final research summary.

## 6. Reference

- `docs/adapromptset/research.md` (directive §11)
- `reports/phaseS_gate_status.md`
- `reports/phaseS_track_selection.json`
