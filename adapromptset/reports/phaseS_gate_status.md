# Phase S Gate Status

Date: 2026-07-22

Phase: EXECUTION

Directive: `research.md` (Phase S — Multi-Track Research Exploration)

## Pre-Gate: Preregistration Completeness

Status: `COMPLETE` (commit 2de8b4d, 2026-07-22)

## Track A — Reference Instability Benchmark

Status: `PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID`

- Date: 2026-07-22
- Gate A: 7/7 conditions PASSED
- CPU-only (0 GPU-hours, 0 SAM calls). Reused Phase R artefacts.
- Key findings:
  - V_ref is 13.8%-33.1% of total variance (corrected decomposition)
  - 80.5%-90.2% of val queries have best-worst gap ≥ 0.05
  - Mean within-query ref std: 0.069-0.101
  - prediction_jaccard_mean |ρ|=0.883 on train (predictability confirmed)
  - Phase R cond3 was INVALID (mixed variance axes)

## Track Selection

Status: `TRACK_B_SELECTED`

- Weighted score: 770/1000
- Routing: "Track A strong + reliability predictable → Track B"
- Commit: 50c82c4

## Track B — Learned Reference Reliability

Status: `PHASES_TRACK_B_NO_GAIN`

- Date: 2026-07-22
- Model: M4_gbm (inner CV Spearman=0.916 on train)
- Weighting: T=0.25, ε=0.0, λ=0.0
- Phase 1: 307s CPU, 7200 train feature rows, 6 models trained
- Phase 2: 309s CPU, 123 SAM calls, 41 val queries × 3 configs
- Phase 3: Gate B evaluation

### Gate B Conditions (1/6 passed)

| # | Condition | Result |
|---|---|---|
| 1 | Mean Dice +0.005 | FAIL (-0.0024) |
| 2 | Within-query ref std -10% | FAIL (2.83%) |
| 3 | Pairwise ref dev -10% | PASS (60.09%) |
| 4 | CVaR_0.1 +0.01 | FAIL (+0.003) |
| 5 | CFR relative -15% | FAIL (0%) |
| 6 | Prediction consistency IoU +0.02 | FAIL |

### Requirements (5/6 passed)

| Requirement | Result |
|---|---|
| Utility prediction Spearman ≥ 0.30 | PASS (0.478) |
| Direction consistent (≥2/3 configs) | PASS (2/3) |
| No query GT used | PASS |
| Single SAM call per query | PASS |
| No weight collapse | PASS (max=0.405) |
| Bootstrap point estimate > 0 | FAIL (-0.0024) |

### Decision

`PHASES_TRACK_B_NO_GAIN` — neither cond1 nor cond2 passed. Reference utility is
predictable (Spearman=0.478) but reliability weighting does not improve Dice.
Pairwise deviation reduced 60% but doesn't translate to Dice gain.

## Track C — Set-Conditioned Robust Prompting

Status: `PHASES_TRACK_C_NO_GAIN`

- Date: 2026-07-22
- Methods implemented: C3 (Pixel-wise Agreement Gate), C4 (Stable Candidate Scoring)
- Hyperparameter grid: α∈{0,1,5}, β∈{0,0.5,1}, γ∈{0,0.5,1} = 27 combos
- Phase 1: 1164s CPU, 80 train queries × 3 configs × 27 combos = 6432 SAM calls
- Phase 2: 195s CPU, 117 SAM calls, 41 val queries × 3 configs
- Phase 3: Gate C evaluation + seed/refcount/LOO tests, 110 SAM calls
- Best hyperparameters: (α=0.0, β=0.0, γ=0.0) = B2_mean baseline

### Key Finding

Neither C3 (pixel-wise agreement gate) nor C4 (stable candidate scoring) improved
over the B2_mean baseline on train. ALL non-zero hyperparameters DECREASED mean Dice
monotonically. The best validation Dice (0.4779) came from the baseline (α=0, β=0, γ=0),
which is effectively B2_mean consensus — NOT a Track C method contribution.

### Gate C Conditions (5/8 passed)

| # | Condition | Result |
|---|---|---|
| 1 | Mean Dice +0.005 vs Track B | FAIL (+0.0047) |
| 2 | CVaR +0.01 vs Track B | FAIL (-0.0031) |
| 3 | Catastrophic failure not increase | PASS (0.0732 = 0.0732) |
| 4 | 3 seeds consistent (≥2/3 beat B2_mean) | PASS (3/3) |
| 5 | Reference count variation stable | PASS |
| 6 | LOO no crash / no big drop | FAIL (7 big drops >0.05) |
| 7 | Params <5M, 1 SAM call, <2x runtime | PASS (0 params, 195s < 309s) |
| 8 | Train-val gap < 0.05 | PASS (0.037) |

### Decision

`PHASES_TRACK_C_NO_GAIN` — cond1 and cond2 both failed. The selected (α=0, β=0, γ=0)
is the B2_mean baseline, meaning C3/C4 interventions consistently degraded performance.
The stability-based candidate re-ranking and agreement-based gating cannot overcome
the prompt interface bottleneck.

## Track D — Direct Reference-Conditioned Segmentation

Status: `START_CONDITION_NOT_FULLY_MET`

- Track A confirmed prompt-interface instability ✓
- Track B/C cannot improve stability ✓
- Raw matching maps have signal but SAM refinement HELPS (consensus Dice 0.48 vs raw <0.3) ✗
- SAM refinement gains unstable across references ✓
- Condition 3 (point prompt conversion loses significantly) NOT met. Track D not started.

## Holdout

Status: `BLOCKED` (Track B failed; directive §11 requires Track B/C/D PASS)

## Cross-Dataset

Status: `BLOCKED` (holdout blocked; directive §12 requires holdout PASS)

## Test Lock

Status: `LOCKED_UNUSED` (directive §22: "不访问 formal test")

## Compute Budget

- Total cap: 210 GPU-hours
- Used: 0.00 GPU-hours (all phases ran on CPU)
- Remaining: 210.00 GPU-hours
- SAM calls: 123 (Track B Phase 2)
