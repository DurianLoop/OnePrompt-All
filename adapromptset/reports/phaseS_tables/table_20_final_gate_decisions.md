# Table 20 — Final Gate Decisions

## 1. Phase S Final State

**State:** `PHASES_BENCHMARK_PAPER_READY`
- Track A: PASS (7/7)
- Track B: NO_GAIN (1/6 cond passed, 5/6 requirements passed)
- Track C: NO_GAIN (5/8 cond passed)
- Track D: NOT_STARTED (start conditions not fully met)
- Holdout: BLOCKED
- Cross-dataset: BLOCKED
- Formal test: LOCKED_UNUSED

**Allowed paper type (per §21):** Benchmark + failure-boundary paper.
**Allowed claims (per `docs/phaseS_claim_registry.md`):**
1. Track A instability benchmark claim
2. Track B Utility-Only claim

## 2. Gate A — Track A Reference Instability Benchmark

**Status:** `PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID` — **7/7 PASSED**

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| A1 | V_ref_frac ≥ 0.10 (P30-N0) | 0.10 | 0.2456 | PASS |
| A2 | Best-worst gap ≥0.05 on ≥50% queries (P30-N0) | 50% | 82.9% | PASS |
| A3 | Mean within-query ref std > 0.05 (k=1, P30-N0) | 0.05 | 0.0893 | PASS |
| A4 | Reference instability present in all 3 configs | 3/3 | 3/3 | PASS |
| A5 | V_ref_frac > 0 at k=6 (all configs) | >0 | 0.0955–0.2455 | PASS |
| A6 | Test-time-safe feature prediction Spearman ≥ 0.5 | 0.5 | 0.8832 | PASS |
| A7 | CVaR_0.1 < mean Dice (tail risk present) | strict < | 0.0738 < 0.4756 | PASS |

**Decision:** Track A PASSED.

## 3. Gate B — Track B Learned Reference Reliability

**Status:** `PHASES_TRACK_B_NO_GAIN` — **1/6 conditions PASSED, 5/6 requirements passed**

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| B1 | Mean Dice weighted > uniform + 0.005 | +0.005 | -0.0024 | **FAIL** |
| B2 | Mean ref std reduction ≥10% | 10% | 2.83% | **FAIL** |
| B3 | Mean pairwise dev reduction ≥10% | 10% | 60.09% | **PASS** |
| B4 | CVaR improvement ≥0 | ≥0 | +0.003 (worse) | FAIL |
| B5 | Direction consistent across ≥2/3 configs | 2/3 | 2/3 | PASS (requirement) |
| B6 | CFR reduction ≥0 | ≥0 | 0.0000 | FAIL |

**Decision rule (preregistered):** cond1 (Dice+0.005) AND cond2 (std -10%) must both PASS.
- cond1 FAIL, cond2 FAIL → **NO_GAIN**.

## 4. Gate C — Track C Set-Conditioned Robust Prompting

**Status:** `PHASES_TRACK_C_NO_GAIN` — **5/8 conditions PASSED**

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| C1 | Mean Dice improvement vs Track B ≥ 0.005 | +0.005 | +0.0047 | **FAIL** |
| C2 | CVaR improvement vs Track B ≥ 0.01 | +0.01 | -0.0031 | **FAIL** |
| C3 | CFR not increase vs Track B | ≤0 | 0.0000 | PASS |
| C4 | 3/3 seeds consistent | 3/3 | 3/3 | PASS |
| C5 | Refcount stable | stable | stable | PASS |
| C6 | LOO drops ≤ 0.05 (≤7 drops) | ≤7 | 7 | **FAIL** |
| C7 | Params / runtime within budget | within | within | PASS |
| C8 | Train-val gap < 0.05 | <0.05 | 0.0259 | PASS |

**Decision rule (preregistered):** cond1 (Dice+0.005) AND cond2 (CVaR+0.01) must both PASS.
- cond1 FAIL, cond2 FAIL → **NO_GAIN**.

## 5. Gate D — Track D Direct Reference-Conditioned Segmentation

**Status:** `START_CONDITION_NOT_FULLY_MET` — **NOT STARTED**

| # | Start condition | Status |
|---|---|---|
| 1 | Track A confirmed instability | MET ✓ |
| 2 | Track B/C cannot improve stability | MET ✓ |
| 3 | Raw matching maps stable but conversion loses | **NOT MET ✗** (refinement GAINS: 0.48 vs <0.3) |
| 4 | SAM refinement gains unstable | MET ✓ |

**Decision:** Track D NOT STARTED (condition 3 not met).

## 6. Holdout Gate

**Status:** **BLOCKED** — Directive §11 requires Track B/C/D PASS, none passed.

| # | Condition | Result | Status |
|---|---|---|---|
| H1 | Holdout Dice ≥ val Dice - 0.02 | not evaluated | BLOCKED |
| H2 | Holdout CFR ≤ val CFR + 0.02 | not evaluated | BLOCKED |
| H3 | Bootstrap CI excludes 0 | not evaluated | BLOCKED |

## 7. Cross-Dataset Gate

**Status:** **BLOCKED** — Directive §12 requires holdout PASS, holdout BLOCKED.

| # | Condition | Result | Status |
|---|---|---|---|
| X1 | V_ref_frac on new dataset ≥ 0.10 | not evaluated | BLOCKED |
| X2 | Mean ref std on new dataset > 0.05 | not evaluated | BLOCKED |
| X3 | Best-worst gap ≥0.05 on ≥50% new queries | not evaluated | BLOCKED |
| X4 | Track B val Spearman on new dataset ≥ 0.3 | not evaluated | BLOCKED |
| X5 | Qualitative trend matches MoNuSAC | not evaluated | BLOCKED |

## 8. Formal Test Gate

**Status:** **LOCKED_UNUSED** — Directive §22 prohibits formal test access.

| # | Condition | Result | Status |
|---|---|---|---|
| FT1 | Formal test Dice within 0.02 of val | not evaluated | LOCKED_UNUSED |
| FT2 | Formal test CFR ≤ val CFR + 0.02 | not evaluated | LOCKED_UNUSED |

## 9. Aggregate Summary

| Gate | Conditions Passed | Decision |
|---|---|---|
| A | 7/7 | **PASS** |
| B | 1/6 cond (5/6 req) | NO_GAIN |
| C | 5/8 | NO_GAIN |
| D | start conditions 3/4 met | NOT_STARTED |
| Holdout | 0/3 evaluated | BLOCKED |
| Cross-dataset | 0/5 evaluated | BLOCKED |
| Formal test | 0/2 evaluated | LOCKED_UNUSED |

## 10. Allowed Paper Type (per §21)

- **Method-track PASS:** requires Track B OR Track C OR Track D to PASS.
- **Track A PASS but no method track PASS:** → allowed paper type = **Benchmark + failure-boundary paper**.

## 11. Allowed Claims (per `docs/phaseS_claim_registry.md`)

1. **Track A claim:** "Valid references induce substantial, measurable segmentation variability under the frozen matching-and-prompting pipeline, and this instability is associated with test-time-safe disagreement features."
2. **Track B Utility-Only claim:** "Reference utility is predictable from test-time-safe features, but weighted consensus does not reduce variance; the utility signal is a scientific finding without a deployable stability gain."

## 12. Reference

- `docs/adapromptset/research.md` (directives §11, §12, §21, §22)
- `docs/phaseS_claim_registry.md`
- `reports/phaseS_gate_status.md`
- `reports/phaseS_trackA_benchmark/gateA_summary.json`
- `reports/phaseS_trackB/gateB_summary.json`
- `reports/phaseS_trackC/gateC_summary.json`
- `reports/phaseS_track_selection.json`
