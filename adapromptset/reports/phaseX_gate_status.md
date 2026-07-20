# Phase X Gate Status

Date: 2026-07-20

Phase: preregistration (no new SAM inference yet).

## Gate X1 — Historical Baseline Audit

Status: `PENDING_PREREGISTRATION_COMMIT`

Through conditions:

- Old 12-action exact definitions recoverable. -> DONE (see
  `reports/phaseX_historical_bank_audit/action_bank_exact_definitions.csv`).
- Strongest fixed exact config recoverable. -> DONE
  (`B_top5_pos3neg`).
- All absolute Dice recoverable. -> DONE.
- Cost metadata recoverable. -> DONE.

Failure code if any artifact is missing: `BLOCKED_HISTORICAL_ARTIFACT_INCOMPLETE`.

## Gate X2 — Candidate Scaling

Status: `CANDIDATE_UNIVERSE_INADEQUATE`.

Two Gate X2 audits were run on 10 MoNuSAC validation episodes:

1. Historical 5-positive pool (`reports/candidate_pilot/candidate_pilot_candidates.csv`):
   pool maxes at 9 positives/9 negatives per episode. Component coverage at
   K+=10 (smallest grid K+ using all 9 distinct positives) is 0.052. Status
   `CANDIDATE_UNIVERSE_INADEQUATE` because coverage at pool max < 0.5.
   Artifacts under `reports/phaseX_candidate_scaling/`.

2. Regenerated 100-positive pool
   (`reports/phaseX_candidate_pilot_100/candidate_pilot_candidates.csv`):
   pool maxes at 100 positives/100 negatives per episode. Even at K+=100,
   component coverage is only 0.433 — still below the 0.5 threshold. Point
   precision drops from 0.500 (K+=1) to 0.407 (K+=100), indicating the
   matcher's foreground score map is diffuse. Negative quality is good
   (background precision 0.93-1.00, false-negative risk 0.03-0.07). Status
   `CANDIDATE_UNIVERSE_INADEQUATE`. Artifacts under
   `reports/phaseX_candidate_scaling_100/`.

Coverage curve (100-positive pool):

| K+ | Mean component coverage | Mean point precision |
|---|---|---|
| 1  | 0.012 | 0.500 |
| 3  | 0.033 | 0.533 |
| 5  | 0.040 | 0.480 |
| 10 | 0.069 | 0.460 |
| 20 | 0.115 | 0.490 |
| 30 | 0.162 | 0.487 |
| 50 | 0.245 | 0.506 |
| 75 | 0.401 | 0.495 |
| 100| 0.433 | 0.407 |

Coverage is monotonically increasing with the biggest jump K=50->75 (+0.156)
and a clear plateau K=75->100 (+0.032). Even at the frozen K+=100 ceiling,
the candidate universe cannot reach 0.5 component coverage.

Per directive section 7, this finding is attributed to the matcher /
candidate generator (DINOv2-L `spread` layout with
`min_positive_distance=4.0`), **not** to the policy. Stage S2
(single-variable K+ scaling on SAM inference) and Gate X3 (Dense Prompt
Viability) cannot proceed under the current candidate generator. The
directive's Section 7 explicitly states: when `CANDIDATE_UNIVERSE_INADEQUATE`,
"the problem should not be attributed to policy, but to matcher/candidate
generator."

Implications for the rest of Phase X:

- Stage S2-S5 (dense prompt scaling on SAM) would use the same inadequate
  candidate universe; absolute Dice gains from K+ scaling will be capped by
  the candidate generator, not by the policy.
- The historical 0.005 oracle gap was measured on this same inadequate
  candidate universe; the gap may reflect matcher saturation rather than
  action-space saturation.
- Gate X3 (`DENSE_PROMPT_VIABILITY_*`) is unlikely to pass without a better
  matcher or candidate generator; this should be recorded but not used to
  skip Stage S2-S5 entirely. Per directive section 25: "不要因为旧的 0.005
  gap 再次提前停止" (do not stop early again because of the old 0.005 gap).

Artifacts (100-positive pool, under `reports/phaseX_candidate_scaling_100/`):

- `positive_scaling_query_metrics.csv`
- `negative_scaling_query_metrics.csv`
- `positive_coverage_curve.csv`
- `negative_quality_curve.csv`
- `component_coverage_by_k.csv`
- `spatial_dispersion_by_k.csv`
- `scaling_summary.md`
- `scaling_summary.json`

Will report one of:

- `CANDIDATE_SCALING_SUPPORTS_LARGE_K`
- `CANDIDATE_SCALING_SATURATES_SMALL_K`
- `CANDIDATE_UNIVERSE_INADEQUATE`
- `CANDIDATE_SCALING_INSUFFICIENT_DATA` (pool maxes below 100 but coverage at
  the pool max is already >= 0.5; regenerate pool to draw a saturation
  conclusion at K+=100)
- `OLD_SMALL_K_BANK_INADEQUATE` (K+=5 coverage is low but K+=100 reaches high)

## Gate X3 — Dense Prompt Viability

Status: `DENSE_PROMPT_VIABILITY_ESTABLISHED`.

Stage S2 (single-variable K+ scaling) ran SAM-B on 10 MoNuSAC validation
episodes with K+ in {1, 3, 5, 10, 20, 30, 50, 75, 100}, selection=Top-K,
negative=none, grouping=one group, calls=1. CPU-only: 59.5 seconds, 0
GPU-hours.

K+ scaling curve:

| K+ | Mean Dice | Mean IoU | Mean AJI | Mean component recall |
|---|---|---|---|---|
| 1  | 0.029 | 0.016 | 0.016 | 0.013 |
| 3  | 0.256 | 0.152 | 0.152 | 0.656 |
| 5  | 0.326 | 0.197 | 0.197 | 0.892 |
| 10 | 0.467 | 0.315 | 0.315 | 0.942 |
| 20 | 0.524 | 0.364 | 0.364 | 0.968 |  <- Dice peak
| 30 | 0.517 | 0.357 | 0.357 | 1.000 |  <- recall saturated
| 50 | 0.442 | 0.286 | 0.286 | 1.000 |
| 75 | 0.397 | 0.248 | 0.248 | 1.000 |
| 100| 0.423 | 0.269 | 0.269 | 1.000 |

Key findings:

1. **Historical 12-action bank was inadequate.** Dice at K=5 (historical max)
   is 0.326. Dice at K=20 is 0.524 — a 60.7% relative improvement. The
   historical bank never tested K+ > 5.
2. **Dice peaks at K=20, then decreases.** Beyond K=20, adding more points
   causes SAM to predict larger masks that include too much background
   (false positives), reducing Dice. This is the "mask over-expansion"
   failure mode.
3. **Component recall saturates at K=30 (1.000).** All GT components are
   hit by SAM's predicted mask from K=30 onwards.
4. **Gate X2 (CANDIDATE_UNIVERSE_INADEQUATE) was misleading.** Even though
   candidate points themselves only cover 0.433 of components at K=100,
   SAM's mask prediction reaches all components by K=30. The candidate
   coverage metric was too conservative — it measured point-level coverage,
   but SAM's mask prediction is much more powerful.
5. **The historical 0.005 oracle gap was an artifact.** The historical
   "oracle" (0.3762) was the best of 12 actions all limited to K+ <= 5.
   The real oracle (K=20) gives 0.524 Dice — 39.6% higher. The 0.005 gap
   was measuring saturation within an inadequate action space, not true
   action-space saturation.

This is the central Phase X finding: the historical conclusion that the
action space was "close to Oracle" was wrong. The action space was
inadequate, and dense prompts (K=20) provide a significant improvement.

Artifacts (under `reports/phaseX_single_variable_kpos/`):

- `single_variable_kpos_query_metrics.csv`
- `single_variable_kpos_curve.csv`
- `single_variable_kpos_summary.md`
- `single_variable_kpos_summary.json`

Will report one of:

- `DENSE_PROMPT_VIABILITY_ESTABLISHED`
- `DENSE_PROMPT_NO_GAIN`
- `DENSE_PROMPT_COST_ONLY_GAIN`
- `DENSE_PROMPT_ABSOLUTE_PERFORMANCE_LOW`
- `DENSE_PROMPT_INSUFFICIENT_DATA`

## Gate X4 — Action-Space Saturation

Status: `PARTIAL` (grouping axis complete, negative + nested bank pending).

### Stage S3 (grouping scaling) — preliminary Gate X4 (grouping axis)

Status: `GROUPING_ONE_GROUP_DOMINATES`.

Stage S3 ran SAM-B on 10 MoNuSAC validation episodes with K+ in
{10, 20, 30, 50, 100} and all 12 frozen grouping families. CPU-only:
4159.8 seconds (69.3 minutes), 0 GPU-hours.

Best grouping per K+:

| K+ | Best grouping | Mean Dice | Mean calls | Next best (non-tied) |
|---|---|---|---|---|
| 10 | G_ONE_GROUP | 0.467 | 1 | G_KMEANS_3 (0.333, 3 calls) |
| 20 | G_ONE_GROUP | 0.524 | 1 | G_KMEANS_3 (0.356, 3 calls) |
| 30 | G_ONE_GROUP | 0.517 | 1 | G_KMEANS_3 (0.479, 3 calls) |
| 50 | G_ONE_GROUP | 0.442 | 1 | G_KMEANS_5 (0.414, 5 calls) |
| 100 | G_ONE_GROUP | 0.423 | 1 | G_KMEANS_3 (0.395, 3 calls) |

Key findings:

1. **G_ONE_GROUP dominates at every K+.** No other grouping beats it by
   >= 0.01 Dice at the S2 peak (K=20). Gain = 0.0000 < 0.010.
2. **G_ONE_POINT_PER_GROUP is the worst at every K+.** Splitting into
   individual points drastically reduces Dice while increasing calls
   (e.g., K=100: 0.315 Dice, 100 calls vs G_ONE_GROUP 0.423 Dice, 1 call).
3. **Multi-group strategies generally hurt Dice.** They produce multiple
   masks that get fused (NMS + union), causing over-segmentation.
   Component recall is often high (1.0) but Dice is low — the
   "cost-only gain" pattern.
4. **Historical claim at directive line 36 CONFIRMED**: "其他分组、聚类
   或多次调用策略无效" (other grouping, clustering, or multi-call
   strategies are invalid). G_ONE_GROUP with 1 call is the best at every
   K+.
5. **BUT** the historical claim was made without testing K+ > 5. The
   finding is: grouping strategies ARE invalid (confirmed), BUT K+
   scaling IS valid (S2 showed K=20 gives 60.7% relative improvement
   over K=5). The 0.005 gap was NOT evidence of action-space saturation
   — it was evidence of an inadequate action space (K+ <= 5).

Artifacts (under `reports/phaseX_grouping_scaling/`):

- `grouping_scaling_query_metrics.csv`
- `grouping_scaling_curve.csv`
- `grouping_scaling_summary.md`
- `grouping_scaling_summary.json`

### Stage S4 (negative scaling) — preliminary Gate X4 (negative axis)

Status: `NOT_RUN_YET`.

### Nested bank saturation — full Gate X4

Status: `NOT_RUN_YET`.

Will report one of:

- `ACTION_SPACE_NOT_SATURATED`
- `ACTION_SPACE_SATURATED`
- `OLD_ACTION_BANK_INADEQUATE`

## Gate X5 — Search-Oracle Adequacy

Status: `NOT_RUN_YET`.

Will report one of:

- `SEARCH_ORACLE_EXCEEDS_BANK_ORACLE`
- `SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE`
- `SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW`

## Gate X6 — Adaptive Value

Status: `NOT_RUN_YET`.

Reopening policy requires *all* of:

1. `D_search_oracle - D_fixed >= 0.01` Dice
2. At least 25% of queries have a
   best structure different from the strongest fixed.
3. The gain is not solely from more calls.
4. Absolute performance reaches a usable level.
5. Action-space saturation is essentially established.
6. Query-specific best config has a predictable relation to morphology /
   candidate statistics.

Pass: `REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE`.

Fail (one of): `PIVOT_STRONG_FIXED_DENSE_PROMPT`,
`PIVOT_PROMPT_INTERFACE_LIMITATION`, `PIVOT_CANDIDATE_LIMITATION`,
`PIVOT_ORACLE_ANALYSIS_EXPANDED_SPACE`.
