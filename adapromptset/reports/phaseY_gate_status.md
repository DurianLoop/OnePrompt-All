# Phase Y Gate Status

Date: 2026-07-21

Phase: preregistration (no new SAM inference yet).

Directive: `AdaPromptSet_PhaseY_Policy_Feasibility.md`

## Pre-Gate: Preregistration Completeness

Status: `PENDING_PREREGISTRATION_COMMIT`

Required artifacts before any new SAM inference:

- `docs/phaseY_goal_read_receipt.md`
- `docs/phaseY_scientific_questions.md`
- `docs/phaseY_data_split_protocol.md`
- `docs/phaseY_reference_protocol.md`
- `docs/phaseY_single_call_action_bank.md`
- `docs/phaseY_oracle_stability_protocol.md`
- `docs/phaseY_bank_policy_protocol.md`
- `docs/phaseY_point_subset_protocol.md`
- `docs/phaseY_statistics_protocol.md`
- `docs/phaseY_claim_registry.md`
- `docs/phaseY_test_lock_notice.md`
- `reports/phaseY_gate_status.md`
- `reports/phaseY_compute_ledger.csv`
- `reports/phaseY_protocol_snapshot.json`
- `reports/phaseY_input_hashes.json`
- `configs/actions/phaseY_full_single_call_bank.yaml`
- `configs/actions/phaseY_compact_policy_bank.yaml`
- `manifests/phaseY/policy_train_queries.csv`
- `manifests/phaseY/policy_validation_queries.csv`
- `manifests/phaseY/confirmatory_holdout_queries.csv`
- `manifests/phaseY/reference_pool.csv`
- `manifests/phaseY/reference_set_A.csv`
- `manifests/phaseY/reference_set_B.csv`
- `reports/phaseY_data_audit/data_audit_summary.md`
- `reports/phaseY_data_audit/split_summary.json`
- `reports/phaseY_reference_audit.json`

Failure code if any artifact is missing: `BLOCKED_PREREGISTRATION_INCOMPLETE`.

## Gate Y1 — Fresh-Query Action Heterogeneity

Status: `PENDING_PREREGISTRATION_COMMIT`

Scope: Policy Validation Queries (41 cases), full audit bank
(210 actions), Reference Sets A and B.

Pass conditions (all must hold):
1. `stable_oracle_gap >= 0.015 Dice`
2. paired bootstrap 95% CI lower bound > 0
3. `>= 30% validation queries` stable best-action family != strongest fixed
4. cross-reference action-rank `median Spearman rho > 0.30`
5. `oracle optimism <= 60% of raw oracle gap`
6. all actions fixed at 1 SAM call (no call-count confound)

Pass: `PHASEY_GATE_Y1_STABLE_HETEROGENEITY`

Fail branches:
- `PHASEY_GATE_Y1_REFERENCE_UNSTABLE` (cross-reference gap collapses)
- `PHASEY_GATE_Y1_SELECTION_BIAS_DOMINATES` (optimism > 60%)
- `PHASEY_GATE_Y1_ORACLE_GAP_SMALL` (gap < 0.015)

On Y1 fail: bank-action policy training is BLOCKED; allowed to pivot to
point-subset stability audit only; holdout/test access forbidden.

## Gate Y2 — Search Oracle Stability

Status: `PENDING_GATE_Y1`

Scope: Policy Train + Validation Queries only (121 cases). Holdout and
formal test are forbidden for search oracle.

Pass conditions:
1. `cross-reference search >= 0.03 Dice` over strongest fixed
2. paired bootstrap CI lower bound > 0
3. `search optimism <= 50% of raw search gap`
4. `median |K_A - K_B| <= 8` OR cross Dice diff <= 0.01
5. `>= 70% queries`: A-search prompt beats strongest fixed on B

Pass: `PHASEY_GATE_Y2_SEARCH_LABELS_STABLE`
Fail: `PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT`

Interpretation rules (section 9.3):
- raw high, cross high -> `SEARCH_ORACLE_REFERENCE_STABLE`
- raw high, cross drops -> `SEARCH_ORACLE_REFERENCE_OVERFIT`
- K unstable, cross Dice stable -> `MULTIPLE_EQUIVALENT_PROMPTS`
- both unstable -> `SEARCH_LABELS_NOT_SUITABLE_FOR_SUPERVISION`

## Gate Y3 — Bank Policy Validation

Status: `PENDING_GATE_Y1`

Scope: compact policy bank (12-32 actions, Pareto-pruned on train-only),
M0-M5 model ladder, 3 seeds per learned model, Validation Queries.

Pass conditions:
1. `Delta Dice >= 0.010` over strongest fixed on validation
2. paired bootstrap 95% CI lower bound > 0
3. `oracle recovery >= 30%` (policy Dice - strongest fixed) / (oracle - strongest fixed)

Pass: `PHASEY_GATE_Y3_BANK_POLICY_VALIDATED`
Fail: `PHASEY_GATE_Y3_BANK_POLICY_INSUFFICIENT`

## Gate Y4 — Point-Subset Learnability

Status: `PENDING_GATE_Y2`

Scope: P0-P5 point-subset ladder, trained on train queries with search
oracle labels, validated on Validation Queries.

Pass conditions (either):
1. `Delta Dice >= 0.015` over strongest fixed, OR
2. `Delta Dice >= 0.0075` over bank policy (Gate Y3 winner)
Plus: paired bootstrap 95% CI lower bound > 0.

Pass: `PHASEY_GATE_Y4_POINT_SUBSET_LEARNABLE`
Fail: `PHASEY_GATE_Y4_POINT_SUBSET_NOT_LEARNABLE`

## Holdout Evaluation

Status: `PENDING_ALL_GATES`

Scope: Confirmatory Holdout (40 cases). Run ONCE on the single frozen
best model from Y3/Y4. No model selection on holdout. No holdout access
before the final frozen run.

## Test Lock

Status: `LOCKED_UNUSED`

MoNuSAC formal test (101 cases), GlaS testA/testB, DDR test, IDRID test
remain locked. No test access for K, action bank, router, features,
model capacity, or seed selection.
