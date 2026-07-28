"""Phase R preregistration builder.

Creates all 13 required preregistration documents per directive section 4,
plus frozen config files and input hashes.

Steps 6-10 of the 23-step execution order:
- Step 6: Freeze prompt configs
- Step 7: Freeze consensus methods
- Step 8: Freeze β, τ grid
- Step 9: Freeze statistics and Gate
- Step 10: Generate hashes

CPU-only. No SAM inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS = PROJECT_ROOT / "docs"
REPORTS = PROJECT_ROOT / "reports"
CONFIGS = PROJECT_ROOT / "configs"
MANIFESTS = PROJECT_ROOT / "manifests" / "phaseR"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
DATE = datetime.now().strftime("%Y-%m-%d")


# ===========================================================================
# Step 6: Frozen prompt configs
# ===========================================================================


FROZEN_PROMPT_CONFIGS = {
    "configs": [
        {
            "name": "P20-N1",
            "action_id": "R_P_TOPK_K20_N1_N_BOTTOMK_G1_C1",
            "k_positive": 20,
            "k_negative": 1,
            "positive_selector": "P_TOPK",
            "negative_selector": "N_BOTTOMK",
            "grouping": "G_ONE_GROUP",
            "sam_calls": 1,
        },
        {
            "name": "P30-N0",
            "action_id": "R_P_TOPK_K30_N0_N_NONE_G1_C1",
            "k_positive": 30,
            "k_negative": 0,
            "positive_selector": "P_TOPK",
            "negative_selector": "N_NONE",
            "grouping": "G_ONE_GROUP",
            "sam_calls": 1,
        },
        {
            "name": "P50-N0",
            "action_id": "R_P_TOPK_K50_N0_N_NONE_G1_C1",
            "k_positive": 50,
            "k_negative": 0,
            "positive_selector": "P_TOPK",
            "negative_selector": "N_NONE",
            "grouping": "G_ONE_GROUP",
            "sam_calls": 1,
        },
    ],
    "frozen_at": NOW,
    "directive_section": "6",
    "notes": "Only these 3 prompt configs are used in Phase R. No other K+/K-/selector/grouping combinations are allowed.",
}


# ===========================================================================
# Step 7: Frozen consensus methods
# ===========================================================================


FROZEN_CONSENSUS_METHODS = {
    "methods": [
        {"id": "B0", "name": "Single Reference", "deployable": True, "formula": "per-reference Dice; report mean/best/worst/std"},
        {"id": "B1", "name": "Random Single Reference", "deployable": True, "formula": "randomly select 1 of 6 references (fixed seed)"},
        {"id": "B2", "name": "Mean Consensus", "deployable": True, "formula": "C_mean(x) = (1/m) * sum_r C_r(x)"},
        {"id": "B3", "name": "Median Consensus", "deployable": True, "formula": "C_median(x) = median_r C_r(x)"},
        {"id": "B4", "name": "Trimmed Mean", "deployable": True, "formula": "remove max and min reference score, then mean (requires >= 5 refs)"},
        {"id": "B5", "name": "Disagreement-Penalized", "deployable": True, "formula": "C_stable(x) = mu(x) - beta * sigma(x)"},
        {"id": "B6", "name": "Quantile Consensus", "deployable": True, "formula": "C_q(x) = Q_tau({C_r(x)})"},
        {"id": "B7", "name": "Reliability-Weighted Mean", "deployable": False, "formula": "C_weighted(x) = sum_r w_r * C_r(x); only after Gate R2 pass"},
    ],
    "frozen_at": NOW,
    "directive_section": "7",
    "notes": "B7 is conditional on Gate R2 pass. All other methods are frozen and cannot be modified.",
}


# ===========================================================================
# Step 8: Frozen β, τ grid
# ===========================================================================


FROZEN_BETA_TAU_GRID = {
    "beta_grid": [0.25, 0.5, 1.0],
    "tau_grid": [0.50, 0.67, 0.83],
    "selection_rule": "beta and tau are selected ONLY on MoNuSAC train=80; validation and holdout cannot participate in selection",
    "frozen_at": NOW,
    "directive_section": "7 (B5, B6)",
}


# ===========================================================================
# Step 9: Frozen statistics and Gate thresholds
# ===========================================================================


FROZEN_STATISTICS = {
    "gate_r1": {
        "conditions": {
            "cond1_30pct_best_worst_gap_ge_0p05": ">= 30% train queries have best-worst Dice gap >= 0.05",
            "cond2_mean_ref_std_ge_0p02": "mean reference std >= 0.02",
            "cond3_direction_consistent_across_seeds": "instability direction consistent across 5 resampling seeds",
            "cond4_one_config_shows_sensitivity": "at least one of P20-N1, P30-N0, P50-N0 shows stable reference sensitivity",
            "cond5_not_code_error": "instability is not caused by code error, coordinate error, or reference-query leakage",
        },
        "pass_state": "PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED",
        "fail_state": "PHASER_GATE_R1_REFERENCE_INSTABILITY_NOT_REPLICATED",
    },
    "gate_r2": {
        "conditions": {
            "cond1_delta_dice_ge_0p010": "Best deterministic consensus vs Random Single: Delta Dice >= 0.010",
            "cond2_bootstrap_ci_lo_gt_0": "paired bootstrap 95% CI lower bound > 0",
            "cond3_variance_reduction_ge_25pct": "reference variance reduction >= 25%",
            "cond4_catastrophic_failure_not_increase": "catastrophic failure rate does not increase",
            "cond5_ab_prediction_overlap_improve": "A/B prediction-mask overlap improves",
            "cond6_one_sam_call": "still 1 SAM call",
            "cond7_no_query_gt": "does not use query GT",
            "cond8_2of3_configs_consistent": "at least 2 of 3 prompt configs directionally consistent",
        },
        "pass_state": "PHASER_GATE_R2_DETERMINISTIC_CONSENSUS_PASS",
        "fail_state": "PHASER_GATE_R2_CONSENSUS_NO_GAIN",
    },
    "gate_r3": {
        "conditions": {
            "cond1_weighted_vs_deterministic_ge_0p005": "Weighted Consensus vs best deterministic: Delta Dice >= 0.005",
            "cond2_weighted_vs_random_ge_0p015": "Weighted Consensus vs Random Single: Delta Dice >= 0.015",
            "cond3_bootstrap_ci_positive": "paired bootstrap CI point estimate > 0",
            "cond4_spearman_rho_ge_0p30": "reference utility prediction Spearman rho >= 0.30",
            "cond5_worst_reference_risk_reduce": "worst-reference risk reduced",
            "cond6_catastrophic_failure_not_increase": "catastrophic failure does not increase",
            "cond7_3seeds_consistent": "3 seeds directionally consistent",
            "cond8_weights_not_collapsed": "weights do not collapse to always select same reference position",
            "cond9_no_query_gt": "does not use query GT",
        },
        "pass_state": "PHASER_GATE_R3_REFERENCE_WEIGHTING_PASS",
        "fail_state": "PHASER_GATE_R3_REFERENCE_WEIGHTING_NOT_LEARNABLE",
    },
    "gate_r4_holdout": {
        "conditions": {
            "cond1_delta_dice_ge_0p0075": "vs Random Single: Delta Dice >= 0.0075",
            "cond2_bootstrap_positive": "paired bootstrap point estimate > 0",
            "cond3_variance_reduction_ge_20pct": "reference variance reduction >= 20%",
            "cond4_catastrophic_failure_not_increase": "catastrophic failure does not increase",
            "cond5_worst_query_tail_improve": "worst-query tail improved",
            "cond6_one_sam_call": "1 SAM call",
            "cond7_no_post_holdout_tuning": "no holdout post-tuning allowed",
        },
        "pass_state": "PHASER_REFERENCE_STABILITY_CONFIRMED",
        "fail_state": "PHASER_HOLDOUT_CONFIRMATION_FAIL",
    },
    "bootstrap_params": {
        "n_resamples": 10000,
        "ci_level": 0.95,
        "unit": "query",
    },
    "catastrophic_failure_threshold": 0.10,
    "frozen_at": NOW,
    "directive_section": "10-13",
}


# ===========================================================================
# Document builders
# ===========================================================================


def build_goal_read_receipt() -> str:
    return f"""# Phase R Goal Read Receipt

Date: {DATE}

## Directive

`D:\\SAM\\adapromptset\\AdaPromptSet_PhaseR_ReferenceStable_Feasibility.md`

## Goal

Execute Phase R — Real-Data Feasibility Study for Reference-Stable Prompting.

## Core Question

Can multi-reference consensus, reference reliability estimation, and stability
constraints reduce reference-induced prompt instability and improve cross-reference
segmentation reliability under the same SAM-call budget?

## Starting Point

Phase Y final state: `PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY`

Dominant bottleneck: `REFERENCE_INSTABILITY`

## Key Constraints

1. No formal test access;
2. No new SAM inference before preregistration commit;
3. 85 GPU-hour budget;
4. 4 Gates (R1-R4) with frozen thresholds;
5. Only claim candidate: reference-stable aggregation reduces variance across references;
6. 23-step execution order must be followed strictly.

## Receipt Status

ACKNOWLEDGED. All 23 sections read and understood.
"""


def build_problem_statement() -> str:
    return f"""# Phase R Problem Statement

Date: {DATE}

## Problem

Phase Y demonstrated that the dominant bottleneck in the AdaPromptSet pipeline
is `REFERENCE_INSTABILITY` — different valid reference images produce substantially
different segmentation quality for the same query, even with the same fixed action.

## Evidence

From Phase R0 historical audit:
- Mean best-worst Dice gap across 6 references: 0.1874
- Mean reference std: 0.0687
- 82.9% of validation queries have best-worst gap >= 0.05
- Exact action agreement rate (A vs B): 4.88%
- Cross-reference search Dice: 0.4105 (below fixed baseline 0.4338)

## Hypothesis

Reference-stable aggregation (mean/median/trimmed/penalized/quantile consensus)
and test-time-safe reference reliability estimation can reduce this variance and
improve cross-reference reliability.

## Research Questions

1. RQ1: Is reference instability stable and replicable on train data?
2. RQ2: Is simple deterministic consensus sufficient to reduce instability?
3. RQ3: Is reference reliability learnable from test-time-safe features?

## Scope

- MoNuSAC only (train=80, validation=41, holdout=40);
- GlaS secondary diagnostic (only if R2 passes);
- DDR optional (only if R2 passes and protocol reusable);
- No formal test;
- No SAM2/MedSAM;
- No RL;
- 1 SAM call per query.
"""


def build_data_roles() -> str:
    return f"""# Phase R Data Roles

Date: {DATE}

## MoNuSAC (Primary)

| Role | Count | Patients | Usage |
|---|---:|---:|---|
| Policy Train Queries | 80 | 18 | R1 instability replication (5 seeds); R2 β/τ selection; R3 reliability model training |
| Policy Validation Queries | 41 | 9 | R2 consensus evaluation; R3 reliability evaluation; frozen A/B |
| Confirmatory Holdout | 40 | 9 | R4 one-shot evaluation (only if R2 or R3 passes) |
| Formal Test | 101 | — | LOCKED. No access. |

## GlaS (Secondary Diagnostic)

| Role | Count | Usage |
|---|---:|---|
| Development | 55 | β/τ config selection (only if MoNuSAC R2 passes) |
| Strict Validation | 30 | Directional evaluation only |
| testA/testB | — | LOCKED. No access. |

## DDR (Optional)

Default: NOT RUN.
Only if MoNuSAC R2 passes AND DDR protocol directly reusable.

## Reference Pool

- Size: 80 cases (same as MoNuSAC train)
- Patients: 18
- Patient-disjoint from all queries
- 6 references per query (A=3, B=3, non-overlapping)

## Data Discipline

- Phase X 10 queries: excluded from primary statistics;
- Phase Y holdout: still unobserved;
- Validation queries: only 1 fixed A/B observation (no multiple observations);
- All splits from Phase Y are reused without modification.
"""


def build_reference_resampling_protocol() -> str:
    return f"""# Phase R Reference Resampling Protocol

Date: {DATE}

## Protocol

Per directive section 5:

### 5.1 References per query

Each MoNuSAC query uses 6 train-only references:
- Reference Set A = 3 references
- Reference Set B = 3 references
- A and B are completely non-overlapping
- References are patient-disjoint from query
- All methods share the same references
- Query is the statistical unit
- Reference episode is NOT an independent sample

### 5.2 Resampling

**Train (80 queries):** 5 different reference resampling seeds
- Seeds: {TRAIN_SEEDS_STR}
- Each seed produces 6 references, split into A=3/B=3
- Used for: instability estimation, reliability training
- NOT used for validation model selection

**Validation (41 queries):** 1 fixed, preregistered A/B split
- Seed: 20260722
- Prevents multiple observations on validation

## Manifests

- `manifests/phaseR/train_resampling_seed_0.csv` through `seed_4.csv`
- `manifests/phaseR/train_resampling_all_seeds.csv` (combined)
- `manifests/phaseR/validation_AB_fixed.csv`

## Verification

- A/B non-overlap: VERIFIED for all query×seed pairs
- Patient-disjoint: VERIFIED for all rows
- Seed diversity: 80/80 queries have different reference sets across 5 seeds
"""


TRAIN_SEEDS_STR = "[20260722, 20261722, 20262722, 20263722, 20264722]"


def build_consensus_methods() -> str:
    return f"""# Phase R Consensus Methods (Frozen)

Date: {DATE}

Per directive section 7. All methods are frozen at preregistration.

## B0. Single Reference

Per-reference Dice. Report: mean, best, worst, std, catastrophic failure rate.
Best reference is privileged diagnostic only — NOT deployable.

## B1. Random Single Reference

Randomly select 1 of 6 references (fixed seed). Deployable high-variance baseline.

## B2. Mean Consensus

C_mean(x) = (1/m) * sum_r C_r(x)

## B3. Median Consensus

C_median(x) = median_r C_r(x)

## B4. Trimmed Mean

Remove max and min reference score per position, then mean. Requires >= 5 references.

## B5. Disagreement-Penalized Consensus

mu(x) = (1/m) * sum_r C_r(x)
sigma(x) = sqrt((1/m) * sum_r (C_r(x) - mu(x))^2)
C_stable(x) = mu(x) - beta * sigma(x)

Frozen beta grid: [0.25, 0.5, 1.0]
beta selected ONLY on train.

## B6. Quantile Consensus

C_q(x) = Q_tau({{C_r(x)}})

Frozen tau grid: [0.50, 0.67, 0.83]
tau selected ONLY on train.

## B7. Reliability-Weighted Mean

w_r = exp(g_r / T) / sum_j exp(g_j / T)
C_weighted(x) = sum_r w_r * C_r(x)

CONDITIONAL: Only trained after Gate R2 passes.

## Frozen Config

See `configs/frozen/phaseR_consensus_methods.yaml` for machine-readable frozen config.
"""


def build_reference_reliability_protocol() -> str:
    return f"""# Phase R Reference Reliability Protocol

Date: {DATE}

Per directive section 8. Only test-time-safe features are allowed.

## 8.1 Reference-only features

- reference foreground area ratio
- foreground component count
- mask compactness
- mask fragmentation
- reference embedding mean/std
- foreground/background prototype separation
- reference stain/intensity statistics

## 8.2 Reference-query pair features

- global embedding cosine
- foreground prototype max/mean score
- background prototype max/mean score
- contrast mean/std/entropy
- top-score decay
- peak count
- candidate dispersion
- estimated foreground ratio
- map smoothness
- high-score connected components
- cross-reference disagreement
- candidate overlap with other references

## 8.3 FORBIDDEN features

- query GT
- query component count GT
- query Dice
- oracle reference ID
- best-reference label from validation
- formal-test information
- search trajectory

## Training labels

y_r = Dice(q, r) — computed ONLY on train queries.
Validation does NOT generate training labels.

## Model ladder (Gate R3)

- M0: Uniform weights (baseline)
- M1: Handcrafted reliability score (e.g., prototype separation - lambda * map entropy)
- M2: Linear regression
- M3: Gradient boosting (lightweight)
- M4: Small MLP (only if M2/M3 insufficient)

## Forbidden models

- Large transformers
- RL reference selector
- SAMIX-style sequential selection
- Validation GT supervision

## Training objective

L = L_utility + lambda * L_rank
L_utility = (y_hat_r - y_r)^2
L_rank = max(0, 1 - (y_hat_i - y_hat_j) * sign(y_i - y_j))
"""


def build_statistics_protocol() -> str:
    return f"""# Phase R Statistics Protocol (Frozen)

Date: {DATE}

Per directive sections 10-13. All thresholds are frozen at preregistration.

## Bootstrap

- n_resamples: 10000
- CI level: 0.95
- Statistical unit: query (not reference episode)

## Catastrophic failure

- Threshold: Dice < 0.10

## Gate R1: Reference Instability Replication (5 conditions)

1. >= 30% train queries have best-worst Dice gap >= 0.05
2. mean reference std >= 0.02
3. instability direction consistent across 5 resampling seeds
4. at least one prompt config shows stable reference sensitivity
5. instability NOT caused by code/coordinate/leakage error

Pass: `PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED`
Fail: `PHASER_GATE_R1_REFERENCE_INSTABILITY_NOT_REPLICATED`

## Gate R2: Deterministic Consensus Feasibility (8 conditions)

1. Best deterministic vs Random Single: Delta Dice >= 0.010
2. paired bootstrap 95% CI lower > 0
3. reference variance reduction >= 25%
4. catastrophic failure does not increase
5. A/B prediction-mask overlap improves
6. still 1 SAM call
7. no query GT
8. at least 2 of 3 prompt configs directionally consistent

Pass: `PHASER_GATE_R2_DETERMINISTIC_CONSENSUS_PASS`
Fail: `PHASER_GATE_R2_CONSENSUS_NO_GAIN`

## Gate R3: Reference Reliability Learnability (9 conditions)

1. Weighted vs best deterministic: Delta Dice >= 0.005
2. Weighted vs Random Single: Delta Dice >= 0.015
3. paired bootstrap CI point estimate > 0
4. reference utility prediction Spearman rho >= 0.30
5. worst-reference risk reduced
6. catastrophic failure does not increase
7. 3 seeds directionally consistent
8. weights do not collapse to same reference position
9. no query GT

Pass: `PHASER_GATE_R3_REFERENCE_WEIGHTING_PASS`
Fail: `PHASER_GATE_R3_REFERENCE_WEIGHTING_NOT_LEARNABLE`

## Gate R4: Confirmatory Holdout (7 conditions)

1. vs Random Single: Delta Dice >= 0.0075
2. paired bootstrap point estimate > 0
3. reference variance reduction >= 20%
4. catastrophic failure does not increase
5. worst-query tail improved
6. 1 SAM call
7. no post-holdout tuning

Pass: `PHASER_REFERENCE_STABILITY_CONFIRMED`
Fail: `PHASER_HOLDOUT_CONFIRMATION_FAIL`
"""


def build_claim_registry() -> str:
    return f"""# Phase R Claim Registry

Date: {DATE}

## Only Allowed Claim Candidate

> Reference-stable aggregation and stability-aware prompt generation can reduce
> performance variance across valid references and improve cross-reference
> segmentation reliability under the same SAM-call budget.

## Allowed Conclusions by Gate Outcome

### Only R1 passes

> Valid references induce substantial segmentation variability under the current
> frozen matching-and-prompting pipeline.

### R2 passes

> Deterministic multi-reference consensus reduces reference-induced variance and
> improves cross-reference reliability under one SAM call.

### R3 passes

> Test-time-safe reference reliability estimation further improves over uniform consensus.

### Holdout passes

> Reference-stable prompting confirms on a fresh internal holdout and is ready
> for a separately frozen formal-test protocol.

## Permanently Forbidden Claims

- "reference instability solved"
- "universal across datasets"
- "SOTA"
- "formal test success"
- "ProSAM/MAUP/SAMIX fully outperformed"
- "reference weighting is model agnostic"
- "mean Dice alone proves robustness"
"""


def build_test_lock_notice() -> str:
    return f"""# Phase R Test Lock Notice

Date: {DATE}

## Locked Test Sets

The following test sets are LOCKED for Phase R:

1. **MoNuSAC formal test** (101 cases) — NO ACCESS
2. **GlaS testA** — NO ACCESS
3. **GlaS testB** — NO ACCESS
4. **IDRID test** — NO ACCESS
5. **DDR test** — NO ACCESS

## Access Conditions

Formal test access requires ALL of:
1. Gate R4 holdout passes;
2. Phase Z freeze documents created;
3. User approval obtained.

## Violation

Any access to locked test sets during Phase R is a protocol violation and
invalidates all results.
"""


def build_gate_status() -> str:
    return f"""# Phase R Gate Status

Date: {DATE}

Phase: PREREGISTRATION (no inference yet)

Directive: `AdaPromptSet_PhaseR_ReferenceStable_Feasibility.md`

## Pre-Gate: Preregistration Completeness

Status: `IN_PROGRESS`

## Gate R1 — Reference Instability Replication

Status: `NOT_STARTED`

## Gate R2 — Deterministic Consensus Feasibility

Status: `NOT_STARTED` (blocked by R1)

## Gate R3 — Reference Reliability Learnability

Status: `NOT_STARTED` (blocked by R2)

## Gate R4 — Confirmatory Holdout

Status: `NOT_STARTED` (blocked by R2 or R3)

## GlaS Secondary Diagnostic

Status: `NOT_STARTED` (blocked by MoNuSAC R2)

## Test Lock

Status: `LOCKED_UNUSED`

## Compute Budget

- Total cap: 85 GPU-hours
- Used: 0.00 GPU-hours
- Remaining: 85.00 GPU-hours
"""


def build_compute_ledger() -> str:
    return """stage,date,gpu_hours,device,description
historical_audit,2026-07-22,0.00,cpu,Phase R0 read-only audit (8 files)
reference_pool_audit,2026-07-22,0.00,cpu,Reference pool sufficiency check
manifest_build,2026-07-22,0.00,cpu,Train resampling + validation A/B manifests
preregistration,2026-07-22,0.00,cpu,All preregistration documents
"""


def build_protocol_snapshot() -> dict:
    return {
        "phase": "R",
        "date": NOW,
        "directive": "AdaPromptSet_PhaseR_ReferenceStable_Feasibility.md",
        "branch": "phaseR-reference-stable-feasibility",
        "starting_state": "PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY",
        "frozen_configs": {
            "prompt_configs": ["P20-N1", "P30-N0", "P50-N0"],
            "consensus_methods": ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"],
            "beta_grid": [0.25, 0.5, 1.0],
            "tau_grid": [0.50, 0.67, 0.83],
            "matcher": "DINOv2-L",
            "segmenter": "SAM-B",
            "sam_calls": 1,
            "prompt_group": 1,
        },
        "data_splits": {
            "train": 80,
            "validation": 41,
            "holdout": 40,
        },
        "reference_resampling": {
            "refs_per_query": 6,
            "train_seeds": [20260722, 20261722, 20262722, 20263722, 20264722],
            "validation_seed": 20260722,
        },
        "compute_budget_gpu_hours": 85,
        "gates": ["R1", "R2", "R3", "R4"],
        "test_locked": True,
    }


# ===========================================================================
# Main
# ===========================================================================


def main() -> int:
    print("=" * 70)
    print("Phase R Preregistration Builder (Steps 6-10)")
    print("=" * 70)
    print()

    # Step 6: Freeze prompt configs
    print("Step 6: Freezing prompt configs...")
    frozen_prompts = CONFIGS / "frozen" / "phaseR_prompt_configs.yaml"
    yaml_content = f"""# Phase R Frozen Prompt Configs
# Frozen at: {NOW}
# Directive section: 6

configs:
"""
    for cfg in FROZEN_PROMPT_CONFIGS["configs"]:
        yaml_content += f"""  - name: {cfg['name']}
    action_id: {cfg['action_id']}
    k_positive: {cfg['k_positive']}
    k_negative: {cfg['k_negative']}
    positive_selector: {cfg['positive_selector']}
    negative_selector: {cfg['negative_selector']}
    grouping: {cfg['grouping']}
    sam_calls: {cfg['sam_calls']}
"""
    _write(frozen_prompts, yaml_content)
    print(f"  {frozen_prompts}")

    # Step 7: Freeze consensus methods
    print("Step 7: Freezing consensus methods...")
    frozen_consensus = CONFIGS / "frozen" / "phaseR_consensus_methods.yaml"
    yaml_content = f"""# Phase R Frozen Consensus Methods
# Frozen at: {NOW}
# Directive section: 7

methods:
"""
    for m in FROZEN_CONSENSUS_METHODS["methods"]:
        yaml_content += f"""  - id: {m['id']}
    name: {m['name']}
    deployable: {str(m['deployable']).lower()}
    formula: {m['formula']}
"""
    _write(frozen_consensus, yaml_content)
    print(f"  {frozen_consensus}")

    # Step 8: Freeze β, τ grid
    print("Step 8: Freezing beta/tau grid...")
    frozen_grid = CONFIGS / "frozen" / "phaseR_beta_tau_grid.yaml"
    yaml_content = f"""# Phase R Frozen Beta/Tau Grid
# Frozen at: {NOW}
# Directive section: 7 (B5, B6)

beta_grid: [0.25, 0.5, 1.0]
tau_grid: [0.50, 0.67, 0.83]
selection_rule: beta and tau selected ONLY on MoNuSAC train=80
"""
    _write(frozen_grid, yaml_content)
    print(f"  {frozen_grid}")

    # Step 9: Freeze statistics and Gate
    print("Step 9: Freezing statistics and Gate thresholds...")
    frozen_stats = CONFIGS / "frozen" / "phaseR_statistics.yaml"
    yaml_content = f"""# Phase R Frozen Statistics and Gate Thresholds
# Frozen at: {NOW}
# Directive sections: 10-13

bootstrap:
  n_resamples: 10000
  ci_level: 0.95
  unit: query

catastrophic_failure_threshold: 0.10

gate_r1:
  cond1: ">= 30% train queries best-worst gap >= 0.05"
  cond2: "mean ref std >= 0.02"
  cond3: "direction consistent across 5 seeds"
  cond4: ">= 1 config shows sensitivity"
  cond5: "not code/coordinate/leakage error"
  pass: PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED
  fail: PHASER_GATE_R1_REFERENCE_INSTABILITY_NOT_REPLICATED

gate_r2:
  cond1: "Delta Dice >= 0.010 vs Random Single"
  cond2: "bootstrap CI lower > 0"
  cond3: "variance reduction >= 25%"
  cond4: "catastrophic failure not increase"
  cond5: "A/B prediction overlap improve"
  cond6: "1 SAM call"
  cond7: "no query GT"
  cond8: ">= 2 of 3 configs consistent"
  pass: PHASER_GATE_R2_DETERMINISTIC_CONSENSUS_PASS
  fail: PHASER_GATE_R2_CONSENSUS_NO_GAIN

gate_r3:
  cond1: "Delta Dice >= 0.005 vs best deterministic"
  cond2: "Delta Dice >= 0.015 vs Random Single"
  cond3: "bootstrap CI > 0"
  cond4: "Spearman rho >= 0.30"
  cond5: "worst-reference risk reduce"
  cond6: "catastrophic failure not increase"
  cond7: "3 seeds consistent"
  cond8: "weights not collapsed"
  cond9: "no query GT"
  pass: PHASER_GATE_R3_REFERENCE_WEIGHTING_PASS
  fail: PHASER_GATE_R3_REFERENCE_WEIGHTING_NOT_LEARNABLE

gate_r4:
  cond1: "Delta Dice >= 0.0075 vs Random Single"
  cond2: "bootstrap > 0"
  cond3: "variance reduction >= 20%"
  cond4: "catastrophic failure not increase"
  cond5: "worst-query tail improve"
  cond6: "1 SAM call"
  cond7: "no post-holdout tuning"
  pass: PHASER_REFERENCE_STABILITY_CONFIRMED
  fail: PHASER_HOLDOUT_CONFIRMATION_FAIL
"""
    _write(frozen_stats, yaml_content)
    print(f"  {frozen_stats}")

    # Create all 13 preregistration documents
    print("\nCreating preregistration documents...")

    docs_to_create = [
        ("docs/phaseR_goal_read_receipt.md", build_goal_read_receipt()),
        ("docs/phaseR_problem_statement.md", build_problem_statement()),
        ("docs/phaseR_data_roles.md", build_data_roles()),
        ("docs/phaseR_reference_resampling_protocol.md", build_reference_resampling_protocol()),
        ("docs/phaseR_consensus_methods.md", build_consensus_methods()),
        ("docs/phaseR_reference_reliability_protocol.md", build_reference_reliability_protocol()),
        ("docs/phaseR_statistics_protocol.md", build_statistics_protocol()),
        ("docs/phaseR_claim_registry.md", build_claim_registry()),
        ("docs/phaseR_test_lock_notice.md", build_test_lock_notice()),
        ("reports/phaseR_gate_status.md", build_gate_status()),
        ("reports/phaseR_compute_ledger.csv", build_compute_ledger()),
    ]

    for rel_path, content in docs_to_create:
        path = PROJECT_ROOT / rel_path
        _write(path, content)
        print(f"  {rel_path}")

    # Protocol snapshot JSON
    snapshot = build_protocol_snapshot()
    snapshot_path = REPORTS / "phaseR_protocol_snapshot.json"
    _write_json(snapshot_path, snapshot)
    print(f"  reports/phaseR_protocol_snapshot.json")

    # Step 10: Generate hashes
    print("\nStep 10: Generating input hashes...")

    files_to_hash = [
        ("directive", PROJECT_ROOT / "AdaPromptSet_PhaseR_ReferenceStable_Feasibility.md"),
        ("frozen_prompt_configs", frozen_prompts),
        ("frozen_consensus_methods", frozen_consensus),
        ("frozen_beta_tau_grid", frozen_grid),
        ("frozen_statistics", frozen_stats),
        ("train_manifest_seed_0", MANIFESTS / "train_resampling_seed_0.csv"),
        ("train_manifest_seed_1", MANIFESTS / "train_resampling_seed_1.csv"),
        ("train_manifest_seed_2", MANIFESTS / "train_resampling_seed_2.csv"),
        ("train_manifest_seed_3", MANIFESTS / "train_resampling_seed_3.csv"),
        ("train_manifest_seed_4", MANIFESTS / "train_resampling_seed_4.csv"),
        ("train_manifest_all", MANIFESTS / "train_resampling_all_seeds.csv"),
        ("validation_AB_fixed", MANIFESTS / "validation_AB_fixed.csv"),
        ("manifest_summary", MANIFESTS / "manifest_summary.json"),
        ("phaseY_train_queries", PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv"),
        ("phaseY_val_queries", PROJECT_ROOT / "manifests" / "phaseY" / "policy_validation_queries.csv"),
        ("phaseY_holdout_queries", PROJECT_ROOT / "manifests" / "phaseY" / "confirmatory_holdout_queries.csv"),
        ("phaseY_reference_pool", PROJECT_ROOT / "manifests" / "phaseY" / "reference_pool.csv"),
    ]

    hashes = {
        "date": NOW,
        "phase": "R",
        "branch": "phaseR-reference-stable-feasibility",
        "files": {},
    }

    for name, path in files_to_hash:
        h = _sha256_file(path)
        hashes["files"][name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": h,
            "exists": path.is_file(),
        }
        print(f"  {name}: {h[:16]}...")

    hashes_path = REPORTS / "phaseR_input_hashes.json"
    _write_json(hashes_path, hashes)
    print(f"\n  Hashes written to: {hashes_path}")

    # Update protocol history
    print("\nUpdating protocol history...")
    history_path = DOCS / "protocol_history.md"
    existing = history_path.read_text(encoding="utf-8") if history_path.is_file() else ""
    new_entry = f"""
## {DATE} Phase R — Preregistration Complete

Directive: `AdaPromptSet_PhaseR_ReferenceStable_Feasibility.md`
Branch: `phaseR-reference-stable-feasibility`

### Phase R0 Historical Audit
- 8 audit files produced under `reports/phaseR_historical_audit/`
- Mean best-worst Dice gap: 0.1874
- Mean reference std: 0.0687
- 82.9% queries with gap >= 0.05
- Reference instability strongly confirmed

### Reference Pool Audit
- Pool size: 80 cases, 18 patients
- Min available refs: 70 (train), 80 (validation)
- Status: PHASER_REFERENCE_POOL_SUFFICIENT

### Manifests
- 5 train resampling seeds (2400 rows total)
- 1 validation fixed A/B (246 rows)
- All verified: A/B non-overlap, patient-disjoint, seed diversity

### Frozen Configs
- Prompt configs: P20-N1, P30-N0, P50-N0
- Consensus methods: B0-B7
- beta grid: [0.25, 0.5, 1.0]
- tau grid: [0.50, 0.67, 0.83]
- Statistics: 4 Gates with frozen thresholds

### Preregistration Documents
- 9 docs/phaseR_*.md files
- 3 reports/phaseR_* files (md, csv, json)
- 4 configs/frozen/phaseR_*.yaml files
- Input hashes generated

### Compute
- Used: 0.00 GPU-hours (CPU-only so far)
- Budget: 85 GPU-hours

### Next
- Gate R1: Reference Instability Replication (train=80, 5 seeds)
"""
    _write(history_path, existing + new_entry)
    print(f"  {history_path}")

    # Update decision log
    decision_path = DOCS / "decision_log.md"
    existing_dec = decision_path.read_text(encoding="utf-8") if decision_path.is_file() else ""
    new_dec = f"""
| {DATE} | Start Phase R — Reference-Stable Prompting Feasibility | Phase Y identified REFERENCE_INSTABILITY as dominant bottleneck; Phase R tests whether consensus and reliability estimation can reduce it | active |
| {DATE} | Freeze Phase R prompt configs, consensus methods, beta/tau grid, and Gate thresholds before any new SAM inference | Directive section 6-9 requires all configs frozen at preregistration; no inference allowed before preregistration commit | locked |
| {DATE} | Phase R0 historical audit confirms reference instability is real and large | Mean best-worst gap=0.1874, mean ref std=0.0687, 82.9% queries gap>=0.05; the small stable oracle gap (0.0061) was an aggregate artifact, not absence of instability | locked |
"""
    _write(decision_path, existing_dec + new_dec)
    print(f"  {decision_path}")

    # Update data exposure log
    exposure_path = DOCS / "data_exposure_log.md"
    existing_exp = exposure_path.read_text(encoding="utf-8") if exposure_path.is_file() else ""
    new_exp = f"""
## {DATE} Phase R Data Exposure

- MoNuSAC train=80: READ (queries and reference pool)
- MoNuSAC validation=41: READ (queries; reference A/B fixed)
- MoNuSAC holdout=40: NOT ACCESSED (locked)
- MoNuSAC test=101: NOT ACCESSED (locked)
- GlaS: NOT ACCESSED (pending R2 pass)
- DDR: NOT ACCESSED (optional, pending R2 pass)
- Phase Y artifacts: READ (all reports/phaseY_* and manifests/phaseY/*)
- Phase R0 audit: READ-ONLY (no new SAM inference)
- Phase R manifests: CREATED (5 train seeds + 1 val fixed)
"""
    _write(exposure_path, existing_exp + new_exp)
    print(f"  {exposure_path}")

    print("\n" + "=" * 70)
    print("Preregistration complete. Ready for preregistration commit.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
