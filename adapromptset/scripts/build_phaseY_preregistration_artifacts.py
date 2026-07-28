"""Phase Y — Generate structured preregistration artifacts.

Generates:
  configs/actions/phaseY_full_single_call_bank.yaml
  configs/actions/phaseY_compact_policy_bank.yaml (placeholder)
  reports/phaseY_protocol_snapshot.json
  reports/phaseY_input_hashes.json
  reports/phaseY_compute_ledger.csv
  reports/phaseY_gate_status.md

Per directive D:\\SAM\\adapromptset\\AdaPromptSet_PhaseY_Policy_Feasibility.md
sections 6, 7, 8, 9, 10.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\SAM\adapromptset")
CONFIG_DIR = PROJECT_ROOT / "configs" / "actions"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Frozen grids per directive section 7.1
K_POSITIVES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
K_NEGATIVES = [0, 1, 3, 5]
POS_SELECTORS = ["P_TOPK", "P_SPREAD", "P_LOCAL_PEAKS"]
NEG_SELECTORS_K0 = ["N_NONE"]
NEG_SELECTORS_KGT0 = ["N_BOTTOMK", "N_HIGH_BACKGROUND"]
GROUPING = "G_ONE_GROUP"
CALLS = 1

# Directive path
DIRECTIVE_PATH = PROJECT_ROOT / "AdaPromptSet_PhaseY_Policy_Feasibility.md"

# Preregistration files to hash
PREREG_FILES = [
    "docs/phaseY_goal_read_receipt.md",
    "docs/phaseY_scientific_questions.md",
    "docs/phaseY_data_split_protocol.md",
    "docs/phaseY_reference_protocol.md",
    "docs/phaseY_single_call_action_bank.md",
    "docs/phaseY_oracle_stability_protocol.md",
    "docs/phaseY_bank_policy_protocol.md",
    "docs/phaseY_point_subset_protocol.md",
    "docs/phaseY_statistics_protocol.md",
    "docs/phaseY_claim_registry.md",
    "docs/phaseY_test_lock_notice.md",
    "reports/phaseY_gate_status.md",
    "reports/phaseY_compute_ledger.csv",
    "reports/phaseY_protocol_snapshot.json",
    "reports/phaseY_input_hashes.json",
    "configs/actions/phaseY_full_single_call_bank.yaml",
    "configs/actions/phaseY_compact_policy_bank.yaml",
    "manifests/phaseY/policy_train_queries.csv",
    "manifests/phaseY/policy_validation_queries.csv",
    "manifests/phaseY/confirmatory_holdout_queries.csv",
    "manifests/phaseY/reference_pool.csv",
    "manifests/phaseY/reference_set_A.csv",
    "manifests/phaseY/reference_set_B.csv",
    "reports/phaseY_data_audit/data_audit_summary.md",
    "reports/phaseY_data_audit/split_summary.json",
    "reports/phaseY_reference_audit.json",
    "scripts/build_phaseY_data_audit_and_splits.py",
    "scripts/build_phaseY_preregistration_artifacts.py",
]


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ---------------------------------------------------------------------------
# 7.1 Full Audit Bank YAML
# ---------------------------------------------------------------------------
def build_full_bank_actions() -> List[Dict[str, Any]]:
    """Enumerate all actions in the full audit bank.

    Action = (positive_selector, K+, negative_selector, K-, G_ONE_GROUP, 1 call)
    When K-=0: negative_selector = N_NONE
    When K->0: negative_selector ∈ {N_BOTTOMK, N_HIGH_BACKGROUND}
    """
    actions: List[Dict[str, Any]] = []
    aid = 0
    for ps in POS_SELECTORS:
        for kp in K_POSITIVES:
            for kn in K_NEGATIVES:
                if kn == 0:
                    neg_selectors = NEG_SELECTORS_K0
                else:
                    neg_selectors = NEG_SELECTORS_KGT0
                for ns in neg_selectors:
                    aid += 1
                    action_id = f"Y_{ps}_K{kp}_N{kn}_{ns}_G1_C1"
                    actions.append(
                        {
                            "action_id": action_id,
                            "action_index": aid,
                            "positive_selector": ps,
                            "k_positive": kp,
                            "negative_selector": ns,
                            "k_negative": kn,
                            "grouping": GROUPING,
                            "sam_calls": CALLS,
                            "point_count": kp + kn,
                        }
                    )
    return actions


def write_full_bank_yaml(actions: List[Dict[str, Any]]) -> str:
    """Write the full audit bank as YAML."""
    lines = []
    lines.append("# Phase Y Single-Call One-Group Full Audit Bank")
    lines.append("# Directive: AdaPromptSet_PhaseY_Policy_Feasibility.md section 7.1")
    lines.append("# Frozen: do NOT modify after preregistration commit.")
    lines.append("# All actions use G_ONE_GROUP and exactly 1 SAM call.")
    lines.append("")
    lines.append("bank_name: phaseY_full_single_call_bank")
    lines.append("bank_version: v1")
    lines.append("frozen_at: preregistration")
    lines.append(f"total_actions: {len(actions)}")
    lines.append("")
    lines.append("frozen_dimensions:")
    lines.append(f"  k_positive: {K_POSITIVES}")
    lines.append(f"  k_negative: {K_NEGATIVES}")
    lines.append(f"  positive_selectors: {POS_SELECTORS}")
    lines.append(f"  negative_selectors_k0: {NEG_SELECTORS_K0}")
    lines.append(f"  negative_selectors_kgt0: {NEG_SELECTORS_KGT0}")
    lines.append(f"  grouping: {GROUPING}")
    lines.append(f"  sam_calls: {CALLS}")
    lines.append("")
    lines.append("constraints:")
    lines.append("  - No query GT in action definition")
    lines.append("  - All points in a single prompt group")
    lines.append("  - Exactly 1 SAM call per action")
    lines.append("  - No mask union / multimask fusion beyond frozen MVP rule")
    lines.append("")
    lines.append("actions:")
    for a in actions:
        lines.append(f"  - action_id: {a['action_id']}")
        lines.append(f"    action_index: {a['action_index']}")
        lines.append(f"    positive_selector: {a['positive_selector']}")
        lines.append(f"    k_positive: {a['k_positive']}")
        lines.append(f"    negative_selector: {a['negative_selector']}")
        lines.append(f"    k_negative: {a['k_negative']}")
        lines.append(f"    grouping: {a['grouping']}")
        lines.append(f"    sam_calls: {a['sam_calls']}")
        lines.append(f"    point_count: {a['point_count']}")
    return "\n".join(lines) + "\n"


def write_compact_bank_yaml_placeholder() -> str:
    """Placeholder for compact policy bank — populated after Pareto pruning
    on train-only queries (post-preregistration, directive section 7.2)."""
    lines = []
    lines.append("# Phase Y Compact Policy Bank (PLACEHOLDER)")
    lines.append("# Directive: AdaPromptSet_PhaseY_Policy_Feasibility.md section 7.2")
    lines.append("#")
    lines.append("# This file is a PLACEHOLDER at preregistration time.")
    lines.append("# The compact bank (12-32 actions) is derived from the full audit bank")
    lines.append("# via train-only Pareto pruning AFTER the preregistration commit.")
    lines.append("# Pruning rules (section 7.2):")
    lines.append("#   1. Drop actions dominated in Dice/calls/runtime by another action")
    lines.append("#   2. Drop actions clearly inferior across all train folds")
    lines.append("#   3. Within a family, if Dice diff < 0.003, keep the simpler action")
    lines.append("#   4. Validation and holdout do NOT participate in pruning")
    lines.append("#   5. Target compact bank size: 12-32 actions")
    lines.append("")
    lines.append("bank_name: phaseY_compact_policy_bank")
    lines.append("bank_version: pending")
    lines.append("status: PLACEHOLDER_PENDING_PARETO_PRUNING")
    lines.append("target_size_min: 12")
    lines.append("target_size_max: 32")
    lines.append("source_bank: phaseY_full_single_call_bank")
    lines.append("pruning_scope: policy_train_queries_only")
    lines.append("actions: []")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Protocol snapshot JSON
# ---------------------------------------------------------------------------
def build_protocol_snapshot(full_bank: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "phase": "phaseY",
        "name": "Phase Y — Fresh-Query Confirmation, Oracle Stability, and Deployable Policy Feasibility",
        "branch": "phaseY-policy-feasibility",
        "preregistration_date": "2026-07-21",
        "directive": str(DIRECTIVE_PATH),
        "directive_sha256": sha256_file(DIRECTIVE_PATH),
        "scope": "MoNuSAC train/validation only; no formal test access",
        "main_question": "Can a query-conditioned, single-call prompt-configuration policy outperform the strongest fixed dense prompt under a strictly matched deployment budget?",
        "frozen_scope": {
            "matcher": "DINOv2-L",
            "segmenter": "SAM-B",
            "sam_calls": 1,
            "prompt_groups": 1,
            "fusion": "none",
            "multimask_rule": "frozen",
            "postprocess": "frozen",
        },
        "prohibited_in_main_experiment": [
            "2-100 SAM calls",
            "one-point-per-group",
            "mask union",
            "KMeans multi-group calls",
            "graph multi-group calls",
            "SAM-L / SAM2 / MedSAM",
            "GlaS / DDR",
            "formal test",
            "PPO / DQN",
            "continuous coordinate generation",
            "GT-guided search outputs as deployable actions",
        ],
        "frozen_grids": {
            "k_positive": K_POSITIVES,
            "k_negative": K_NEGATIVES,
            "positive_selectors": POS_SELECTORS,
            "negative_selectors_k0": NEG_SELECTORS_K0,
            "negative_selectors_kgt0": NEG_SELECTORS_KGT0,
            "grouping": GROUPING,
            "sam_calls": CALLS,
        },
        "full_audit_bank": {
            "total_actions": len(full_bank),
            "bank_name": "phaseY_full_single_call_bank",
            "bank_version": "v1",
            "config_path": "configs/actions/phaseY_full_single_call_bank.yaml",
        },
        "compact_policy_bank": {
            "status": "PLACEHOLDER_PENDING_PARETO_PRUNING",
            "target_size_min": 12,
            "target_size_max": 32,
            "config_path": "configs/actions/phaseY_compact_policy_bank.yaml",
            "pruning_scope": "policy_train_queries_only",
        },
        "data_split": {
            "source": "MoNuSAC processed manifest (310 cases)",
            "split_unit": "patient",
            "target_train": 80,
            "target_validation": 40,
            "target_holdout": 40,
            "actual_train": 80,
            "actual_validation": 41,
            "actual_holdout": 40,
            "patient_level_disjoint": True,
            "phasex_excluded_patients": 10,
            "test_locked": True,
            "manifests_dir": "manifests/phaseY/",
        },
        "reference_sets": {
            "set_A_size": 3,
            "set_B_size": 3,
            "disjoint_per_query": True,
            "patient_disjoint_from_query": True,
            "references_from_train_only": True,
            "fixed_seed": 20260721,
        },
        "gates": {
            "Y1": {
                "name": "Fresh-Query Action Heterogeneity",
                "pass_state": "PHASEY_GATE_Y1_STABLE_HETEROGENEITY",
                "fail_states": [
                    "PHASEY_GATE_Y1_REFERENCE_UNSTABLE",
                    "PHASEY_GATE_Y1_SELECTION_BIAS_DOMINATES",
                    "PHASEY_GATE_Y1_ORACLE_GAP_SMALL",
                ],
                "conditions": [
                    "stable_oracle_gap >= 0.015 Dice",
                    "paired bootstrap 95% CI lower bound > 0",
                    ">=30% validation queries have stable best-action family != strongest fixed",
                    "cross-reference action-rank median Spearman rho > 0.30",
                    "oracle optimism <= 60% of raw oracle gap",
                    "all actions fixed at 1 SAM call (no call-count confound)",
                ],
            },
            "Y2": {
                "name": "Search Oracle Stability",
                "pass_state": "PHASEY_GATE_Y2_SEARCH_LABELS_STABLE",
                "fail_state": "PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT",
                "conditions": [
                    "cross-reference search >= 0.03 Dice over strongest fixed",
                    "paired bootstrap CI lower bound > 0",
                    "search optimism <= 50% of raw search gap",
                    "median |K_A - K_B| <= 8 OR cross Dice diff <= 0.01",
                    ">=70% queries: A-search prompt beats strongest fixed on B",
                ],
            },
            "Y3": {
                "name": "Bank Policy Validation",
                "pass_state": "PHASEY_GATE_Y3_BANK_POLICY_VALIDATED",
                "fail_state": "PHASEY_GATE_Y3_BANK_POLICY_INSUFFICIENT",
                "conditions": [
                    "Delta Dice >= 0.010 over strongest fixed on validation",
                    "paired bootstrap 95% CI lower bound > 0",
                    "oracle recovery >= 30%",
                ],
            },
            "Y4": {
                "name": "Point-Subset Learnability",
                "pass_state": "PHASEY_GATE_Y4_POINT_SUBSET_LEARNABLE",
                "fail_state": "PHASEY_GATE_Y4_POINT_SUBSET_NOT_LEARNABLE",
                "conditions": [
                    "Delta Dice >= 0.015 over strongest fixed, OR",
                    "Delta Dice >= 0.0075 over bank policy",
                    "paired bootstrap 95% CI lower bound > 0",
                ],
            },
        },
        "model_ladders": {
            "bank_action": ["M0_global_prior", "M1_morphology_heuristic", "M2_multiclass_classifier", "M3_per_action_regression", "M4_pairwise_ranking", "M5_hierarchical_router"],
            "point_subset": ["P0_topk_contrast", "P1_pointwise_regression", "P2_pairwise_ranking", "P3_diversity_reranker", "P4_set_encoder", "P5_learned_greedy_selector"],
        },
        "statistics": {
            "unit": "query",
            "reference_episodes_not_independent": True,
            "bootstrap_method": "paired patient-cluster bootstrap",
            "bootstrap_iterations": 10000,
            "ci_level": 0.95,
            "multiple_testing": "Benjamini-Hochberg FDR at 0.05",
            "historical_10_episodes": "EXPLORATORY_PHASEX_EVIDENCE — excluded from confirmatory stats",
        },
        "compute_caps_gpu_hours": {
            "data_audit_and_split": 0.0,
            "preregistration": 0.0,
            "utility_matrix_Y1": 40.0,
            "search_oracle_Y2": 30.0,
            "bank_policy_Y3": 40.0,
            "point_subset_Y4": 30.0,
            "holdout_evaluation": 10.0,
            "statistics_and_figures": 5.0,
            "total_cap": 155.0,
        },
        "historical_anchors": {
            "phasex_status": "REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE",
            "top20_n1_fixed_dice": 0.5411,
            "a_large_bank_oracle_dice": 0.6119,
            "search_oracle_dice": 0.7256,
            "phasex_episode_count": 10,
            "label": "EXPLORATORY_PHASEX_EVIDENCE",
        },
    }


# ---------------------------------------------------------------------------
# Input hashes JSON
# ---------------------------------------------------------------------------
def build_input_hashes(full_bank_yaml: str, compact_bank_yaml: str) -> Dict[str, Any]:
    hashes: Dict[str, Any] = {
        "phase": "phaseY",
        "preregistration_date": "2026-07-21",
        "input_hashes": {},
    }
    for rel in PREREG_FILES:
        abs_path = PROJECT_ROOT / rel
        h = sha256_file(abs_path)
        hashes["input_hashes"][rel.replace("\\", "/")] = {
            "path": str(abs_path),
            "sha256": h,
            "exists": abs_path.exists(),
        }
    return hashes


# ---------------------------------------------------------------------------
# Compute ledger CSV
# ---------------------------------------------------------------------------
def write_compute_ledger() -> str:
    rows = [
        {"stage": "data_audit_and_split", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 5.0, "status": "COMPLETE", "notes": "read-only manifest processing"},
        {"stage": "preregistration", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 1.0, "status": "COMPLETE", "notes": "doc and config generation"},
        {"stage": "utility_matrix_Y1", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 0.0, "status": "PENDING", "notes": "Gate Y1: 210 actions x 161 queries x 2 ref sets"},
        {"stage": "search_oracle_Y2", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 0.0, "status": "PENDING", "notes": "Gate Y2: GT-guided search on train+val only"},
        {"stage": "bank_policy_Y3", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 0.0, "status": "PENDING", "notes": "Gate Y3: M0-M5 on compact bank"},
        {"stage": "point_subset_Y4", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 0.0, "status": "PENDING", "notes": "Gate Y4: P0-P5 point-subset models"},
        {"stage": "holdout_evaluation", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 0.0, "status": "PENDING", "notes": "single frozen-model run on 40 holdout queries"},
        {"stage": "statistics_and_figures", "phase": "phaseY", "gpu_hours": 0.0, "cpu_seconds": 0.0, "status": "PENDING", "notes": "18 tables + 20 figures"},
    ]
    out = []
    fields = ["stage", "phase", "gpu_hours", "cpu_seconds", "status", "notes"]
    for r in rows:
        out.append(",".join(str(r[f]) for f in fields))
    return "stage,phase,gpu_hours,cpu_seconds,status,notes\n" + "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Gate status MD
# ---------------------------------------------------------------------------
def write_gate_status_md(full_bank: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# Phase Y Gate Status")
    lines.append("")
    lines.append("Date: 2026-07-21")
    lines.append("")
    lines.append("Phase: preregistration (no new SAM inference yet).")
    lines.append("")
    lines.append("Directive: `AdaPromptSet_PhaseY_Policy_Feasibility.md`")
    lines.append("")
    lines.append("## Pre-Gate: Preregistration Completeness")
    lines.append("")
    lines.append("Status: `PENDING_PREREGISTRATION_COMMIT`")
    lines.append("")
    lines.append("Required artifacts before any new SAM inference:")
    lines.append("")
    prereq = [
        "docs/phaseY_goal_read_receipt.md",
        "docs/phaseY_scientific_questions.md",
        "docs/phaseY_data_split_protocol.md",
        "docs/phaseY_reference_protocol.md",
        "docs/phaseY_single_call_action_bank.md",
        "docs/phaseY_oracle_stability_protocol.md",
        "docs/phaseY_bank_policy_protocol.md",
        "docs/phaseY_point_subset_protocol.md",
        "docs/phaseY_statistics_protocol.md",
        "docs/phaseY_claim_registry.md",
        "docs/phaseY_test_lock_notice.md",
        "reports/phaseY_gate_status.md",
        "reports/phaseY_compute_ledger.csv",
        "reports/phaseY_protocol_snapshot.json",
        "reports/phaseY_input_hashes.json",
        "configs/actions/phaseY_full_single_call_bank.yaml",
        "configs/actions/phaseY_compact_policy_bank.yaml",
        "manifests/phaseY/policy_train_queries.csv",
        "manifests/phaseY/policy_validation_queries.csv",
        "manifests/phaseY/confirmatory_holdout_queries.csv",
        "manifests/phaseY/reference_pool.csv",
        "manifests/phaseY/reference_set_A.csv",
        "manifests/phaseY/reference_set_B.csv",
        "reports/phaseY_data_audit/data_audit_summary.md",
        "reports/phaseY_data_audit/split_summary.json",
        "reports/phaseY_reference_audit.json",
    ]
    for p in prereq:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("Failure code if any artifact is missing: `BLOCKED_PREREGISTRATION_INCOMPLETE`.")
    lines.append("")
    lines.append("## Gate Y1 — Fresh-Query Action Heterogeneity")
    lines.append("")
    lines.append("Status: `PENDING_PREREGISTRATION_COMMIT`")
    lines.append("")
    lines.append("Scope: Policy Validation Queries (41 cases), full audit bank")
    lines.append(f"({len(full_bank)} actions), Reference Sets A and B.")
    lines.append("")
    lines.append("Pass conditions (all must hold):")
    lines.append("1. `stable_oracle_gap >= 0.015 Dice`")
    lines.append("2. paired bootstrap 95% CI lower bound > 0")
    lines.append("3. `>= 30% validation queries` stable best-action family != strongest fixed")
    lines.append("4. cross-reference action-rank `median Spearman rho > 0.30`")
    lines.append("5. `oracle optimism <= 60% of raw oracle gap`")
    lines.append("6. all actions fixed at 1 SAM call (no call-count confound)")
    lines.append("")
    lines.append("Pass: `PHASEY_GATE_Y1_STABLE_HETEROGENEITY`")
    lines.append("")
    lines.append("Fail branches:")
    lines.append("- `PHASEY_GATE_Y1_REFERENCE_UNSTABLE` (cross-reference gap collapses)")
    lines.append("- `PHASEY_GATE_Y1_SELECTION_BIAS_DOMINATES` (optimism > 60%)")
    lines.append("- `PHASEY_GATE_Y1_ORACLE_GAP_SMALL` (gap < 0.015)")
    lines.append("")
    lines.append("On Y1 fail: bank-action policy training is BLOCKED; allowed to pivot to")
    lines.append("point-subset stability audit only; holdout/test access forbidden.")
    lines.append("")
    lines.append("## Gate Y2 — Search Oracle Stability")
    lines.append("")
    lines.append("Status: `PENDING_GATE_Y1`")
    lines.append("")
    lines.append("Scope: Policy Train + Validation Queries only (121 cases). Holdout and")
    lines.append("formal test are forbidden for search oracle.")
    lines.append("")
    lines.append("Pass conditions:")
    lines.append("1. `cross-reference search >= 0.03 Dice` over strongest fixed")
    lines.append("2. paired bootstrap CI lower bound > 0")
    lines.append("3. `search optimism <= 50% of raw search gap`")
    lines.append("4. `median |K_A - K_B| <= 8` OR cross Dice diff <= 0.01")
    lines.append("5. `>= 70% queries`: A-search prompt beats strongest fixed on B")
    lines.append("")
    lines.append("Pass: `PHASEY_GATE_Y2_SEARCH_LABELS_STABLE`")
    lines.append("Fail: `PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT`")
    lines.append("")
    lines.append("Interpretation rules (section 9.3):")
    lines.append("- raw high, cross high -> `SEARCH_ORACLE_REFERENCE_STABLE`")
    lines.append("- raw high, cross drops -> `SEARCH_ORACLE_REFERENCE_OVERFIT`")
    lines.append("- K unstable, cross Dice stable -> `MULTIPLE_EQUIVALENT_PROMPTS`")
    lines.append("- both unstable -> `SEARCH_LABELS_NOT_SUITABLE_FOR_SUPERVISION`")
    lines.append("")
    lines.append("## Gate Y3 — Bank Policy Validation")
    lines.append("")
    lines.append("Status: `PENDING_GATE_Y1`")
    lines.append("")
    lines.append("Scope: compact policy bank (12-32 actions, Pareto-pruned on train-only),")
    lines.append("M0-M5 model ladder, 3 seeds per learned model, Validation Queries.")
    lines.append("")
    lines.append("Pass conditions:")
    lines.append("1. `Delta Dice >= 0.010` over strongest fixed on validation")
    lines.append("2. paired bootstrap 95% CI lower bound > 0")
    lines.append("3. `oracle recovery >= 30%` (policy Dice - strongest fixed) / (oracle - strongest fixed)")
    lines.append("")
    lines.append("Pass: `PHASEY_GATE_Y3_BANK_POLICY_VALIDATED`")
    lines.append("Fail: `PHASEY_GATE_Y3_BANK_POLICY_INSUFFICIENT`")
    lines.append("")
    lines.append("## Gate Y4 — Point-Subset Learnability")
    lines.append("")
    lines.append("Status: `PENDING_GATE_Y2`")
    lines.append("")
    lines.append("Scope: P0-P5 point-subset ladder, trained on train queries with search")
    lines.append("oracle labels, validated on Validation Queries.")
    lines.append("")
    lines.append("Pass conditions (either):")
    lines.append("1. `Delta Dice >= 0.015` over strongest fixed, OR")
    lines.append("2. `Delta Dice >= 0.0075` over bank policy (Gate Y3 winner)")
    lines.append("Plus: paired bootstrap 95% CI lower bound > 0.")
    lines.append("")
    lines.append("Pass: `PHASEY_GATE_Y4_POINT_SUBSET_LEARNABLE`")
    lines.append("Fail: `PHASEY_GATE_Y4_POINT_SUBSET_NOT_LEARNABLE`")
    lines.append("")
    lines.append("## Holdout Evaluation")
    lines.append("")
    lines.append("Status: `PENDING_ALL_GATES`")
    lines.append("")
    lines.append("Scope: Confirmatory Holdout (40 cases). Run ONCE on the single frozen")
    lines.append("best model from Y3/Y4. No model selection on holdout. No holdout access")
    lines.append("before the final frozen run.")
    lines.append("")
    lines.append("## Test Lock")
    lines.append("")
    lines.append("Status: `LOCKED_UNUSED`")
    lines.append("")
    lines.append("MoNuSAC formal test (101 cases), GlaS testA/testB, DDR test, IDRID test")
    lines.append("remain locked. No test access for K, action bank, router, features,")
    lines.append("model capacity, or seed selection.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 7.1 Full audit bank
    full_bank = build_full_bank_actions()
    full_bank_yaml = write_full_bank_yaml(full_bank)
    (CONFIG_DIR / "phaseY_full_single_call_bank.yaml").write_text(full_bank_yaml, encoding="utf-8")

    # 7.2 Compact policy bank placeholder
    compact_bank_yaml = write_compact_bank_yaml_placeholder()
    (CONFIG_DIR / "phaseY_compact_policy_bank.yaml").write_text(compact_bank_yaml, encoding="utf-8")

    # Protocol snapshot
    snapshot = build_protocol_snapshot(full_bank)
    (REPORTS_DIR / "phaseY_protocol_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Input hashes
    hashes = build_input_hashes(full_bank_yaml, compact_bank_yaml)
    (REPORTS_DIR / "phaseY_input_hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Compute ledger
    ledger = write_compute_ledger()
    (REPORTS_DIR / "phaseY_compute_ledger.csv").write_text(ledger, encoding="utf-8")

    # Gate status
    gate_md = write_gate_status_md(full_bank)
    (REPORTS_DIR / "phaseY_gate_status.md").write_text(gate_md, encoding="utf-8")

    # Console
    print("=" * 60)
    print("Phase Y Preregistration Artifacts — GENERATED")
    print("=" * 60)
    print(f"Full audit bank: {len(full_bank)} actions")
    print(f"  -> {CONFIG_DIR / 'phaseY_full_single_call_bank.yaml'}")
    print(f"  -> {CONFIG_DIR / 'phaseY_compact_policy_bank.yaml'} (placeholder)")
    print(f"  -> {REPORTS_DIR / 'phaseY_protocol_snapshot.json'}")
    print(f"  -> {REPORTS_DIR / 'phaseY_input_hashes.json'}")
    print(f"  -> {REPORTS_DIR / 'phaseY_compute_ledger.csv'}")
    print(f"  -> {REPORTS_DIR / 'phaseY_gate_status.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
