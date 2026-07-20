from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GATE_F_MIN_EPISODES = 10


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(project_root)).replace("\\", "/") if path.exists() else str(path),
        "exists": path.exists(),
    }


def _glas_readiness(project_root: Path) -> dict[str, Any]:
    manifest_path = project_root / "manifests" / "glas_manifest.summary.json"
    protocol_path = project_root / "manifests" / "episodes" / "glas" / "protocol_metadata.json"
    candidate_path = project_root / "reports" / "candidate_pilot_glas" / "candidate_pilot_summary.json"
    oracle_path = project_root / "reports" / "oracle_gap_pilot_glas" / "oracle_gap_pilot_summary.json"
    manifest = _read_json_or_none(manifest_path) or {}
    protocol = _read_json_or_none(protocol_path) or {}
    candidate = _read_json_or_none(candidate_path) or {}
    oracle = _read_json_or_none(oracle_path) or {}

    episode_count = int(oracle.get("episode_count") or 0)
    gate_min = int(oracle.get("gate_min_episodes") or GATE_F_MIN_EPISODES)
    if not oracle:
        status = "BLOCKED_MISSING_ORACLE_PILOT"
    elif episode_count < gate_min:
        status = "BLOCKED_INSUFFICIENT_EPISODES"
    elif oracle.get("status") in {"PASS", "PASS_ORACLE_GAP"} or oracle.get("gate_f_status") in {"PASS", "PASS_ORACLE_GAP"}:
        status = "READY_FOR_PREDECLARED_GATE_F_REVIEW"
    else:
        status = str(oracle.get("status") or oracle.get("gate_f_status") or "BLOCKED_GATE_F_NOT_PASSED")

    role_counts = protocol.get("role_counts") or manifest.get("role_counts") or {}
    return {
        "dataset": "GlaS",
        "role": "first_clean_morphology_shift_extension",
        "status": status,
        "formal_policy_training_allowed": False,
        "official_test_preserved": True,
        "source_split_counts": manifest.get("split_counts", {}),
        "role_counts": role_counts,
        "episode_counts": protocol.get("episode_counts", {}),
        "candidate_smoke": {
            "status": candidate.get("status", "MISSING"),
            "episode_count": candidate.get("episode_count", 0),
            "formal_gate_c": candidate.get("formal_gate_c"),
        },
        "gate": {
            "name": "GlaS Gate F readiness",
            "status": oracle.get("status") or oracle.get("gate_f_status") or "MISSING",
            "episode_count": episode_count,
            "gate_min_episodes": gate_min,
            "dice_oracle_gap": oracle.get("dice_oracle_gap"),
            "gate_f_min_dice_gap": oracle.get("gate_f_min_dice_gap", 0.01),
        },
        "artifacts": {
            "manifest": _artifact(manifest_path, project_root),
            "protocol": _artifact(protocol_path, project_root),
            "candidate_pilot": _artifact(candidate_path, project_root),
            "oracle_gap_pilot": _artifact(oracle_path, project_root),
        },
        "allowed_next_actions": [
            "predeclare_glas_trainval_diagnostic_without_official_test",
            "rerun_glas_readiness_after_new_episode_manifest",
            "keep_glas_as_smoke_only_if_min_episodes_cannot_be_met",
        ],
        "prohibited_next_actions": [
            "promote_glas_smoke_to_gate_f",
            "use_testA_or_testB_for_action_or_threshold_design",
            "open_glas_policy_training_before_predeclared_gate_f_pass",
            "use_glas_to_reopen_monusac_policy",
        ],
        "supplemental_experiments_when_blocked": [
            {
                "trigger": "episode_count_below_gate_minimum",
                "experiment": "predeclared_trainval_diagnostic",
                "rule": "Use source-train data only; preserve official testA/testB; report whether enough query-level units exist before any policy work.",
                "stop_if": "No non-leaky protocol reaches the predeclared minimum query units.",
            }
        ],
    }


def _ddr_readiness(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "ddr_protocol_audit" / "ddr_protocol_audit.json"
    audit = _read_json_or_none(path) or {}
    status = (
        "READY_FOR_INFRASTRUCTURE_ONLY"
        if audit.get("status") == "PASS_PROTOCOL_DEFINED"
        else "BLOCKED_PROTOCOL_AUDIT_MISSING_OR_FAILED"
    )
    return {
        "dataset": "DDR",
        "role": "conditional_eye_lesion_domain_boundary",
        "status": status,
        "formal_policy_training_allowed": False,
        "stage1_mask_rule": (audit.get("stage1_union") or {}).get("mask_rule"),
        "stage1_allowed_use": (audit.get("stage1_union") or {}).get("allowed_use", "infrastructure_only"),
        "stage2_intents": (audit.get("stage2_four_intents") or {}).get("intents", []),
        "artifacts": {"protocol_audit": _artifact(path, project_root)},
        "allowed_next_actions": [
            "ddr_stage1_union_loader_smoke",
            "ddr_stage1_coordinate_and_mask_merge_smoke",
            "predeclare_ddr_stage2_four_intent_protocol",
        ],
        "prohibited_next_actions": [
            "formal_ddr_policy_training_without_protocol",
            "claim_union_mask_as_preferred_formal_task",
            "use_idrid_observed_test_behavior_for_ddr_thresholds",
        ],
        "supplemental_experiments_when_blocked": [
            {
                "trigger": "pathology_gate_f_failure_may_be_domain_shape_specific",
                "experiment": "stage1_infrastructure_smoke_only",
                "rule": "Exercise loaders, mask union, coordinate transforms, and candidate diagnostics without policy labels.",
                "stop_if": "Protocol audit or label pairing fails.",
            }
        ],
    }


def _idrid_readiness() -> dict[str, Any]:
    return {
        "dataset": "IDRID",
        "role": "recorded_non_tunable_boundary",
        "status": "READ_ONLY_NON_TUNABLE",
        "formal_policy_training_allowed": False,
        "allowed_next_actions": [
            "historical_negative_result_summary",
            "read_only_boundary_table_if_protocol_history_supports_it",
        ],
        "prohibited_next_actions": [
            "use_idrid_for_candidate_design",
            "use_idrid_for_action_bank_design",
            "use_idrid_for_policy_feature_selection",
            "use_idrid_for_threshold_or_budget_tuning",
        ],
        "supplemental_experiments_when_blocked": [
            {
                "trigger": "fresh_unobserved_subset_cannot_be_proven",
                "experiment": "none",
                "rule": "Keep IDRID as read-only historical or boundary evidence.",
                "stop_if": "Any proposed action would depend on observed IDRID test behavior.",
            }
        ],
    }


def _cpm17_readiness(project_root: Path) -> dict[str, Any]:
    workspace_root = project_root.parent
    cpm_root = workspace_root / "external" / "research_data" / "raw" / "CPM17"
    cpm15_root = cpm_root / "cpm15"
    image_like = []
    mat_like = []
    if cpm15_root.exists():
        image_like = [
            path
            for path in cpm15_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        ]
        mat_like = [path for path in cpm15_root.rglob("*.mat") if path.is_file()]
    status = "DEFERRED_MISSING_OR_INCOMPLETE"
    if len(image_like) >= 32 and len(mat_like) >= 32:
        status = "PENDING_MANIFEST_AND_LICENSE_AUDIT"
    return {
        "dataset": "CPM-17",
        "role": "deferred_external_nuclei_extension",
        "status": status,
        "formal_policy_training_allowed": False,
        "observed_local_subset": {
            "root": str(cpm15_root),
            "exists": cpm15_root.exists(),
            "image_like_count": len(image_like),
            "mat_label_count": len(mat_like),
        },
        "allowed_next_actions": [
            "write_missing_asset_report",
            "parser_development_on_cpm15_only_if_labeled_exploratory",
        ],
        "prohibited_next_actions": [
            "call_cpm15_formal_cpm17_evaluation",
            "download_or_repair_cpm17_without_user_approval",
            "include_cpm17_in_formal_gate_before_manifest_license_pass",
        ],
        "supplemental_experiments_when_blocked": [
            {
                "trigger": "complete_public_version_missing",
                "experiment": "missing_asset_report",
                "rule": "Record expected source, license, 32/32 split assumptions, and parser requirements.",
                "stop_if": "Network or download approval is absent.",
            }
        ],
    }


def build_new_domain_pilot_readiness(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    datasets = {
        "glas": _glas_readiness(project_root),
        "ddr": _ddr_readiness(project_root),
        "idrid": _idrid_readiness(),
        "cpm17": _cpm17_readiness(project_root),
    }
    blocked = [
        name
        for name, item in datasets.items()
        if str(item["status"]).startswith(("BLOCKED", "DEFERRED"))
    ]
    infrastructure_ready = [
        name for name, item in datasets.items() if item["status"] == "READY_FOR_INFRASTRUCTURE_ONLY"
    ]
    return {
        "schema_version": "adapromptset.new_domain_pilot_readiness.v1",
        "status": "READY_WITH_BLOCKED_FORMAL_DATASETS" if infrastructure_ready else "BLOCKED_NO_FORMAL_NEW_DOMAIN_READY",
        "policy": {
            "purpose": "Decide which local datasets can support Gate-F-safe extensions after the MoNuSAC policy pivot.",
            "does_not_reopen_monusac_policy": True,
            "does_not_train_policy": True,
        },
        "datasets": datasets,
        "blocked_datasets": blocked,
        "infrastructure_ready_datasets": infrastructure_ready,
        "global_allowed_next_actions": [
            "fixed_rule_oracle_analysis_assets",
            "predeclare_glas_trainval_diagnostic_without_official_test",
            "ddr_stage1_infrastructure_smoke",
            "cpm17_missing_asset_report",
        ],
        "global_prohibited_next_actions": [
            "monusac_policy_training",
            "posthoc_monusac_action_redesign",
            "test_time_action_search",
            "promote_insufficient_episode_smoke_to_gate_pass",
            "formal_new_domain_policy_training_without_predeclared_gate_f",
        ],
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# New-Domain Pilot Readiness",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This audit is read-only: it does not reopen MoNuSAC policy training and does not train a new policy.",
        "",
        "## Dataset Decisions",
        "",
        "| Dataset | Status | Formal policy training | Key next action |",
        "|---|---|---:|---|",
    ]
    for name in ["glas", "ddr", "idrid", "cpm17"]:
        item = summary["datasets"][name]
        allowed = item.get("allowed_next_actions", [])
        key_action = allowed[0] if allowed else "none"
        lines.append(
            f"| {item['dataset']} | `{item['status']}` | "
            f"{str(item['formal_policy_training_allowed']).lower()} | `{key_action}` |"
        )
    lines.extend(["", "## Gate Supplements", ""])
    for name in ["glas", "ddr", "idrid", "cpm17"]:
        item = summary["datasets"][name]
        lines.append(f"### {item['dataset']}")
        for experiment in item.get("supplemental_experiments_when_blocked", []):
            lines.append(f"- `{experiment['experiment']}` when `{experiment['trigger']}`: {experiment['rule']}")
        lines.append("")
    lines.extend(["## Global Prohibited Next Actions", ""])
    lines.extend(f"- `{action}`" for action in summary["global_prohibited_next_actions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_new_domain_pilot_readiness(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    summary = build_new_domain_pilot_readiness(project_root)
    outputs = {
        "json": project_root / "reports" / "new_domain_pilot_readiness.json",
        "markdown": project_root / "reports" / "new_domain_pilot_readiness.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Gate-F-safe readiness for local new-domain pilots.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_new_domain_pilot_readiness(args.project_root)


if __name__ == "__main__":
    main()
