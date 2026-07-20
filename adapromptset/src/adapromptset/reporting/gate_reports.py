from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapromptset.data.audit import compute_file_sha256


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _source_hashes(project_root: Path, paths: list[Path]) -> list[dict[str, str]]:
    hashes: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        hashes.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": compute_file_sha256(path),
            }
        )
    return hashes


def _status_to_pass_fail(status: str) -> str:
    upper = status.upper()
    if upper == "PASS" or upper.startswith("PASS_"):
        return "PASS"
    if upper.startswith("FAIL") or upper.startswith("PIVOT"):
        return "FAIL"
    if upper.startswith("PENDING"):
        return "PENDING"
    if "PROHIBITED" in upper or upper.startswith("NOT_RUN"):
        return "NOT_RUN"
    return "UNKNOWN"


def _markdown_table(items: dict[str, Any]) -> list[str]:
    lines = ["| Field | Value |", "|---|---|"]
    for key, value in items.items():
        if isinstance(value, (dict, list)):
            display = "`" + json.dumps(value, sort_keys=True) + "`"
        else:
            display = f"`{value}`"
        lines.append(f"| {key} | {display} |")
    return lines


def _write_report(project_root: Path, report: dict[str, Any]) -> list[Path]:
    out_dir = project_root / "reports" / "gates"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gate_{str(report['gate']).lower()}_{report['name']}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    title = f"Gate {report['gate']}: {report['title']}"
    lines = [
        f"# {title}",
        "",
        f"Status: `{report['status']}`",
        f"Pass/fail: `{report['pass_fail']}`",
        "",
        "## Rule Applied",
        "",
        report["rule_applied"],
        "",
        "## Diagnostics",
        "",
    ]
    lines.extend(_markdown_table(report["diagnostics"]))
    lines.extend(["", "## Commands", ""])
    lines.extend(f"- `{command}`" for command in report["commands"])
    lines.extend(["", "## Input Hashes", ""])
    if report["input_hashes"]:
        lines.extend(
            f"- `{item['path']}`: `{item['sha256']}`" for item in report["input_hashes"]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Allowed Next Actions", ""])
    lines.extend(f"- `{item}`" for item in report["allowed_next_actions"])
    lines.extend(["", "## Prohibited Next Actions", ""])
    lines.extend(f"- `{item}`" for item in report["prohibited_next_actions"])
    lines.extend(["", "## Deviations", ""])
    if report["deviations"]:
        lines.extend(f"- {item}" for item in report["deviations"])
    else:
        lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, md_path]


def _gate_a(project_root: Path) -> dict[str, Any]:
    paths = sorted((project_root / "reports" / "dataset_audit").glob("*.json"))
    audits = [_read_json(path) for path in paths]
    passed = all(bool(item.get("gate", {}).get("passed")) for item in audits) if audits else False
    issues = {
        item.get("dataset", "unknown"): item.get("gate", {}).get("issues", [])
        for item in audits
    }
    return {
        "gate": "A",
        "name": "data_integrity",
        "title": "Data Integrity",
        "status": "PASS" if passed else "FAIL",
        "pass_fail": "PASS" if passed else "FAIL",
        "input_hashes": _source_hashes(project_root, paths + [project_root / "reports" / "gate_status.md"]),
        "commands": ["scripts/audit_data.py --config configs/experiments/e0_data_audit.yaml"],
        "diagnostics": {
            "dataset_count": len(audits),
            "datasets": [item.get("dataset", "unknown") for item in audits],
            "issues": issues,
        },
        "rule_applied": "All formal cases require verified image-mask pairs, hashes, and zero strict train/test identity overlap.",
        "allowed_next_actions": ["gate_b_protocol_exposure_audit"] if passed else ["repair_manifest"],
        "prohibited_next_actions": ["formal_experiments_with_failed_manifest"] if not passed else [],
        "deviations": [],
    }


def _gate_b(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "gate_b_status.md"
    text = _read_text(path)
    failed = "| FAIL |" in text
    passed = bool(text.strip()) and not failed
    return {
        "gate": "B",
        "name": "protocol_exposure",
        "title": "Protocol and Exposure Integrity",
        "status": "PASS" if passed else "FAIL",
        "pass_fail": "PASS" if passed else "FAIL",
        "input_hashes": _source_hashes(project_root, [path]),
        "commands": ["scripts/build_episodes.py"],
        "diagnostics": {"status_table_present": bool(text.strip()), "contains_fail": failed},
        "rule_applied": "Reference-query episodes must be deterministic, versioned, and free of test-support leakage.",
        "allowed_next_actions": ["candidate_quality_gate"] if passed else ["repair_episode_protocol"],
        "prohibited_next_actions": ["baseline_or_policy_on_leaky_episodes"] if not passed else [],
        "deviations": [],
    }


def _gate_c(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "candidate_pilot" / "candidate_pilot_summary.json"
    glas_path = project_root / "reports" / "candidate_pilot_glas" / "candidate_pilot_summary.json"
    payload = _read_json(path) if path.is_file() else {}
    status = str(payload.get("formal_gate_c", payload.get("status", "MISSING")))
    diagnostics = {
        "mean_precision": payload.get("mean_precision"),
        "mean_random_precision": payload.get("mean_random_precision"),
        "mean_component_coverage": payload.get("mean_component_coverage"),
    }
    deviations = []
    if glas_path.is_file():
        glas = _read_json(glas_path)
        diagnostics["glas_status"] = glas.get("formal_gate_c", glas.get("status"))
        deviations.append("GlaS evidence is a strict one-episode extension smoke, not a formal GlaS Gate F claim.")
    return {
        "gate": "C",
        "name": "candidate_quality",
        "title": "Candidate Quality",
        "status": status,
        "pass_fail": _status_to_pass_fail(status),
        "input_hashes": _source_hashes(project_root, [path, glas_path, project_root / "reports" / "gate_c_status.md"]),
        "commands": ["scripts/run_candidate_pilot.py"],
        "diagnostics": diagnostics,
        "rule_applied": "Positive candidates must beat a support-matched random baseline and candidate validation must report coordinate/blank-background issues.",
        "allowed_next_actions": ["segmenter_execution_gate"] if status.startswith("PASS") else ["candidate_rescue_diagnostics"],
        "prohibited_next_actions": ["policy_training_from_failed_candidates"] if not status.startswith("PASS") else [],
        "deviations": deviations,
    }


def _gate_d(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "sam_smoke" / "sam_smoke_summary.json"
    payload = _read_json(path) if path.is_file() else {}
    status = str(payload.get("formal_gate_d", payload.get("status", "MISSING")))
    return {
        "gate": "D",
        "name": "segmenter_execution",
        "title": "Segmenter Execution",
        "status": status,
        "pass_fail": _status_to_pass_fail(status),
        "input_hashes": _source_hashes(project_root, [path, project_root / "reports" / "gate_d_status.md"]),
        "commands": ["scripts/run_sam_smoke.py"],
        "diagnostics": {
            "component_dice": payload.get("component_dice"),
            "semantic_union_dice": payload.get("semantic_union_dice"),
            "call_count": payload.get("call_count"),
            "prompt_group_count": payload.get("prompt_group_count"),
        },
        "rule_applied": "Privileged prompt diagnostics must verify SAM-B wrapper correctness, labels, call accounting, and deterministic repeated prompts.",
        "allowed_next_actions": ["action_bank_validity_gate"] if status.startswith("PASS") else ["repair_segmenter_wrapper"],
        "prohibited_next_actions": ["action_utility_generation_with_failed_segmenter"] if not status.startswith("PASS") else [],
        "deviations": ["Semantic-union Dice is diagnostic for one-point prompts and is not used as the wrapper pass criterion."],
    }


def _gate_e(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "action_bank" / "action_bank_summary.json"
    payload = _read_json(path) if path.is_file() else {}
    valid_count = int(payload.get("valid_action_count", 0) or 0)
    status = "PASS_RESCUE_SUBSET" if valid_count else "FAIL"
    return {
        "gate": "E",
        "name": "action_bank_validity",
        "title": "Action-Bank Validity",
        "status": status,
        "pass_fail": _status_to_pass_fail(status),
        "input_hashes": _source_hashes(project_root, [path, project_root / "reports" / "gate_e_status.md"]),
        "commands": ["scripts/build_action_bank.py --bank-version v2"],
        "diagnostics": {
            "bank_name": payload.get("name"),
            "bank_hash": payload.get("bank_hash"),
            "action_count": payload.get("action_count"),
            "valid_action_count": payload.get("valid_action_count"),
        },
        "rule_applied": "Actions must be predeclared, versioned, hashable, cost-accounted, and built from the shared candidate pool.",
        "allowed_next_actions": ["oracle_gap_gate"] if valid_count else ["repair_action_bank"],
        "prohibited_next_actions": ["post_hoc_action_bank_redesign_after_gate_f"],
        "deviations": ["Current bank is a rescue subset; full KMeans/SynPo/GF-SAM/safe-anchor matrix remains incomplete."],
    }


def _gate_f(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "oracle_gap_pilot_v2" / "oracle_gap_pilot_summary.json"
    decision_path = project_root / "reports" / "decision_checkpoint" / "phase5_decision.json"
    payload = _read_json(path) if path.is_file() else {}
    decision = _read_json(decision_path) if decision_path.is_file() else {}
    status = str(payload.get("gate_f_status", payload.get("status", "MISSING")))
    return {
        "gate": "F",
        "name": "oracle_gap",
        "title": "Oracle Gap",
        "status": status,
        "pass_fail": _status_to_pass_fail(status),
        "input_hashes": _source_hashes(project_root, [path, decision_path, project_root / "reports" / "gate_f_status.md"]),
        "commands": [
            "scripts/run_oracle_gap_pilot.py --action-bank configs/actions/prompt_set_bank_v2.json",
            "scripts/write_decision_checkpoint.py",
        ],
        "diagnostics": {
            "episode_count": payload.get("episode_count"),
            "record_count": payload.get("record_count"),
            "mean_oracle_dice": payload.get("mean_oracle_dice"),
            "mean_best_fixed_dice": payload.get("mean_best_fixed_dice"),
            "dice_oracle_gap": payload.get("dice_oracle_gap"),
            "aji_available": payload.get("aji_available"),
            "mean_oracle_aji": payload.get("mean_oracle_aji"),
            "mean_best_fixed_aji": payload.get("mean_best_fixed_aji"),
            "aji_oracle_gap": payload.get("aji_oracle_gap"),
            "gate_f_min_dice_gap": payload.get("gate_f_min_dice_gap"),
            "top_oracle_action_fraction": payload.get("top_oracle_action_fraction"),
            "best_fixed_action": payload.get("best_fixed_action_id", payload.get("best_fixed_action")),
            "phase5_status": decision.get("phase5_status"),
        },
        "rule_applied": "Policy learning may continue only if the predeclared oracle gap and action-heterogeneity criteria pass after at most one rescue bank.",
        "allowed_next_actions": decision.get("allowed_next_actions", ["fixed_rule_action_oracle_analysis", "predeclared_new_domain_pilot"]),
        "prohibited_next_actions": decision.get(
            "prohibited_next_actions",
            ["monusac_policy_training", "post_hoc_monusac_action_bank_redesign"],
        ),
        "deviations": ["Gate F failed after the one allowed MoNuSAC rescue bank; MoNuSAC policy training remains prohibited."],
    }


def _post_f_gate(
    *,
    gate: str,
    name: str,
    title: str,
    status: str,
    rule: str,
    project_root: Path,
) -> dict[str, Any]:
    decision_path = project_root / "reports" / "decision_checkpoint" / "phase5_decision.json"
    decision = _read_json(decision_path) if decision_path.is_file() else {}
    return {
        "gate": gate,
        "name": name,
        "title": title,
        "status": status,
        "pass_fail": "NOT_RUN",
        "input_hashes": _source_hashes(project_root, [decision_path]),
        "commands": [],
        "diagnostics": {"phase5_status": decision.get("phase5_status", "UNKNOWN")},
        "rule_applied": rule,
        "allowed_next_actions": decision.get("allowed_next_actions", ["fixed_rule_action_oracle_analysis"]),
        "prohibited_next_actions": decision.get(
            "prohibited_next_actions",
            ["monusac_policy_training", "post_hoc_monusac_action_bank_redesign"],
        ),
        "deviations": ["Not executed because Gate F selected the current oracle-analysis pivot."],
    }


def build_standard_gate_reports(project_root: Path) -> list[Path]:
    reports = [
        _gate_a(project_root),
        _gate_b(project_root),
        _gate_c(project_root),
        _gate_d(project_root),
        _gate_e(project_root),
        _gate_f(project_root),
        _post_f_gate(
            gate="G",
            name="utility_label_reliability",
            title="Utility-Label Reliability",
            status="NOT_RUN_AFTER_GATE_F_PIVOT",
            rule="Utility-label reliability is evaluated before policy training only when Gate F permits policy learning.",
            project_root=project_root,
        ),
        _post_f_gate(
            gate="H",
            name="policy_learning",
            title="Policy Learning",
            status="PROHIBITED_BY_GATE_F_PIVOT",
            rule="Policy training is prohibited when Gate F fails after the one allowed action-bank rescue.",
            project_root=project_root,
        ),
        _post_f_gate(
            gate="I",
            name="frozen_formal_test",
            title="Frozen Formal Test",
            status="NOT_RUN_NO_FROZEN_POLICY_PROTOCOL",
            rule="Frozen formal testing requires a locked method or approved pivot protocol; no MoNuSAC policy checkpoint exists under the current evidence.",
            project_root=project_root,
        ),
    ]
    written: list[Path] = []
    for report in reports:
        written.extend(_write_report(project_root, report))
    return written
