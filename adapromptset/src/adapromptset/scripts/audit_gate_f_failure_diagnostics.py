from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_valid(row: dict[str, Any]) -> bool:
    return str(row.get("valid", "")).lower() in {"true", "1", "yes"}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in {None, ""}:
        return default
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in {None, ""}:
        return default
    return int(float(value))


def _coverage_diagnostic(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        return {
            "status": "MISSING_CANDIDATE_METRICS",
            "mean_component_coverage_at_max_pool": None,
            "mean_point_precision_at_max_pool": None,
            "max_positive_limit": None,
        }
    max_limit = max(_int(row, "positive_limit") for row in candidate_rows)
    rows = [row for row in candidate_rows if _int(row, "positive_limit") == max_limit]
    mean_coverage = mean(_float(row, "component_coverage") for row in rows)
    mean_precision = mean(_float(row, "point_precision") for row in rows)
    mean_random = mean(_float(row, "random_precision") for row in rows)
    status = "LOW_COMPONENT_COVERAGE" if mean_coverage < 0.60 else "COVERAGE_OK"
    return {
        "status": status,
        "max_positive_limit": max_limit,
        "episode_count": len({str(row["episode_id"]) for row in rows}),
        "mean_component_coverage_at_max_pool": mean_coverage,
        "mean_point_precision_at_max_pool": mean_precision,
        "mean_random_precision_at_max_pool": mean_random,
        "precision_delta_over_random": mean_precision - mean_random,
        "component_coverage_warning_threshold": 0.60,
    }


def _valid_by_episode(utility_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in utility_rows:
        if _is_valid(row):
            grouped[str(row["episode_id"])].append(row)
    return grouped


def _proposal_diagnostic(utility_rows: list[dict[str, Any]], *, tie_tolerance: float = 0.005) -> dict[str, Any]:
    margins: list[float] = []
    ranges: list[float] = []
    for rows in _valid_by_episode(utility_rows).values():
        dices = sorted((_float(row, "dice") for row in rows), reverse=True)
        if not dices:
            continue
        ranges.append(dices[0] - dices[-1])
        margins.append(dices[0] - dices[1] if len(dices) > 1 else 0.0)
    near_tie_fraction = sum(1 for value in margins if value < tie_tolerance) / len(margins) if margins else 0.0
    mean_margin = mean(margins) if margins else 0.0
    return {
        "status": "ACTIONS_NEAR_TIED" if near_tie_fraction >= 0.50 else "ACTIONS_SEPARATED",
        "tie_tolerance": tie_tolerance,
        "episode_count": len(margins),
        "mean_top2_dice_margin": mean_margin,
        "near_tie_episode_fraction": near_tie_fraction,
        "mean_within_episode_dice_range": mean(ranges) if ranges else 0.0,
    }


def _sam_sensitivity_diagnostic(utility_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in utility_rows if _is_valid(row)]
    by_calls: dict[int, list[float]] = defaultdict(list)
    for row in valid:
        by_calls[_int(row, "sam_calls")].append(_float(row, "dice"))
    if not by_calls:
        return {
            "status": "MISSING_UTILITY_RECORDS",
            "best_one_call_dice": None,
            "best_high_call_dice": None,
            "best_high_call_minus_best_one_call": None,
        }
    one_call = by_calls.get(1, [])
    high_call = [score for calls, scores in by_calls.items() if calls > 1 for score in scores]
    best_one = max(one_call) if one_call else max(score for scores in by_calls.values() for score in scores)
    best_high = max(high_call) if high_call else best_one
    delta = best_high - best_one
    return {
        "status": "NO_HIGHER_CALL_GAIN" if delta <= 0.005 else "HIGHER_CALL_GAIN_OBSERVED",
        "call_counts_observed": sorted(by_calls),
        "best_one_call_dice": best_one,
        "best_high_call_dice": best_high,
        "best_high_call_minus_best_one_call": delta,
        "high_call_gain_threshold": 0.005,
    }


def analyze_gate_f_failure(
    candidate_rows: list[dict[str, Any]],
    utility_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage = _coverage_diagnostic(candidate_rows)
    proposal = _proposal_diagnostic(utility_rows)
    sam = _sam_sensitivity_diagnostic(utility_rows)
    flags: list[str] = []
    if coverage["status"] == "LOW_COMPONENT_COVERAGE":
        flags.append("candidate_pool_limited_coverage")
    if proposal["status"] == "ACTIONS_NEAR_TIED":
        flags.append("actions_are_near_tied")
    if sam["status"] == "NO_HIGHER_CALL_GAIN":
        flags.append("no_clear_sam_call_grouping_gain")
    return {
        "status": "PASS_DIAGNOSTIC_ONLY",
        "scope": "MoNuSAC Gate F post-failure diagnostics from existing validation artifacts",
        "policy_training_allowed": False,
        "post_hoc_monusac_action_redesign_allowed": False,
        "coverage_diagnostic": coverage,
        "proposal_diagnostic": proposal,
        "sam_sensitivity_diagnostic": sam,
        "interpretation_flags": flags,
        "allowed_next_actions": [
            "fixed_rule_action_oracle_analysis",
            "representation_analysis",
            "predeclared_new_domain_pilot",
        ],
        "prohibited_next_actions": [
            "monusac_policy_training",
            "post_hoc_monusac_action_bank_redesign",
            "test_time_action_search",
        ],
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    coverage = summary["coverage_diagnostic"]
    proposal = summary["proposal_diagnostic"]
    sam = summary["sam_sensitivity_diagnostic"]
    lines = [
        "# Gate F Failure Diagnostics",
        "",
        "Policy training remains prohibited for the current MoNuSAC evidence.",
        "",
        "| Diagnostic | Status | Key value |",
        "|---|---|---:|",
        f"| Candidate coverage | `{coverage['status']}` | {coverage['mean_component_coverage_at_max_pool']} |",
        f"| Proposal separation | `{proposal['status']}` | {proposal['mean_top2_dice_margin']:.6f} top-2 Dice margin |",
        f"| SAM call/group sensitivity | `{sam['status']}` | {sam['best_high_call_minus_best_one_call']:.6f} high-call delta |",
        "",
        "Interpretation flags:",
        "",
    ]
    lines.extend(f"- `{flag}`" for flag in summary["interpretation_flags"])
    lines.extend(
        [
            "",
            "Allowed next actions:",
            "",
            "- fixed-rule/action-oracle analysis",
            "- representation analysis",
            "- separately predeclared new-domain pilot",
            "",
            "Prohibited next actions:",
            "",
            "- MoNuSAC policy training",
            "- post-hoc MoNuSAC action-bank redesign",
            "- test-time action search",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gate_f_failure_diagnostics(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    candidate_rows = _read_csv(project_root / "reports" / "candidate_pilot" / "candidate_pilot_metrics.csv")
    utility_rows = _read_csv(project_root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv")
    summary = analyze_gate_f_failure(candidate_rows, utility_rows)
    out_dir = project_root / "reports" / "gate_f_failure_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_f_failure_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "gate_f_failure_diagnostics.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Gate F failure diagnostics from existing pilot artifacts.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_gate_f_failure_diagnostics(args.project_root)


if __name__ == "__main__":
    main()
