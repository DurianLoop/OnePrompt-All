from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _is_valid(record: dict[str, Any]) -> bool:
    return record.get("valid") is True or str(record.get("valid", "")).lower() in {"true", "1"}


def _float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    if value in {"", None}:
        return default
    return float(value)


def _int(record: dict[str, Any], key: str, default: int = 0) -> int:
    value = record.get(key, default)
    if value in {"", None}:
        return default
    return int(float(value))


def _valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _is_valid(record)]


def _by_episode(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _valid_records(records):
        grouped[str(record["episode_id"])].append(record)
    return grouped


def _by_action(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _valid_records(records):
        grouped[str(record["action_id"])].append(record)
    return grouped


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = sum(x * x for x in dx)
    denom_y = sum(y * y for y in dy)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / ((denom_x * denom_y) ** 0.5)


def _best_fixed_action_id(records: list[dict[str, Any]]) -> str:
    action_means = {
        action_id: _mean([_float(row, "dice") for row in items])
        for action_id, items in _by_action(records).items()
    }
    return max(action_means, key=action_means.get)


def _candidate_by_episode(
    candidate_rows: list[dict[str, Any]],
    *,
    positive_limit: int = 5,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    fallback: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        episode_id = str(row["episode_id"])
        fallback[episode_id] = row
        if _int(row, "positive_limit") == positive_limit:
            selected[episode_id] = row
    for episode_id, row in fallback.items():
        selected.setdefault(episode_id, row)
    return selected


def episode_representation_rows(
    utility_records: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    positive_limit: int = 5,
    top_margin_tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    best_fixed_action = _best_fixed_action_id(utility_records)
    candidates = _candidate_by_episode(candidate_rows, positive_limit=positive_limit)
    rows: list[dict[str, Any]] = []
    for episode_id, items in sorted(_by_episode(utility_records).items()):
        ranked = sorted(items, key=lambda row: _float(row, "dice"), reverse=True)
        oracle = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]
        fixed = next(row for row in items if str(row["action_id"]) == best_fixed_action)
        candidate = candidates.get(episode_id, {})
        top2_margin = _float(oracle, "dice") - _float(second, "dice")
        regret = _float(oracle, "dice") - _float(fixed, "dice")
        rows.append(
            {
                "episode_id": episode_id,
                "oracle_action_id": str(oracle["action_id"]),
                "oracle_family": str(oracle["action_family"]),
                "best_fixed_action_id": best_fixed_action,
                "oracle_dice": _float(oracle, "dice"),
                "best_fixed_dice": _float(fixed, "dice"),
                "regret_to_oracle": regret,
                "top2_dice_margin": top2_margin,
                "near_tie_top2": 1 if top2_margin <= top_margin_tolerance else 0,
                "oracle_differs_from_best_fixed": 1 if str(oracle["action_id"]) != best_fixed_action else 0,
                "oracle_sam_calls": _int(oracle, "sam_calls"),
                "best_fixed_sam_calls": _int(fixed, "sam_calls"),
                f"candidate_point_precision_at_{positive_limit}": _float(candidate, "point_precision"),
                f"candidate_component_coverage_at_{positive_limit}": _float(candidate, "component_coverage"),
                f"candidate_random_precision_at_{positive_limit}": _float(candidate, "random_precision"),
                "candidate_negative_blank_fraction": _float(candidate, "negative_blank_fraction"),
            }
        )
    return rows


def summarize_representation_pivot(
    utility_records: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    positive_limit: int = 5,
    top_margin_tolerance: float = 0.005,
) -> dict[str, Any]:
    rows = episode_representation_rows(
        utility_records,
        candidate_rows,
        positive_limit=positive_limit,
        top_margin_tolerance=top_margin_tolerance,
    )
    precision_key = f"candidate_point_precision_at_{positive_limit}"
    coverage_key = f"candidate_component_coverage_at_{positive_limit}"
    random_key = f"candidate_random_precision_at_{positive_limit}"
    regrets = [float(row["regret_to_oracle"]) for row in rows]
    margins = [float(row["top2_dice_margin"]) for row in rows]
    precisions = [float(row[precision_key]) for row in rows]
    coverages = [float(row[coverage_key]) for row in rows]
    best_fixed = rows[0]["best_fixed_action_id"] if rows else ""
    mean_regret = _mean(regrets)
    near_tie_fraction = _mean([float(row["near_tie_top2"]) for row in rows])
    oracle_differs_fraction = _mean([float(row["oracle_differs_from_best_fixed"]) for row in rows])
    insufficient = mean_regret < 0.01 or near_tie_fraction >= 0.5
    return {
        "status": "PIVOT_REPRESENTATION_ANALYSIS",
        "episode_count": len(rows),
        "action_count": len(_by_action(utility_records)),
        "positive_limit": positive_limit,
        "top_margin_tolerance": top_margin_tolerance,
        "best_fixed_action_id": best_fixed,
        "mean_regret_to_oracle": mean_regret,
        "mean_top2_dice_margin": _mean(margins),
        "near_tie_episode_fraction": near_tie_fraction,
        "oracle_differs_from_best_fixed_fraction": oracle_differs_fraction,
        "mean_candidate_point_precision": _mean(precisions),
        "mean_candidate_component_coverage": _mean(coverages),
        "mean_candidate_random_precision": _mean([float(row[random_key]) for row in rows]),
        "mean_candidate_negative_blank_fraction": _mean([float(row["candidate_negative_blank_fraction"]) for row in rows]),
        "pearson_precision_vs_regret": _pearson(precisions, regrets),
        "pearson_coverage_vs_regret": _pearson(coverages, regrets),
        "representation_claim": "oracle_analysis_only_insufficient_separation"
        if insufficient
        else "oracle_analysis_candidate_features_require_larger_validation",
        "allowed_next_actions": [
            "fixed_rule_action_oracle_analysis",
            "representation_analysis_without_policy_training",
            "predeclared_new_domain_pilot",
        ],
        "prohibited_next_actions": [
            "monusac_policy_training",
            "posthoc_monusac_action_redesign",
            "test_time_action_search",
        ],
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Representation Pivot Analysis",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Episodes | {summary['episode_count']} |",
        f"| Actions | {summary['action_count']} |",
        f"| Best fixed action | `{summary['best_fixed_action_id']}` |",
        f"| Mean regret to oracle | {summary['mean_regret_to_oracle']:.6f} |",
        f"| Mean top-2 Dice margin | {summary['mean_top2_dice_margin']:.6f} |",
        f"| Near-tie episode fraction | {summary['near_tie_episode_fraction']:.3f} |",
        f"| Oracle differs from best fixed fraction | {summary['oracle_differs_from_best_fixed_fraction']:.3f} |",
        f"| Mean candidate point precision | {summary['mean_candidate_point_precision']:.6f} |",
        f"| Mean candidate component coverage | {summary['mean_candidate_component_coverage']:.6f} |",
        "",
        f"Representation claim: `{summary['representation_claim']}`",
        "",
        "Interpretation:",
        "",
        "- This report is computed from existing MoNuSAC validation pilot artifacts only.",
        "- It is a Gate-F-pivot analysis artifact, not a policy-training result.",
        "- Small oracle margins and near-ties are treated as evidence against starting MoNuSAC policy training.",
        "",
        "Allowed Next Actions:",
    ]
    lines.extend(f"- `{item}`" for item in summary["allowed_next_actions"])
    lines.extend(["", "Prohibited Next Actions:"])
    lines.extend(f"- `{item}`" for item in summary["prohibited_next_actions"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_representation_pivot_analysis(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    utility_path = project_root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv"
    candidate_path = project_root / "reports" / "candidate_pilot" / "candidate_pilot_metrics.csv"
    out_dir = project_root / "reports" / "representation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    utility_records = _read_csv(utility_path)
    candidate_rows = _read_csv(candidate_path)
    rows = episode_representation_rows(utility_records, candidate_rows)
    summary = summarize_representation_pivot(utility_records, candidate_rows)
    (out_dir / "representation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "episode_representation.csv", rows)
    _write_markdown(out_dir / "representation_report.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze observable representation limits after the Gate F pivot.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_representation_pivot_analysis(args.project_root)


if __name__ == "__main__":
    main()
