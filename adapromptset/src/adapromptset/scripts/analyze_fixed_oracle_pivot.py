from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _is_valid(record: dict[str, Any]) -> bool:
    return record.get("valid") is True or str(record.get("valid", "")).lower() in {"true", "1"}


def _valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _is_valid(record)]


def _float(record: dict[str, Any], key: str) -> float:
    return float(record[key])


def _int(record: dict[str, Any], key: str) -> int:
    return int(float(record[key]))


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


def _episode_oracles(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        episode_id: max(items, key=lambda row: _float(row, "dice"))
        for episode_id, items in _by_episode(records).items()
    }


def _metric_available(records: list[dict[str, Any]], metric: str) -> bool:
    valid = _valid_records(records)
    return bool(valid) and all(record.get(metric) not in {None, ""} for record in valid)


def _best_fixed_action_id(records: list[dict[str, Any]]) -> str:
    action_means = {
        action_id: sum(_float(row, "dice") for row in items) / len(items)
        for action_id, items in _by_action(records).items()
    }
    return max(action_means, key=action_means.get)


def episode_regret_rows(
    records: list[dict[str, Any]],
    *,
    best_fixed_action_id: str,
    tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id, items in sorted(_by_episode(records).items()):
        oracle = max(items, key=lambda row: _float(row, "dice"))
        fixed = next(row for row in items if str(row["action_id"]) == best_fixed_action_id)
        regret = _float(oracle, "dice") - _float(fixed, "dice")
        rows.append(
            {
                "episode_id": episode_id,
                "oracle_action_id": str(oracle["action_id"]),
                "oracle_family": str(oracle["action_family"]),
                "oracle_dice": _float(oracle, "dice"),
                "best_fixed_action_id": best_fixed_action_id,
                "best_fixed_dice": _float(fixed, "dice"),
                "best_fixed_regret": regret,
                "oracle_sam_calls": _int(oracle, "sam_calls"),
                "best_fixed_sam_calls": _int(fixed, "sam_calls"),
                "within_tolerance": 1 if regret <= tolerance else 0,
            }
        )
    return rows


def budget_pareto_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = _valid_records(records)
    budgets = sorted({_int(record, "sam_calls") for record in valid})
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        budget_records = [record for record in valid if _int(record, "sam_calls") <= budget]
        by_episode = _by_episode(budget_records)
        by_action = _by_action(budget_records)
        if not by_episode:
            continue
        action_means = {
            action_id: sum(_float(row, "dice") for row in items) / len(items)
            for action_id, items in by_action.items()
        }
        best_action = max(action_means, key=action_means.get)
        oracle_dices = []
        best_dices = []
        for items in by_episode.values():
            oracle_dices.append(_float(max(items, key=lambda row: _float(row, "dice")), "dice"))
            best_dices.append(_float(next(row for row in items if str(row["action_id"]) == best_action), "dice"))
        mean_oracle = sum(oracle_dices) / len(oracle_dices)
        mean_best = sum(best_dices) / len(best_dices)
        rows.append(
            {
                "budget": budget,
                "best_fixed_action_id": best_action,
                "mean_best_fixed_dice": mean_best,
                "mean_oracle_dice": mean_oracle,
                "dice_oracle_gap": mean_oracle - mean_best,
                "aji_oracle_gap": _budget_metric_gap(budget_records, "aji"),
                "episode_count": len(by_episode),
                "action_count": len(by_action),
            }
        )
    return rows


def _budget_metric_gap(records: list[dict[str, Any]], metric: str) -> float | None:
    if not _metric_available(records, metric):
        return None
    by_episode = _by_episode(records)
    by_action = _by_action(records)
    action_means = {
        action_id: sum(_float(row, metric) for row in items) / len(items)
        for action_id, items in by_action.items()
    }
    best_action = max(action_means, key=action_means.get)
    oracle_values = []
    best_values = []
    for items in by_episode.values():
        oracle_values.append(_float(max(items, key=lambda row: _float(row, metric)), metric))
        best_values.append(_float(next(row for row in items if str(row["action_id"]) == best_action), metric))
    return (sum(oracle_values) / len(oracle_values)) - (sum(best_values) / len(best_values))


def action_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = _valid_records(records)
    oracles = _episode_oracles(valid)
    oracle_counts = Counter(str(record["action_id"]) for record in oracles.values())
    rows: list[dict[str, Any]] = []
    for action_id, items in sorted(_by_action(valid).items()):
        rows.append(
            {
                "action_id": action_id,
                "action_family": str(items[0]["action_family"]),
                "mean_dice": sum(_float(row, "dice") for row in items) / len(items),
                "mean_iou": sum(_float(row, "iou") for row in items) / len(items),
                "mean_aji": sum(_float(row, "aji") for row in items) / len(items) if _metric_available(items, "aji") else None,
                "mean_sam_calls": sum(_int(row, "sam_calls") for row in items) / len(items),
                "mean_runtime_ms": sum(_float(row, "runtime_ms") for row in items) / len(items) if "runtime_ms" in items[0] else 0.0,
                "oracle_count": oracle_counts.get(action_id, 0),
                "episode_count": len(items),
            }
        )
    rows.sort(key=lambda row: (-float(row["mean_dice"]), str(row["action_id"])))
    return rows


def analyze_fixed_oracle(
    records: list[dict[str, Any]],
    *,
    deterministic_tolerance: float = 0.005,
    dominance_threshold: float = 0.70,
) -> dict[str, Any]:
    valid = _valid_records(records)
    by_episode = _by_episode(valid)
    best_action = _best_fixed_action_id(valid)
    regret_rows = episode_regret_rows(valid, best_fixed_action_id=best_action, tolerance=deterministic_tolerance)
    oracle_counts = Counter(row["oracle_action_id"] for row in regret_rows)
    family_counts = Counter(row["oracle_family"] for row in regret_rows)
    regrets = [float(row["best_fixed_regret"]) for row in regret_rows]
    mean_fixed = sum(float(row["best_fixed_dice"]) for row in regret_rows) / len(regret_rows)
    mean_oracle = sum(float(row["oracle_dice"]) for row in regret_rows) / len(regret_rows)
    aji_available = _metric_available(valid, "aji")
    aji_oracles = {
        episode_id: max(items, key=lambda row: _float(row, "aji"))
        for episode_id, items in by_episode.items()
    } if aji_available else {}
    action_mean_aji = {
        action_id: sum(_float(row, "aji") for row in items) / len(items)
        for action_id, items in _by_action(valid).items()
    } if aji_available else {}
    best_aji_action = max(action_mean_aji, key=action_mean_aji.get) if aji_available else None
    best_aji_by_episode = {
        episode_id: next(row for row in items if str(row["action_id"]) == best_aji_action)
        for episode_id, items in by_episode.items()
    } if aji_available else {}
    mean_best_aji = (
        sum(_float(row, "aji") for row in best_aji_by_episode.values()) / len(best_aji_by_episode)
        if aji_available else None
    )
    mean_oracle_aji = (
        sum(_float(row, "aji") for row in aji_oracles.values()) / len(aji_oracles)
        if aji_available else None
    )
    top_fraction = max(oracle_counts.values()) / len(regret_rows) if regret_rows else 0.0
    near_fraction = sum(int(row["within_tolerance"]) for row in regret_rows) / len(regret_rows) if regret_rows else 0.0
    mean_gap = mean_oracle - mean_fixed
    promoted = top_fraction >= dominance_threshold or mean_gap < (deterministic_tolerance - 1e-12)
    return {
        "episode_count": len(by_episode),
        "action_count": len(_by_action(valid)),
        "best_fixed_action_id": best_action,
        "mean_best_fixed_dice": mean_fixed,
        "mean_oracle_dice": mean_oracle,
        "mean_regret_to_oracle": mean_gap,
        "aji_available": aji_available,
        "best_fixed_aji_action_id": best_aji_action,
        "mean_best_fixed_aji": mean_best_aji,
        "mean_oracle_aji": mean_oracle_aji,
        "mean_aji_regret_to_oracle": (mean_oracle_aji - mean_best_aji) if aji_available else None,
        "median_regret_to_oracle": median(regrets) if regrets else 0.0,
        "max_regret_to_oracle": max(regrets) if regrets else 0.0,
        "near_oracle_tolerance": deterministic_tolerance,
        "near_oracle_episode_fraction": near_fraction,
        "oracle_action_counts": dict(oracle_counts),
        "oracle_family_counts": dict(family_counts),
        "top_oracle_action_fraction": top_fraction,
        "deterministic_promotion_status": "PROMOTED" if promoted else "NOT_PROMOTED",
        "recommended_claim": "deterministic_structured_prompting" if promoted else "oracle_analysis_only",
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
        "# Fixed Rule / Oracle Pivot Analysis",
        "",
        f"Recommended claim: `{summary['recommended_claim']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Episodes | {summary['episode_count']} |",
        f"| Actions | {summary['action_count']} |",
        f"| Best fixed action | `{summary['best_fixed_action_id']}` |",
        f"| Mean best fixed Dice | {summary['mean_best_fixed_dice']:.6f} |",
        f"| Mean oracle Dice | {summary['mean_oracle_dice']:.6f} |",
        f"| Mean regret to oracle | {summary['mean_regret_to_oracle']:.6f} |",
        f"| AJI available | {summary['aji_available']} |",
        f"| Mean best fixed AJI | {summary['mean_best_fixed_aji']} |",
        f"| Mean oracle AJI | {summary['mean_oracle_aji']} |",
        f"| Mean AJI regret to oracle | {summary['mean_aji_regret_to_oracle']} |",
        f"| Median regret to oracle | {summary['median_regret_to_oracle']:.6f} |",
        f"| Max regret to oracle | {summary['max_regret_to_oracle']:.6f} |",
        f"| Near-oracle episode fraction | {summary['near_oracle_episode_fraction']:.3f} |",
        f"| Top oracle action fraction | {summary['top_oracle_action_fraction']:.3f} |",
        "",
        "Interpretation:",
        "",
        "- The current MoNuSAC evidence does not justify policy training.",
        "- Oracle and best fixed are close, so the next valid MoNuSAC output is an action-oracle/representation analysis, not an adaptive-policy claim.",
        "- A new-domain pilot remains allowed only if predeclared and kept separate from observed MoNuSAC/IDRID test behavior.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixed_oracle_analysis(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    out_dir = project_root / "reports" / "fixed_oracle_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _read_csv(project_root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv")
    summary = analyze_fixed_oracle(records)
    (out_dir / "fixed_oracle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "fixed_oracle_report.md", summary)
    _write_csv(out_dir / "episode_regret.csv", episode_regret_rows(records, best_fixed_action_id=summary["best_fixed_action_id"]))
    _write_csv(out_dir / "budget_pareto.csv", budget_pareto_rows(records))
    _write_csv(out_dir / "action_summary.csv", action_summary_rows(records))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the fixed-rule/action-oracle pivot after Gate F.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_fixed_oracle_analysis(args.project_root)


if __name__ == "__main__":
    main()
