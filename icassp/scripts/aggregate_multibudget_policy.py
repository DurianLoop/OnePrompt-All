"""Aggregate the five nested folds of the four-action budget experiment."""

from __future__ import annotations

import csv
import json
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_risk_controlled_response_gate import bootstrap_delta, write_csv  # noqa: E402


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("icassp/runs/multibudget_negative_policy_v1"))
    parser.add_argument("--frozen",type=Path,default=Path("runs/research_direction/nested_cv5_negative_budget"))
    args=parser.parse_args()
    root, frozen = args.root, args.frozen
    all_rows = []
    for fold in range(5):
        learned = {r["episode_id"]: r for r in read(root / f"fold_{fold}/test_learned_predictions.csv")}
        anchor = {r["episode_id"]: r for r in read(root / f"fold_{fold}/test_fixed_predictions.csv")}
        two = {r["episode_id"]: r for r in read(frozen / f"fold_{fold}/policy/test_learned_predictions.csv")}
        if set(learned) != set(anchor) or set(learned) != set(two):
            raise AssertionError(f"Episode mismatch in fold {fold}")
        for episode in sorted(learned):
            m, a, t = learned[episode], anchor[episode], two[episode]
            all_rows.append({"fold": fold, "episode_id": episode, "query_case_id": m["query_case_id"],
                "anchor_action": a["action"], "anchor_dice": float(a["dice"]),
                "two_action": t["action"], "two_action_dice": float(t["dice"]),
                "four_action": m["action"], "four_action_dice": float(m["dice"]),
                "four_action_switch": int(m["action"] != a["action"]),
                "four_action_uses_gt5": int(m["action"] in {"neg10", "neg20"})})
    write_csv(root / "combined_outer_test_episodes.csv", all_rows)
    grouped = defaultdict(list)
    for row in all_rows: grouped[row["query_case_id"]].append(row)
    if any(len(rows) != 5 for rows in grouped.values()):
        raise AssertionError("Every outer query must have five support-reference episodes")
    queries = []
    for query, rows in sorted(grouped.items()):
        queries.append({"query_case_id": query, "references": 5,
            "anchor_dice": sum(x["anchor_dice"] for x in rows) / 5,
            "two_action_dice": sum(x["two_action_dice"] for x in rows) / 5,
            "four_action_dice": sum(x["four_action_dice"] for x in rows) / 5,
            "four_action_any_gt5": int(any(x["four_action_uses_gt5"] for x in rows)),
            "four_action_any_switch": int(any(x["four_action_switch"] for x in rows))})
    write_csv(root / "query_level_outer_test.csv", queries)
    counts = Counter(str(x["four_action"]) for x in all_rows)
    summary = {"protocol": "four_action_negative_budget_policy_v1", "status": "COMPLETE",
        "outer_test_query_count": len(queries), "outer_test_reference_episode_count": len(all_rows),
        "four_action_distribution": dict(sorted(counts.items())),
        "four_action_episode_gt5_rate": sum(x["four_action_uses_gt5"] for x in all_rows) / len(all_rows),
        "four_action_query_gt5_rate": sum(x["four_action_any_gt5"] for x in queries) / len(queries),
        "four_action_vs_anchor": bootstrap_delta(queries, "four_action_dice", "anchor_dice"),
        "four_action_vs_two_action": bootstrap_delta(queries, "four_action_dice", "two_action_dice"),
        "two_action_vs_anchor": bootstrap_delta(queries, "two_action_dice", "anchor_dice"),
        "selection_note": "Each fold selected tree complexity and gain threshold on its inner validation split; outer test labels were not used for selection."}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
