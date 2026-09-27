"""Fixed-action, adaptive-action, and oracle ceiling ablation for four budgets."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_risk_controlled_response_gate import bootstrap_delta, write_csv  # noqa: E402


def read(path):
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def main():
    source = Path("runs/research_direction/nested_cv5_negative_budget")
    policy = Path("icassp/runs/multibudget_negative_policy_v1/combined_outer_test_episodes.csv")
    selected = {x["episode_id"]: x for x in read(policy)}
    episodes = []
    for fold in range(5):
        grouped = defaultdict(dict)
        for row in read(source / f"fold_{fold}/negative_test/per_case.csv"):
            grouped[row["episode_id"]][f"neg{int(float(row['negative_count']))}"] = float(row["dice"])
        for episode, values in grouped.items():
            if set(values) != {"neg0", "neg5", "neg10", "neg20"}: raise AssertionError(f"Incomplete action outcomes: {episode}")
            picked = selected[episode]
            episodes.append({"episode_id": episode, "query_case_id": picked["query_case_id"], "fold": fold,
                **values, "adaptive_dice": float(picked["four_action_dice"]), "oracle_dice": max(values.values())})
    by_q = defaultdict(list)
    for row in episodes: by_q[row["query_case_id"]].append(row)
    if any(len(rows) != 5 for rows in by_q.values()): raise AssertionError("Expected five references per query")
    queries = []
    for query, rows in sorted(by_q.items()):
        queries.append({"query_case_id": query, "references": 5,
            **{action: sum(r[action] for r in rows) / 5 for action in ("neg0", "neg5", "neg10", "neg20")},
            "adaptive_dice": sum(r["adaptive_dice"] for r in rows) / 5,
            "oracle_dice": sum(r["oracle_dice"] for r in rows) / 5})
    out = Path("icassp/runs/multibudget_negative_policy_v1")
    write_csv(out / "fixed_action_query_ablation.csv", queries)
    fixed_means = {action: sum(x[action] for x in queries) / len(queries) for action in ("neg0", "neg5", "neg10", "neg20")}
    best_fixed = max(fixed_means, key=fixed_means.get)
    summary = {"protocol": "four_action_fixed_budget_ablation_v1", "status": "COMPLETE", "query_count": len(queries),
        "fixed_action_mean_dice": fixed_means, "best_fixed_action": best_fixed,
        "adaptive_mean_dice": sum(x["adaptive_dice"] for x in queries) / len(queries),
        "oracle_episode_action_mean_dice": sum(x["oracle_dice"] for x in queries) / len(queries),
        "adaptive_vs_best_fixed": bootstrap_delta(queries, "adaptive_dice", best_fixed),
        "adaptive_vs_neg0": bootstrap_delta(queries, "adaptive_dice", "neg0"),
        "adaptive_vs_neg5": bootstrap_delta(queries, "adaptive_dice", "neg5"),
        "adaptive_vs_neg10": bootstrap_delta(queries, "adaptive_dice", "neg10"),
        "adaptive_vs_neg20": bootstrap_delta(queries, "adaptive_dice", "neg20"),
        "oracle_minus_adaptive": bootstrap_delta(queries, "oracle_dice", "adaptive_dice"),
        "oracle_note": "Oracle chooses a budget after observing each episode's ground truth and is reported only as an unattainable ceiling."}
    (out / "fixed_action_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
