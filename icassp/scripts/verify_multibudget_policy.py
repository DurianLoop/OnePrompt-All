"""Integrity checks for the completed four-action nested-CV experiment."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path("icassp/runs/multibudget_negative_policy_v1")


def rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def main():
    report = {"protocol": "four_action_negative_budget_policy_v1", "checks": []}
    combined = rows(ROOT / "combined_outer_test_episodes.csv")
    queries = rows(ROOT / "query_level_outer_test.csv")
    actions = Counter(x["four_action"] for x in combined)
    def check(name, condition, detail):
        report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition: raise AssertionError(f"{name}: {detail}")
    check("episode_count", len(combined) == 185, f"found {len(combined)}")
    check("query_count", len(queries) == 37, f"found {len(queries)}")
    check("five_references_per_query", all(int(x["references"]) == 5 for x in queries), "all query rows declare five references")
    check("all_four_actions_exposed", set(actions) == {"neg0", "neg5", "neg10", "neg20"}, dict(actions))
    check("expanded_actions_used", actions["neg10"] + actions["neg20"] == 26, dict(actions))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    ablation = json.loads((ROOT / "fixed_action_ablation_summary.json").read_text(encoding="utf-8"))
    check("four_beats_two_ci", summary["four_action_vs_two_action"]["ci95_low"] > 0, summary["four_action_vs_two_action"])
    check("four_beats_best_fixed_ci", ablation["adaptive_vs_best_fixed"]["ci95_low"] > 0, ablation["adaptive_vs_best_fixed"])
    check("oracle_is_ceiling", ablation["oracle_episode_action_mean_dice"] > ablation["adaptive_mean_dice"], "oracle diagnostic ceiling exceeds deployable policy")
    report["status"] = "PASS"
    (ROOT / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
