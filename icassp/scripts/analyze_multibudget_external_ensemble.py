"""Exploratory cross-domain majority ensemble of five source-fold policies."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_risk_controlled_response_gate import bootstrap_delta, write_csv  # noqa: E402


def read(path):
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def main():
    root = Path("icassp/runs/multibudget_external_monusac_v1")
    folds = [{x["episode_id"]: x for x in read(root / f"fold_{f}/learned_predictions.csv")} for f in range(5)]
    two = {x["episode_id"]: x for x in read("runs/research_direction/monusac_prompt_policy/negative_gate_ref1_frozen_v2/learned_predictions.csv")}
    if any(set(model) != set(folds[0]) for model in folds[1:]) or set(two) != set(folds[0]): raise AssertionError("External episode mismatch")
    rows, distribution = [], Counter()
    for episode in sorted(folds[0]):
        vote_actions = [model[episode]["action"] for model in folds]
        # Deterministic majority; ties prefer the smaller prompt budget.
        chosen = max(set(vote_actions), key=lambda x: (vote_actions.count(x), -int(x[3:])))
        selected = next(model[episode] for model in folds if model[episode]["action"] == chosen)
        distribution[chosen] += 1
        rows.append({"episode_id": episode, "query_case_id": selected["query_case_id"], "ensemble_action": chosen,
            "ensemble_dice": float(selected["dice"]), "two_action_dice": float(two[episode]["dice"]),
            "votes": "|".join(vote_actions)})
    write_csv(root / "external_majority_ensemble.csv", rows)
    summary = {"protocol": "four_action_external_majority_ensemble_v1", "status": "COMPLETE", "scope": "exploratory external stress test; source-fold ensemble, not a new tuned target-domain model",
        "episodes": len(rows), "action_distribution": dict(sorted(distribution.items())),
        "ensemble_mean_dice": sum(x["ensemble_dice"] for x in rows) / len(rows), "two_action_reference_mean_dice": sum(x["two_action_dice"] for x in rows) / len(rows),
        "ensemble_vs_two_action": bootstrap_delta(rows, "ensemble_dice", "two_action_dice"),
        "decision_rule": "five frozen MoNuSeg policies vote; tie -> smaller budget; no MoNuSAC labels are used in the rule."}
    (root / "external_ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
