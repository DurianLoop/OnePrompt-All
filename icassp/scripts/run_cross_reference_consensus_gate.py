"""Nested-CV test of a cross-reference consensus response gate.

An optional background-prompt action is accepted only when its model-predicted
gain is corroborated by a quorum of the five support references for the same
query.  All hyperparameters are selected from inner validation episodes before
outer-test labels are opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_risk_controlled_response_gate import (  # noqa: E402
    BETAS, THRESHOLDS, bootstrap_delta, dice_by_episode, read_csv, wilson_upper, write_csv,
)


def evaluate(rows, outcomes, anchor, beta, threshold, quorum):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_case_id"]].append(row)
    selected = []
    for query_id, group in grouped.items():
        if len(group) != 5:
            raise AssertionError(f"Expected five references for {query_id}, got {len(group)}")
        scores = [float(row["policy_predicted_gain"]) - beta * float(row["policy_predicted_gain_std"]) for row in group]
        eligible = sum(score > threshold for score in scores) >= quorum
        for row, score in zip(group, scores):
            episode, alternative = row["episode_id"], row["policy_best_action"]
            if alternative == anchor:
                raise ValueError(f"Alternative equals anchor for {episode}")
            switch = eligible and score > threshold
            anchor_dice, alt_dice = outcomes[episode][anchor], outcomes[episode][alternative]
            selected.append({
                "episode_id": episode, "query_case_id": query_id,
                "anchor_action": anchor, "alternative_action": alternative,
                "predicted_gain_mean": float(row["policy_predicted_gain"]),
                "predicted_gain_std": float(row["policy_predicted_gain_std"]),
                "beta": beta, "threshold": threshold, "quorum": quorum,
                "risk_adjusted_score": score, "query_eligible": int(eligible), "switched": int(switch),
                "anchor_dice": anchor_dice, "alternative_dice": alt_dice,
                "selected_dice": alt_dice if switch else anchor_dice,
                "actual_switch_gain": alt_dice - anchor_dice,
                "harmful_switch": int(switch and alt_dice < anchor_dice),
            })
    switched, harmful = sum(x["switched"] for x in selected), sum(x["harmful_switch"] for x in selected)
    return selected, {
        "mean_dice": sum(x["selected_dice"] for x in selected) / len(selected), "episodes": len(selected),
        "switched": switched, "episode_coverage": switched / len(selected), "harmful_switches": harmful,
        "harmful_switch_rate": harmful / switched if switched else 0.0,
        "harmful_switch_wilson_upper_95": wilson_upper(harmful, switched),
        "queries_eligible": sum(any(x["query_eligible"] for x in selected if x["query_case_id"] == q) for q in grouped),
    }


def choose(rows, outcomes, anchor):
    grid = []
    for beta in BETAS:
        for threshold in THRESHOLDS:
            for quorum in range(1, 6):
                _, metrics = evaluate(rows, outcomes, anchor, beta, threshold, quorum)
                feasible = metrics["switched"] >= 3 and metrics["harmful_switch_wilson_upper_95"] <= 0.50
                grid.append({"beta": beta, "threshold": threshold, "quorum": quorum, "feasible": feasible, **metrics})
    feasible = [x for x in grid if x["feasible"]]
    if not feasible:
        return {"beta": None, "threshold": None, "quorum": None, "feasible": False,
                "selection_source": "inner_validation_anchor_fallback"}, grid
    best = sorted(feasible, key=lambda x: (-x["mean_dice"], x["harmful_switch_wilson_upper_95"], -x["quorum"], -x["beta"], -x["threshold"]))[0]
    return {**best, "selection_source": "inner_validation_cross_reference_consensus"}, grid


def action_summary(rows):
    switched, harmful = sum(int(x["switched"]) for x in rows), sum(int(x["harmful_switch"]) for x in rows)
    return {"episodes": len(rows), "switched": switched, "episode_coverage": switched / len(rows),
            "harmful_switches": harmful, "harmful_switch_rate": harmful / switched if switched else 0.0,
            "harmful_switch_wilson_upper_95": wilson_upper(harmful, switched)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("runs/research_direction/nested_cv5_negative_budget"))
    parser.add_argument("--output-root", type=Path, default=Path("icassp/runs/cross_reference_consensus_gate_v1"))
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Preserving existing experiment: {args.output_root}")
    choices = {}
    # Commit every validation-derived choice before test labels are read.
    for fold in range(5):
        root = args.source_root / f"fold_{fold}"
        model = json.loads((root / "policy/model.json").read_text(encoding="utf-8"))
        anchor = model["anchor_selected_on_validation"]
        choice, grid = choose(read_csv(root / "policy/val_learned_predictions.csv"), dice_by_episode(root / "negative_val/per_case.csv"), anchor)
        output = args.output_root / f"fold_{fold}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "selection.json").write_text(json.dumps({"fold": fold, "anchor_action": anchor, "candidate_grid": grid,
            "chosen": choice, "test_labels_read": False, "protocol": "cross_reference_consensus_gate_v1"}, indent=2), encoding="utf-8")
        choices[fold] = (anchor, choice, float(model["selection"]["gain_threshold"]))
    risk_all, mean_all = [], []
    for fold in range(5):
        root = args.source_root / f"fold_{fold}"
        rows, outcomes = read_csv(root / "policy/test_learned_predictions.csv"), dice_by_episode(root / "negative_test/per_case.csv")
        anchor, choice, frozen_threshold = choices[fold]
        if choice["beta"] is None:
            risk, risk_metrics = evaluate(rows, outcomes, anchor, 0.0, float("inf"), 1)
        else:
            risk, risk_metrics = evaluate(rows, outcomes, anchor, choice["beta"], choice["threshold"], choice["quorum"])
        mean, mean_metrics = evaluate(rows, outcomes, anchor, 0.0, frozen_threshold, 1)
        for row in risk: row.update(fold=fold, policy="cross_reference_consensus")
        for row in mean: row.update(fold=fold, policy="mean_only_frozen")
        write_csv(args.output_root / f"fold_{fold}/outer_test_consensus.csv", risk)
        write_csv(args.output_root / f"fold_{fold}/outer_test_mean_only.csv", mean)
        (args.output_root / f"fold_{fold}/outer_test_metrics.json").write_text(json.dumps({"consensus": risk_metrics, "mean_only": mean_metrics}, indent=2), encoding="utf-8")
        risk_all.extend(risk); mean_all.extend(mean)
    by_q_r, by_q_m = defaultdict(list), defaultdict(list)
    for row in risk_all: by_q_r[row["query_case_id"]].append(row)
    for row in mean_all: by_q_m[row["query_case_id"]].append(row)
    if set(by_q_r) != set(by_q_m) or any(len(x) != 5 for x in by_q_r.values()): raise AssertionError("Outer queries must have five supports")
    query_rows = []
    for query in sorted(by_q_r):
        risk, mean = by_q_r[query], by_q_m[query]
        query_rows.append({"query_case_id": query, "references": 5,
            "anchor_dice": sum(x["anchor_dice"] for x in risk) / 5,
            "mean_only_dice": sum(x["selected_dice"] for x in mean) / 5,
            "consensus_dice": sum(x["selected_dice"] for x in risk) / 5,
            "mean_only_any_switch": int(any(x["switched"] for x in mean)),
            "consensus_any_switch": int(any(x["switched"] for x in risk))})
    write_csv(args.output_root / "query_level_outer_test.csv", query_rows)
    summary = {"protocol": "cross_reference_consensus_gate_v1", "status": "COMPLETE",
        "fold_choices": {str(f): choices[f][1] for f in range(5)}, "outer_test_query_count": len(query_rows),
        "outer_test_reference_episode_count": len(risk_all), "consensus_actions": action_summary(risk_all),
        "mean_only_actions": action_summary(mean_all), "consensus_vs_anchor": bootstrap_delta(query_rows, "consensus_dice", "anchor_dice"),
        "consensus_vs_mean_only": bootstrap_delta(query_rows, "consensus_dice", "mean_only_dice"),
        "mean_only_vs_anchor": bootstrap_delta(query_rows, "mean_only_dice", "anchor_dice"),
        "consensus_query_coverage": sum(x["consensus_any_switch"] for x in query_rows) / len(query_rows),
        "mean_only_query_coverage": sum(x["mean_only_any_switch"] for x in query_rows) / len(query_rows),
        "selection_commitment": "All fold selection.json files were written before any outer-test label file was opened."}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
