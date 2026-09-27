"""Leakage-safe evaluation of a risk-controlled optional negative-prompt gate.

The existing nested-CV run trains the reward model and exposes its tree-wise
uncertainty.  This script makes a *new policy decision* using only inner
validation labels, persists that choice, and only then reads outer-test labels.
It never retrains a model or changes the frozen prompt-generation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
THRESHOLDS = (-0.02, 0.0, 0.005, 0.01, 0.02, 0.05)
Z_ONE_SIDED_95 = 1.6448536269514722


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def action_name(value: str) -> str:
    return f"neg{int(float(value))}"


def dice_by_episode(path: Path) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for row in read_csv(path):
        table[row["episode_id"]][action_name(row["negative_count"])] = float(row["dice"])
    incomplete = [episode for episode, actions in table.items() if not {"neg0", "neg5"}.issubset(actions)]
    if incomplete:
        raise ValueError(f"Missing neg0/neg5 outcomes in {path}: {incomplete[:3]}")
    return dict(table)


def wilson_upper(harmful: int, switched: int) -> float:
    if switched == 0:
        return 1.0
    p = harmful / switched
    z2 = Z_ONE_SIDED_95**2
    centre = (p + z2 / (2 * switched)) / (1 + z2 / switched)
    radius = Z_ONE_SIDED_95 * math.sqrt((p * (1 - p) + z2 / (4 * switched)) / switched) / (1 + z2 / switched)
    return min(1.0, centre + radius)


def evaluate(rows: list[dict[str, str]], outcomes: dict[str, dict[str, float]], anchor: str, beta: float, threshold: float) -> tuple[list[dict[str, object]], dict[str, float]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        episode = row["episode_id"]
        alternative = row["policy_best_action"]
        if alternative == anchor:
            raise ValueError(f"Policy best action unexpectedly equals anchor for {episode}")
        if alternative not in outcomes[episode]:
            raise ValueError(f"No outcome for {alternative} in {episode}")
        mu, sigma = float(row["policy_predicted_gain"]), float(row["policy_predicted_gain_std"])
        score = mu - beta * sigma
        switch = score > threshold
        anchor_dice, alternative_dice = outcomes[episode][anchor], outcomes[episode][alternative]
        selected_dice = alternative_dice if switch else anchor_dice
        selected.append({
            "episode_id": episode,
            "query_case_id": row["query_case_id"],
            "anchor_action": anchor,
            "alternative_action": alternative,
            "predicted_gain_mean": mu,
            "predicted_gain_std": sigma,
            "beta": beta,
            "threshold": threshold,
            "risk_adjusted_score": score,
            "switched": int(switch),
            "anchor_dice": anchor_dice,
            "alternative_dice": alternative_dice,
            "selected_dice": selected_dice,
            "actual_switch_gain": alternative_dice - anchor_dice,
            "harmful_switch": int(switch and alternative_dice < anchor_dice),
        })
    switched = sum(int(x["switched"]) for x in selected)
    harmful = sum(int(x["harmful_switch"]) for x in selected)
    return selected, {
        "mean_dice": sum(float(x["selected_dice"]) for x in selected) / len(selected),
        "episodes": len(selected),
        "switched": switched,
        "episode_coverage": switched / len(selected),
        "harmful_switches": harmful,
        "harmful_switch_rate": harmful / switched if switched else 0.0,
        "harmful_switch_wilson_upper_95": wilson_upper(harmful, switched),
    }


def choose_fold(rows: list[dict[str, str]], outcomes: dict[str, dict[str, float]], anchor: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for beta in BETAS:
        for threshold in THRESHOLDS:
            _, metrics = evaluate(rows, outcomes, anchor, beta, threshold)
            feasible = metrics["switched"] >= 3 and metrics["harmful_switch_wilson_upper_95"] <= 0.50
            candidates.append({"beta": beta, "threshold": threshold, "feasible": feasible, **metrics})
    feasible = [x for x in candidates if x["feasible"]]
    if feasible:
        # Higher validation Dice, then safer bound, then more conservative beta/tau.
        chosen = sorted(feasible, key=lambda x: (-x["mean_dice"], x["harmful_switch_wilson_upper_95"], -x["beta"], -x["threshold"]))[0]
        chosen = {**chosen, "selection_source": "inner_validation_risk_constrained_grid"}
    else:
        chosen = {
            "beta": None, "threshold": None, "feasible": False, "selection_source": "inner_validation_anchor_fallback",
            "mean_dice": sum(v[anchor] for v in outcomes.values()) / len(outcomes), "episodes": len(outcomes),
            "switched": 0, "episode_coverage": 0.0, "harmful_switches": 0,
            "harmful_switch_rate": 0.0, "harmful_switch_wilson_upper_95": 1.0,
        }
    return chosen, candidates


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def bootstrap_delta(query_rows: list[dict[str, object]], field: str, reference: str, seed: int = 20260715, samples: int = 10000) -> dict[str, float]:
    deltas = [float(row[field]) - float(row[reference]) for row in query_rows]
    rng = random.Random(seed)
    draws = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(samples)]
    return {"mean_delta": sum(deltas) / len(deltas), "ci95_low": percentile(draws, 0.025), "ci95_high": percentile(draws, 0.975), "bootstrap_samples": samples, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("runs/research_direction/nested_cv5_negative_budget"))
    parser.add_argument("--output-root", type=Path, default=Path("icassp/runs/risk_controlled_response_gate_v1"))
    args = parser.parse_args()
    source_root, output_root = args.source_root, args.output_root
    if output_root.exists():
        raise FileExistsError(f"Output already exists; preserving prior experiment: {output_root}")

    # Phase 1: all policy choices are committed using validation data only.
    choices: dict[int, dict[str, object]] = {}
    for fold in range(5):
        fold_root = source_root / f"fold_{fold}"
        model = json.loads((fold_root / "policy/model.json").read_text(encoding="utf-8"))
        anchor = model["anchor_selected_on_validation"]
        val_rows = read_csv(fold_root / "policy/val_learned_predictions.csv")
        val_outcomes = dice_by_episode(fold_root / "negative_val/per_case.csv")
        chosen, grid = choose_fold(val_rows, val_outcomes, anchor)
        payload = {"fold": fold, "anchor_action": anchor, "candidate_grid": grid, "chosen": chosen,
                   "test_labels_read": False, "protocol": "risk_controlled_response_gate_v1"}
        fold_output = output_root / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=False)
        (fold_output / "selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        choices[fold] = {"anchor": anchor, "chosen": chosen, "model_threshold": float(model["selection"]["gain_threshold"])}

    # Phase 2: outer labels are now read exactly once, after selections are persisted.
    all_test_rows: list[dict[str, object]] = []
    all_mean_rows: list[dict[str, object]] = []
    for fold in range(5):
        fold_root = source_root / f"fold_{fold}"
        test_rows = read_csv(fold_root / "policy/test_learned_predictions.csv")
        outcomes = dice_by_episode(fold_root / "negative_test/per_case.csv")
        anchor, chosen = choices[fold]["anchor"], choices[fold]["chosen"]
        if chosen["beta"] is None:
            risk_rows, risk_metrics = evaluate(test_rows, outcomes, anchor, 0.0, float("inf"))
        else:
            risk_rows, risk_metrics = evaluate(test_rows, outcomes, anchor, float(chosen["beta"]), float(chosen["threshold"]))
        mean_rows, mean_metrics = evaluate(test_rows, outcomes, anchor, 0.0, float(choices[fold]["model_threshold"]))
        saved = {row["episode_id"]: row["selected_action"] for row in test_rows}
        observed = {str(row["episode_id"]): (str(row["alternative_action"]) if int(row["switched"]) else str(row["anchor_action"])) for row in mean_rows}
        if saved != observed:
            raise AssertionError(f"Mean-only reconstruction mismatches frozen policy in fold {fold}")
        for row in risk_rows:
            row["fold"] = fold
            row["policy"] = "risk_controlled"
        for row in mean_rows:
            row["fold"] = fold
            row["policy"] = "mean_only_frozen"
        write_csv(output_root / f"fold_{fold}/outer_test_risk_controlled.csv", risk_rows)
        write_csv(output_root / f"fold_{fold}/outer_test_mean_only.csv", mean_rows)
        (output_root / f"fold_{fold}/outer_test_metrics.json").write_text(json.dumps({"risk_controlled": risk_metrics, "mean_only": mean_metrics}, indent=2), encoding="utf-8")
        all_test_rows.extend(risk_rows)
        all_mean_rows.extend(mean_rows)

    risk_by_query: dict[str, list[dict[str, object]]] = defaultdict(list)
    mean_by_query: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_test_rows: risk_by_query[str(row["query_case_id"])].append(row)
    for row in all_mean_rows: mean_by_query[str(row["query_case_id"])].append(row)
    if set(risk_by_query) != set(mean_by_query) or any(len(v) != 5 for v in risk_by_query.values()):
        raise AssertionError("Expected exactly five reference episodes for every outer query")
    query_rows: list[dict[str, object]] = []
    for query_id in sorted(risk_by_query):
        risk, mean = risk_by_query[query_id], mean_by_query[query_id]
        anchor_dice = sum(float(x["anchor_dice"]) for x in risk) / 5
        query_rows.append({
            "query_case_id": query_id, "references": 5,
            "anchor_dice": anchor_dice,
            "mean_only_dice": sum(float(x["selected_dice"]) for x in mean) / 5,
            "risk_controlled_dice": sum(float(x["selected_dice"]) for x in risk) / 5,
            "mean_only_any_switch": int(any(int(x["switched"]) for x in mean)),
            "risk_controlled_any_switch": int(any(int(x["switched"]) for x in risk)),
        })
    write_csv(output_root / "query_level_outer_test.csv", query_rows)
    def action_summary(rows: list[dict[str, object]]) -> dict[str, float]:
        switched = sum(int(row["switched"]) for row in rows)
        harmful = sum(int(row["harmful_switch"]) for row in rows)
        return {"episodes": len(rows), "switched": switched, "episode_coverage": switched / len(rows),
                "harmful_switches": harmful, "harmful_switch_rate": harmful / switched if switched else 0.0,
                "harmful_switch_wilson_upper_95": wilson_upper(harmful, switched)}
    summary = {
        "protocol": "risk_controlled_response_gate_v1", "status": "COMPLETE",
        "fold_choices": {str(fold): choices[fold]["chosen"] for fold in range(5)},
        "outer_test_query_count": len(query_rows), "outer_test_reference_episode_count": len(all_test_rows),
        "risk_controlled_actions": action_summary(all_test_rows), "mean_only_actions": action_summary(all_mean_rows),
        "risk_controlled_vs_anchor": bootstrap_delta(query_rows, "risk_controlled_dice", "anchor_dice"),
        "risk_controlled_vs_mean_only": bootstrap_delta(query_rows, "risk_controlled_dice", "mean_only_dice"),
        "mean_only_vs_anchor": bootstrap_delta(query_rows, "mean_only_dice", "anchor_dice"),
        "risk_controlled_query_coverage": sum(int(x["risk_controlled_any_switch"]) for x in query_rows) / len(query_rows),
        "mean_only_query_coverage": sum(int(x["mean_only_any_switch"]) for x in query_rows) / len(query_rows),
        "selection_commitment": "All fold selection.json files were written before any outer-test label file was opened.",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
