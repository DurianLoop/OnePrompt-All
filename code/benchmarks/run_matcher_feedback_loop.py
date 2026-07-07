from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from matcher_feedback_learning import classification_metrics, learn_scalar_threshold
from run_matcher_learning_casebook import load_case_rows, resolve_path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate iterative human feedback for Matcher proposal policy adaptation.")
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--feature", default="feedback_learning_score")
    parser.add_argument("--max-feedback-rounds", type=int, default=5)
    parser.add_argument("--feedback-per-round", type=int, default=1)
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_feedback_loop")
    return parser.parse_args()


def learn_policy(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    values = [float(row.get(feature, 0.0)) for row in rows]
    labels = [int(row["label"]) for row in rows]
    return {"feature": feature, **learn_scalar_threshold(values, labels).as_dict()}


def predict(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[int]:
    feature = str(policy["feature"])
    threshold = float(policy["threshold"])
    return [1 if float(row.get(feature, 0.0)) >= threshold else 0 for row in rows]


def select_uncertain_feedback(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    revealed_indices: set[int],
    feedback_per_round: int,
) -> list[int]:
    feature = str(policy["feature"])
    threshold = float(policy["threshold"])
    candidates = [
        (abs(float(row.get(feature, 0.0)) - threshold), idx)
        for idx, row in enumerate(rows)
        if idx not in revealed_indices
    ]
    candidates.sort(key=lambda item: item[0])
    return [idx for _, idx in candidates[: max(int(feedback_per_round), 0)]]


def macro_by_round(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds = sorted({int(row["round"]) for row in curves})
    summary: list[dict[str, Any]] = []
    for round_id in rounds:
        subset = [row for row in curves if int(row["round"]) == round_id]
        summary.append({
            "round": round_id,
            "mean_f1": round(float(np.mean([float(row["f1"]) for row in subset])), 6),
            "mean_precision": round(float(np.mean([float(row["precision"]) for row in subset])), 6),
            "mean_recall": round(float(np.mean([float(row["recall"]) for row in subset])), 6),
            "mean_accuracy": round(float(np.mean([float(row["accuracy"]) for row in subset])), 6),
            "mean_labeled_feedback": round(float(np.mean([float(row["labeled_feedback"]) for row in subset])), 3),
        })
    return summary


def main() -> None:
    args = parse_args()
    cases_json = resolve_path(args.cases_json)
    cases = json.loads(cases_json.read_text(encoding="utf-8")).get("cases", [])
    rows_by_case = {
        str(case["case_name"]): load_case_rows(case, cases_json.parent, use_pseudo_labels=False)
        for case in cases
    }
    rows_by_case = {name: rows for name, rows in rows_by_case.items() if rows}
    curves: list[dict[str, Any]] = []
    feedback_events: list[dict[str, Any]] = []

    for test_case, test_rows in rows_by_case.items():
        train_rows = [row for name, rows in rows_by_case.items() if name != test_case for row in rows]
        revealed_indices: set[int] = set()
        labels = [int(row["label"]) for row in test_rows]
        for round_id in range(int(args.max_feedback_rounds) + 1):
            feedback_rows = [test_rows[idx] for idx in sorted(revealed_indices)]
            policy = learn_policy(train_rows + feedback_rows, args.feature)
            preds = predict(test_rows, policy)
            metrics = classification_metrics(labels, preds)
            curves.append({
                "test_case": test_case,
                "round": round_id,
                "threshold": policy["threshold"],
                "labeled_feedback": len(revealed_indices),
                **metrics,
            })
            if round_id >= int(args.max_feedback_rounds):
                break
            selected = select_uncertain_feedback(test_rows, policy, revealed_indices, args.feedback_per_round)
            for idx in selected:
                revealed_indices.add(idx)
                row = test_rows[idx]
                feedback_events.append({
                    "test_case": test_case,
                    "round_added": round_id + 1,
                    "group_id": int(row["group_id"]),
                    "label": int(row["label"]),
                    "score": float(row.get(args.feature, 0.0)),
                })

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "feedback_loop_curves.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["test_case", "round", "threshold", "labeled_feedback", "n", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curves)
    with (out_dir / "feedback_events.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["test_case", "round_added", "group_id", "label", "score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feedback_events)

    round_summary = macro_by_round(curves)
    result = {
        "cases_json": str(cases_json.relative_to(WORKSPACE_ROOT)) if cases_json.is_relative_to(WORKSPACE_ROOT) else str(cases_json),
        "fold_count": len(rows_by_case),
        "feature": args.feature,
        "max_feedback_rounds": args.max_feedback_rounds,
        "feedback_per_round": args.feedback_per_round,
        "curves": curves,
        "round_summary": round_summary,
        "feedback_events": feedback_events,
    }
    (out_dir / "feedback_loop_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Matcher Iterative Feedback Loop",
        "",
        f"- Folds: `{result['fold_count']}`",
        f"- Feature: `{args.feature}`",
        "",
        "| Round | Mean F1 | Mean Precision | Mean Recall | Mean labeled feedback |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in round_summary:
        lines.append(
            f"| {row['round']} | {row['mean_f1']:.6f} | {row['mean_precision']:.6f} | "
            f"{row['mean_recall']:.6f} | {row['mean_labeled_feedback']:.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'feedback_loop_results.json'}")


if __name__ == "__main__":
    main()
