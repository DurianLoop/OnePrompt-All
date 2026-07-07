from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from matcher_feedback_learning import (
    classification_metrics,
    learn_scalar_threshold,
    predict_decision_stump,
    predict_logistic,
    rows_to_matrix,
    standardize_train_test,
    train_decision_stump,
    train_logistic_regression,
)
from run_matcher_learning_casebook import load_case_rows, resolve_path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-case-out validation for Matcher learning policies.")
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--feature-keys", default="feedback_learning_score,instance_similarity,sam_score,cluster_coverage,prompt_coverage,mask_area_pct,bbox_area_pct")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_learning_loco")
    return parser.parse_args()


def evaluate_threshold(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], feature: str) -> tuple[dict[str, Any], list[int]]:
    train_values = [float(row.get(feature, 0.0)) for row in train_rows]
    train_labels = [int(row["label"]) for row in train_rows]
    policy = learn_scalar_threshold(train_values, train_labels).as_dict()
    policy = {"feature": feature, **policy}
    preds = [1 if float(row.get(feature, 0.0)) >= float(policy["threshold"]) else 0 for row in test_rows]
    return policy, preds


def macro_average(values: list[dict[str, Any]], key: str) -> float:
    return round(float(np.mean([float(row[key]) for row in values])), 6) if values else 0.0


def main() -> None:
    args = parse_args()
    cases_json = resolve_path(args.cases_json)
    cases_payload = json.loads(cases_json.read_text(encoding="utf-8"))
    cases = cases_payload.get("cases", [])
    if len(cases) < 2:
        raise ValueError("LOCO needs at least two cases")
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        rows = load_case_rows(case, cases_json.parent, use_pseudo_labels=False)
        if rows:
            rows_by_case[str(case["case_name"])] = rows
    feature_keys = [key.strip() for key in args.feature_keys.split(",") if key.strip()]
    folds: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for test_case, test_rows in rows_by_case.items():
        train_rows = [row for case_name, rows in rows_by_case.items() if case_name != test_case for row in rows]
        if not train_rows:
            continue
        labels = [int(row["label"]) for row in test_rows]

        fold_models: list[tuple[str, dict[str, Any], list[int]]] = []
        feedback_policy, feedback_preds = evaluate_threshold(train_rows, test_rows, "feedback_learning_score")
        instance_policy, instance_preds = evaluate_threshold(train_rows, test_rows, "instance_similarity")
        fold_models.append(("feedback_threshold", feedback_policy, feedback_preds))
        fold_models.append(("instance_threshold", instance_policy, instance_preds))

        x_train, y_train = rows_to_matrix(train_rows, feature_keys)
        x_test, _ = rows_to_matrix(test_rows, feature_keys)
        x_train_s, x_test_s, mean, std = standardize_train_test(x_train, x_test)
        logistic_model = train_logistic_regression(x_train_s, y_train)
        _, logistic_preds_np = predict_logistic(logistic_model, x_test_s)
        logistic_policy = {
            "feature_keys": feature_keys,
            "weights": [round(float(v), 6) for v in logistic_model["weights"]],
            "bias": round(float(logistic_model["bias"]), 6),
            "mean": [round(float(v), 6) for v in mean],
            "std": [round(float(v), 6) for v in std],
        }
        fold_models.append(("logistic_regression", logistic_policy, logistic_preds_np.astype(int).tolist()))

        stump_policy = train_decision_stump(train_rows, feature_keys)
        stump_preds = predict_decision_stump(stump_policy, test_rows)
        fold_models.append(("decision_stump", stump_policy, stump_preds))

        model_metrics: dict[str, Any] = {}
        for model_name, policy, preds in fold_models:
            metrics = classification_metrics(labels, preds)
            model_metrics[model_name] = {"policy": policy, "metrics": metrics}
            for row, pred in zip(test_rows, preds):
                prediction_rows.append({
                    "test_case": test_case,
                    "model": model_name,
                    "group_id": int(row["group_id"]),
                    "label": int(row["label"]),
                    "prediction": int(pred),
                    "feedback_learning_score": float(row.get("feedback_learning_score", 0.0)),
                    "instance_similarity": float(row.get("instance_similarity", 0.0)),
                    "sam_score": float(row.get("sam_score", 0.0)),
                    "cluster_coverage": float(row.get("cluster_coverage", 0.0)),
                })
        folds.append({
            "test_case": test_case,
            "train_cases": [name for name in rows_by_case if name != test_case],
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "models": model_metrics,
        })

    macro_f1_by_model: dict[str, float] = {}
    for model in ["feedback_threshold", "instance_threshold", "logistic_regression", "decision_stump"]:
        macro_f1_by_model[model] = macro_average([fold["models"][model]["metrics"] for fold in folds], "f1")
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "loco_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["test_case", "model", "group_id", "label", "prediction", "feedback_learning_score", "instance_similarity", "sam_score", "cluster_coverage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    result = {
        "cases_json": str(cases_json.relative_to(WORKSPACE_ROOT)) if cases_json.is_relative_to(WORKSPACE_ROOT) else str(cases_json),
        "fold_count": len(folds),
        "row_count": sum(len(rows) for rows in rows_by_case.values()),
        "feature_keys": feature_keys,
        "macro_f1_by_model": macro_f1_by_model,
        "folds": folds,
        "artifacts": {
            "loco_predictions_csv": str((out_dir / "loco_predictions.csv").relative_to(WORKSPACE_ROOT)) if out_dir.is_relative_to(WORKSPACE_ROOT) else str(out_dir / "loco_predictions.csv"),
            "summary": str((out_dir / "summary.md").relative_to(WORKSPACE_ROOT)) if out_dir.is_relative_to(WORKSPACE_ROOT) else str(out_dir / "summary.md"),
        },
    }
    (out_dir / "loco_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Matcher Learning Leave-One-Case-Out",
        "",
        f"- Folds: `{result['fold_count']}`",
        f"- Rows: `{result['row_count']}`",
        "",
        "| Model | Macro F1 |",
        "| --- | ---: |",
    ]
    for model, f1 in macro_f1_by_model.items():
        lines.append(f"| {model} | {f1:.6f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'loco_results.json'}")


if __name__ == "__main__":
    main()
