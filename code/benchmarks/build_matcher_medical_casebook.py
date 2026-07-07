from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code" / "benchmarks"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from matcher_feedback_learning import (  # noqa: E402
    classification_metrics,
    learn_best_scalar_policy,
    predict_decision_stump,
    train_decision_stump,
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(WORKSPACE_ROOT):
        return str(resolved.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    return str(resolved)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as img:
        if size is not None and img.size != size:
            img = img.resize(size, Image.NEAREST)
        arr = np.asarray(img)
    if arr.ndim == 3:
        return np.any(arr > 0, axis=2)
    return arr > 0


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    inter = float(np.logical_and(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = float(np.logical_or(pred, target).sum())
    dice = (2.0 * inter / (pred_sum + target_sum)) if (pred_sum + target_sum) else 1.0
    iou = (inter / union) if union else 1.0
    return dice, iou


def feedback_like_score(row: dict[str, Any]) -> float:
    area = float(row.get("mask_area_pct", 0.0))
    bbox = float(row.get("bbox_area_pct", 0.0))
    area_penalty = max(0.0, area - 25.0) / 25.0
    bbox_penalty = max(0.0, bbox - 30.0) / 30.0
    return round(
        float(row.get("filter_score", 0.0))
        + 0.10 * float(row.get("sam_score", 0.0))
        + 0.10 * float(row.get("cluster_coverage", 0.0))
        + 0.05 * float(row.get("prompt_coverage", 0.0))
        - 0.15 * area_penalty
        - 0.15 * bbox_penalty,
        6,
    )


def load_manual_labels(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for item in labels:
        case_name = str(item["case_name"])
        group_id = int(item["group_id"])
        result[(case_name, group_id)] = item
    return result


def load_case_rows(
    row: dict[str, str],
    run_dir: Path,
    positive_dice: float,
    manual_labels: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    case_dir = run_dir / row["pilot_id"]
    scores_path = case_dir / "stage3_filtering" / "filtered_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    target = load_mask(resolve_path(row["query_mask_path"]))
    target_size = (target.shape[1], target.shape[0])
    rows: list[dict[str, Any]] = []
    for metric in payload["metrics"]:
        pred = load_mask(resolve_path(metric["mask_path"]), size=target_size)
        dice, iou = dice_iou(pred, target)
        item = dict(metric)
        group_id = int(metric["group_id"])
        auto_label = 1 if dice >= positive_dice else 0
        manual_item = (manual_labels or {}).get((row["pilot_id"], group_id))
        if manual_item:
            label_text = str(manual_item.get("label", "")).lower()
            label = 1 if label_text in {"1", "accept", "positive", "pos"} else 0
            label_source = str(manual_item.get("label_source", "manual_accept_reject"))
            review_reason = str(manual_item.get("reason", "manual review label"))
        else:
            label = auto_label
            label_source = f"dice_ge_{positive_dice:g}"
            review_reason = "auto Dice threshold label"
        item.update(
            {
                "case_name": row["pilot_id"],
                "dataset": row["dataset"],
                "support_case_id": row["support_case_id"],
                "query_case_id": row["query_case_id"],
                "label": label,
                "label_source": label_source,
                "review_label": "accept" if label == 1 else "reject",
                "review_reason": review_reason,
                "dice": round(float(dice), 6),
                "iou": round(float(iou), 6),
                "stage3_keep": 1 if str(metric.get("filter_status")) == "retained" else 0,
            }
        )
        if "overlay_path" not in item and item.get("mask_path"):
            item["overlay_path"] = str(item["mask_path"]).replace("proposal_masks", "proposal_overlays")
        item["feedback_learning_score"] = feedback_like_score(item)
        item["instance_similarity"] = float(item.get("filter_score", 0.0))
        rows.append(item)
    return rows


def select_best_by_feature(rows: list[dict[str, Any]], feature: str, direction: str = "ge") -> dict[str, Any] | None:
    if not rows:
        return None
    reverse = direction != "lt"
    return sorted(rows, key=lambda item: float(item.get(feature, 0.0)), reverse=reverse)[0]


def mean(values: list[float]) -> float:
    return round(float(sum(values) / len(values)), 6) if values else 0.0


def per_case_best(rows_by_case: dict[str, list[dict[str, Any]]], feature: str, direction: str = "ge") -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for case_name, rows in rows_by_case.items():
        best = select_best_by_feature(rows, feature, direction)
        if best is None:
            continue
        selected.append(
            {
                "case_name": case_name,
                "group_id": int(best["group_id"]),
                "dice": float(best["dice"]),
                "iou": float(best["iou"]),
                "feature_value": float(best.get(feature, 0.0)),
            }
        )
    return {
        "feature": feature,
        "direction": direction,
        "case_count": len(selected),
        "dice_mean": mean([float(item["dice"]) for item in selected]),
        "iou_mean": mean([float(item["iou"]) for item in selected]),
        "selected": selected,
    }


def retained_per_case(rows_by_case: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for case_name, rows in rows_by_case.items():
        retained = [row for row in rows if int(row["stage3_keep"]) == 1]
        if retained:
            best = max(retained, key=lambda item: float(item["dice"]))
            selected.append({"case_name": case_name, "group_id": int(best["group_id"]), "dice": float(best["dice"]), "iou": float(best["iou"])})
        else:
            selected.append({"case_name": case_name, "group_id": 0, "dice": 0.0, "iou": 0.0})
    return {
        "case_count": len(selected),
        "dice_mean": mean([float(item["dice"]) for item in selected]),
        "iou_mean": mean([float(item["iou"]) for item in selected]),
        "selected": selected,
    }


def evaluate_loco(rows_by_case: dict[str, list[dict[str, Any]]], feature_keys: list[str]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    predictions: list[int] = []
    labels: list[int] = []
    for test_case, test_rows in rows_by_case.items():
        train_rows = [row for case_name, rows in rows_by_case.items() if case_name != test_case for row in rows]
        if not train_rows:
            continue
        if len({int(row["label"]) for row in train_rows}) < 2:
            continue
        model = train_decision_stump(train_rows, feature_keys)
        preds = predict_decision_stump(model, test_rows)
        truth = [int(row["label"]) for row in test_rows]
        predictions.extend(preds)
        labels.extend(truth)
        folds.append(
            {
                "test_case": test_case,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "policy": model,
                "metrics": classification_metrics(truth, preds),
            }
        )
    return {
        "fold_count": len(folds),
        "metrics": classification_metrics(labels, predictions) if labels else {},
        "folds": folds,
    }


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_name",
        "dataset",
        "group_id",
        "label",
        "label_source",
        "review_label",
        "review_reason",
        "dice",
        "iou",
        "stage3_keep",
        "policy_keep",
        "stump_keep",
        "feedback_learning_score",
        "filter_score",
        "sam_score",
        "cluster_coverage",
        "prompt_coverage",
        "mask_area_pct",
        "bbox_area_pct",
        "filter_status",
        "reject_reasons",
        "mask_path",
        "overlay_path",
    ]
    with (out_dir / "medical_casebook_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "medical_casebook_policy.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Matcher Medical Casebook",
        "",
        f"- Cases: `{payload['case_count']}`",
        f"- Labeled rows: `{payload['labeled_count']}`",
        f"- Positive / negative: `{payload['positive_count']}` / `{payload['negative_count']}`",
        f"- Positive rule: `{payload['label_rule']}`",
        f"- Best policy: `{payload['best_policy']['model']}` / `{payload['best_policy']['feature']}`",
        f"- Best policy F1: `{payload['best_policy']['metrics']['f1']}`",
        f"- Stage3 F1: `{payload['baseline_stage3']['metrics']['f1']}`",
        f"- Oracle best Dice mean: `{payload['oracle_best_per_case']['dice_mean']}`",
        f"- Learned best Dice mean: `{payload['learned_best_per_case']['dice_mean']}`",
        f"- Stage3 retained Dice mean: `{payload['baseline_stage3']['retained_per_case']['dice_mean']}`",
        "",
        "## Model Metrics",
        "",
        "| Model | Precision | Recall | F1 | Accuracy | Kept |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["model_metrics"]:
        metrics = item["metrics"]
        lines.append(
            f"| {item['model']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['accuracy']} | {item['kept_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Case Dice",
            "",
            "| Strategy | Dice mean | IoU mean |",
            "| --- | ---: | ---: |",
            f"| Oracle best proposal | {payload['oracle_best_per_case']['dice_mean']} | {payload['oracle_best_per_case']['iou_mean']} |",
            f"| Stage3 retained best | {payload['baseline_stage3']['retained_per_case']['dice_mean']} | {payload['baseline_stage3']['retained_per_case']['iou_mean']} |",
            f"| Learned best by policy | {payload['learned_best_per_case']['dice_mean']} | {payload['learned_best_per_case']['iou_mean']} |",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a medical proposal casebook and learn a simple Matcher policy.")
    parser.add_argument("--manifest", default="runs/research_direction/matcher_medical_pilot/medical_matcher_pilot_manifest.csv")
    parser.add_argument("--run-dir", default="runs/research_direction/matcher_medical_pilot/run")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_medical_pilot/casebook")
    parser.add_argument("--positive-dice", type=float, default=0.30)
    parser.add_argument("--feature-keys", default="feedback_learning_score,filter_score,sam_score,cluster_coverage,prompt_coverage,mask_area_pct,bbox_area_pct")
    parser.add_argument("--labels-json", default="")
    parser.add_argument("--dataset-filter", default="", help="Optional comma-separated dataset filter, e.g. IDRID or MonuSeg.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = resolve_path(args.manifest)
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    labels_json = resolve_path(args.labels_json) if args.labels_json else None
    manual_labels = load_manual_labels(labels_json)
    dataset_filter = {item.strip() for item in args.dataset_filter.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    for manifest_row in read_manifest(manifest):
        if dataset_filter and manifest_row.get("dataset") not in dataset_filter:
            continue
        rows.extend(load_case_rows(manifest_row, run_dir, float(args.positive_dice), manual_labels=manual_labels))
    if not rows:
        raise ValueError("No medical proposal rows found")
    if len({int(row["label"]) for row in rows}) < 2:
        raise ValueError("Need both positive and negative labels")
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_case.setdefault(str(row["case_name"]), []).append(row)
    feature_keys = [key.strip() for key in args.feature_keys.split(",") if key.strip()]

    scalar = learn_best_scalar_policy(rows, feature_keys=feature_keys)["best_policy"]
    scalar_preds = [1 if float(row.get(scalar["feature"], 0.0)) >= float(scalar["threshold"]) else 0 for row in rows]
    scalar_metrics = classification_metrics([int(row["label"]) for row in rows], scalar_preds)
    stump = train_decision_stump(rows, feature_keys)
    stump_preds = predict_decision_stump(stump, rows)
    stump_metrics = classification_metrics([int(row["label"]) for row in rows], stump_preds)
    stage3_preds = [int(row["stage3_keep"]) for row in rows]
    stage3_metrics = classification_metrics([int(row["label"]) for row in rows], stage3_preds)

    model_metrics = [
        {"model": "stage3_retained_rule", "metrics": stage3_metrics, "kept_count": sum(stage3_preds), "policy": {"source": "filter_status == retained"}},
        {"model": "scalar_threshold", "metrics": scalar_metrics, "kept_count": sum(scalar_preds), "policy": scalar},
        {"model": "decision_stump", "metrics": stump_metrics, "kept_count": sum(stump_preds), "policy": stump},
    ]
    best_model = max(model_metrics[1:], key=lambda item: (float(item["metrics"]["f1"]), float(item["metrics"]["accuracy"])))
    if best_model["model"] == "decision_stump":
        best_feature = str(stump["feature"])
        best_direction = str(stump.get("direction", "ge"))
    else:
        best_feature = str(scalar["feature"])
        best_direction = "ge"
    for row, scalar_pred, stump_pred in zip(rows, scalar_preds, stump_preds):
        row["policy_keep"] = scalar_pred
        row["stump_keep"] = stump_pred

    payload = {
        "manifest": rel(manifest),
        "run_dir": rel(run_dir),
        "labels_json": rel(labels_json) if labels_json else "",
        "dataset_filter": sorted(dataset_filter),
        "case_count": len(rows_by_case),
        "labeled_count": len(rows),
        "positive_count": sum(1 for row in rows if int(row["label"]) == 1),
        "negative_count": sum(1 for row in rows if int(row["label"]) == 0),
        "label_rule": f"labels_json={rel(labels_json)}" if labels_json else f"dice >= {float(args.positive_dice):g}",
        "feature_keys": feature_keys,
        "best_policy": {
            "model": best_model["model"],
            "feature": best_feature,
            "direction": best_direction,
            "metrics": best_model["metrics"],
            "policy": best_model["policy"],
        },
        "model_metrics": model_metrics,
        "baseline_stage3": {
            "metrics": stage3_metrics,
            "retained_per_case": retained_per_case(rows_by_case),
        },
        "oracle_best_per_case": per_case_best(rows_by_case, "dice", "ge"),
        "learned_best_per_case": per_case_best(rows_by_case, best_feature, best_direction),
        "leave_one_case_out": evaluate_loco(rows_by_case, feature_keys),
        "artifacts": {
            "rows_csv": rel(out_dir / "medical_casebook_rows.csv"),
            "summary": rel(out_dir / "summary.md"),
        },
    }
    write_outputs(out_dir, rows, payload)
    print(f"Wrote {rel(out_dir / 'medical_casebook_policy.json')}")


if __name__ == "__main__":
    main()
