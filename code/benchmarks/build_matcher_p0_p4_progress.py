from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from matcher_feedback_learning import (
    classification_metrics,
    learn_best_scalar_policy,
    predict_decision_stump,
    predict_logistic,
    rows_to_matrix,
    standardize_train_test,
    train_decision_stump,
    train_logistic_regression,
)


ROOT = Path(__file__).resolve().parents[2]
FEATURE_KEYS = [
    "feedback_learning_score",
    "filter_score",
    "sam_score",
    "cluster_coverage",
    "prompt_coverage",
    "mask_area_pct",
    "bbox_area_pct",
]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(value)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def to_int(row: dict[str, Any], key: str) -> int:
    try:
        return int(float(row.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def typed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_float = ["dice", "iou", *FEATURE_KEYS]
    numeric_int = ["group_id", "label", "stage3_keep", "policy_keep", "stump_keep"]
    converted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in numeric_float:
            item[key] = to_float(item, key)
        for key in numeric_int:
            item[key] = to_int(item, key)
        converted.append(item)
    return converted


def group_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_name"]), []).append(row)
    return grouped


def stage3_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return classification_metrics([to_int(row, "label") for row in rows], [to_int(row, "stage3_keep") for row in rows])


def scalar_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy = learn_best_scalar_policy(rows, feature_keys=FEATURE_KEYS)["best_policy"]
    preds = [1 if to_float(row, str(policy["feature"])) >= float(policy["threshold"]) else 0 for row in rows]
    return {
        "policy": policy,
        "metrics": classification_metrics([to_int(row, "label") for row in rows], preds),
        "kept_count": int(sum(preds)),
    }


def p0_review_package(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conflict_rows = [
        row
        for row in rows
        if (to_int(row, "label") == 1 and to_int(row, "stage3_keep") == 0)
        or (to_int(row, "label") == 0 and to_int(row, "stage3_keep") == 1)
    ]
    priority = sorted(conflict_rows, key=lambda item: (to_float(item, "dice"), to_float(item, "feedback_learning_score")), reverse=True)[:20]
    review_items = []
    for row in priority:
        mask_path = str(row.get("mask_path", ""))
        overlay_path = mask_path.replace("proposal_masks", "proposal_overlays")
        review_items.append(
            {
                "case_name": row["case_name"],
                "dataset": row["dataset"],
                "group_id": to_int(row, "group_id"),
                "suggested_reason": "positive_but_filtered" if to_int(row, "label") == 1 else "negative_but_retained",
                "dice": round(to_float(row, "dice"), 6),
                "stage3_keep": to_int(row, "stage3_keep"),
                "feedback_learning_score": round(to_float(row, "feedback_learning_score"), 6),
                "mask_path": mask_path,
                "overlay_path": overlay_path,
            }
        )
    return {
        "pending_review_count": len(rows),
        "stage3_conflict_count": len(conflict_rows),
        "positive_but_filtered": sum(1 for row in conflict_rows if to_int(row, "label") == 1),
        "negative_but_retained": sum(1 for row in conflict_rows if to_int(row, "label") == 0),
        "priority_review_items": review_items,
    }


def p1_domain_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        domain_rows = [row for row in rows if str(row["dataset"]) == dataset]
        stage3 = stage3_metrics(domain_rows)
        learned = scalar_metrics(domain_rows)
        domains[dataset] = {
            "row_count": len(domain_rows),
            "positive_count": sum(1 for row in domain_rows if to_int(row, "label") == 1),
            "negative_count": sum(1 for row in domain_rows if to_int(row, "label") == 0),
            "stage3_metrics": stage3,
            "learned_metrics": learned["metrics"],
            "learned_policy": learned["policy"],
            "learned_kept_count": learned["kept_count"],
        }
    return domains


def p2_feedback_simulation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_by_case(rows)
    global_policy = scalar_metrics(rows)["policy"]
    pivot = float(global_policy["threshold"])
    feature = str(global_policy["feature"])
    rounds: list[dict[str, Any]] = []
    for budget in [1, 2, 4, 8, 12]:
        labeled: list[dict[str, Any]] = []
        for case_rows in grouped.values():
            ordered = sorted(case_rows, key=lambda row: abs(to_float(row, feature) - pivot))
            labeled.extend(ordered[: min(budget, len(ordered))])
        if len({to_int(row, "label") for row in labeled}) < 2:
            continue
        learned = scalar_metrics(labeled)["policy"]
        preds = [1 if to_float(row, str(learned["feature"])) >= float(learned["threshold"]) else 0 for row in rows]
        metrics = classification_metrics([to_int(row, "label") for row in rows], preds)
        rounds.append(
            {
                "labels_per_case": budget,
                "labeled_rows": len(labeled),
                "positive_labels": sum(1 for row in labeled if to_int(row, "label") == 1),
                "negative_labels": sum(1 for row in labeled if to_int(row, "label") == 0),
                "policy": learned,
                "metrics_on_all_rows": metrics,
                "kept_count": int(sum(preds)),
            }
        )
    return {"seed_policy": global_policy, "rounds": rounds}


def p3_expansion_plan(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    idrid_images = sorted((ROOT / "code" / "datasets" / "IDRID" / "DR_Training_Set" / "Fundus Images").glob("*.jpg"))
    monuseg_images = sorted((ROOT / "code" / "datasets" / "MonuSeg" / "kmms" / "images").glob("*.png"))
    available_idrid_queries = max(len(idrid_images) - 1, 0)
    available_monuseg_queries = max(len(monuseg_images) - 1, 0)
    planned_idrid = min(30, available_idrid_queries)
    planned_monuseg = min(20, available_monuseg_queries)
    return {
        "current_cases": len(manifest_rows),
        "available_idrid_queries": available_idrid_queries,
        "available_monuseg_queries": available_monuseg_queries,
        "planned_idrid_cases": planned_idrid,
        "planned_monuseg_cases": planned_monuseg,
        "planned_cases": planned_idrid + planned_monuseg,
        "one_click_command": (
            "D:\\SAM\\conda_envs\\sam_gpu\\python.exe -B "
            "code\\benchmarks\\run_matcher_medical_one_click.py "
            "--out-dir runs\\research_direction\\matcher_medical_expanded_50 "
            f"--idrid-cases {planned_idrid} --monuseg-cases {planned_monuseg}"
        ),
    }


def logistic_loco(rows_by_case: dict[str, list[dict[str, Any]]]) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    all_labels: list[int] = []
    all_preds: list[int] = []
    folds: list[dict[str, Any]] = []
    for test_case, test_rows in rows_by_case.items():
        train_rows = [row for name, case_rows in rows_by_case.items() if name != test_case for row in case_rows]
        if len({to_int(row, "label") for row in train_rows}) < 2:
            continue
        x_train, y_train = rows_to_matrix(train_rows, FEATURE_KEYS)
        x_test, _ = rows_to_matrix(test_rows, FEATURE_KEYS)
        x_train, x_test, mean, std = standardize_train_test(x_train, x_test)
        model = train_logistic_regression(x_train, y_train, lr=0.1, steps=1000, l2=0.02)
        probs, preds = predict_logistic(model, x_test, threshold=0.5)
        labels = [to_int(row, "label") for row in test_rows]
        pred_list = [int(value) for value in preds.tolist()]
        all_labels.extend(labels)
        all_preds.extend(pred_list)
        folds.append(
            {
                "test_case": test_case,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "metrics": classification_metrics(labels, pred_list),
                "probability_mean": round(float(np.asarray(probs).mean()), 6),
                "standardize_mean": [round(float(value), 6) for value in mean.tolist()],
                "standardize_std": [round(float(value), 6) for value in std.tolist()],
            }
        )
    return all_labels, all_preds, folds


def p4_model_upgrade(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_case = group_by_case(rows)
    loco_payload: dict[str, Any] = {}
    for model_name in ["scalar_threshold", "decision_stump"]:
        all_labels: list[int] = []
        all_preds: list[int] = []
        folds: list[dict[str, Any]] = []
        for test_case, test_rows in rows_by_case.items():
            train_rows = [row for name, case_rows in rows_by_case.items() if name != test_case for row in case_rows]
            if len({to_int(row, "label") for row in train_rows}) < 2:
                continue
            if model_name == "scalar_threshold":
                policy = scalar_metrics(train_rows)["policy"]
                preds = [1 if to_float(row, str(policy["feature"])) >= float(policy["threshold"]) else 0 for row in test_rows]
            else:
                policy = train_decision_stump(train_rows, FEATURE_KEYS)
                preds = predict_decision_stump(policy, test_rows)
            labels = [to_int(row, "label") for row in test_rows]
            all_labels.extend(labels)
            all_preds.extend(preds)
            folds.append(
                {
                    "test_case": test_case,
                    "train_count": len(train_rows),
                    "test_count": len(test_rows),
                    "policy": policy,
                    "metrics": classification_metrics(labels, preds),
                }
            )
        loco_payload[model_name] = {"metrics": classification_metrics(all_labels, all_preds), "folds": folds}
    labels, preds, folds = logistic_loco(rows_by_case)
    loco_payload["logistic_regression"] = {"metrics": classification_metrics(labels, preds), "folds": folds}
    return {"loco": {name: {"f1": value["metrics"]["f1"], **value["metrics"]} for name, value in loco_payload.items()}, "folds": loco_payload}


def best_case_proposal(rows: list[dict[str, Any]], case_name: str) -> dict[str, Any]:
    case_rows = [row for row in rows if str(row["case_name"]) == case_name]
    if not case_rows:
        return {}
    row = max(case_rows, key=lambda item: to_float(item, "dice"))
    mask_path = str(row.get("mask_path", ""))
    return {
        "group_id": to_int(row, "group_id"),
        "dice": round(to_float(row, "dice"), 6),
        "iou": round(to_float(row, "iou"), 6),
        "stage3_keep": to_int(row, "stage3_keep"),
        "feedback_learning_score": round(to_float(row, "feedback_learning_score"), 6),
        "mask_path": mask_path,
        "overlay_path": mask_path.replace("proposal_masks", "proposal_overlays"),
    }


def case_point_detail(
    *,
    case_name: str,
    manifest_row: dict[str, Any],
    expanded_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_dir = expanded_root / "run" / case_name
    points_payload = json.loads((run_dir / "stage1" / "points.json").read_text(encoding="utf-8"))
    sampling_payload = json.loads((run_dir / "stage2_sampling" / "prompt_groups.json").read_text(encoding="utf-8"))
    top_points = [
        {
            "rank": int(point["rank"]),
            "x_original": round(float(point["x_original"]), 3),
            "y_original": round(float(point["y_original"]), 3),
            "similarity": round(float(point["similarity"]), 6),
            "selected_by_threshold": bool(point.get("selected_by_threshold", False)),
        }
        for point in points_payload.get("points", [])[:10]
    ]
    stage2_points = {int(point["rank"]): point for point in sampling_payload.get("points", [])}
    for point in top_points:
        stage2_point = stage2_points.get(int(point["rank"]), {})
        point["cluster_id"] = int(stage2_point.get("cluster_id", -99))
    prompt_groups = []
    for group in sampling_payload.get("prompt_groups", [])[:8]:
        prompt_groups.append(
            {
                "group_id": int(group["group_id"]),
                "cluster_id": int(group.get("cluster_id", -99)),
                "prompt_points": [
                    {
                        "x": round(float(point["x"]), 3),
                        "y": round(float(point["y"]), 3),
                        "rank": int(point.get("rank", 0)),
                    }
                    for point in group.get("prompt_points", [])
                ],
            }
        )
    return {
        "case_name": case_name,
        "dataset": manifest_row["dataset"],
        "support_case_id": manifest_row["support_case_id"],
        "query_case_id": manifest_row["query_case_id"],
        "reference_image": manifest_row["support_image_path"],
        "reference_mask": manifest_row["support_mask_path"],
        "reference_box_xyxy": json.loads(manifest_row["support_box_xyxy"]),
        "query_image": manifest_row["query_image_path"],
        "query_mask": manifest_row["query_mask_path"],
        "stage1_topk_overlay": rel(run_dir / "stage1" / "topk_overlay.png"),
        "stage1_heatmap": rel(run_dir / "stage1" / "similarity_heatmap.png"),
        "stage2_cluster_overlay": rel(run_dir / "stage2_sampling" / "cluster_overlay.png"),
        "stage2_proposal_grid": rel(run_dir / "stage2_proposals" / "proposal_grid.png"),
        "stage3_retained_grid": rel(run_dir / "stage3_filtering" / "retained_grid.png"),
        "stage1_threshold": points_payload.get("threshold"),
        "stage1_selected_count": points_payload.get("selected_count"),
        "stage2_cluster_count": sampling_payload.get("cluster_count"),
        "stage2_prompt_group_count": sampling_payload.get("prompt_group_count"),
        "top_points": top_points,
        "prompt_groups": prompt_groups,
        "best_proposal": best_case_proposal(rows, case_name),
    }


def build_point_details(manifest_rows: list[dict[str, Any]], expanded_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_case = {row["pilot_id"]: row for row in manifest_rows}
    case_names = [case for case in ["idrid_11", "monuseg_04", "idrid_01"] if case in manifest_by_case]
    return [
        case_point_detail(case_name=case_name, manifest_row=manifest_by_case[case_name], expanded_root=expanded_root, rows=rows)
        for case_name in case_names
    ]


def write_point_csv(out_dir: Path, point_details: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for detail in point_details:
        for point in detail["top_points"]:
            rows.append(
                {
                    "case_name": detail["case_name"],
                    "dataset": detail["dataset"],
                    "reference_image": detail["reference_image"],
                    "query_image": detail["query_image"],
                    "rank": point["rank"],
                    "x_original": point["x_original"],
                    "y_original": point["y_original"],
                    "similarity": point["similarity"],
                    "cluster_id": point["cluster_id"],
                }
            )
    write_csv(
        out_dir / "point_details.csv",
        rows,
        ["case_name", "dataset", "reference_image", "query_image", "rank", "x_original", "y_original", "similarity", "cluster_id"],
    )


def write_domain_csv(out_dir: Path, payload: dict[str, Any]) -> None:
    rows = []
    for domain, item in payload.items():
        rows.append(
            {
                "dataset": domain,
                "row_count": item["row_count"],
                "positive_count": item["positive_count"],
                "negative_count": item["negative_count"],
                "stage3_f1": item["stage3_metrics"]["f1"],
                "learned_f1": item["learned_metrics"]["f1"],
                "learned_feature": item["learned_policy"]["feature"],
                "learned_threshold": item["learned_policy"]["threshold"],
            }
        )
    write_csv(
        out_dir / "domain_policy_summary.csv",
        rows,
        ["dataset", "row_count", "positive_count", "negative_count", "stage3_f1", "learned_f1", "learned_feature", "learned_threshold"],
    )


def write_loco_csv(out_dir: Path, payload: dict[str, Any]) -> None:
    rows = []
    for model, metrics in payload["loco"].items():
        rows.append(
            {
                "model": model,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
            }
        )
    write_csv(out_dir / "model_upgrade_loco.csv", rows, ["model", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"])


def write_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    p1 = payload["p1_domain_policy"]
    p2_rounds = payload["p2_feedback_simulation"]["rounds"]
    p4 = payload["p4_model_upgrade"]["loco"]
    lines = [
        "# P0-P4 Matcher Research Progress",
        "",
        "## P0 Review Package",
        "",
        f"- Pending proposal reviews: `{payload['p0_review_package']['pending_review_count']}`",
        f"- Stage3 conflicts: `{payload['p0_review_package']['stage3_conflict_count']}`",
        f"- Positive but filtered: `{payload['p0_review_package']['positive_but_filtered']}`",
        f"- Negative but retained: `{payload['p0_review_package']['negative_but_retained']}`",
        "",
        "## P1 Domain Policies",
        "",
        "| Domain | Rows | Pos/Neg | Stage3 F1 | Learned F1 | Learned policy |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, item in p1.items():
        lines.append(
            f"| {domain} | {item['row_count']} | {item['positive_count']}/{item['negative_count']} | "
            f"{item['stage3_metrics']['f1']} | {item['learned_metrics']['f1']} | "
            f"{item['learned_policy']['feature']} >= {item['learned_policy']['threshold']} |"
        )
    lines.extend(["", "## P2 Feedback Simulation", "", "| Labels per case | Labeled rows | Pos/Neg | F1 on all rows | Kept |", "| ---: | ---: | ---: | ---: | ---: |"])
    for item in p2_rounds:
        lines.append(
            f"| {item['labels_per_case']} | {item['labeled_rows']} | {item['positive_labels']}/{item['negative_labels']} | "
            f"{item['metrics_on_all_rows']['f1']} | {item['kept_count']} |"
        )
    lines.extend(["", "## P3 Expansion Readiness", ""])
    p3 = payload["p3_expansion_plan"]
    lines.extend(
        [
            f"- Current cases: `{p3['current_cases']}`",
            f"- Available IDRID queries: `{p3['available_idrid_queries']}`",
            f"- Available MonuSeg queries: `{p3['available_monuseg_queries']}`",
            f"- Planned 50-case split: `{p3['planned_idrid_cases']}` IDRID + `{p3['planned_monuseg_cases']}` MonuSeg",
            f"- Command: `{p3['one_click_command']}`",
            "",
            "## P4 Model Upgrade LOCO",
            "",
            "| Model | Precision | Recall | F1 | Accuracy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, metrics in p4.items():
        lines.append(f"| {model} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['accuracy']} |")
    lines.extend(["", "## Point Details", ""])
    for detail in payload["point_details"]:
        first = detail["top_points"][0]
        lines.append(
            f"- `{detail['case_name']}`: reference `{detail['reference_image']}`, query `{detail['query_image']}`, "
            f"top-1 point `({first['x_original']}, {first['y_original']})`, similarity `{first['similarity']}`."
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P0-P4 Matcher research progress with detailed traceability.")
    parser.add_argument("--expanded-root", default="runs/research_direction/matcher_medical_expanded")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_p0_p4_progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expanded_root = resolve(args.expanded_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(expanded_root / "medical_matcher_pilot_manifest.csv")
    rows = typed_rows(read_csv(expanded_root / "casebook" / "medical_casebook_rows.csv"))
    point_details = build_point_details(manifest_rows, expanded_root, rows)
    payload = {
        "source": {
            "expanded_root": rel(expanded_root),
            "manifest": rel(expanded_root / "medical_matcher_pilot_manifest.csv"),
            "casebook_rows": rel(expanded_root / "casebook" / "medical_casebook_rows.csv"),
        },
        "p0_review_package": p0_review_package(rows),
        "p1_domain_policy": p1_domain_policy(rows),
        "p2_feedback_simulation": p2_feedback_simulation(rows),
        "p3_expansion_plan": p3_expansion_plan(manifest_rows),
        "p4_model_upgrade": p4_model_upgrade(rows),
        "point_details": point_details,
    }
    (out_dir / "p0_p4_progress.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_point_csv(out_dir, point_details)
    write_domain_csv(out_dir, payload["p1_domain_policy"])
    write_loco_csv(out_dir, payload["p4_model_upgrade"])
    write_summary(out_dir, payload)
    print(f"Wrote {rel(out_dir / 'p0_p4_progress.json')}")


if __name__ == "__main__":
    main()
