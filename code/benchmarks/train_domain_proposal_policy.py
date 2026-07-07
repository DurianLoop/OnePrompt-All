from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from matcher_feedback_learning import (
    classification_metrics,
    learn_best_scalar_policy,
    predict_decision_stump,
    train_decision_stump,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

BASE_FEATURES = [
    "feedback_learning_score",
    "filter_score",
    "sam_score",
    "cluster_coverage",
    "prompt_coverage",
    "mask_area_pct",
    "bbox_area_pct",
    "small_mask_score",
    "small_bbox_score",
]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(WORKSPACE_ROOT):
        return str(resolved.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    return str(resolved)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    return int(float(value))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["label"] = as_int(item.get("label", 0))
    item["stage3_keep"] = as_int(item.get("stage3_keep", 0))
    item["dice"] = as_float(item.get("dice", 0.0))
    item["mask_area_pct"] = as_float(item.get("mask_area_pct", 0.0))
    item["bbox_area_pct"] = as_float(item.get("bbox_area_pct", 0.0))
    item["small_mask_score"] = 100.0 - item["mask_area_pct"]
    item["small_bbox_score"] = 100.0 - item["bbox_area_pct"]
    for feature in BASE_FEATURES:
        item[feature] = as_float(item.get(feature, 0.0))
    return item


def case_count(rows: list[dict[str, Any]]) -> int:
    return len({str(row["case_name"]) for row in rows})


def per_case_best(rows: list[dict[str, Any]], keep_fn: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_name"])].append(row)

    best_scores: list[float] = []
    selected_cases = 0
    selected_proposals = 0
    for case_rows in grouped.values():
        selected = [row for row in case_rows if keep_fn(row)]
        selected_proposals += len(selected)
        if selected:
            selected_cases += 1
            best_scores.append(max(as_float(row.get("dice", 0.0)) for row in selected))
        else:
            best_scores.append(0.0)

    mean = sum(best_scores) / max(len(best_scores), 1)
    return {
        "case_count": len(grouped),
        "selected_cases": selected_cases,
        "selected_proposals": selected_proposals,
        "dice_mean": round(mean, 6),
        "dice_max": round(max(best_scores) if best_scores else 0.0, 6),
    }


def label_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    labels = [as_int(row["label"]) for row in rows]
    preds = [as_int(row[pred_key]) for row in rows]
    return classification_metrics(labels, preds)


def apply_scalar_policy(rows: list[dict[str, Any]], policy: dict[str, Any], pred_key: str) -> None:
    feature = str(policy["feature"])
    threshold = as_float(policy["threshold"])
    for row in rows:
        row[f"{pred_key}_score"] = as_float(row.get(feature, 0.0))
        row[pred_key] = 1 if row[f"{pred_key}_score"] >= threshold else 0


def evaluate_split(
    *,
    domain: str,
    split: str,
    train_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    scalar_policy: dict[str, Any],
    stump_policy: dict[str, Any],
) -> dict[str, Any]:
    metrics_stage3 = label_metrics(rows, "stage3_keep")
    metrics_scalar = label_metrics(rows, "domain_scalar_keep")
    metrics_stump = label_metrics(rows, "domain_stump_keep")
    dice_stage3 = per_case_best(rows, lambda row: as_int(row["stage3_keep"]) == 1)
    dice_scalar = per_case_best(rows, lambda row: as_int(row["domain_scalar_keep"]) == 1)
    dice_stump = per_case_best(rows, lambda row: as_int(row["domain_stump_keep"]) == 1)
    dice_oracle = per_case_best(rows, lambda row: True)
    positives = sum(as_int(row["label"]) for row in rows)
    return {
        "domain": domain,
        "split": split,
        "train_cases": case_count(train_rows),
        "train_proposals": len(train_rows),
        "apply_cases": case_count(rows),
        "apply_proposals": len(rows),
        "positives": positives,
        "negatives": len(rows) - positives,
        "stage3_f1": metrics_stage3["f1"],
        "stage3_precision": metrics_stage3["precision"],
        "stage3_recall": metrics_stage3["recall"],
        "scalar_f1": metrics_scalar["f1"],
        "scalar_precision": metrics_scalar["precision"],
        "scalar_recall": metrics_scalar["recall"],
        "stump_f1": metrics_stump["f1"],
        "stump_precision": metrics_stump["precision"],
        "stump_recall": metrics_stump["recall"],
        "scalar_feature": scalar_policy["feature"],
        "scalar_threshold": scalar_policy["threshold"],
        "stump_feature": stump_policy["feature"],
        "stump_threshold": stump_policy["threshold"],
        "stump_direction": stump_policy["direction"],
        "stage3_dice_mean": dice_stage3["dice_mean"],
        "scalar_dice_mean": dice_scalar["dice_mean"],
        "stump_dice_mean": dice_stump["dice_mean"],
        "oracle_dice_mean": dice_oracle["dice_mean"],
        "scalar_selected_cases": dice_scalar["selected_cases"],
        "scalar_selected_proposals": dice_scalar["selected_proposals"],
        "stump_selected_cases": dice_stump["selected_cases"],
        "stump_selected_proposals": dice_stump["selected_proposals"],
    }


def train_domain(
    *,
    domain: str,
    train_rows_all: list[dict[str, Any]],
    target_rows_all: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_rows = [enrich(row) for row in train_rows_all if row["dataset"] == domain]
    target_rows = [enrich(row) for row in target_rows_all if row["dataset"] == domain]
    if not train_rows:
        raise ValueError(f"No training rows for {domain}")
    if not target_rows:
        raise ValueError(f"No target rows for {domain}")

    scalar_result = learn_best_scalar_policy(train_rows, feature_keys=BASE_FEATURES)
    scalar_policy = dict(scalar_result["best_policy"])
    stump_policy = train_decision_stump(train_rows, BASE_FEATURES)

    apply_scalar_policy(train_rows, scalar_policy, "domain_scalar_keep")
    apply_scalar_policy(target_rows, scalar_policy, "domain_scalar_keep")
    train_stump_preds = predict_decision_stump(stump_policy, train_rows)
    target_stump_preds = predict_decision_stump(stump_policy, target_rows)
    for row, pred in zip(train_rows, train_stump_preds):
        row["domain_stump_keep"] = pred
    for row, pred in zip(target_rows, target_stump_preds):
        row["domain_stump_keep"] = pred

    train_cases = {str(row["case_name"]) for row in train_rows}
    heldout_rows = [row for row in target_rows if str(row["case_name"]) not in train_cases]
    overlap_rows = [row for row in target_rows if str(row["case_name"]) in train_cases]

    split_results = [
        evaluate_split(
            domain=domain,
            split="train_self",
            train_rows=train_rows,
            rows=train_rows,
            scalar_policy=scalar_policy,
            stump_policy=stump_policy,
        ),
        evaluate_split(
            domain=domain,
            split="target_50_all",
            train_rows=train_rows,
            rows=target_rows,
            scalar_policy=scalar_policy,
            stump_policy=stump_policy,
        ),
        evaluate_split(
            domain=domain,
            split="target_50_overlap",
            train_rows=train_rows,
            rows=overlap_rows,
            scalar_policy=scalar_policy,
            stump_policy=stump_policy,
        ),
        evaluate_split(
            domain=domain,
            split="target_50_heldout",
            train_rows=train_rows,
            rows=heldout_rows,
            scalar_policy=scalar_policy,
            stump_policy=stump_policy,
        ),
    ]

    model = {
        "domain": domain,
        "feature_keys": BASE_FEATURES,
        "scalar_policies": scalar_result["policies"],
        "scalar_policy": scalar_policy,
        "stump_policy": stump_policy,
        "train_case_names": sorted(train_cases),
        "target_case_count": case_count(target_rows),
        "heldout_case_count": case_count(heldout_rows),
        "split_results": split_results,
    }
    domain_dir = out_dir / domain.lower()
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "domain_policy.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_rows: list[dict[str, Any]] = []
    for row in target_rows:
        prediction_rows.append(
            {
                "domain": domain,
                "case_name": row["case_name"],
                "group_id": row["group_id"],
                "label": row["label"],
                "dice": row["dice"],
                "stage3_keep": row["stage3_keep"],
                "domain_scalar_keep": row["domain_scalar_keep"],
                "domain_stump_keep": row["domain_stump_keep"],
                "domain_scalar_score": row["domain_scalar_keep_score"],
                "split": "overlap" if str(row["case_name"]) in train_cases else "heldout",
                "mask_path": row.get("mask_path", ""),
                "overlay_path": row.get("overlay_path", ""),
            }
        )
    return split_results, prediction_rows, model


def write_markdown(out_dir: Path, summary_rows: list[dict[str, Any]], model_payload: dict[str, Any]) -> None:
    lines = [
        "# 分域 Feedback-Learned proposal policy 验证",
        "",
        "## 实验目的",
        "",
        "本实验回应“如果在单独某一个领域训练一次，是否可以把该领域中对应病灶 proposal 更稳定地标注出来”。当前训练对象不是端到端分割网络，而是 proposal keep/reject 策略：Stage1/Stage2 仍负责产生候选点和候选 mask，学习模块负责从候选池中选择更可信的 proposal。",
        "",
        "## 主结果",
        "",
        "| Domain | Split | Cases | Proposals | Stage3 F1 | Scalar F1 | Stump F1 | Stage3 Dice | Scalar Dice | Oracle Dice | Learned rule |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {domain} | {split} | {apply_cases} | {apply_proposals} | {stage3_f1:.6f} | {scalar_f1:.6f} | {stump_f1:.6f} | {stage3_dice_mean:.6f} | {scalar_dice_mean:.6f} | {oracle_dice_mean:.6f} | {scalar_feature} >= {scalar_threshold} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "1. 这一步已经把“手写 Stage3 规则”推进为“按领域学习的筛选策略”。",
            "2. 如果同域 heldout F1 和 Dice 稳定高于 Stage3，就说明领域内少量标注能迁移到更多 case 的候选筛选。",
            "3. 这还不能直接声称“训练一次即可标注所有病灶”，因为上游候选池必须先覆盖目标病灶；下一步需要扩大人工标签、加入更多特征或轻量学习器，并报告 heldout/LOCO/bootstrap。",
            "",
            "## 输出文件",
            "",
            f"- `{rel(out_dir / 'domain_policy_transfer_summary.csv')}`",
            f"- `{rel(out_dir / 'domain_policy_predictions.csv')}`",
            f"- `{rel(out_dir / 'domain_policy_transfer_results.json')}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train domain-specific proposal keep/reject policies and apply them to 50 cases.")
    parser.add_argument(
        "--train-rows",
        default="runs/research_direction/feedback_learned_matcher_mainline/overall_20/medical_casebook_rows.csv",
    )
    parser.add_argument(
        "--target-rows",
        default="runs/research_direction/matcher_medical_expanded_50/casebook/medical_casebook_rows.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/research_direction/feedback_learned_matcher_mainline/domain_policy_transfer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_csv(resolve_path(args.train_rows))
    target_rows = read_csv(resolve_path(args.target_rows))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = sorted({row["dataset"] for row in train_rows})
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for domain in domains:
        split_rows, domain_predictions, model = train_domain(
            domain=domain,
            train_rows_all=train_rows,
            target_rows_all=target_rows,
            out_dir=out_dir,
        )
        summary_rows.extend(split_rows)
        prediction_rows.extend(domain_predictions)
        models[domain] = model

    summary_fields = [
        "domain",
        "split",
        "train_cases",
        "train_proposals",
        "apply_cases",
        "apply_proposals",
        "positives",
        "negatives",
        "stage3_f1",
        "stage3_precision",
        "stage3_recall",
        "scalar_f1",
        "scalar_precision",
        "scalar_recall",
        "stump_f1",
        "stump_precision",
        "stump_recall",
        "scalar_feature",
        "scalar_threshold",
        "stump_feature",
        "stump_threshold",
        "stump_direction",
        "stage3_dice_mean",
        "scalar_dice_mean",
        "stump_dice_mean",
        "oracle_dice_mean",
        "scalar_selected_cases",
        "scalar_selected_proposals",
        "stump_selected_cases",
        "stump_selected_proposals",
    ]
    write_csv(out_dir / "domain_policy_transfer_summary.csv", summary_rows, summary_fields)
    write_csv(
        out_dir / "domain_policy_predictions.csv",
        prediction_rows,
        [
            "domain",
            "case_name",
            "group_id",
            "label",
            "dice",
            "stage3_keep",
            "domain_scalar_keep",
            "domain_stump_keep",
            "domain_scalar_score",
            "split",
            "mask_path",
            "overlay_path",
        ],
    )
    payload = {
        "train_rows": rel(resolve_path(args.train_rows)),
        "target_rows": rel(resolve_path(args.target_rows)),
        "domains": models,
        "summary_rows": summary_rows,
    }
    (out_dir / "domain_policy_transfer_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_dir, summary_rows, payload)
    print(f"Wrote {rel(out_dir / 'domain_policy_transfer_summary.csv')}")


if __name__ == "__main__":
    main()
