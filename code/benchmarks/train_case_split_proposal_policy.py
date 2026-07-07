from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from matcher_feedback_learning import classification_metrics, learn_best_scalar_policy, predict_decision_stump, train_decision_stump


ROOT = Path(__file__).resolve().parents[2]

FEATURES = [
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


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
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


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def inum(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    return int(float(value))


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["label"] = inum(item.get("label", 0))
    item["stage3_keep"] = inum(item.get("stage3_keep", 0))
    item["dice"] = fnum(item.get("dice", 0.0))
    item["mask_area_pct"] = fnum(item.get("mask_area_pct", 0.0))
    item["bbox_area_pct"] = fnum(item.get("bbox_area_pct", 0.0))
    item["small_mask_score"] = 100.0 - item["mask_area_pct"]
    item["small_bbox_score"] = 100.0 - item["bbox_area_pct"]
    for feature in FEATURES:
        item[feature] = fnum(item.get(feature, 0.0))
    return item


def case_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["case_name"]) for row in rows})


def make_stratified_case_folds(rows: list[dict[str, Any]], fold_count: int) -> list[set[str]]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["dataset"]), str(row["case_name"]))
        if key in seen:
            continue
        seen.add(key)
        by_domain[str(row["dataset"])].append(str(row["case_name"]))
    folds = [set() for _ in range(fold_count)]
    for domain_cases in by_domain.values():
        for idx, name in enumerate(sorted(domain_cases)):
            folds[idx % fold_count].add(name)
    return folds


def apply_scalar(rows: list[dict[str, Any]], policy: dict[str, Any], key: str) -> None:
    feature = str(policy["feature"])
    threshold = fnum(policy["threshold"])
    for row in rows:
        row[f"{key}_score"] = fnum(row.get(feature, 0.0))
        row[key] = 1 if row[f"{key}_score"] >= threshold else 0


def per_case_best(rows: list[dict[str, Any]], keep_fn: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_name"])].append(row)
    scores: list[float] = []
    selected_cases = 0
    selected_proposals = 0
    for case_rows in grouped.values():
        kept = [row for row in case_rows if keep_fn(row)]
        selected_proposals += len(kept)
        if kept:
            selected_cases += 1
            scores.append(max(fnum(row.get("dice", 0.0)) for row in kept))
        else:
            scores.append(0.0)
    return {
        "case_count": len(grouped),
        "selected_cases": selected_cases,
        "selected_proposals": selected_proposals,
        "dice_mean": round(sum(scores) / max(len(scores), 1), 6),
        "dice_max": round(max(scores) if scores else 0.0, 6),
    }


def evaluate_predictions(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    metrics = classification_metrics([inum(row["label"]) for row in rows], [inum(row[pred_key]) for row in rows])
    dice = per_case_best(rows, lambda row: inum(row[pred_key]) == 1)
    return {**metrics, "dice_mean": dice["dice_mean"], "selected_cases": dice["selected_cases"], "selected_proposals": dice["selected_proposals"]}


def oracle_dice(rows: list[dict[str, Any]]) -> float:
    return float(per_case_best(rows, lambda row: True)["dice_mean"])


def train_predict_overall(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scalar_result = learn_best_scalar_policy(train_rows, feature_keys=FEATURES)
    scalar_policy = dict(scalar_result["best_policy"])
    stump_policy = train_decision_stump(train_rows, FEATURES)
    apply_scalar(test_rows, scalar_policy, "scalar_keep")
    stump_preds = predict_decision_stump(stump_policy, test_rows)
    for row, pred in zip(test_rows, stump_preds):
        row["stump_keep"] = pred
    meta = {"scalar_policy": scalar_policy, "stump_policy": stump_policy}
    return test_rows, meta


def train_predict_domain(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    policies: dict[str, Any] = {}
    domains = sorted({str(row["dataset"]) for row in test_rows})
    for domain in domains:
        domain_train = [row for row in train_rows if str(row["dataset"]) == domain]
        domain_test = [row for row in test_rows if str(row["dataset"]) == domain]
        scalar_result = learn_best_scalar_policy(domain_train, feature_keys=FEATURES)
        scalar_policy = dict(scalar_result["best_policy"])
        stump_policy = train_decision_stump(domain_train, FEATURES)
        apply_scalar(domain_test, scalar_policy, "scalar_keep")
        stump_preds = predict_decision_stump(stump_policy, domain_test)
        for row, pred in zip(domain_test, stump_preds):
            row["stump_keep"] = pred
        policies[domain] = {"scalar_policy": scalar_policy, "stump_policy": stump_policy}
        out_rows.extend(domain_test)
    return out_rows, policies


def run_setting(rows: list[dict[str, Any]], folds: list[set[str]], setting: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    fold_meta: dict[str, Any] = {}
    all_cases = set(case_names(rows))
    for fold_idx, test_cases in enumerate(folds):
        train_cases = all_cases - test_cases
        train_rows = [dict(row) for row in rows if str(row["case_name"]) in train_cases]
        test_rows = [dict(row) for row in rows if str(row["case_name"]) in test_cases]
        if setting == "overall_5fold":
            predicted, meta = train_predict_overall(train_rows, test_rows)
        elif setting == "domain_5fold":
            predicted, meta = train_predict_domain(train_rows, test_rows)
        else:
            raise ValueError(setting)
        fold_meta[str(fold_idx)] = meta
        stage3 = evaluate_predictions(predicted, "stage3_keep")
        scalar = evaluate_predictions(predicted, "scalar_keep")
        stump = evaluate_predictions(predicted, "stump_keep")
        fold_rows.append(
            {
                "setting": setting,
                "fold": fold_idx,
                "train_cases": len(train_cases),
                "test_cases": len(test_cases),
                "train_proposals": len(train_rows),
                "test_proposals": len(test_rows),
                "test_positive": sum(inum(row["label"]) for row in test_rows),
                "test_negative": len(test_rows) - sum(inum(row["label"]) for row in test_rows),
                "stage3_f1": stage3["f1"],
                "scalar_f1": scalar["f1"],
                "stump_f1": stump["f1"],
                "stage3_dice_mean": stage3["dice_mean"],
                "scalar_dice_mean": scalar["dice_mean"],
                "stump_dice_mean": stump["dice_mean"],
                "oracle_dice_mean": oracle_dice(test_rows),
                "test_case_names": ";".join(sorted(test_cases)),
            }
        )
        for row in predicted:
            prediction_rows.append(
                {
                    "setting": setting,
                    "fold": fold_idx,
                    "case_name": row["case_name"],
                    "dataset": row["dataset"],
                    "group_id": row["group_id"],
                    "label": row["label"],
                    "dice": row["dice"],
                    "stage3_keep": row["stage3_keep"],
                    "scalar_keep": row["scalar_keep"],
                    "stump_keep": row["stump_keep"],
                    "mask_path": row.get("mask_path", ""),
                    "overlay_path": row.get("overlay_path", ""),
                }
            )
    return fold_rows, prediction_rows, fold_meta


def summarize(setting: str, rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage3 = evaluate_predictions(prediction_rows, "stage3_keep")
    scalar = evaluate_predictions(prediction_rows, "scalar_keep")
    stump = evaluate_predictions(prediction_rows, "stump_keep")
    return {
        "setting": setting,
        "folds": len(rows),
        "cases": len(set(row["case_name"] for row in prediction_rows)),
        "proposals": len(prediction_rows),
        "positive": sum(inum(row["label"]) for row in prediction_rows),
        "negative": len(prediction_rows) - sum(inum(row["label"]) for row in prediction_rows),
        "stage3_f1": stage3["f1"],
        "scalar_f1": scalar["f1"],
        "stump_f1": stump["f1"],
        "stage3_dice_mean": stage3["dice_mean"],
        "scalar_dice_mean": scalar["dice_mean"],
        "stump_dice_mean": stump["dice_mean"],
        "oracle_dice_mean": oracle_dice(prediction_rows),
        "test_count_per_case": ";".join(f"{case}:{count}" for case, count in sorted(Counter(row["case_name"] for row in prediction_rows).items())),
    }


def write_summary_md(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 50-case case-level 5-fold proposal policy",
        "",
        "This experiment uses all 50 cases and all 600 proposal labels. Each fold trains on 40 cases and tests on 10 cases; across 5 folds every case is tested exactly once.",
        "",
        "| Setting | Cases | Proposals | Pos/Neg | Stage3 F1 | Scalar F1 | Stump F1 | Stage3 Dice | Scalar Dice | Oracle Dice |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {setting} | {cases} | {proposals} | {positive}/{negative} | {stage3_f1:.6f} | {scalar_f1:.6f} | {stump_f1:.6f} | {stage3_dice_mean:.6f} | {scalar_dice_mean:.6f} | {oracle_dice_mean:.6f} |".format(**row)
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run case-level 5-fold train/test on all 50 medical proposal labels.")
    parser.add_argument("--rows", default="runs/research_direction/matcher_medical_expanded_50/casebook/medical_casebook_rows.csv")
    parser.add_argument("--out-dir", default="runs/research_direction/feedback_learned_matcher_mainline/case_split_50")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [enrich(row) for row in read_csv(resolve(args.rows))]
    folds = make_stratified_case_folds(rows, args.folds)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fold_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_meta: dict[str, Any] = {"rows": rel(resolve(args.rows)), "fold_count": args.folds, "settings": {}}
    summary_rows: list[dict[str, Any]] = []
    for setting in ("overall_5fold", "domain_5fold"):
        fold_rows, predictions, meta = run_setting(rows, folds, setting)
        all_fold_rows.extend(fold_rows)
        all_predictions.extend(predictions)
        all_meta["settings"][setting] = meta
        summary_rows.append(summarize(setting, fold_rows, predictions))

    write_csv(
        out_dir / "case_split_50_folds.csv",
        all_fold_rows,
        [
            "setting",
            "fold",
            "train_cases",
            "test_cases",
            "train_proposals",
            "test_proposals",
            "test_positive",
            "test_negative",
            "stage3_f1",
            "scalar_f1",
            "stump_f1",
            "stage3_dice_mean",
            "scalar_dice_mean",
            "stump_dice_mean",
            "oracle_dice_mean",
            "test_case_names",
        ],
    )
    write_csv(
        out_dir / "case_split_50_predictions.csv",
        all_predictions,
        [
            "setting",
            "fold",
            "case_name",
            "dataset",
            "group_id",
            "label",
            "dice",
            "stage3_keep",
            "scalar_keep",
            "stump_keep",
            "mask_path",
            "overlay_path",
        ],
    )
    write_csv(
        out_dir / "case_split_50_summary.csv",
        summary_rows,
        [
            "setting",
            "folds",
            "cases",
            "proposals",
            "positive",
            "negative",
            "stage3_f1",
            "scalar_f1",
            "stump_f1",
            "stage3_dice_mean",
            "scalar_dice_mean",
            "stump_dice_mean",
            "oracle_dice_mean",
            "test_count_per_case",
        ],
    )
    (out_dir / "case_split_50_models.json").write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(out_dir, summary_rows)
    print(f"Wrote {rel(out_dir / 'case_split_50_summary.csv')}")


if __name__ == "__main__":
    main()
