from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


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


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def label_reason(label: str, dice: float, iou: float, stage3_keep: str) -> str:
    if label == "accept":
        status = "Stage3 kept" if stage3_keep == "1" else "Stage3 dropped"
        return f"GT Dice {dice:.3f} / IoU {iou:.3f}; overlaps query mask enough for positive proposal; {status}."
    return f"GT Dice {dice:.3f} / IoU {iou:.3f}; insufficient query-mask overlap, background leakage, or incomplete target."


def build_labels(rows: list[dict[str, str]], positive_dice: float) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        dice = to_float(row, "dice")
        iou = to_float(row, "iou")
        label = "accept" if dice >= positive_dice else "reject"
        case_name = row["case_name"]
        group_id = int(row["group_id"])
        cases.setdefault(
            case_name,
            {
                "dataset": row.get("dataset", ""),
                "positive_group_ids": [],
                "negative_group_ids": [],
            },
        )
        if label == "accept":
            cases[case_name]["positive_group_ids"].append(group_id)
        else:
            cases[case_name]["negative_group_ids"].append(group_id)
        labels.append(
            {
                "case_name": case_name,
                "dataset": row.get("dataset", ""),
                "group_id": group_id,
                "label": label,
                "label_source": f"gt_dice_review_ge_{positive_dice:g}",
                "dice": round(dice, 6),
                "iou": round(iou, 6),
                "stage3_keep": int(row.get("stage3_keep", 0)),
                "feedback_learning_score": to_float(row, "feedback_learning_score"),
                "overlay_path": row.get("overlay_path", ""),
                "mask_path": row.get("mask_path", ""),
                "reason": label_reason(label, dice, iou, row.get("stage3_keep", "0")),
            }
        )
    return {
        "label_source": "gt-assisted accept/reject review seed",
        "positive_rule": f"dice >= {positive_dice:g}",
        "case_count": len(cases),
        "proposal_count": len(labels),
        "positive_count": sum(1 for item in labels if item["label"] == "accept"),
        "negative_count": sum(1 for item in labels if item["label"] == "reject"),
        "cases": cases,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build auditable accept/reject labels for medical Matcher proposals.")
    parser.add_argument("--casebook-rows", default="runs/research_direction/matcher_medical_expanded/casebook/medical_casebook_rows.csv")
    parser.add_argument("--out-json", default="runs/research_direction/feedback_learned_matcher_mainline/manual_labels_20.json")
    parser.add_argument("--positive-dice", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_path = resolve_path(args.casebook_rows)
    out_json = resolve_path(args.out_json)
    rows = read_csv(rows_path)
    payload = build_labels(rows, args.positive_dice)
    payload["source_rows"] = rel(rows_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {rel(out_json)}")


if __name__ == "__main__":
    main()
