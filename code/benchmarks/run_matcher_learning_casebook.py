from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from matcher_feedback_learning import apply_scalar_policy, learn_best_scalar_policy, score_stage3_proposal


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str, base: Path | None = None) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        if base is not None and (base / path).exists():
            path = base / path
        else:
            path = WORKSPACE_ROOT / path
    return path.resolve()


def load_proposal_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("proposals", "metrics", "feature_metrics"):
        if isinstance(payload.get(key), list):
            return payload, payload[key]
    raise ValueError(f"No proposal list found in {path}")


def labels_from_payload(label_payload: dict[str, Any]) -> dict[int, int]:
    labels: dict[int, int] = {}
    for value in label_payload.get("positive_group_ids", []):
        labels[int(value)] = 1
    for value in label_payload.get("negative_group_ids", []):
        labels[int(value)] = 0
    return labels


def pseudo_labels_from_retained(payload: dict[str, Any], proposals: list[dict[str, Any]]) -> dict[int, int]:
    retained = {int(value) for value in payload.get("retained_group_ids", [])}
    labels: dict[int, int] = {}
    for proposal in proposals:
        group_id = int(proposal.get("group_id", -1))
        if group_id in retained:
            labels[group_id] = 1
        elif proposal.get("reject_reasons") or proposal.get("filter_status") == "rejected":
            labels[group_id] = 0
    return labels


def load_case_rows(case: dict[str, Any], cases_base: Path, use_pseudo_labels: bool) -> list[dict[str, Any]]:
    case_name = str(case["case_name"])
    feature_path = resolve_path(str(case["stage3_feature_json"]), cases_base)
    payload, proposals = load_proposal_payload(feature_path)
    if case.get("labels_json"):
        labels_path = resolve_path(str(case["labels_json"]), cases_base)
        labels = labels_from_payload(json.loads(labels_path.read_text(encoding="utf-8")))
        label_source = str(labels_path)
    elif use_pseudo_labels:
        labels = pseudo_labels_from_retained(payload, proposals)
        label_source = "pseudo_retained_vs_rejected"
    else:
        labels = {}
        label_source = "none"

    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        group_id = int(proposal.get("group_id", -1))
        if group_id not in labels:
            continue
        row = dict(proposal)
        row["case_name"] = case_name
        row["group_id"] = group_id
        row["label"] = int(labels[group_id])
        row["label_source"] = label_source
        row["feedback_learning_score"] = score_stage3_proposal(row)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a cross-case Matcher proposal policy from labeled Stage3 outputs.")
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--feature-keys", default="feedback_learning_score,instance_similarity,sam_score,cluster_coverage")
    parser.add_argument("--use-pseudo-labels", action="store_true")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_learning_casebook")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_json = resolve_path(args.cases_json)
    cases_payload = json.loads(cases_json.read_text(encoding="utf-8"))
    cases = cases_payload.get("cases", [])
    if not cases:
        raise ValueError("--cases-json must contain a non-empty 'cases' list")
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(load_case_rows(case, cases_json.parent, args.use_pseudo_labels))
    if not rows:
        raise ValueError("No labeled rows found")
    feature_keys = [key.strip() for key in args.feature_keys.split(",") if key.strip()]
    learned = learn_best_scalar_policy(rows, feature_keys=feature_keys)
    applied = apply_scalar_policy(rows, learned["best_policy"])
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_name",
        "group_id",
        "label",
        "feedback_learning_score",
        "instance_similarity",
        "sam_score",
        "cluster_coverage",
        "prompt_coverage",
        "mask_area_pct",
        "bbox_area_pct",
        "policy_score",
        "policy_keep",
        "label_source",
    ]
    with (out_dir / "casebook_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(applied)

    result = {
        "cases_json": str(cases_json.relative_to(WORKSPACE_ROOT)) if cases_json.is_relative_to(WORKSPACE_ROOT) else str(cases_json),
        "case_count": len(cases),
        "labeled_count": len(rows),
        "positive_count": sum(1 for row in rows if int(row["label"]) == 1),
        "negative_count": sum(1 for row in rows if int(row["label"]) == 0),
        "feature_keys": feature_keys,
        "best_policy": learned["best_policy"],
        "all_policies": learned["policies"],
        "selected_rows": [
            {"case_name": row["case_name"], "group_id": row["group_id"], "policy_score": row["policy_score"]}
            for row in applied
            if row["policy_keep"]
        ],
        "artifacts": {
            "casebook_rows_csv": str((out_dir / "casebook_rows.csv").relative_to(WORKSPACE_ROOT)) if out_dir.is_relative_to(WORKSPACE_ROOT) else str(out_dir / "casebook_rows.csv"),
            "summary": str((out_dir / "summary.md").relative_to(WORKSPACE_ROOT)) if out_dir.is_relative_to(WORKSPACE_ROOT) else str(out_dir / "summary.md"),
        },
    }
    (out_dir / "casebook_policy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Matcher Learning Casebook",
        "",
        f"- Cases: `{result['case_count']}`",
        f"- Labeled rows: `{result['labeled_count']}`",
        f"- Positive / negative: `{result['positive_count']}` / `{result['negative_count']}`",
        f"- Best feature: `{result['best_policy']['feature']}`",
        f"- Best threshold: `{result['best_policy']['threshold']}`",
        f"- F1: `{result['best_policy']['f1']}`",
        f"- Selected rows: `{len(result['selected_rows'])}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'casebook_policy.json'}")


if __name__ == "__main__":
    main()
