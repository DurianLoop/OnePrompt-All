from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path(r"D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation")
LESION_INTENTS = ("EX", "HE", "MA", "SE")
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LABEL_SUFFIXES = {".tif", ".tiff", ".png"}


def _files_by_stem(path: Path, suffixes: set[str]) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {
        item.stem: item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in suffixes
    }


def _split_summary(root: Path, split: str) -> dict[str, Any]:
    image_by_stem = _files_by_stem(root / split / "image", IMAGE_SUFFIXES)
    label_by_intent = {
        intent: _files_by_stem(root / split / "label" / intent, LABEL_SUFFIXES)
        for intent in LESION_INTENTS
    }
    missing: list[dict[str, str]] = []
    complete = 0
    for case_id in sorted(image_by_stem):
        case_complete = True
        for intent, labels in label_by_intent.items():
            if case_id not in labels:
                missing.append({"case_id": case_id, "intent": intent})
                case_complete = False
        if case_complete:
            complete += 1
    return {
        "image_count": len(image_by_stem),
        "label_counts": {intent: len(labels) for intent, labels in label_by_intent.items()},
        "complete_case_count": complete,
        "missing_label_count": len(missing),
        "missing_labels": missing[:20],
    }


def analyze_ddr_protocol(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    splits = {split: _split_summary(data_root, split) for split in SPLITS}
    total_missing = sum(summary["missing_label_count"] for summary in splits.values())
    status = "PASS_PROTOCOL_DEFINED" if total_missing == 0 else "FAIL_INCOMPLETE_LABELS"
    return {
        "schema_version": "adapromptset.ddr_protocol_audit.v1",
        "status": status,
        "data_root": str(data_root),
        "splits": splits,
        "total_image_count": sum(summary["image_count"] for summary in splits.values()),
        "total_complete_case_count": sum(summary["complete_case_count"] for summary in splits.values()),
        "total_missing_label_count": total_missing,
        "stage1_union": {
            "name": "lesion_all_union",
            "allowed_use": "infrastructure_only",
            "mask_rule": "binary_union_of_EX_HE_MA_SE",
            "uses": [
                "loader_validation",
                "mask_merge_validation",
                "coordinate_tests",
                "candidate_diagnostics",
                "segmenter_smoke",
                "runtime_estimation",
            ],
            "prohibited_use": "formal_scientific_claim",
        },
        "stage2_four_intents": {
            "name": "four_reference_defined_intents",
            "intents": list(LESION_INTENTS),
            "reference_rule": "reference mask contains exactly one lesion intent",
            "query_rule": "positive-query protocol first; query must contain the same target lesion for the selected intent",
            "metrics": [
                "per_intent_dice",
                "per_intent_iou",
                "macro_average",
                "lesion_size_stratified_results",
                "empty_target_handling_fixed_before_formal_evaluation",
            ],
        },
        "formal_policy_training_allowed": False,
        "allowed_next_actions": [
            "stage1_union_loader_and_mask_merge_smoke",
            "predeclared_stage2_four_intent_protocol",
        ],
        "prohibited_next_actions": [
            "ddr_formal_policy_training_before_pathology_gate_decision",
            "using_idrid_observed_test_behavior_for_ddr_thresholds",
            "claiming_union_mask_as_preferred_formal_task",
        ],
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DDR Protocol Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Stage 1 is infrastructure only; formal policy training remains prohibited before the predeclared gates allow DDR work.",
        "",
        "## Stage 1",
        "",
        f"- Mask rule: `{summary['stage1_union']['mask_rule']}`",
        f"- Allowed use: `{summary['stage1_union']['allowed_use']}`",
        f"- Prohibited use: `{summary['stage1_union']['prohibited_use']}`",
        "",
        "## Stage 2",
        "",
        f"- Intents: `{', '.join(summary['stage2_four_intents']['intents'])}`",
        f"- Reference rule: {summary['stage2_four_intents']['reference_rule']}",
        f"- Query rule: {summary['stage2_four_intents']['query_rule']}",
        "",
        "## Pairing Audit",
        "",
        "| Split | Images | Complete cases | EX | HE | MA | SE | Missing labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = summary["splits"][split]
        counts = item["label_counts"]
        lines.append(
            f"| {split} | {item['image_count']} | {item['complete_case_count']} | "
            f"{counts['EX']} | {counts['HE']} | {counts['MA']} | {counts['SE']} | {item['missing_label_count']} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- DDR union masks may be used for loader, merge, coordinate, candidate, segmenter-smoke, and runtime checks.",
            "- DDR four-intent evaluation must be predeclared before any formal run.",
            "- IDRID observed test behavior must not tune DDR candidates, thresholds, actions, or claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ddr_protocol_audit(
    *,
    project_root: Path = PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    summary = analyze_ddr_protocol(data_root)
    out_dir = project_root / "reports" / "ddr_protocol_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ddr_protocol_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "ddr_protocol_audit.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and record the DDR Stage 1/Stage 2 protocol boundary.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    write_ddr_protocol_audit(project_root=args.project_root, data_root=args.data_root)


if __name__ == "__main__":
    main()
