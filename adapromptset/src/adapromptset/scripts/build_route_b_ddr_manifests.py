from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path(r"D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation")
LESION_INTENTS = ("EX", "HE", "MA", "SE")
ALLOWED_SPLITS = ("train", "valid")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LABEL_SUFFIXES = {".tif", ".tiff", ".png"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _files_by_stem(path: Path, suffixes: set[str]) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {
        item.stem: item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in suffixes
    }


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _mask_stats(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        mask = image.convert("L")
        width, height = mask.size
        histogram = mask.histogram()
        total = sum(histogram)
        foreground = total - histogram[0]
        return {
            "mask_width": width,
            "mask_height": height,
            "mask_foreground_pixels": foreground,
            "mask_area_ratio": foreground / total if total else 0.0,
            "mask_nonempty": foreground > 0,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_splits(splits: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(split.lower() for split in splits))
    disallowed = sorted(set(normalized) - set(ALLOWED_SPLITS))
    if disallowed:
        raise ValueError(f"Route B DDR development manifests may only use train/valid; disallowed={disallowed}")
    return normalized


def _rows_for_split(data_root: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Disallowed split for DDR development: {split}")

    image_by_stem = _files_by_stem(data_root / split / "image", IMAGE_SUFFIXES)
    label_by_intent = {
        intent: _files_by_stem(data_root / split / "label" / intent, LABEL_SUFFIXES)
        for intent in LESION_INTENTS
    }
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    for image_id, image_path in image_by_stem.items():
        image_width, image_height = _image_size(image_path)
        image_hash = _sha256(image_path)
        for intent in LESION_INTENTS:
            mask_path = label_by_intent[intent].get(image_id)
            if mask_path is None:
                issues.append({"split": split, "image_id": image_id, "intent": intent, "issue": "missing_mask"})
                continue
            mask_stats = _mask_stats(mask_path)
            if (mask_stats["mask_width"], mask_stats["mask_height"]) != (image_width, image_height):
                issues.append({"split": split, "image_id": image_id, "intent": intent, "issue": "dimension_mismatch"})
            rows.append(
                {
                    "dataset": "DDR",
                    "split": split,
                    "protocol_role": "development_train" if split == "train" else "development_validation",
                    "image_id": image_id,
                    "intent": intent,
                    "query_unit_id": f"DDR:{split}:{image_id}:{intent}",
                    "image_path": str(image_path),
                    "image_sha256": image_hash,
                    "mask_path": str(mask_path),
                    "mask_sha256": _sha256(mask_path),
                    "image_width": image_width,
                    "image_height": image_height,
                    **mask_stats,
                    "formal_test_locked": True,
                }
            )
    return rows, issues


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_ids = {row["image_id"] for row in rows}
    positive_rows = [row for row in rows if row["mask_nonempty"]]
    all_counts = Counter(row["intent"] for row in rows)
    positive_counts = Counter(row["intent"] for row in positive_rows)
    return {
        "image_count": len(image_ids),
        "all_intent_rows": len(rows),
        "positive_query_rows": len(positive_rows),
        "all_intent_counts": {intent: all_counts[intent] for intent in LESION_INTENTS},
        "positive_query_counts": {intent: positive_counts[intent] for intent in LESION_INTENTS},
        "mean_positive_area_ratio": (
            sum(float(row["mask_area_ratio"]) for row in positive_rows) / len(positive_rows) if positive_rows else 0.0
        ),
    }


def build_route_b_ddr_manifests(
    *,
    project_root: Path = PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
) -> dict[str, Any]:
    splits = _validate_splits(splits)
    manifest_dir = project_root / "manifests" / "phase4plus"
    report_dir = project_root / "reports" / "route_b_ddr_manifest"
    all_fieldnames = [
        "dataset",
        "split",
        "protocol_role",
        "image_id",
        "intent",
        "query_unit_id",
        "image_path",
        "image_sha256",
        "mask_path",
        "mask_sha256",
        "image_width",
        "image_height",
        "mask_width",
        "mask_height",
        "mask_foreground_pixels",
        "mask_area_ratio",
        "mask_nonempty",
        "formal_test_locked",
    ]

    outputs: dict[str, str] = {}
    split_summaries: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    total_rows: list[dict[str, Any]] = []
    total_positive_rows: list[dict[str, Any]] = []

    for split in splits:
        rows, split_issues = _rows_for_split(data_root, split)
        positive_rows = [row for row in rows if row["mask_nonempty"]]
        issues.extend(split_issues)
        total_rows.extend(rows)
        total_positive_rows.extend(positive_rows)
        all_path = manifest_dir / f"ddr_route_b_{split}_all_intents.csv"
        positive_path = manifest_dir / f"ddr_route_b_{split}_positive_queries.csv"
        _write_csv(all_path, rows, all_fieldnames)
        _write_csv(positive_path, positive_rows, all_fieldnames)
        outputs[f"{split}_all_intents"] = str(all_path)
        outputs[f"{split}_positive_queries"] = str(positive_path)
        split_summaries[split] = _summarize_split(rows)

    zero_positive = [
        {"split": split, "intent": intent, "issue": "zero_positive_queries"}
        for split, summary in split_summaries.items()
        for intent, count in summary["positive_query_counts"].items()
        if count == 0
    ]
    issues.extend(zero_positive)

    status = "PASS_DDR_ROUTE_B_DEV_MANIFEST" if not issues else "FAIL_DDR_ROUTE_B_DEV_MANIFEST"
    manifest_hashes = {name: _sha256(Path(path)) for name, path in outputs.items()}
    summary = {
        "schema_version": "adapromptset.route_b.ddr_dev_manifest.v1",
        "date": "2026-07-19",
        "status": status,
        "route": "B_reference_promptability_benchmark",
        "dataset": "DDR",
        "data_root": str(data_root),
        "generated_splits": list(splits),
        "allowed_splits": list(ALLOWED_SPLITS),
        "formal_test_access": False,
        "test_split_requested": False,
        "test_files_read": False,
        "formal_test_locked": True,
        "intents": list(LESION_INTENTS),
        "positive_query_protocol": True,
        "split_summaries": split_summaries,
        "total_all_intent_rows": len(total_rows),
        "total_positive_query_rows": len(total_positive_rows),
        "total_image_units": len({(row["split"], row["image_id"]) for row in total_rows}),
        "statistical_unit_note": "Primary independent unit is image/case; image-intent rows are repeated task views and must not be treated as independent patients.",
        "issues": issues,
        "outputs": outputs,
        "output_sha256": manifest_hashes,
        "allowed_next_actions": [
            "Route B train/valid benchmark infrastructure",
            "train-only promptability label generation after protocol freeze",
            "valid-only gate evaluation after preregistered metrics",
        ],
        "prohibited_next_actions": [
            "DDR test access",
            "formal DDR claim",
            "threshold/action/prompt tuning from DDR test",
        ],
    }

    _write_json(report_dir / "ddr_route_b_manifest_report.json", summary)
    _write_markdown(report_dir / "ddr_route_b_manifest_report.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Route B DDR Development Manifest Report",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Route: `{summary['route']}`",
        "",
        f"Formal test access: `{str(summary['formal_test_access']).lower()}`",
        f"DDR test files read: `{str(summary['test_files_read']).lower()}`",
        "",
        "## Split Summary",
        "",
        "| Split | Images | All intent rows | Positive queries | EX+ | HE+ | MA+ | SE+ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in summary["generated_splits"]:
        item = summary["split_summaries"][split]
        counts = item["positive_query_counts"]
        lines.append(
            f"| {split} | {item['image_count']} | {item['all_intent_rows']} | {item['positive_query_rows']} | "
            f"{counts['EX']} | {counts['HE']} | {counts['MA']} | {counts['SE']} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Positive-query manifests include only non-empty intent masks.",
            "- All-intent manifests retain empty masks for prevalence accounting.",
            "- DDR test remains locked and unused.",
            "- Image-intent rows are repeated task views of the same image and are not independent patient units.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, output in summary["outputs"].items():
        lines.append(f"- `{name}`: `{output}`")
    if summary["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in summary["issues"]:
            lines.append(f"- `{issue}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Route B DDR train/valid manifests without touching DDR test.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(ALLOWED_SPLITS))
    args = parser.parse_args()
    build_route_b_ddr_manifests(
        project_root=args.project_root,
        data_root=args.data_root,
        splits=tuple(args.splits),
    )


if __name__ == "__main__":
    main()
