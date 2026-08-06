"""Organize SURF dataset 2.0 with the legacy OnePrompt-All benchmark.

Creates manifest-only benchmark views.  Source images and GT masks stay in
their audited locations; the evaluation manifest contains only absolute paths
and provenance fields.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surf-v2", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


FIELDS = ["index", "benchmark_split", "source_scope", "source_group", "source_slot", "class_name", "target_instance_count", "image", "union_mask", "instance_mask_dir", "difficulty_level", "episode_id", "source_dataset"]


def legacy_rows(root: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8-sig")))
    return [{
        "benchmark_split": "legacy-220", "source_scope": "legacy", "source_group": row["source_split"], "source_slot": row["source_index"],
        "class_name": row["class_name"], "target_instance_count": row["target_instance_count"], "image": row["image"], "union_mask": row["union_mask"],
        "instance_mask_dir": row["instance_mask_dir"], "difficulty_level": row.get("difficulty_level", "mixed"), "episode_id": row["episode_id"], "source_dataset": row["source_split"],
    } for row in rows]


def scope_rows(root: Path, scope: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for manifest in sorted((root / scope).glob("group_*/manifest.csv")):
        group_root = manifest.parent; group_name = group_root.name
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            result.append({
                "benchmark_split": f"surf-v2-{scope.lower()}", "source_scope": scope.lower(), "source_group": group_name, "source_slot": row["slot"],
                "class_name": row["semantic_class_name"], "target_instance_count": row["target_instance_count"],
                "image": str((group_root / row["image"]).resolve()), "union_mask": str((group_root / row["union_mask"]).resolve()),
                "instance_mask_dir": str((group_root / row["instance_mask_dir"]).resolve()), "difficulty_level": "filtered" if scope == "Filtered" else "unfiltered",
                "episode_id": row["episode_id"], "source_dataset": row["source_dataset"],
            })
    return result


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for index, row in enumerate(rows, 1): writer.writerow({"index": index, **row})


def main() -> None:
    cfg = args(); cfg.out_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_rows(cfg.legacy); filtered = scope_rows(cfg.surf_v2, "Filtered"); unfiltered = scope_rows(cfg.surf_v2, "Unfiltered")
    unfiltered_multi = [row for row in unfiltered if int(row["target_instance_count"]) >= 2]
    oneprompt = legacy + filtered
    catalog = legacy + filtered + unfiltered
    # Two positions per audited group provide a small but class-diverse
    # screening split (40 images) before an hours-long 2220 x 5 full run.
    screening: list[dict[str, str]] = []
    for group in sorted({row["source_group"] for row in filtered}):
        members = [row for row in filtered if row["source_group"] == group]
        screening.extend([members[0], members[len(members) // 2]])
    write_manifest(cfg.out_dir / "oneprompt_multiinstance_2220" / "manifest.csv", oneprompt)
    write_manifest(cfg.out_dir / "all_catalog_4220" / "manifest.csv", catalog)
    write_manifest(cfg.out_dir / "unfiltered_multiinstance_6" / "manifest.csv", unfiltered_multi)
    write_manifest(cfg.out_dir / "filtered_stratified_screening_40" / "manifest.csv", screening)
    summary = {
        "all_catalog_images": len(catalog), "oneprompt_multiinstance_images": len(oneprompt), "legacy_images": len(legacy),
        "filtered_multiinstance_images": len(filtered), "filtered_stratified_screening_images": len(screening), "unfiltered_images": len(unfiltered), "unfiltered_multiinstance_images": len(unfiltered_multi),
        "protocol": "OnePrompt-All evaluation uses only the 2220-image multi-instance manifest; unfiltered single-instance episodes remain catalog-only.",
    }
    (cfg.out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (cfg.out_dir / "README.md").write_text(
        "# SURF v2 unified benchmark\n\n"
        "- `all_catalog_4220`: every audited episode, including unfiltered single-instance examples.\n"
        "- `oneprompt_multiinstance_2220`: the valid OnePrompt-All benchmark (legacy 220 + Filtered 2000).\n"
        "- `filtered_stratified_screening_40`: two fixed episodes per Filtered group for quick batch screening.\n"
        "- `unfiltered_multiinstance_6`: supplementary multi-instance episodes only.\n\n"
        "All manifests are provenance-only and use absolute source paths. Ground truth is used only for reproducible prompt sampling and evaluation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
