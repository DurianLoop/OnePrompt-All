"""Create a manifest-only merged OnePrompt-All 20+200 benchmark.

No image or annotation is copied.  The manifest uses absolute source paths so
the 220-image benchmark remains auditable and cannot accidentally expose GT to
the interactive application.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banana", type=Path, required=True)
    parser.add_argument("--broccoli", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def source_rows(root: Path, source_split: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8-sig")))
    result: list[dict[str, str]] = []
    for row in rows:
        result.append({
            "source_split": source_split,
            "source_index": row["index"],
            "class_name": row["class_name"],
            "target_instance_count": row["target_instance_count"],
            "image": str((root / row["image"]).resolve()),
            "union_mask": str((root / row["union_mask"]).resolve()),
            "instance_mask_dir": str((root / row["instance_mask_dir"]).resolve()),
            "difficulty_level": row.get("difficulty_level", "mixed"),
            "episode_id": row["episode_id"],
        })
    return result


def main() -> None:
    cfg = args(); cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = source_rows(cfg.banana, "banana-20") + source_rows(cfg.broccoli, "broccoli-200")
    for index, row in enumerate(rows, 1):
        row["index"] = str(index)
    fields = ["index", "source_split", "source_index", "class_name", "target_instance_count", "image", "union_mask", "instance_mask_dir", "difficulty_level", "episode_id"]
    with (cfg.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {
        "name": "OnePrompt-All merged benchmark 220",
        "images": len(rows),
        "splits": {"banana-20": 20, "broccoli-200": 200},
        "classes": {"banana": 20, "broccoli": 200},
        "data_materialization": "manifest only; source images and GT stay in original datasets",
    }
    (cfg.out_dir / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (cfg.out_dir / "README.md").write_text(
        "# OnePrompt-All 220 benchmark\n\n"
        "This manifest merges the 20-image banana and 200-image broccoli episodes. "
        "Each episode is a single image containing multiple instances of one class. "
        "Absolute paths make source provenance explicit; annotations are for click simulation and scoring only.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
