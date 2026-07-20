from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_MANIFEST = (
    WORKSPACE_ROOT
    / "runs"
    / "research_direction"
    / "dataset_audit"
    / "monuseg_official_manifest.csv"
)
DEFAULT_OUT = (
    WORKSPACE_ROOT
    / "runs"
    / "research_direction"
    / "monuseg_official_screening"
    / "official_monuseg_screening_manifest.csv"
)


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "pilot_id",
        "dataset",
        "domain",
        "support_case_id",
        "support_split",
        "support_image_path",
        "support_mask_path",
        "query_case_id",
        "query_split",
        "query_image_path",
        "query_mask_path",
        "query_width",
        "query_height",
        "query_nuclei_region_count",
        "query_mask_area_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def select_rows(rows: list[dict[str, str]], split: str, limit: int) -> list[dict[str, str]]:
    selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["sample_id"])
    if limit > 0:
        selected = selected[:limit]
    return selected


def build_rows(
    official_rows: list[dict[str, str]],
    support_split: str,
    query_split: str,
    support_limit: int,
    query_limit: int,
) -> list[dict[str, Any]]:
    supports = select_rows(official_rows, support_split, support_limit)
    queries = select_rows(official_rows, query_split, query_limit)
    if not supports:
        raise ValueError(f"No support rows found for split={support_split}")
    if not queries:
        raise ValueError(f"No query rows found for split={query_split}")

    rows: list[dict[str, Any]] = []
    for support in supports:
        for query in queries:
            if support["sample_id"] == query["sample_id"]:
                continue
            rows.append(
                {
                    "pilot_id": safe_id(f"{support['sample_id']}__to__{query['sample_id']}"),
                    "dataset": "MonuSeg",
                    "domain": "pathology_nuclei",
                    "support_case_id": support["sample_id"],
                    "support_split": support["split"],
                    "support_image_path": support["image_path"],
                    "support_mask_path": support["mask_path"],
                    "query_case_id": query["sample_id"],
                    "query_split": query["split"],
                    "query_image_path": query["image_path"],
                    "query_mask_path": query["mask_path"],
                    "query_width": query["width"],
                    "query_height": query["height"],
                    "query_nuclei_region_count": query["nuclei_region_count"],
                    "query_mask_area_ratio": query["mask_area_ratio"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a matcher screening manifest from official MoNuSeg image-level data.")
    parser.add_argument("--official-manifest", default=str(DEFAULT_OFFICIAL_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--support-split", choices=["train", "test"], default="train")
    parser.add_argument("--query-split", choices=["train", "test"], default="test")
    parser.add_argument("--support-limit", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=0, help="0 means all query rows in the selected split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_manifest = resolve_path(args.official_manifest)
    rows = build_rows(
        official_rows=read_csv(official_manifest),
        support_split=args.support_split,
        query_split=args.query_split,
        support_limit=args.support_limit,
        query_limit=args.query_limit,
    )
    out = resolve_path(args.out)
    write_csv(out, rows)
    print(f"wrote {out}")
    print(f"rows={len(rows)} support_split={args.support_split} query_split={args.query_split} support_limit={args.support_limit} query_limit={args.query_limit}")
    if rows:
        print(f"first={rows[0]['pilot_id']}")


if __name__ == "__main__":
    main()
