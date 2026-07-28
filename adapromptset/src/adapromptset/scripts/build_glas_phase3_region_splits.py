from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "manifests" / "phase2" / "glas" / "case_assignments.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "manifests" / "phase3"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase3_glas" / "split"
DEFAULT_SEED = 20260719
DEFAULT_CALIBRATION_COUNT = 40
DEFAULT_SELECTION_COUNT = 15
DEFAULT_REFERENCES_PER_QUERY = 3


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_glass_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [
        WORKSPACE_ROOT / "external" / "research_data" / "extracted" / "GlaS" / path,
        WORKSPACE_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_text)


def _component_count(mask_path: Path) -> int:
    image = Image.open(mask_path).convert("L")
    arr = np.asarray(image) > 0
    seen = np.zeros_like(arr, dtype=bool)
    height, width = arr.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not arr[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(x, y)]
            seen[y, x] = True
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and arr[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
    return count


def _bin_component_count(values: list[int], value: int) -> str:
    ordered = sorted(values)
    low = ordered[len(ordered) // 3]
    high = ordered[(2 * len(ordered)) // 3]
    if value <= low:
        return "low_component_count"
    if value <= high:
        return "mid_component_count"
    return "high_component_count"


def _stratified_select_selection(rows: list[dict[str, Any]], selection_count: int, seed: int) -> set[str]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["grade"], row["component_count_bin"])].append(row)
    raw_targets = {key: len(items) * selection_count / len(rows) for key, items in strata.items()}
    targets = {key: int(raw_targets[key]) for key in strata}
    while sum(targets.values()) < selection_count:
        key = max(strata, key=lambda item: (raw_targets[item] - targets[item], len(strata[item]), item))
        targets[key] += 1
    while sum(targets.values()) > selection_count:
        key = max(targets, key=lambda item: (targets[item] - raw_targets[item], targets[item], item))
        targets[key] -= 1
    selected: set[str] = set()
    for key, items in sorted(strata.items()):
        ordered = sorted(items, key=lambda row: row["case_id"])
        random.Random(f"{seed}:phase3-selection:{key[0]}:{key[1]}").shuffle(ordered)
        selected.update(row["case_id"] for row in ordered[: targets[key]])
    return selected


def _reference_rows(rows: list[dict[str, Any]], references_per_query: int, seed: int) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for query in sorted(rows, key=lambda row: row["case_id"]):
        eligible = [row for row in rows if row["case_id"] != query["case_id"]]
        same_grade = [row for row in eligible if row["grade"] == query["grade"]]
        pool = same_grade if len(same_grade) >= references_per_query else eligible
        selected = random.Random(f"{seed}:phase3:{query['phase3_role']}:{query['case_id']}:refs").sample(
            sorted(pool, key=lambda row: row["case_id"]),
            references_per_query,
        )
        for index, support in enumerate(selected, start=1):
            references.append(
                {
                    "episode_id": f"glas_phase3_{query['phase3_role']}_{query['case_id']}_ref{index}_{support['case_id']}",
                    "dataset": "GlaS",
                    "phase3_role": query["phase3_role"],
                    "support_case_id": support["case_id"],
                    "support_phase3_role": support["phase3_role"],
                    "support_grade": support["grade"],
                    "support_image_path": support["image_path"],
                    "support_mask_path": support["mask_path"],
                    "query_case_id": query["case_id"],
                    "query_grade": query["grade"],
                    "query_image_path": query["image_path"],
                    "query_mask_path": query["mask_path"],
                    "query_component_count": query["component_count"],
                    "query_component_count_bin": query["component_count_bin"],
                    "reference_index": index,
                    "grade_matched_reference_fallback": "false" if pool is same_grade else "true",
                }
            )
    return references


def build_outputs(
    *,
    assignments_csv: Path = DEFAULT_ASSIGNMENTS,
    out_dir: Path = DEFAULT_OUT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    seed: int = DEFAULT_SEED,
    calibration_count: int = DEFAULT_CALIBRATION_COUNT,
    selection_count: int = DEFAULT_SELECTION_COUNT,
    references_per_query: int = DEFAULT_REFERENCES_PER_QUERY,
) -> dict[str, Any]:
    assignments = _read_csv(assignments_csv)
    utility = [row for row in assignments if row["phase2_role"] == "utility_train"]
    if len(utility) != calibration_count + selection_count:
        raise ValueError(f"expected {calibration_count + selection_count} utility rows, got {len(utility)}")
    if any(row["source_split"] != "train" for row in utility):
        raise ValueError("phase3 internal split can only use official-train utility rows")
    counts = {row["case_id"]: _component_count(_resolve_glass_path(row["mask_path"])) for row in utility}
    count_values = list(counts.values())
    enriched: list[dict[str, Any]] = []
    for row in utility:
        component_count = counts[row["case_id"]]
        enriched.append(
            {
                **row,
                "component_count": component_count,
                "component_count_bin": _bin_component_count(count_values, component_count),
            }
        )
    selection_ids = _stratified_select_selection(enriched, selection_count, seed)
    output_rows: list[dict[str, Any]] = []
    for row in enriched:
        role = "region_selection" if row["case_id"] in selection_ids else "region_calibration"
        output_rows.append({**row, "phase3_role": role})
    calibration_rows = [row for row in output_rows if row["phase3_role"] == "region_calibration"]
    selection_rows = [row for row in output_rows if row["phase3_role"] == "region_selection"]
    if len(calibration_rows) != calibration_count or len(selection_rows) != selection_count:
        raise AssertionError("unexpected phase3 split counts")
    references = _reference_rows(output_rows, references_per_query, seed)
    artifacts = {
        "calibration_csv": "manifests/phase3/glas_region_calibration_40.csv",
        "selection_csv": "manifests/phase3/glas_region_selection_15.csv",
        "references_csv": "manifests/phase3/glas_region_references.csv",
        "report_json": "reports/phase3_glas/split/region_split_report.json",
        "report_md": "reports/phase3_glas/split/region_split_report.md",
    }
    summary = {
        "schema_version": "adapromptset.phase3.glas_region_split.v1",
        "status": "PASS_PHASE3_GLAS_REGION_40_15_SPLIT",
        "seed": seed,
        "source_assignments": str(assignments_csv),
        "source_assignments_sha256": _sha256(assignments_csv),
        "counts": {
            "region_calibration": len(calibration_rows),
            "region_selection": len(selection_rows),
            "references": len(references),
            "locked_test_cases_used": 0,
        },
        "grade_counts": {
            role: dict(sorted(Counter(row["grade"] for row in output_rows if row["phase3_role"] == role).items()))
            for role in ("region_calibration", "region_selection")
        },
        "component_bin_counts": {
            role: dict(
                sorted(Counter(row["component_count_bin"] for row in output_rows if row["phase3_role"] == role).items())
            )
            for role in ("region_calibration", "region_selection")
        },
        "references_per_query": references_per_query,
        "testA_testB_access": False,
        "artifacts": artifacts,
    }
    _write_csv(out_dir / "glas_region_calibration_40.csv", calibration_rows)
    _write_csv(out_dir / "glas_region_selection_15.csv", selection_rows)
    _write_csv(out_dir / "glas_region_references.csv", references)
    _write_json(report_dir / "region_split_report.json", summary)
    lines = [
        "# GlaS Phase 3 Region Split Report",
        "",
        f"Status: `{summary['status']}`",
        f"Seed: `{seed}`",
        "",
        "## Counts",
        "",
        f"- region calibration: `{len(calibration_rows)}`",
        f"- region selection: `{len(selection_rows)}`",
        f"- reference episodes: `{len(references)}`",
        "- locked testA/testB used: `0`",
        "",
        "## Artifacts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in artifacts.items()],
    ]
    (report_dir / "region_split_report.md").parent.mkdir(parents=True, exist_ok=True)
    (report_dir / "region_split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 3 GlaS region calibration/selection split.")
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(build_outputs(assignments_csv=args.assignments_csv, out_dir=args.out_dir, report_dir=args.report_dir, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
