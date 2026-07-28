from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SIZE = 518
PATCH_SIZE = 14
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
DEFAULT_SEED = "route-b-ddr-failure-taxonomy-v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            pixels: list[tuple[int, int]] = []
            stack = [(x, y)]
            seen[y, x] = True
            while stack:
                cx, cy = stack.pop()
                pixels.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            components.append(pixels)
    return components


def _point_precision(points: list[tuple[int, int]], mask: np.ndarray) -> float:
    if not points:
        return 0.0
    return sum(bool(mask[y, x]) for x, y in points) / len(points)


def _component_coverage(points: list[tuple[int, int]], mask: np.ndarray) -> float:
    components = _components(mask)
    if not components:
        return 0.0
    covered = 0
    for component in components:
        if any((x, y) in set(component) for x, y in points):
            covered += 1
    return covered / len(components)


def _random_grid_points(query_unit_id: str, *, count: int, seed: str) -> list[tuple[int, int]]:
    rng = random.Random(f"{seed}:{query_unit_id}")
    all_centers = [
        (min(IMAGE_SIZE - 1, col * PATCH_SIZE + PATCH_SIZE // 2), min(IMAGE_SIZE - 1, row * PATCH_SIZE + PATCH_SIZE // 2))
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    ]
    return rng.sample(all_centers, count)


def _flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if float(row["point_precision"]) == 0.0:
        flags.append("no_candidate_hit")
    if float(row["component_coverage"]) == 0.0:
        flags.append("no_component_covered")
    if float(row["point_precision"]) <= float(row["random_point_precision"]):
        flags.append("random_not_beaten")
    if float(row["mask_area_ratio"]) < 0.005:
        flags.append("small_sparse_target")
    if int(row["component_count"]) > 30:
        flags.append("fragmented_target")
    if float(row["s_fg_top_margin"]) < 0.02:
        flags.append("low_score_margin")
    if float(row["s_fg_entropy"]) > 0.995:
        flags.append("diffuse_matching_map")
    return flags


def analyze_route_b_ddr_failure_taxonomy(
    *,
    project_root: Path = PROJECT_ROOT,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    smoke_dir = project_root / "reports" / "route_b_ddr_stratified_matcher"
    query_metrics = _read_csv(smoke_dir / "query_metrics.csv")
    selected = {row["query_unit_id"]: row for row in _read_csv(smoke_dir / "selected_episodes.csv")}
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for metric in query_metrics:
        selected_row = selected.get(metric["query_unit_id"])
        if selected_row is None:
            issues.append({"query_unit_id": metric["query_unit_id"], "issue": "missing_selected_episode_row"})
            continue
        mask_path = Path(selected_row.get("query_mask_path", ""))
        if "\\test\\" in str(mask_path).replace("/", "\\").lower():
            issues.append({"query_unit_id": metric["query_unit_id"], "issue": "test_path_detected"})
            continue
        mask = _load_mask(mask_path)
        random_points = _random_grid_points(metric["query_unit_id"], count=int(metric["positive_count"]), seed=seed)
        components = _components(mask)
        foreground = int(mask.sum())
        row = {
            "dataset": "DDR",
            "split": metric["split"],
            "intent": metric["intent"],
            "query_unit_id": metric["query_unit_id"],
            "query_image_id": metric["query_image_id"],
            "point_precision": float(metric["point_precision"]),
            "component_coverage": float(metric["component_coverage"]),
            "random_point_precision": _point_precision(random_points, mask),
            "random_component_coverage": _component_coverage(random_points, mask),
            "mask_foreground_pixels": foreground,
            "mask_area_ratio": foreground / float(IMAGE_SIZE * IMAGE_SIZE),
            "component_count": len(components),
            "s_fg_top_score": float(metric["s_fg_top_score"]),
            "s_fg_top_margin": float(metric["s_fg_top_margin"]),
            "s_fg_entropy": float(metric["s_fg_entropy"]),
            "uses_query_gt_for_generation": False,
            "formal_test_row": False,
        }
        row["failure_flags"] = ";".join(_flags(row))
        rows.append(row)

    flag_counts: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(flag for flag in row["failure_flags"].split(";") if flag)
    status = "PASS_ROUTE_B_DDR_FAILURE_TAXONOMY" if rows and not issues else "FAIL_ROUTE_B_DDR_FAILURE_TAXONOMY"
    summary = {
        "schema_version": "adapromptset.route_b.ddr_failure_taxonomy.v1",
        "date": "2026-07-19",
        "status": status,
        "route": "B_reference_promptability_benchmark",
        "dataset": "DDR",
        "source": "reports/route_b_ddr_stratified_matcher/query_metrics.csv",
        "n": len(rows),
        "formal_test_access": False,
        "test_files_read": False,
        "uses_query_gt_for_generation": False,
        "uses_query_gt_for_evaluation": True,
        "random_control_seed": seed,
        "mean_point_precision": mean(float(row["point_precision"]) for row in rows) if rows else 0.0,
        "mean_random_point_precision": mean(float(row["random_point_precision"]) for row in rows) if rows else 0.0,
        "mean_component_coverage": mean(float(row["component_coverage"]) for row in rows) if rows else 0.0,
        "mean_random_component_coverage": mean(float(row["random_component_coverage"]) for row in rows) if rows else 0.0,
        "failure_flag_counts": dict(sorted(flag_counts.items())),
        "issues": issues,
        "claim_boundary": "control_aware_development_taxonomy_not_benchmark_result",
        "next_recommended_action": "register Route B failure-taxonomy benchmark assets and add prompt-interface controls before any formal test",
    }
    out_dir = project_root / "reports" / "route_b_ddr_failure_taxonomy"
    _write_csv(out_dir / "failure_taxonomy.csv", rows)
    _write_json(out_dir / "failure_taxonomy_summary.json", summary)
    _write_markdown(out_dir / "failure_taxonomy_summary.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Route B DDR Failure Taxonomy",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Rows: `{summary['n']}`",
        f"Formal test access: `{str(summary['formal_test_access']).lower()}`",
        f"DDR test files read: `{str(summary['test_files_read']).lower()}`",
        "",
        "## Aggregate",
        "",
        f"- Mean DINOv2 point precision: `{summary['mean_point_precision']}`",
        f"- Mean random point precision: `{summary['mean_random_point_precision']}`",
        f"- Mean DINOv2 component coverage: `{summary['mean_component_coverage']}`",
        f"- Mean random component coverage: `{summary['mean_random_component_coverage']}`",
        "",
        "## Failure Flags",
        "",
    ]
    for flag, count in summary["failure_flag_counts"].items():
        lines.append(f"- `{flag}`: `{count}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a control-aware development taxonomy, not a formal benchmark result.",
            "- Query GT is used only for evaluation and failure taxonomy labels.",
            "- DDR test remains locked.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Route B DDR stratified matcher failures with random controls.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    analyze_route_b_ddr_failure_taxonomy(project_root=args.project_root, seed=args.seed)


if __name__ == "__main__":
    main()
