from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
IMAGE_SIZE = 518
GRID = {
    "contrast_quantile": [0.70, 0.80, 0.90],
    "min_component_area_ratio": [0.001, 0.005],
    "closing_radius": [0, 3, 5],
    "top_component_count": [1, 3],
}
Q_TO_LIMIT = {0.70: 20, 0.80: 10, 0.90: 5}
BOX_FAMILIES = ("R03_automatic_union_box", "R04_automatic_largest_component_box", "R05_automatic_top3_component_boxes")
MASK_FAMILIES = ("R00_direct_matching_mask",)


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


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _load_mask(path_text: str) -> np.ndarray:
    image = Image.open(_resolve_glass_path(path_text)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def _components(mask: np.ndarray) -> list[dict[str, Any]]:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    records: list[dict[str, Any]] = []
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
            xs = [px for px, _ in pixels]
            ys = [py for _, py in pixels]
            records.append({"area": len(pixels), "bbox": (min(xs), min(ys), max(xs), max(ys)), "pixels": pixels})
    return sorted(records, key=lambda row: row["area"], reverse=True)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _box_mask(box: tuple[int, int, int, int] | None) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    if box is None:
        return mask
    x0, y0, x1, y1 = box
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _box_iou(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _random_box_like(box: tuple[int, int, int, int] | None, seed_text: str) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    rng = random.Random(seed_text)
    rx0 = rng.randrange(0, max(1, IMAGE_SIZE - width + 1))
    ry0 = rng.randrange(0, max(1, IMAGE_SIZE - height + 1))
    return rx0, ry0, rx0 + width - 1, ry0 + height - 1


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int((pred & target).sum()) / denom


def _recall(pred: np.ndarray, target: np.ndarray) -> float:
    target_sum = int(target.sum())
    return int((pred & target).sum()) / target_sum if target_sum else 0.0


def _precision(pred: np.ndarray, target: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    return int((pred & target).sum()) / pred_sum if pred_sum else 0.0


def _component_coverage(pred: np.ndarray, target: np.ndarray) -> float:
    comps = _components(target)
    if not comps:
        return 0.0
    covered = 0
    for comp in comps:
        if any(pred[y, x] for x, y in comp["pixels"]):
            covered += 1
    return covered / len(comps)


def _candidate_points(candidate_rows: list[dict[str, str]], query_case_id: str, positive_limit: int) -> list[tuple[int, int, float]]:
    rows = [
        row
        for row in candidate_rows
        if row["episode_id"].startswith(f"glas_phase2_val_{query_case_id}_")
        or row["episode_id"].startswith(f"glas_phase3_region_calibration_{query_case_id}_")
        or row["episode_id"].startswith(f"glas_phase3_region_selection_{query_case_id}_")
    ]
    rows = [row for row in rows if row["kind"] == "positive" and int(row["positive_limit"]) == positive_limit]
    scored: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        scored[(int(row["x"]), int(row["y"]))].append(float(row["score"]))
    aggregate = [(x, y, max(scores)) for (x, y), scores in scored.items()]
    return sorted(aggregate, key=lambda item: item[2], reverse=True)[:positive_limit]


def _proposal_mask(points: list[tuple[int, int, float]], radius: int) -> np.ndarray:
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(image)
    disk_radius = max(8, 10 + radius)
    for x, y, _ in points:
        draw.ellipse((x - disk_radius, y - disk_radius, x + disk_radius, y + disk_radius), fill=255)
    return np.asarray(image) > 0


def _filter_components(mask: np.ndarray, min_area_ratio: float, top_k: int | None = None) -> np.ndarray:
    min_area = int(min_area_ratio * IMAGE_SIZE * IMAGE_SIZE)
    comps = [comp for comp in _components(mask) if comp["area"] >= min_area]
    if top_k is not None:
        comps = comps[:top_k]
    out = np.zeros_like(mask, dtype=bool)
    for comp in comps:
        for x, y in comp["pixels"]:
            out[y, x] = True
    return out


def _config_rows() -> list[dict[str, Any]]:
    rows = []
    for q in GRID["contrast_quantile"]:
        for area in GRID["min_component_area_ratio"]:
            for radius in GRID["closing_radius"]:
                for k in GRID["top_component_count"]:
                    rows.append(
                        {
                            "contrast_quantile": q,
                            "positive_limit": Q_TO_LIMIT[q],
                            "min_component_area_ratio": area,
                            "closing_radius": radius,
                            "top_component_count": k,
                        }
                    )
    return rows


def _proposal_for_family(mask: np.ndarray, family: str, top_component_count: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if family == "R00_direct_matching_mask":
        return mask, []
    comps = _components(mask)
    if family == "R03_automatic_union_box":
        box = _bbox(mask)
        return _box_mask(box), [] if box is None else [box]
    if family == "R04_automatic_largest_component_box":
        box = comps[0]["bbox"] if comps else None
        return _box_mask(box), [] if box is None else [box]
    if family == "R05_automatic_top3_component_boxes":
        boxes = [comp["bbox"] for comp in comps[:top_component_count]]
        out = np.zeros_like(mask, dtype=bool)
        for box in boxes:
            out |= _box_mask(box)
        return out, boxes
    raise ValueError(f"unsupported R1 family: {family}")


def _evaluate_family(
    *,
    episode_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    family: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    unique_queries: dict[str, dict[str, str]] = {}
    for row in episode_rows:
        unique_queries.setdefault(row["query_case_id"], row)
    rows = []
    for query_case_id, row in sorted(unique_queries.items()):
        target = _load_mask(row["query_mask_path"])
        target_box = _bbox(target)
        points = _candidate_points(candidate_rows, query_case_id, int(config["positive_limit"]))
        sparse = _proposal_mask(points, int(config["closing_radius"]))
        sparse = _filter_components(sparse, float(config["min_component_area_ratio"]), top_k=None)
        proposal, boxes = _proposal_for_family(sparse, family, int(config["top_component_count"]))
        proposal_box = _bbox(proposal)
        random_box = _random_box_like(proposal_box, f"phase3-r1:{family}:{query_case_id}")
        random_mask = _box_mask(random_box)
        is_box = family in BOX_FAMILIES
        metric_value = _box_iou(proposal_box, target_box) if is_box else _dice(proposal, target)
        control_value = _box_iou(random_box, target_box) if is_box else _dice(random_mask, target)
        rows.append(
            {
                "query_case_id": query_case_id,
                "query_grade": row.get("query_grade", ""),
                "family": family,
                "metric_type": "box_iou" if is_box else "mask_dice",
                "metric_value": metric_value,
                "control_value": control_value,
                "improvement": metric_value - control_value,
                "win": metric_value > control_value,
                "foreground_recall": _recall(proposal, target),
                "precision": _precision(proposal, target),
                "component_coverage": _component_coverage(proposal, target),
                "empty": int(proposal.sum()) == 0,
                "area_ratio": float(proposal.sum()) / float(IMAGE_SIZE * IMAGE_SIZE),
                "background_spill": float((proposal & ~target).sum()) / float(max(1, proposal.sum())),
                "box_count": len(boxes),
                **config,
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    is_box = family in BOX_FAMILIES
    improvement = mean(row["improvement"] for row in rows)
    win_rate = sum(1 for row in rows if row["win"]) / len(rows)
    recall = mean(row["foreground_recall"] for row in rows)
    empty_rate = sum(1 for row in rows if row["empty"]) / len(rows)
    raw_dice = mean(row["metric_value"] for row in rows)
    if is_box:
        passed = improvement >= 0.10 and win_rate >= 0.70 and recall >= 0.70 and empty_rate <= 0.10
    else:
        passed = raw_dice >= 0.30 and improvement >= 0.15 and recall >= 0.60 and empty_rate <= 0.10
    return {
        "family": family,
        "route": "box" if is_box else "mask",
        "query_count": len(rows),
        "mean_metric": raw_dice,
        "mean_control": mean(row["control_value"] for row in rows),
        "mean_improvement": improvement,
        "win_rate": win_rate,
        "foreground_recall": recall,
        "precision": mean(row["precision"] for row in rows),
        "component_coverage": mean(row["component_coverage"] for row in rows),
        "empty_rate": empty_rate,
        "area_ratio": mean(row["area_ratio"] for row in rows),
        "status": "PASS" if passed else "FAIL",
    }


def _select_configs(selection_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grid_results: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for family in (*MASK_FAMILIES, *BOX_FAMILIES):
        best: tuple[float, float, dict[str, Any]] | None = None
        for config in _config_rows():
            rows = _evaluate_family(episode_rows=selection_rows, candidate_rows=candidate_rows, family=family, config=config)
            summary = _summary(rows, family)
            score = summary["mean_metric"]
            simplicity = -(
                int(config["positive_limit"])
                + int(config["closing_radius"])
                + int(config["top_component_count"])
                + int(float(config["min_component_area_ratio"]) * 1000)
            )
            grid_results.append({**summary, **config, "selection_score": score, "selection_simplicity": simplicity})
            if best is None or score > best[0] + 0.005 or (abs(score - best[0]) <= 0.005 and simplicity > best[1]):
                best = (score, simplicity, dict(config))
        if best is None:
            raise AssertionError("no config evaluated")
        selected[family] = best[2]
    return grid_results, selected


def run_gate_r1(
    *,
    project_root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    phase3_references_csv: Path | None = None,
    internal_candidate_csv: Path | None = None,
    strict_val_episodes_csv: Path | None = None,
    strict_candidate_csv: Path | None = None,
) -> int:
    out_dir = out_dir or project_root / "reports" / "phase3_glas" / "gate_r1_region_proposal"
    phase3_references_csv = phase3_references_csv or project_root / "manifests" / "phase3" / "glas_region_references.csv"
    internal_candidate_csv = internal_candidate_csv or project_root / "reports" / "phase3_glas" / "region_candidate_calibration" / "candidate_pilot_candidates.csv"
    strict_val_episodes_csv = strict_val_episodes_csv or project_root / "manifests" / "phase2" / "glas" / "episodes_val.csv"
    strict_candidate_csv = strict_candidate_csv or project_root / "reports" / "phase2_glas" / "candidates_redesign_v1" / "candidate_pilot_candidates.csv"
    references = _read_csv(phase3_references_csv)
    selection_rows = [row for row in references if row["phase3_role"] == "region_selection"]
    internal_candidates = _read_csv(internal_candidate_csv)
    grid_results, selected = _select_configs(selection_rows, internal_candidates)
    strict_rows = _read_csv(strict_val_episodes_csv)
    strict_candidates = _read_csv(strict_candidate_csv)
    query_rows: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    for family, config in selected.items():
        rows = _evaluate_family(episode_rows=strict_rows, candidate_rows=strict_candidates, family=family, config=config)
        query_rows.extend(rows)
        family_summaries.append({**_summary(rows, family), **config})
    final_state = "PHASE3_GATE_R1_PASS" if any(row["status"] == "PASS" for row in family_summaries) else "STOP_GLAS_REGION_PROPOSAL_FAILURE"
    _write_csv(out_dir / "parameter_grid_results.csv", grid_results)
    _write_json(out_dir / "selected_region_configs.json", selected)
    _write_csv(out_dir / "r1_query_level_metrics.csv", query_rows)
    _write_csv(out_dir / "r1_family_summary.csv", family_summaries)
    summary = {
        "schema_version": "adapromptset.phase3.gate_r1.v1",
        "status": final_state,
        "testA_testB_access": False,
        "selection_pool_query_count": len({row["query_case_id"] for row in selection_rows}),
        "strict_validation_query_count": len({row["query_case_id"] for row in strict_rows}),
        "selected_configs": selected,
        "family_summaries": family_summaries,
        "dense_map_status": "sparse_candidate_evidence_from_frozen_local_peaks_redesign",
        "artifacts": {
            "parameter_grid_results": "reports/phase3_glas/gate_r1_region_proposal/parameter_grid_results.csv",
            "selected_region_configs": "reports/phase3_glas/gate_r1_region_proposal/selected_region_configs.json",
            "query_level_metrics": "reports/phase3_glas/gate_r1_region_proposal/r1_query_level_metrics.csv",
            "family_summary": "reports/phase3_glas/gate_r1_region_proposal/r1_family_summary.csv",
        },
    }
    _write_json(out_dir / "r1_summary.json", summary)
    _write_md(
        out_dir / "r1_summary.md",
        [
            "# GlaS Phase 3 Gate R1 Region Proposal Summary",
            "",
            f"Status: `{final_state}`",
            "",
            "| Family | Route | Metric | Improvement | Win rate | Recall | Empty | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            *[
                f"| {row['family']} | {row['route']} | `{row['mean_metric']:.6f}` | `{row['mean_improvement']:.6f}` | `{row['win_rate']:.6f}` | `{row['foreground_recall']:.6f}` | `{row['empty_rate']:.6f}` | `{row['status']}` |"
                for row in family_summaries
            ],
            "",
            "No SAM inference was run for Gate R1. GlaS testA/testB were not accessed.",
        ],
    )
    return 0 if final_state == "PHASE3_GATE_R1_PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GlaS Phase 3 Gate R1 region proposal quality.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--phase3-references-csv", type=Path, default=None)
    parser.add_argument("--internal-candidate-csv", type=Path, default=None)
    parser.add_argument("--strict-val-episodes-csv", type=Path, default=None)
    parser.add_argument("--strict-candidate-csv", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(
        run_gate_r1(
            project_root=args.project_root,
            out_dir=args.out_dir,
            phase3_references_csv=args.phase3_references_csv,
            internal_candidate_csv=args.internal_candidate_csv,
            strict_val_episodes_csv=args.strict_val_episodes_csv,
            strict_candidate_csv=args.strict_candidate_csv,
        )
    )


if __name__ == "__main__":
    main()
