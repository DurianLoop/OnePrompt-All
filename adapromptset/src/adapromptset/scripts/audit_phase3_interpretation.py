from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_FAMILY = "P3_component_grouped_centroids"


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


def _to_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def _to_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def _group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["segmenter"], row["prompt_family"], row["query_case_id"]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _control_decomposition(query_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in query_rows:
        by_key[(row["segmenter"], row["prompt_family"])].append(row)
    for (segmenter, family), group in sorted(by_key.items()):
        p = [_to_float(row, "privileged_dice") for row in group]
        c1 = [_to_float(row, "c1_random_dice") for row in group]
        c2 = [_to_float(row, "c2_mismatched_dice") for row in group]
        matched = [_to_float(row, "matched_control_dice") for row in group]
        p_minus_c1 = [a - b for a, b in zip(p, c1, strict=True)]
        p_minus_c2 = [a - b for a, b in zip(p, c2, strict=True)]
        c1_minus_c2 = [a - b for a, b in zip(c1, c2, strict=True)]
        rows.append(
            {
                "segmenter": segmenter,
                "prompt_family": family,
                "query_count": len(group),
                "mean_privileged_dice": mean(p),
                "mean_c1_random_dice": mean(c1),
                "mean_c2_alternative_reference_dice": mean(c2),
                "mean_matched_control_dice": mean(matched),
                "mean_privileged_minus_c1": mean(p_minus_c1),
                "mean_privileged_minus_c2": mean(p_minus_c2),
                "mean_c1_minus_c2": mean(c1_minus_c2),
                "win_rate_vs_c1": sum(1 for value in p_minus_c1 if value > 0) / len(group),
                "win_rate_vs_c2": sum(1 for value in p_minus_c2 if value > 0) / len(group),
                "c2_label": "alternative_reference_control",
            }
        )
    return rows


def _area_bias(query_rows: list[dict[str, str]], morphology_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    morphology_by_query = {
        (row["segmenter"], row["query_case_id"]): row
        for row in morphology_rows
        if row.get("prompt_family") == PRIMARY_FAMILY
    }
    rows: list[dict[str, Any]] = []
    for row in query_rows:
        if row["prompt_family"] != PRIMARY_FAMILY:
            continue
        morph = morphology_by_query.get((row["segmenter"], row["query_case_id"]), {})
        target_area = float(morph.get("foreground_area_ratio", "0") or 0)
        pred_area = max(0.0, target_area + _to_float(row, "mask_area_bias"))
        dice = _to_float(row, "privileged_dice")
        intersection = dice * (target_area + pred_area) / 2.0
        false_positive = max(0.0, pred_area - intersection)
        false_negative = max(0.0, target_area - intersection)
        recall = intersection / target_area if target_area > 0 else 0.0
        precision = intersection / pred_area if pred_area > 0 else 0.0
        rows.append(
            {
                "segmenter": row["segmenter"],
                "query_case_id": row["query_case_id"],
                "query_grade": row.get("query_grade", ""),
                "target_area_ratio": target_area,
                "predicted_area_ratio": pred_area,
                "area_bias": _to_float(row, "mask_area_bias"),
                "dice": dice,
                "estimated_intersection_area_ratio": intersection,
                "estimated_false_positive_area_ratio": false_positive,
                "estimated_false_negative_area_ratio": false_negative,
                "estimated_component_recall_proxy": recall,
                "estimated_precision_proxy": precision,
                "component_count": int(float(morph.get("component_count", "0") or 0)),
                "foreground_area_ratio": target_area,
                "privileged_catastrophic": _to_bool(row, "privileged_catastrophic"),
            }
        )
    return rows


def _component_recall(area_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in area_rows:
        rows.append(
            {
                "segmenter": row["segmenter"],
                "query_case_id": row["query_case_id"],
                "component_count": row["component_count"],
                "foreground_area_ratio": row["foreground_area_ratio"],
                "p3_dice": row["dice"],
                "estimated_component_recall_proxy": row["estimated_component_recall_proxy"],
                "estimated_false_positive_area_ratio": row["estimated_false_positive_area_ratio"],
                "estimated_false_negative_area_ratio": row["estimated_false_negative_area_ratio"],
                "note": "proxy_from_dice_and_area_bias_no_group_masks_saved",
            }
        )
    return rows


def _overlap_analysis(query_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in query_rows:
        if row["prompt_family"] != PRIMARY_FAMILY:
            continue
        dice = _to_float(row, "privileged_dice")
        iou = _to_float(row, "privileged_iou")
        rows.append(
            {
                "segmenter": row["segmenter"],
                "query_case_id": row["query_case_id"],
                "p3_dice": dice,
                "p3_iou": iou,
                "dice_iou_consistency_gap": abs(dice - (2 * iou / (1 + iou) if iou > -1 else 0.0)),
                "group_overlap_available": False,
                "note": "Phase2D did not save per-group P3 masks; overlap is limited to fused Dice/IoU consistency.",
            }
        )
    return rows


def _failure_taxonomy(query_rows: list[dict[str, str]], area_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    area_by_key = {(row["segmenter"], row["query_case_id"]): row for row in area_rows}
    rows: list[dict[str, Any]] = []
    for row in query_rows:
        if row["prompt_family"] != PRIMARY_FAMILY:
            continue
        area = area_by_key[(row["segmenter"], row["query_case_id"])]
        margin = _to_float(row, "privileged_margin")
        reason = "non_catastrophic_control_advantage"
        if _to_bool(row, "privileged_catastrophic"):
            if area["predicted_area_ratio"] <= 0.0005:
                reason = "empty_or_tiny_output"
            elif area["predicted_area_ratio"] >= 0.95:
                reason = "image_wide_output"
            elif area["dice"] < 0.20:
                reason = "low_dice_catastrophic"
            else:
                reason = "catastrophic_other"
        elif margin < 0 and area["estimated_false_negative_area_ratio"] > area["estimated_false_positive_area_ratio"]:
            reason = "under_coverage_false_negative_dominant"
        elif margin < 0:
            reason = "control_advantage_without_catastrophe"
        rows.append(
            {
                "segmenter": row["segmenter"],
                "query_case_id": row["query_case_id"],
                "query_grade": row.get("query_grade", ""),
                "p3_margin": margin,
                "p3_dice": area["dice"],
                "component_count": area["component_count"],
                "foreground_area_ratio": area["foreground_area_ratio"],
                "predicted_area_ratio": area["predicted_area_ratio"],
                "false_positive_area_ratio": area["estimated_false_positive_area_ratio"],
                "false_negative_area_ratio": area["estimated_false_negative_area_ratio"],
                "privileged_catastrophic": _to_bool(row, "privileged_catastrophic"),
                "failure_type": reason,
            }
        )
    return rows


def run_audit(
    *,
    project_root: Path = PROJECT_ROOT,
    phase2d_dir: Path | None = None,
    out_dir: Path | None = None,
) -> int:
    phase2d_dir = phase2d_dir or project_root / "reports" / "phase2_glas" / "gate_d_confirmation"
    out_dir = out_dir or project_root / "reports" / "phase3_interpretation_audit"
    query_rows = _read_csv(phase2d_dir / "query_level_metrics.csv")
    morphology_rows = _read_csv(phase2d_dir / "morphology_analysis.csv")

    decomposition_rows = _control_decomposition(query_rows)
    area_rows = _area_bias(query_rows, morphology_rows)
    component_rows = _component_recall(area_rows)
    overlap_rows = _overlap_analysis(query_rows)
    failure_rows = _failure_taxonomy(query_rows, area_rows)

    _write_csv(out_dir / "phase2d_control_decomposition.csv", decomposition_rows)
    _write_csv(out_dir / "p3_area_bias.csv", area_rows)
    _write_csv(out_dir / "p3_component_recall.csv", component_rows)
    _write_csv(out_dir / "p3_overlap_analysis.csv", overlap_rows)
    _write_csv(out_dir / "p3_failure_taxonomy.csv", failure_rows)

    p3_by_segmenter = {row["segmenter"]: row for row in decomposition_rows if row["prompt_family"] == PRIMARY_FAMILY}
    p2_by_segmenter = {
        row["segmenter"]: row
        for row in decomposition_rows
        if row["prompt_family"] == "P2_fixed_multi_centroid_one_group"
    }
    p4_by_segmenter = {
        row["segmenter"]: row
        for row in decomposition_rows
        if row["prompt_family"] == "P4_union_mask_bounding_box"
    }
    fail_counts = Counter(row["failure_type"] for row in failure_rows)
    high_component_rows = [row for row in failure_rows if int(row["component_count"]) >= 3]
    low_component_rows = [row for row in failure_rows if int(row["component_count"]) < 3]
    summary = {
        "schema_version": "adapromptset.phase3.interpretation_audit.v1",
        "status": "PHASE3_INTERPRETATION_AUDIT_COMPLETE",
        "source_phase2d_state": "STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED",
        "interpretation_state": "PIVOT_GLAS_REGION_PROMPT_REQUIRED",
        "scientific_correction": "grouped sparse point prompting failed; region prompting remains viable as indicated by the privileged box upper bound",
        "c2_relabel": "alternative_reference_control",
        "c2_rationale": "C2 uses another GlaS gland reference from the frozen support manifest, so it is not a true negative control.",
        "p3_control_decomposition": p3_by_segmenter,
        "p2_control_decomposition": p2_by_segmenter,
        "p4_box_upper_bound": p4_by_segmenter,
        "failure_type_counts": dict(sorted(fail_counts.items())),
        "high_component_mean_margin": mean(row["p3_margin"] for row in high_component_rows) if high_component_rows else 0.0,
        "low_component_mean_margin": mean(row["p3_margin"] for row in low_component_rows) if low_component_rows else 0.0,
        "p3_group_overlap_limitation": "per-group P3 masks were not saved in Phase 2D; only fused Dice/IoU consistency and area-derived proxies can be audited without rerunning inference.",
        "testA_testB_access": False,
        "artifacts": {
            "control_decomposition": "reports/phase3_interpretation_audit/phase2d_control_decomposition.csv",
            "area_bias": "reports/phase3_interpretation_audit/p3_area_bias.csv",
            "component_recall": "reports/phase3_interpretation_audit/p3_component_recall.csv",
            "overlap_analysis": "reports/phase3_interpretation_audit/p3_overlap_analysis.csv",
            "failure_taxonomy": "reports/phase3_interpretation_audit/p3_failure_taxonomy.csv",
        },
    }
    _write_json(out_dir / "interpretation_summary.json", summary)
    _write_md(
        out_dir / "phase2d_control_decomposition.md",
        [
            "# Phase 2D Control Decomposition",
            "",
            "C2 is interpreted as an alternative-reference control, not a true negative control, because it uses another GlaS gland reference from the frozen support manifest.",
            "",
            "| Segmenter | Family | Priv-C1 | Priv-C2 | C1-C2 | Win vs C1 | Win vs C2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['segmenter']} | {row['prompt_family']} | `{row['mean_privileged_minus_c1']:.6f}` | `{row['mean_privileged_minus_c2']:.6f}` | `{row['mean_c1_minus_c2']:.6f}` | `{row['win_rate_vs_c1']:.6f}` | `{row['win_rate_vs_c2']:.6f}` |"
                for row in decomposition_rows
            ],
        ],
    )
    _write_md(
        out_dir / "interpretation_summary.md",
        [
            "# Phase 3 Interpretation Audit Summary",
            "",
            "Status: `PHASE3_INTERPRETATION_AUDIT_COMPLETE`",
            "",
            "Interpretation state: `PIVOT_GLAS_REGION_PROMPT_REQUIRED`",
            "",
            "The historical operational Gate-D state remains `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED` under the registered P3 grouped-point viability protocol.",
            "",
            "Scientific correction: grouped sparse point prompting failed; region prompting remains viable as indicated by the privileged box upper bound.",
            "",
            "Key evidence:",
            f"- SAM-B P4 mean Dice `{p4_by_segmenter['SAM-B']['mean_privileged_dice']:.6f}`, margin `{p4_by_segmenter['SAM-B']['mean_privileged_minus_c1']:.6f}` vs C1 and `{p4_by_segmenter['SAM-B']['mean_privileged_minus_c2']:.6f}` vs C2.",
            f"- SAM-L P4 mean Dice `{p4_by_segmenter['SAM-L']['mean_privileged_dice']:.6f}`, margin `{p4_by_segmenter['SAM-L']['mean_privileged_minus_c1']:.6f}` vs C1 and `{p4_by_segmenter['SAM-L']['mean_privileged_minus_c2']:.6f}` vs C2.",
            f"- SAM-B P3 mean matched-control margin `{p3_by_segmenter['SAM-B']['mean_privileged_dice'] - p3_by_segmenter['SAM-B']['mean_matched_control_dice']:.6f}`.",
            f"- SAM-L P3 mean matched-control margin `{p3_by_segmenter['SAM-L']['mean_privileged_dice'] - p3_by_segmenter['SAM-L']['mean_matched_control_dice']:.6f}`.",
            "",
            f"Failure taxonomy counts: `{dict(sorted(fail_counts.items()))}`",
            "",
            "No new SAM inference was run for this audit. GlaS testA/testB were not accessed.",
        ],
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 2D controls and grouped-point failure interpretation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--phase2d-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(run_audit(project_root=args.project_root, phase2d_dir=args.phase2d_dir, out_dir=args.out_dir))


if __name__ == "__main__":
    main()
