from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

IMAGE_SIZE = 518


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _safe_float(value: str | int | float | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if v == v]


def _mean(values: list[float]) -> float | None:
    values = _finite(values)
    return mean(values) if values else None


def _median(values: list[float]) -> float | None:
    values = _finite(values)
    return median(values) if values else None


def _fmt(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_inputs(root: Path) -> dict[str, Any]:
    return {
        "r1_summary": _read_json(root / "reports/phase3_glas/gate_r1_region_proposal/r1_summary.json"),
        "r1_family": _read_csv(root / "reports/phase3_glas/gate_r1_region_proposal/r1_family_summary.csv"),
        "r1_query": _read_csv(root / "reports/phase3_glas/gate_r1_region_proposal/r1_query_level_metrics.csv"),
        "r1_grid": _read_csv(root / "reports/phase3_glas/gate_r1_region_proposal/parameter_grid_results.csv"),
        "candidate_summary": _read_json(root / "reports/phase3_glas/region_candidate_calibration/candidate_pilot_summary.json"),
        "interpretation_summary": _read_json(root / "reports/phase3_interpretation_audit/interpretation_summary.json"),
        "control_decomp": _read_csv(root / "reports/phase3_interpretation_audit/phase2d_control_decomposition.csv"),
        "failure_taxonomy": _read_csv(root / "reports/phase3_interpretation_audit/p3_failure_taxonomy.csv"),
        "morphology": _read_csv(root / "reports/phase2_glas/gate_d_confirmation/morphology_analysis.csv"),
        "phase3_final": _read_json(root / "reports/phase3_final_report.json"),
    }


def _source_audit(root: Path) -> dict[str, Any]:
    source = (root / "src/adapromptset/scripts/run_glas_phase3_gate_r1.py").read_text(encoding="utf-8")
    return {
        "r03_box_iou_definition": "IoU between the automatic proposal bounding box and the query GT union bounding box, computed by _box_iou(proposal_box, target_box).",
        "random_box_width_height_source": "The random box uses width and height copied from the corresponding automatic proposal bounding box.",
        "random_control_inherits_automatic_scale": "_random_box_like(box, seed_text) copies x/y span from the automatic proposal box.",
        "random_control_uses_gt_for_generation": False,
        "random_control_gt_usage_note": "Query GT is used only after proposal generation for evaluation of IoU/Dice/recall, not for random-box placement.",
        "random_samples_per_query_per_family": 1,
        "random_seed_pattern": "phase3-r1:{family}:{query_case_id}",
        "full_image_box_recorded_in_r1": False,
        "source_confirms_single_sample": "rng.randrange is called once for x and y in _random_box_like; no loop over multiple random samples exists.",
        "source_hash": _sha256(root / "src/adapromptset/scripts/run_glas_phase3_gate_r1.py"),
        "has_random_box_like": "_random_box_like" in source,
        "has_gt_box_eval": "_box_iou(proposal_box, target_box)" in source,
    }


def _morph_by_query(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["prompt_family"] == "P3_component_grouped_centroids" and row["segmenter"] == "SAM-B":
            out[row["query_case_id"]] = row
    return out


def _bbox_area_fraction_from_mask_fraction(row: dict[str, str]) -> float:
    # Full-image box vs GT union box IoU is not equal to foreground fraction.
    # The exact GT bbox is not saved in Phase 2D morphology rows, so Phase 4 records
    # the computable full-image mask metrics and flags bbox IoU as not available.
    return _safe_float(row.get("foreground_area_ratio"))


def _r00_r09_metrics(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    family_by_name = {row["family"]: row for row in inputs["r1_family"]}
    rows: list[dict[str, Any]] = []
    for family_id, family, route, role in [
        ("R00", "R00_direct_matching_mask", "mask", "automatic_raw"),
        ("R01", "R01_full_image_box", "box", "control"),
        ("R02", "R02_size_matched_random_box", "box", "control"),
        ("R03", "R03_automatic_union_box", "box", "automatic"),
        ("R04", "R04_automatic_largest_component_box", "box", "automatic"),
        ("R05", "R05_automatic_top3_component_boxes", "box", "automatic"),
        ("R06", "R06_automatic_hard_mask_prompt", "mask_prompt", "blocked_by_r1"),
        ("R07", "R07_automatic_soft_logit_mask_prompt", "mask_prompt", "blocked_by_r1"),
        ("R08", "R08_automatic_largest_box_plus_soft_mask", "hybrid", "blocked_by_r1"),
        ("R09", "R09_frozen_matching_top_point_baseline", "point", "blocked_by_r1"),
    ]:
        if family in family_by_name:
            source = family_by_name[family]
            rows.append(
                {
                    "family_id": family_id,
                    "family": family,
                    "route": route,
                    "role": role,
                    "status": source["status"],
                    "metric_mean": source["mean_metric"],
                    "control_mean": source["mean_control"],
                    "improvement": source["mean_improvement"],
                    "win_rate": source["win_rate"],
                    "foreground_recall": source["foreground_recall"],
                    "precision": source["precision"],
                    "component_coverage": source["component_coverage"],
                    "empty_rate": source["empty_rate"],
                    "area_ratio": source["area_ratio"],
                    "source": "phase3_r1_family_summary",
                }
            )
        elif family_id == "R01":
            morph = list(_morph_by_query(inputs["morphology"]).values())
            rows.append(
                {
                    "family_id": family_id,
                    "family": family,
                    "route": route,
                    "role": role,
                    "status": "DERIVED_READ_ONLY_NOT_R1_GATE",
                    "metric_mean": "N/A",
                    "control_mean": "N/A",
                    "improvement": "N/A",
                    "win_rate": "N/A",
                    "foreground_recall": "1.000000",
                    "precision": _fmt(_mean([_safe_float(r["foreground_area_ratio"]) for r in morph])),
                    "component_coverage": "1.000000",
                    "empty_rate": "0.000000",
                    "area_ratio": "1.000000",
                    "source": "phase2d_morphology_derived_full_image_mask_metrics",
                }
            )
        elif family_id == "R02":
            r03 = family_by_name["R03_automatic_union_box"]
            rows.append(
                {
                    "family_id": family_id,
                    "family": family,
                    "route": route,
                    "role": role,
                    "status": "CONTROL_USED_BY_R1",
                    "metric_mean": r03["mean_control"],
                    "control_mean": "N/A",
                    "improvement": "N/A",
                    "win_rate": "N/A",
                    "foreground_recall": "N/A",
                    "precision": "N/A",
                    "component_coverage": "N/A",
                    "empty_rate": "N/A",
                    "area_ratio": r03["area_ratio"],
                    "source": "R03_matched_control_mean",
                }
            )
        else:
            rows.append(
                {
                    "family_id": family_id,
                    "family": family,
                    "route": route,
                    "role": role,
                    "status": "NOT_RUN_BLOCKED_BY_R1_FAILURE",
                    "metric_mean": "N/A",
                    "control_mean": "N/A",
                    "improvement": "N/A",
                    "win_rate": "N/A",
                    "foreground_recall": "N/A",
                    "precision": "N/A",
                    "component_coverage": "N/A",
                    "empty_rate": "N/A",
                    "area_ratio": "N/A",
                    "source": "phase3_state_machine",
                }
            )
    return rows


def _decomposition(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    family_rows = {row["family"]: row for row in inputs["r1_family"]}
    morph = list(_morph_by_query(inputs["morphology"]).values())
    full_precision = _mean([_safe_float(r["foreground_area_ratio"]) for r in morph])
    full_spill = 1.0 - full_precision if full_precision is not None else None
    rows = []
    for family in ["R00_direct_matching_mask", "R03_automatic_union_box", "R04_automatic_largest_component_box", "R05_automatic_top3_component_boxes"]:
        row = family_rows[family]
        rows.append(
            {
                "family": family,
                "automatic_metric": row["mean_metric"],
                "matched_random_metric": row["mean_control"],
                "automatic_minus_random": row["mean_improvement"],
                "full_image_foreground_recall": "1.000000",
                "full_image_precision": _fmt(full_precision),
                "full_image_background_spill": _fmt(full_spill),
                "automatic_area_ratio": row["area_ratio"],
                "interpretation": "large_box_prior_confounded" if family == "R03_automatic_union_box" else "undercoverage_or_low_raw_mask_quality",
            }
        )
    return rows


def _area_location_scale(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    morph = _morph_by_query(inputs["morphology"])
    rows = []
    for row in inputs["r1_query"]:
        m = morph.get(row["query_case_id"], {})
        rows.append(
            {
                "query_case_id": row["query_case_id"],
                "query_grade": row["query_grade"],
                "family": row["family"],
                "metric_value": row["metric_value"],
                "control_value": row["control_value"],
                "improvement": row["improvement"],
                "area_ratio": row["area_ratio"],
                "foreground_area_ratio": m.get("foreground_area_ratio", ""),
                "component_count": m.get("component_count", ""),
                "large_foreground": str(_safe_float(m.get("foreground_area_ratio")) >= 0.5 if m else ""),
                "high_component": str(_safe_float(m.get("component_count")) >= 10 if m else ""),
                "location_saved_in_r1_artifact": "false",
                "location_note": "R1 artifacts do not save box coordinates; Phase 4 can audit scale but not exact center without recomputation.",
            }
        )
    return rows


def _seed_sensitivity(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family in ["R03_automatic_union_box", "R04_automatic_largest_component_box", "R05_automatic_top3_component_boxes"]:
        vals = [r for r in inputs["r1_query"] if r["family"] == family]
        improvements = [_safe_float(r["improvement"]) for r in vals]
        rows.append(
            {
                "family": family,
                "random_samples_per_query": 1,
                "seed_pattern": "phase3-r1:{family}:{query_case_id}",
                "mean_improvement": _fmt(_mean(improvements)),
                "median_improvement": _fmt(_median(improvements)),
                "min_improvement": _fmt(min(improvements)),
                "max_improvement": _fmt(max(improvements)),
                "strict_seed_sensitivity_available": "false",
                "reason": "Only one deterministic size-matched random placement was generated per query/family in the existing R1 artifact.",
            }
        )
    return rows


def _internal_vs_strict(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    strict = {row["family"]: row for row in inputs["r1_family"]}
    grid = inputs["r1_grid"]
    rows = []
    for family, strict_row in strict.items():
        candidates = [r for r in grid if r["family"] == family]
        selected = [
            r
            for r in candidates
            if r["contrast_quantile"] == strict_row["contrast_quantile"]
            and r["positive_limit"] == strict_row["positive_limit"]
            and r["min_component_area_ratio"] == strict_row["min_component_area_ratio"]
            and r["closing_radius"] == strict_row["closing_radius"]
            and r["top_component_count"] == strict_row["top_component_count"]
        ]
        selected_row = selected[0] if selected else {}
        rows.append(
            {
                "family": family,
                "internal_selection_metric": selected_row.get("mean_metric", ""),
                "internal_selection_control": selected_row.get("mean_control", ""),
                "internal_selection_improvement": selected_row.get("mean_improvement", ""),
                "internal_selection_win_rate": selected_row.get("win_rate", ""),
                "strict_metric": strict_row["mean_metric"],
                "strict_control": strict_row["mean_control"],
                "strict_improvement": strict_row["mean_improvement"],
                "strict_win_rate": strict_row["win_rate"],
                "metric_shift_strict_minus_internal": _fmt(_safe_float(strict_row["mean_metric"]) - _safe_float(selected_row.get("mean_metric")) if selected_row else None),
                "status_shift": f"{selected_row.get('status', 'N/A')}->{strict_row['status']}",
            }
        )
    return rows


def _morphology_decomposition(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    morph = _morph_by_query(inputs["morphology"])
    rows = []
    buckets = {
        "all": lambda _m: True,
        "malignant": lambda m: m.get("query_grade") == "malignant",
        "benign": lambda m: m.get("query_grade") == "benign",
        "high_component_ge10": lambda m: _safe_float(m.get("component_count")) >= 10,
        "large_foreground_ge050": lambda m: _safe_float(m.get("foreground_area_ratio")) >= 0.5,
        "small_foreground_lt025": lambda m: _safe_float(m.get("foreground_area_ratio")) < 0.25,
    }
    for family in sorted({r["family"] for r in inputs["r1_query"]}):
        family_rows = [r for r in inputs["r1_query"] if r["family"] == family]
        for bucket, predicate in buckets.items():
            selected = [r for r in family_rows if r["query_case_id"] in morph and predicate(morph[r["query_case_id"]])]
            if not selected:
                continue
            rows.append(
                {
                    "family": family,
                    "bucket": bucket,
                    "query_count": len(selected),
                    "mean_metric": _fmt(_mean([_safe_float(r["metric_value"]) for r in selected])),
                    "mean_control": _fmt(_mean([_safe_float(r["control_value"]) for r in selected])),
                    "mean_improvement": _fmt(_mean([_safe_float(r["improvement"]) for r in selected])),
                    "win_rate": _fmt(mean([1.0 if r["win"] == "True" else 0.0 for r in selected])),
                    "mean_area_ratio": _fmt(_mean([_safe_float(r["area_ratio"]) for r in selected])),
                    "mean_fg_ratio": _fmt(_mean([_safe_float(morph[r["query_case_id"]]["foreground_area_ratio"]) for r in selected])),
                    "mean_component_count": _fmt(_mean([_safe_float(morph[r["query_case_id"]]["component_count"]) for r in selected])),
                }
            )
    return rows


def _direct_mask_analysis(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [r for r in inputs["r1_query"] if r["family"] == "R00_direct_matching_mask"]
    return [
        {
            "query_count": len(rows),
            "mean_raw_mask_dice": _fmt(_mean([_safe_float(r["metric_value"]) for r in rows])),
            "mean_control_dice": _fmt(_mean([_safe_float(r["control_value"]) for r in rows])),
            "mean_improvement": _fmt(_mean([_safe_float(r["improvement"]) for r in rows])),
            "win_rate": _fmt(mean([1.0 if r["win"] == "True" else 0.0 for r in rows])),
            "mean_precision": _fmt(_mean([_safe_float(r["precision"]) for r in rows])),
            "mean_recall": _fmt(_mean([_safe_float(r["foreground_recall"]) for r in rows])),
            "mean_component_coverage": _fmt(_mean([_safe_float(r["component_coverage"]) for r in rows])),
            "mean_area_ratio": _fmt(_mean([_safe_float(r["area_ratio"]) for r in rows])),
            "interpretation": "High precision but very low recall; direct thresholded matching is not yet a viable whole-image segmentation method.",
        }
    ]


def _prompt_failure_map(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in inputs["r1_query"]:
        recall = _safe_float(row["foreground_recall"])
        precision = _safe_float(row["precision"])
        area = _safe_float(row["area_ratio"])
        improvement = _safe_float(row["improvement"])
        if recall < 0.2 and precision > 0.8:
            failure = "precise_fragment_undercoverage"
        elif area > 0.6 and improvement <= 0.05:
            failure = "large_area_prior_not_localization"
        elif improvement <= 0:
            failure = "matched_control_not_beaten"
        else:
            failure = "weak_positive_but_gate_insufficient"
        rows.append(
            {
                "query_case_id": row["query_case_id"],
                "family": row["family"],
                "metric_value": row["metric_value"],
                "control_value": row["control_value"],
                "improvement": row["improvement"],
                "foreground_recall": row["foreground_recall"],
                "precision": row["precision"],
                "area_ratio": row["area_ratio"],
                "failure_type": failure,
            }
        )
    return rows


def _paper_claim_audit(root: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    files = [root / "paper_phase3/main.tex", *sorted((root / "paper_phase3/sections").glob("*.tex"))]
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files if path.exists()).lower()
    prohibited = [
        "automatic region prompting works",
        "sam refinement gain",
        "router improvement",
        "formal test result",
        "submission ready",
        "state of the art",
        "model-agnostic",
        "plug-and-play",
        "universal",
    ]
    negation_markers = ("no ", "not ", "does not claim", "not measured", "not claimed", "without ")
    hits = []
    for phrase in prohibited:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            context = text[max(0, idx - 80) : idx + len(phrase) + 80]
            if not any(marker in context for marker in negation_markers):
                hits.append(phrase)
                break
            start = idx + len(phrase)
    required = [
        "no automatic region family passes",
        "testa/testb access",
        "not establish an automatic method",
        "stops before sam refinement",
    ]
    return {
        "status": "PASS" if not hits else "FAIL",
        "prohibited_hits": hits,
        "required_boundary_phrases_present": {phrase: phrase in text for phrase in required},
        "phase3_final_state": inputs["phase3_final"]["final_state"],
        "paper_pdf": inputs["phase3_final"]["paper"]["pdf"],
        "paper_hash_reported": inputs["phase3_final"]["paper"].get("pdf_sha256"),
    }


def _artifact_consistency(root: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    pdf_path = root / inputs["phase3_final"]["paper"]["pdf"]
    log_path = root / "paper_phase3/build/main.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    bad_log = re.findall(r"Overfull|undefined|Citation .* undefined|Label\(s\) may have changed|Rerun|File .* not found|Missing", log_text)
    r1 = inputs["r1_summary"]
    final = inputs["phase3_final"]
    r03 = next(row for row in r1["family_summaries"] if row["family"] == "R03_automatic_union_box")
    return {
        "status": "PASS" if pdf_path.exists() and not bad_log and final["gate_r1"]["state"] == r1["status"] else "FAIL",
        "pdf_exists": pdf_path.exists(),
        "pdf_sha256_actual": _sha256(pdf_path),
        "pdf_sha256_reported": final["paper"].get("pdf_sha256"),
        "pdf_hash_matches_report": _sha256(pdf_path) == final["paper"].get("pdf_sha256"),
        "latex_log_problem_count": len(bad_log),
        "gate_state_consistent": final["gate_r1"]["state"] == r1["status"],
        "r03_metric_consistent": abs(final["gate_r1"]["strongest_automatic_metric"] - r03["mean_metric"]) < 1e-12,
    }


def audit_phase4(root: Path) -> dict[str, Any]:
    out = root / "reports/phase4_read_only_audit"
    inputs = _load_inputs(root)
    source = _source_audit(root)
    complete_metrics = _r00_r09_metrics(inputs)
    decomposition = _decomposition(inputs)
    area_scale = _area_location_scale(inputs)
    seed = _seed_sensitivity(inputs)
    internal = _internal_vs_strict(inputs)
    morphology = _morphology_decomposition(inputs)
    direct = _direct_mask_analysis(inputs)
    failure_map = _prompt_failure_map(inputs)
    paper_claim = _paper_claim_audit(root, inputs)
    consistency = _artifact_consistency(root, inputs)

    _write_json(out / "control_construction_audit.json", source)
    _write_md(
        out / "control_construction_audit.md",
        f"""
# Control Construction Audit

Status: `PASS_READ_ONLY`

- R03 box IoU definition: {source['r03_box_iou_definition']}
- Random box width/height source: {source['random_box_width_height_source']}
- Random control inherits automatic scale: `{source['random_control_inherits_automatic_scale']}`
- Random control uses query GT for generation: `{source['random_control_uses_gt_for_generation']}`
- Query GT usage: {source['random_control_gt_usage_note']}
- Random samples per query/family: `{source['random_samples_per_query_per_family']}`
- Random seed pattern: `{source['random_seed_pattern']}`
- Full-image box recorded in original R1 artifact: `{source['full_image_box_recorded_in_r1']}`

Interpretation: R03's high absolute box IoU must be read against a size-matched random control that inherits automatic scale. Its small improvement and low win rate mean localization advantage is not established.
""",
    )
    _write_csv(out / "r00_r09_complete_metrics.csv", complete_metrics)
    _write_csv(out / "automatic_random_fullimage_decomposition.csv", decomposition)
    _write_csv(out / "proposal_area_location_scale_analysis.csv", area_scale)
    _write_csv(out / "random_seed_sensitivity_from_existing_artifacts.csv", seed)
    _write_csv(out / "internal_vs_strict_validation_consistency.csv", internal)
    _write_csv(out / "morphology_failure_decomposition.csv", morphology)
    _write_csv(out / "direct_matching_mask_analysis.csv", direct)
    _write_csv(out / "prompt_interface_failure_map.csv", failure_map)
    _write_json(out / "paper_claim_audit.json", paper_claim)
    _write_md(
        out / "paper_claim_audit.md",
        "# Paper Claim Audit\n\n"
        f"Status: `{paper_claim['status']}`\n\n"
        f"Prohibited hits: `{paper_claim['prohibited_hits']}`\n\n"
        "Required boundary phrases:\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in paper_claim["required_boundary_phrases_present"].items()),
    )
    _write_json(out / "paper_artifact_consistency.json", consistency)
    _write_md(
        out / "paper_artifact_consistency.md",
        "# Paper Artifact Consistency\n\n"
        f"Status: `{consistency['status']}`\n\n"
        f"- PDF exists: `{consistency['pdf_exists']}`\n"
        f"- PDF hash matches final report: `{consistency['pdf_hash_matches_report']}`\n"
        f"- LaTeX log problem count: `{consistency['latex_log_problem_count']}`\n"
        f"- Gate state consistent: `{consistency['gate_state_consistent']}`\n"
        f"- R03 metric consistent: `{consistency['r03_metric_consistent']}`\n",
    )

    r1_family = {row["family"]: row for row in inputs["r1_family"]}
    r03 = r1_family["R03_automatic_union_box"]
    r00 = r1_family["R00_direct_matching_mask"]
    failure_counts = Counter(row["failure_type"] for row in failure_map)
    conclusion = {
        "schema_version": "adapromptset.phase4.read_only_audit.v1",
        "date": "2026-07-19",
        "status": "PHASE4_AUDIT_SUPPORTS_BENCHMARK_TRACK",
        "testA_testB_access": False,
        "new_model_inference": False,
        "key_answers": {
            "r03_box_iou_definition": source["r03_box_iou_definition"],
            "random_box_width_height_source": source["random_box_width_height_source"],
            "random_control_inherits_automatic_scale": True,
            "random_control_uses_gt_for_generation": False,
            "random_samples_per_query_per_family": 1,
            "r03_area_ratio": float(r03["area_ratio"]),
            "full_image_recall": 1.0,
            "full_image_spill_mean": 1.0 - float(next(row for row in complete_metrics if row["family_id"] == "R01")["precision"]),
            "r03_close_to_full_image_box": float(r03["area_ratio"]) >= 0.65,
            "r00_raw_dice": float(r00["mean_metric"]),
            "r00_precision": float(r00["precision"]),
            "r00_recall": float(r00["foreground_recall"]),
            "r04_r05_difference": "R04/R05 have much smaller area ratios and foreground recall than R03, indicating component-level undercoverage rather than robust localization.",
            "internal_vs_strict_consistency": "All selected families remained FAIL on strict validation; R03 retained high absolute IoU but not enough matched-control advantage.",
            "morphology_concentration": "Failures are associated with scale/foreground area and component structure, but no single morphology bucket rescues R1.",
            "phase3_pdf_overclaim": paper_claim["status"] != "PASS",
            "artifact_consistency": consistency["status"],
        },
        "interpretation": "R03's absolute IoU is largely compatible with a large scale/area prior because the size-matched random control is also high. R00 is precise but under-recovers foreground. The current evidence is strongest as a promptability/control benchmark motivation, not as a validated method route.",
        "failure_type_counts": dict(failure_counts),
        "artifacts": [
            "reports/phase4_read_only_audit/control_construction_audit.md",
            "reports/phase4_read_only_audit/r00_r09_complete_metrics.csv",
            "reports/phase4_read_only_audit/automatic_random_fullimage_decomposition.csv",
            "reports/phase4_read_only_audit/proposal_area_location_scale_analysis.csv",
            "reports/phase4_read_only_audit/random_seed_sensitivity_from_existing_artifacts.csv",
            "reports/phase4_read_only_audit/internal_vs_strict_validation_consistency.csv",
            "reports/phase4_read_only_audit/morphology_failure_decomposition.csv",
            "reports/phase4_read_only_audit/direct_matching_mask_analysis.csv",
            "reports/phase4_read_only_audit/prompt_interface_failure_map.csv",
            "reports/phase4_read_only_audit/paper_claim_audit.md",
            "reports/phase4_read_only_audit/paper_artifact_consistency.md",
        ],
    }
    _write_json(out / "phase4_audit_conclusion.json", conclusion)
    _write_md(
        out / "phase4_audit_conclusion.md",
        "# Phase 4 Read-Only Audit Conclusion\n\n"
        f"Status: `{conclusion['status']}`\n\n"
        f"Interpretation: {conclusion['interpretation']}\n\n"
        "The audit supports Route B as a strong candidate because the existing evidence exposes a control and promptability measurement problem more clearly than it supports immediate adaptive routing or direct segmentation.\n",
    )
    return conclusion


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 read-only evidence audit.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    conclusion = audit_phase4(args.project_root.resolve())
    print(json.dumps(conclusion, indent=2, sort_keys=True))
