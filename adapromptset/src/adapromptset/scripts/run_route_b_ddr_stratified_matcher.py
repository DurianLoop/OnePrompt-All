from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from adapromptset.candidates.points import (
    build_candidate_pools,
    component_coverage,
    point_precision,
    validate_candidate_pools,
)
from adapromptset.matching.evidence import (
    IMAGE_SIZE,
    PATCH_SIZE,
    extract_dinov2_features,
    load_dinov2_model,
    matching_scores_from_features,
    patch_weights_from_mask,
    score_map_global_features,
)
from adapromptset.matching.runtime import checkpoint_record
from adapromptset.scripts.run_route_b_ddr_matcher_smoke import _load_mask_tensor, _load_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
INTENTS = ("EX", "HE", "MA", "SE")
SPLITS = ("train", "valid")
DEFAULT_QUERIES_PER_INTENT = 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _episode_path(project_root: Path, split: str) -> Path:
    return project_root / "manifests" / "phase4plus" / f"ddr_route_b_{split}_reference_query_episodes.csv"


def _contains_test_path(row: dict[str, str]) -> bool:
    path_values = [
        row.get("query_image_path", ""),
        row.get("query_mask_path", ""),
        row.get("reference_image_path", ""),
        row.get("reference_mask_path", ""),
    ]
    lowered = [value.replace("/", "\\").lower() for value in path_values]
    return any("\\test\\" in value or value.endswith("\\test") for value in lowered)


def _select_rows(project_root: Path, queries_per_intent: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    for split in SPLITS:
        rows = _read_csv(_episode_path(project_root, split))
        first_ref_rows = [row for row in rows if row.get("reference_rank") in {"1", 1}]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sorted(first_ref_rows, key=lambda item: item["query_unit_id"]):
            grouped[row["intent"]].append(row)
        for intent in INTENTS:
            candidates = grouped[intent]
            if len(candidates) < queries_per_intent:
                issues.append({"split": split, "intent": intent, "issue": "insufficient_stratified_queries"})
            selected.extend(candidates[:queries_per_intent])
    return selected, issues


def _score_features(maps: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    return {name: score_map_global_features(value.to(dtype=torch.float32)) for name, value in sorted(maps.items())}


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    precision = [float(row["point_precision"]) for row in rows]
    coverage = [float(row["component_coverage"]) for row in rows]
    by_split_intent: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for intent in INTENTS:
            bucket = [row for row in rows if row["split"] == split and row["intent"] == intent]
            if not bucket:
                continue
            key = f"{split}:{intent}"
            by_split_intent[key] = {
                "n": len(bucket),
                "mean_point_precision": mean(float(row["point_precision"]) for row in bucket),
                "mean_component_coverage": mean(float(row["component_coverage"]) for row in bucket),
            }
    return {
        "n": len(rows),
        "mean_point_precision": mean(precision),
        "median_point_precision": median(precision),
        "mean_component_coverage": mean(coverage),
        "median_component_coverage": median(coverage),
        "zero_precision_fraction": sum(value == 0.0 for value in precision) / len(precision),
        "zero_component_coverage_fraction": sum(value == 0.0 for value in coverage) / len(coverage),
        "by_split_intent": by_split_intent,
    }


def run_route_b_ddr_stratified_matcher(
    *,
    project_root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    queries_per_intent: int = DEFAULT_QUERIES_PER_INTENT,
    dry_run: bool = False,
    device_text: str = "auto",
) -> dict[str, Any]:
    if queries_per_intent < 1:
        raise ValueError("queries_per_intent must be >= 1")
    out_dir = out_dir or (project_root / "reports" / "route_b_ddr_stratified_matcher")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected, selection_issues = _select_rows(project_root, queries_per_intent)
    test_path_issues = [
        {"split": row["split"], "query_unit_id": row["query_unit_id"], "issue": "test_path_detected"}
        for row in selected
        if _contains_test_path(row)
    ]
    issues = selection_issues + test_path_issues
    selected_fieldnames = [
        "split",
        "intent",
        "query_unit_id",
        "query_image_id",
        "reference_unit_id",
        "reference_image_id",
        "reference_source_split",
    ]
    _write_csv(
        out_dir / "selected_episodes.csv",
        [{key: row[key] for key in selected_fieldnames} for row in selected],
        selected_fieldnames,
    )
    matcher_path = project_root.parent / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    base_payload: dict[str, Any] = {
        "schema_version": "adapromptset.route_b.ddr_stratified_matcher.v1",
        "date": "2026-07-19",
        "route": "B_reference_promptability_benchmark",
        "dataset": "DDR",
        "queries_per_intent": queries_per_intent,
        "selected_episode_count": len(selected),
        "selected_counts": dict(Counter(f"{row['split']}:{row['intent']}" for row in selected)),
        "formal_test_access": False,
        "test_files_read": False,
        "uses_query_gt_for_generation": False,
        "uses_query_gt_for_evaluation": not dry_run,
        "segmenter_inference": False,
        "issues": issues,
        "outputs": {
            "selected_episodes": str(out_dir / "selected_episodes.csv"),
            "query_metrics": str(out_dir / "query_metrics.csv"),
        },
    }
    if dry_run:
        payload = {**base_payload, "status": "DRY_RUN"}
        _write_json(out_dir / "stratified_matcher_summary.json", payload)
        return payload
    if issues:
        payload = {**base_payload, "status": "FAIL_ROUTE_B_DDR_STRATIFIED_MATCHER_SELECTION"}
        _write_json(out_dir / "stratified_matcher_summary.json", payload)
        return payload

    matcher = checkpoint_record(matcher_path, role="matcher")
    device = torch.device(
        "cuda:0" if device_text == "auto" and torch.cuda.is_available() else ("cpu" if device_text == "auto" else device_text)
    )
    start = time.perf_counter()
    model = load_dinov2_model(matcher_path, workspace_root=project_root.parent, device=device)
    query_rows: list[dict[str, Any]] = []
    for row in selected:
        support_image = _load_rgb(Path(row["reference_image_path"]))
        query_image = _load_rgb(Path(row["query_image_path"]))
        support_mask = _load_mask_tensor(Path(row["reference_mask_path"]))
        query_mask = _load_mask_tensor(Path(row["query_mask_path"]))
        with torch.inference_mode():
            support_features = extract_dinov2_features(model, support_image, device)
            query_features = extract_dinov2_features(model, query_image, device)
        fg_weights = patch_weights_from_mask(support_mask, patch_size=PATCH_SIZE, foreground=True)
        bg_weights = patch_weights_from_mask(support_mask, patch_size=PATCH_SIZE, foreground=False)
        maps = matching_scores_from_features(
            support_features=support_features.cpu(),
            query_features=query_features.cpu(),
            foreground_weights=fg_weights,
            background_weights=bg_weights,
            grid_size=IMAGE_SIZE // PATCH_SIZE,
        )
        pools = build_candidate_pools(
            maps["s_fg"].tolist(),
            maps["s_bg"].tolist(),
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            positive_limit=5,
            negative_limit=5,
            positive_layout="spread",
            min_positive_distance=4.0,
        )
        candidates = list(pools.positives + pools.negatives)
        validation_issues = validate_candidate_pools(candidates, image_size=(IMAGE_SIZE, IMAGE_SIZE))
        features = _score_features(maps)
        query_rows.append(
            {
                "dataset": "DDR",
                "split": row["split"],
                "intent": row["intent"],
                "query_unit_id": row["query_unit_id"],
                "query_image_id": row["query_image_id"],
                "reference_unit_id": row["reference_unit_id"],
                "reference_image_id": row["reference_image_id"],
                "reference_source_split": row["reference_source_split"],
                "matcher": "DINOv2-L",
                "segmenter": "none",
                "row_family": "direct_matching",
                "representation": "point_candidates_from_matching",
                "point_precision": point_precision(candidates, query_mask.bool().tolist()),
                "component_coverage": component_coverage(candidates, query_mask.bool().tolist()),
                "positive_count": len(pools.positives),
                "negative_count": len(pools.negatives),
                "candidate_validation_issue_count": len(validation_issues),
                "s_fg_top_score": features["s_fg"]["top_score"],
                "s_fg_top_margin": features["s_fg"]["top_margin"],
                "s_fg_entropy": features["s_fg"]["matching_entropy"],
                "s_bg_top_score": features["s_bg"]["top_score"],
                "uses_query_gt_for_generation": False,
                "uses_query_gt_for_evaluation": True,
                "formal_test_row": False,
            }
        )
    elapsed = time.perf_counter() - start
    _write_csv(out_dir / "query_metrics.csv", query_rows)
    status = "PASS_ROUTE_B_DDR_STRATIFIED_MATCHER_SMOKE"
    payload = {
        **base_payload,
        "status": status,
        "matcher": "DINOv2-L",
        "matcher_checkpoint": str(matcher_path),
        "matcher_checkpoint_sha256": matcher["sha256"],
        "device": str(device),
        "runtime_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 if device.type == "cuda" else 0.0,
        "summary": _summaries(query_rows),
        "claim_boundary": "stratified_development_smoke_only_not_benchmark_result",
    }
    _write_json(out_dir / "stratified_matcher_summary.json", payload)
    _write_markdown(out_dir / "stratified_matcher_summary.md", payload)
    return payload


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    lines = [
        "# Route B DDR Stratified Matcher Smoke",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Selected episodes: `{payload['selected_episode_count']}`",
        f"Formal test access: `{str(payload['formal_test_access']).lower()}`",
        f"DDR test files read: `{str(payload['test_files_read']).lower()}`",
        f"Segmenter inference: `{str(payload['segmenter_inference']).lower()}`",
        "",
        "## Aggregate",
        "",
        f"- Mean point precision: `{summary.get('mean_point_precision', 'N/A')}`",
        f"- Mean component coverage: `{summary.get('mean_component_coverage', 'N/A')}`",
        f"- Zero precision fraction: `{summary.get('zero_precision_fraction', 'N/A')}`",
        f"- Zero component coverage fraction: `{summary.get('zero_component_coverage_fraction', 'N/A')}`",
        "",
        "## Boundary",
        "",
        "- This is a stratified development smoke, not a benchmark result.",
        "- Query GT is used only for evaluation metrics.",
        "- DDR test remains locked.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a stratified DDR DINOv2 matcher development smoke.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--queries-per-intent", type=int, default=DEFAULT_QUERIES_PER_INTENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_route_b_ddr_stratified_matcher(
        project_root=args.project_root,
        out_dir=args.out_dir,
        queries_per_intent=args.queries_per_intent,
        dry_run=args.dry_run,
        device_text=args.device,
    )


if __name__ == "__main__":
    main()
