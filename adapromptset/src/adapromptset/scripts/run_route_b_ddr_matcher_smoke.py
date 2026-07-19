from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_EPISODES = Path("manifests/phase4plus/ddr_route_b_valid_reference_query_episodes.csv")


def _read_first_episode(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty episode CSV: {path}")
    row = rows[0]
    if row.get("split") not in {"train", "valid"}:
        raise ValueError(f"disallowed split for Route B DDR smoke: {row.get('split')}")
    return row


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


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def _load_mask_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    values = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    return (values > 0).to(torch.float32)


def _score_features(maps: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    return {name: score_map_global_features(value.to(dtype=torch.float32)) for name, value in sorted(maps.items())}


def run_route_b_ddr_matcher_smoke(
    *,
    project_root: Path = PROJECT_ROOT,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    device_text: str = "auto",
) -> dict[str, Any]:
    workspace_root = project_root.parent
    episodes_csv = episodes_csv or (project_root / DEFAULT_EPISODES)
    out_dir = out_dir or (project_root / "reports" / "route_b_ddr_development_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    row = _read_first_episode(episodes_csv)
    matcher_path = workspace_root / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    matcher = {"sha256": "NOT_COMPUTED_DRY_RUN"} if dry_run else checkpoint_record(matcher_path, role="matcher")
    common_payload: dict[str, Any] = {
        "schema_version": "adapromptset.route_b.ddr_matcher_smoke.v1",
        "date": "2026-07-19",
        "route": "B_reference_promptability_benchmark",
        "dataset": "DDR",
        "status": "DRY_RUN" if dry_run else "RUNNING",
        "episode_csv": str(episodes_csv),
        "split": row["split"],
        "intent": row["intent"],
        "query_unit_id": row["query_unit_id"],
        "query_image_id": row["query_image_id"],
        "reference_unit_id": row["reference_unit_id"],
        "reference_image_id": row["reference_image_id"],
        "reference_source_split": row["reference_source_split"],
        "matcher": "DINOv2-L",
        "matcher_checkpoint": str(matcher_path),
        "matcher_checkpoint_sha256": matcher["sha256"],
        "formal_test_access": False,
        "test_files_read": False,
        "uses_query_gt_for_generation": False,
        "uses_query_gt_for_evaluation": not dry_run,
        "segmenter_inference": False,
    }
    if dry_run:
        _write_json(out_dir / "route_b_ddr_matcher_smoke.json", common_payload)
        return common_payload

    device = torch.device(
        "cuda:0" if device_text == "auto" and torch.cuda.is_available() else ("cpu" if device_text == "auto" else device_text)
    )
    start = time.perf_counter()
    model = load_dinov2_model(matcher_path, workspace_root=workspace_root, device=device)
    support_image = _load_rgb(Path(row["reference_image_path"]))
    query_image = _load_rgb(Path(row["query_image_path"]))
    support_mask = _load_mask_tensor(Path(row["reference_mask_path"]))
    query_mask = _load_mask_tensor(Path(row["query_mask_path"]))
    with torch.inference_mode():
        support_features = extract_dinov2_features(model, support_image, device)
        query_features = extract_dinov2_features(model, query_image, device)
    fg_weights = patch_weights_from_mask(support_mask, patch_size=PATCH_SIZE, foreground=True)
    bg_weights = patch_weights_from_mask(support_mask, patch_size=PATCH_SIZE, foreground=False)
    grid_size = IMAGE_SIZE // PATCH_SIZE
    maps = matching_scores_from_features(
        support_features=support_features.cpu(),
        query_features=query_features.cpu(),
        foreground_weights=fg_weights,
        background_weights=bg_weights,
        grid_size=grid_size,
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
    issues = validate_candidate_pools(candidates, image_size=(IMAGE_SIZE, IMAGE_SIZE))
    elapsed = time.perf_counter() - start
    candidate_rows = [
        {
            "kind": item.kind,
            "grid_row": item.grid_index[0],
            "grid_col": item.grid_index[1],
            "x": item.coordinate[0],
            "y": item.coordinate[1],
            "score": item.score,
        }
        for item in candidates
    ]
    _write_csv(out_dir / "route_b_ddr_matcher_candidates.csv", candidate_rows)
    payload = {
        **common_payload,
        "status": "PASS_ROUTE_B_DDR_MATCHER_SMOKE" if not issues else "FAIL_ROUTE_B_DDR_MATCHER_SMOKE",
        "device": str(device),
        "runtime_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 if device.type == "cuda" else 0.0,
        "score_map_features": _score_features(maps),
        "candidate_metrics": {
            "point_precision": point_precision(candidates, query_mask.bool().tolist()),
            "component_coverage": component_coverage(candidates, query_mask.bool().tolist()),
            "positive_count": len(pools.positives),
            "negative_count": len(pools.negatives),
            "validation_issues": issues,
        },
        "outputs": {
            "candidate_csv": str(out_dir / "route_b_ddr_matcher_candidates.csv"),
        },
        "claim_boundary": "development_smoke_only_not_benchmark_result",
    }
    _write_json(out_dir / "route_b_ddr_matcher_smoke.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-episode Route B DDR DINOv2 matcher development smoke.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_route_b_ddr_matcher_smoke(
        project_root=args.project_root,
        episodes_csv=args.episodes_csv,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        device_text=args.device,
    )


if __name__ == "__main__":
    main()
