from __future__ import annotations

# ruff: noqa: I001

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
    build_evidence_metadata,
    extract_dinov2_features,
    load_dinov2_model,
    matching_scores_from_features,
    patch_weights_from_mask,
    score_map_global_features,
)
from adapromptset.matching.runtime import checkpoint_record


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAM_ROOT = PROJECT_ROOT.parent


def read_first_episode(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty episode CSV: {path}")
    return rows[0]


def resolve_episode_paths(row: dict[str, str], workspace_root: Path) -> dict[str, Path]:
    dataset = row.get("dataset", "MoNuSAC")
    dataset_roots = {
        "MoNuSAC": workspace_root / "code" / "datasets" / "MoNuSAC",
        "MoNuSeg": workspace_root / "code" / "datasets" / "MonuSeg" / "official",
        "GlaS": workspace_root / "external" / "research_data" / "extracted" / "GlaS",
    }
    try:
        dataset_root = dataset_roots[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported episode dataset: {dataset}") from exc
    def resolve_path(path_text: str) -> Path:
        path = Path(path_text)
        if path.is_absolute():
            return path
        # Episode manifests store paths relative to their declared dataset
        # root.  Keep that contract even in dry runs where synthetic paths do
        # not exist yet; falling back to ``workspace_root / path`` silently
        # changes MoNuSAC's parent and makes path provenance ambiguous.
        return dataset_root / path

    return {
        "support_image": resolve_path(row["support_image_path"]),
        "support_mask": resolve_path(row["support_mask_path"]),
        "query_image": resolve_path(row["query_image_path"]),
        "query_mask": resolve_path(row["query_mask_path"]),
    }


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def load_mask_tensor(path: Path, *, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    values = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(size[1], size[0])
    return (values > 0).to(torch.float32)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_ledger(project_root: Path, gpu_hours: float) -> None:
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "3-real-matcher-smoke",
                "command": "scripts/run_real_matcher_smoke.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": "Single validation episode DINOv2-L forward smoke; formal Gate C still requires larger pilot.",
            }
        )


def _write_gate_c_status(project_root: Path, status: str) -> None:
    path = project_root / "reports" / "gate_c_status.md"
    lines = [
        "# Gate C Status",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
        "| synthetic candidate kernel smoke | PASS | score-map mechanics only; no query GT used for candidate generation |",
        f"| real DINOv2-L matcher smoke | {status} | single MoNuSAC validation episode only |",
        "| formal matcher candidate gate | PENDING_FULL_CANDIDATE_PILOT | requires predeclared multi-episode candidate quality pilot |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_score_maps(maps: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    return {
        name: score_map_global_features(value.to(dtype=torch.float32))
        for name, value in sorted(maps.items())
    }


def run_smoke(
    *,
    project_root: Path = PROJECT_ROOT,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    device_text: str = "auto",
) -> int:
    workspace_root = project_root.parent
    episodes_csv = episodes_csv or (project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv")
    out_dir = out_dir or (project_root / "reports" / "real_matcher_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    row = read_first_episode(episodes_csv)
    paths = resolve_episode_paths(row, workspace_root)
    matcher_path = workspace_root / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    matcher = checkpoint_record(matcher_path, role="matcher")
    metadata = build_evidence_metadata(
        dataset=row["dataset"],
        episode_id=row["episode_id"],
        support_case_id=row["support_case_id"],
        query_case_id=row["query_case_id"],
        support_image_path=row["support_image_path"],
        support_mask_path=row["support_mask_path"],
        query_image_path=row["query_image_path"],
        matcher_checkpoint_sha256=str(matcher["sha256"]),
        feature_grid_size=IMAGE_SIZE // PATCH_SIZE,
    )
    if dry_run:
        payload = {
            "status": "DRY_RUN",
            "formal_gate_c": "PENDING_FULL_CANDIDATE_PILOT",
            "episode": row["episode_id"],
            "evidence_metadata": metadata,
        }
        (out_dir / "real_matcher_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    device = torch.device("cuda:0" if device_text == "auto" and torch.cuda.is_available() else ("cpu" if device_text == "auto" else device_text))
    start = time.perf_counter()
    model = load_dinov2_model(matcher_path, workspace_root=workspace_root, device=device)
    support_image = load_rgb(paths["support_image"])
    query_image = load_rgb(paths["query_image"])
    support_mask = load_mask_tensor(paths["support_mask"], size=(IMAGE_SIZE, IMAGE_SIZE))
    query_mask = load_mask_tensor(paths["query_mask"], size=(IMAGE_SIZE, IMAGE_SIZE))
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
    metrics = {
        "point_precision": point_precision(candidates, query_mask.bool().tolist()),
        "component_coverage": component_coverage(candidates, query_mask.bool().tolist()),
        "positive_count": len(pools.positives),
        "negative_count": len(pools.negatives),
        "validation_issues": issues,
    }
    torch.save({name: value.to(torch.float32) for name, value in maps.items()}, out_dir / "score_maps.pt")
    _write_csv(
        out_dir / "candidates.csv",
        [
            {
                "kind": item.kind,
                "grid_row": item.grid_index[0],
                "grid_col": item.grid_index[1],
                "x": item.coordinate[0],
                "y": item.coordinate[1],
                "score": item.score,
            }
            for item in candidates
        ],
    )
    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.type == "cuda" else 0.0
    status = "PASS" if not issues and metrics["positive_count"] > 0 else "FAIL"
    payload = {
        "status": status,
        "formal_gate_c": "PENDING_FULL_CANDIDATE_PILOT",
        "episode": row["episode_id"],
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "gpu_hours_estimate": round(gpu_hours, 6),
        "evidence_metadata": metadata,
        "score_map_features": summarize_score_maps(maps),
        "metrics": metrics,
        "artifacts": {
            "score_maps": str(out_dir / "score_maps.pt"),
            "candidates": str(out_dir / "candidates.csv"),
        },
        "notes": [
            "Single validation episode smoke only.",
            "Query mask is used only for diagnostic metrics, not for candidate generation or cache metadata.",
        ],
    }
    (out_dir / "real_matcher_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_gate_c_status(project_root, status)
    _append_ledger(project_root, gpu_hours)
    return 0 if status == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one validation-only DINOv2-L matcher evidence smoke.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    raise SystemExit(
        run_smoke(
            project_root=args.project_root,
            episodes_csv=args.episodes_csv,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
            device_text=args.device,
        )
    )


if __name__ == "__main__":
    main()
