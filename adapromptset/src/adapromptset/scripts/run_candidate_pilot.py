from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from adapromptset.candidates.points import (
    Candidate,
    build_candidate_pools,
    component_coverage,
    point_precision,
    validate_candidate_pools,
)
from adapromptset.evaluation.gate_c import summarize_candidate_pilot
from adapromptset.matching.evidence import (
    IMAGE_SIZE,
    PATCH_SIZE,
    extract_dinov2_features,
    load_dinov2_model,
    matching_scores_from_features,
    patch_weights_from_mask,
)
from adapromptset.matching.runtime import checkpoint_record
from adapromptset.scripts.run_real_matcher_smoke import (
    load_mask_tensor,
    load_rgb,
    resolve_episode_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRECISION_MARGIN = 0.05
POSITIVE_LIMITS = (1, 3, 5)


def dataset_slug(dataset_name: str) -> str:
    known = {"MoNuSAC": "monusac", "MoNuSeg": "monuseg", "GlaS": "glas"}
    return known.get(dataset_name, dataset_name.lower())


def read_episodes(path: Path, max_episodes: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty episode CSV: {path}")
    return rows[:max_episodes]


def random_positive_candidates(
    *,
    seed_text: str,
    count: int,
    image_size: tuple[int, int],
    grid_size: int,
) -> list[Candidate]:
    rng = random.Random(seed_text)
    width, height = image_size
    patch_width = width // grid_size
    patch_height = height // grid_size
    seen: set[tuple[int, int]] = set()
    items: list[Candidate] = []
    while len(items) < count:
        row = rng.randrange(grid_size)
        col = rng.randrange(grid_size)
        if (row, col) in seen:
            continue
        seen.add((row, col))
        x = min(width - 1, col * patch_width + patch_width // 2)
        y = min(height - 1, row * patch_height + patch_height // 2)
        items.append(Candidate(kind="positive", grid_index=(row, col), coordinate=(x, y), score=0.0))
    return items


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def negative_blank_fraction(candidates: list[Candidate], query_image_path: Path) -> float:
    negatives = [item for item in candidates if item.kind == "negative"]
    if not negatives:
        return 0.0
    image = Image.open(query_image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    pixels = image.load()
    blank = 0
    for item in negatives:
        x, y = item.coordinate
        r, g, b = pixels[x, y]
        if max(r, g, b) >= 245 or max(r, g, b) <= 10:
            blank += 1
    return blank / len(negatives)


def _append_ledger(project_root: Path, gpu_hours: float, *, dataset_name: str) -> None:
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "3-candidate-pilot",
                "command": "scripts/run_candidate_pilot.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": f"{dataset_name} validation multi-episode DINOv2-L candidate quality pilot.",
            }
        )


def _write_gate_c_status(project_root: Path, summary: dict[str, Any], *, dataset_name: str) -> None:
    slug = dataset_slug(dataset_name)
    path = project_root / "reports" / ("gate_c_status.md" if slug == "monusac" else f"gate_c_status_{slug}.md")
    issues = "; ".join(summary.get("issues", [])) or "none"
    lines = [
        f"# Gate C Status: {dataset_name}",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
        "| synthetic candidate kernel smoke | PASS | score-map mechanics only; no query GT used for candidate generation |",
        f"| real DINOv2-L matcher smoke | PASS | {dataset_name} validation episode path is dataset-aware |",
        f"| formal matcher candidate gate | {summary['status']} | {issues} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(
    *,
    project_root: Path = PROJECT_ROOT,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    max_episodes: int = 5,
    device_text: str = "auto",
    dataset_name: str = "MoNuSAC",
    positive_limits: tuple[int, ...] = POSITIVE_LIMITS,
    positive_layout: str = "spread",
    min_positive_distance: float = 4.0,
) -> int:
    slug = dataset_slug(dataset_name)
    out_dir = out_dir or (project_root / "reports" / ("candidate_pilot" if slug == "monusac" else f"candidate_pilot_{slug}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    predeclared_rules = {
        "min_episodes": max_episodes,
        "precision_margin": PRECISION_MARGIN,
        "positive_limits": list(positive_limits),
        "positive_layout": positive_layout,
        "min_positive_distance": min_positive_distance,
        "dataset_role": "MoNuSAC validation only" if slug == "monusac" else f"{dataset_name} validation extension",
        "test_split_used": False,
        "query_mask_use": "diagnostic metrics and random-baseline evaluation only",
    }
    if dry_run:
        payload = {
            "status": "DRY_RUN",
            "formal_gate_c": "PENDING_EXECUTION",
            "predeclared_rules": predeclared_rules,
        }
        (out_dir / "candidate_pilot_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    workspace_root = project_root.parent
    episodes_csv = episodes_csv or (project_root / "manifests" / "episodes" / slug / "episodes_val.csv")
    episodes = read_episodes(episodes_csv, max_episodes)
    matcher_path = workspace_root / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    matcher = checkpoint_record(matcher_path, role="matcher")
    device = torch.device("cuda:0" if device_text == "auto" and torch.cuda.is_available() else ("cpu" if device_text == "auto" else device_text))
    start = time.perf_counter()
    model = load_dinov2_model(matcher_path, workspace_root=workspace_root, device=device)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    grid_size = IMAGE_SIZE // PATCH_SIZE
    for episode in episodes:
        paths = resolve_episode_paths(episode, workspace_root)
        support_image = load_rgb(paths["support_image"])
        query_image = load_rgb(paths["query_image"])
        support_mask = load_mask_tensor(paths["support_mask"], size=(IMAGE_SIZE, IMAGE_SIZE))
        query_mask = load_mask_tensor(paths["query_mask"], size=(IMAGE_SIZE, IMAGE_SIZE))
        with torch.inference_mode():
            support_features = extract_dinov2_features(model, support_image, device)
            query_features = extract_dinov2_features(model, query_image, device)
        maps = matching_scores_from_features(
            support_features=support_features.cpu(),
            query_features=query_features.cpu(),
            foreground_weights=patch_weights_from_mask(support_mask, patch_size=PATCH_SIZE, foreground=True),
            background_weights=patch_weights_from_mask(support_mask, patch_size=PATCH_SIZE, foreground=False),
            grid_size=grid_size,
        )
        query_bool = query_mask.bool().tolist()
        for positive_limit in positive_limits:
            pools = build_candidate_pools(
                maps["s_fg"].tolist(),
                maps["s_bg"].tolist(),
                image_size=(IMAGE_SIZE, IMAGE_SIZE),
                positive_limit=positive_limit,
                negative_limit=positive_limit,
                positive_layout=positive_layout,
                min_positive_distance=min_positive_distance,
            )
            candidates = list(pools.positives + pools.negatives)
            issues = validate_candidate_pools(candidates, image_size=(IMAGE_SIZE, IMAGE_SIZE))
            random_candidates = random_positive_candidates(
                seed_text=f"{episode['episode_id']}:{positive_limit}",
                count=positive_limit,
                image_size=(IMAGE_SIZE, IMAGE_SIZE),
                grid_size=grid_size,
            )
            row = {
                "episode_id": episode["episode_id"],
                "query_case_id": episode["query_case_id"],
                "support_case_id": episode["support_case_id"],
                "positive_limit": positive_limit,
                "point_precision": point_precision(candidates, query_bool),
                "component_coverage": component_coverage(candidates, query_bool),
                "random_precision": point_precision(random_candidates, query_bool),
                "validation_issue_count": len(issues),
                "negative_blank_fraction": negative_blank_fraction(candidates, paths["query_image"]),
            }
            rows.append(row)
            for item in candidates:
                candidate_rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "positive_limit": positive_limit,
                        "kind": item.kind,
                        "grid_row": item.grid_index[0],
                        "grid_col": item.grid_index[1],
                        "x": item.coordinate[0],
                        "y": item.coordinate[1],
                        "score": item.score,
                    }
                )
    summary = summarize_candidate_pilot(rows, precision_margin=PRECISION_MARGIN, min_episodes=max_episodes)
    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.type == "cuda" else 0.0
    summary.update(
        {
            "formal_gate_c": summary["status"],
            "device": str(device),
            "elapsed_seconds": round(elapsed, 3),
            "gpu_hours_estimate": round(gpu_hours, 6),
            "predeclared_rules": predeclared_rules,
            "matcher_checkpoint_sha256": matcher["sha256"],
            "artifacts": {
                "metrics_csv": str(out_dir / "candidate_pilot_metrics.csv"),
                "candidates_csv": str(out_dir / "candidate_pilot_candidates.csv"),
            },
        }
    )
    _write_csv(out_dir / "candidate_pilot_metrics.csv", rows)
    _write_csv(out_dir / "candidate_pilot_candidates.csv", candidate_rows)
    (out_dir / "candidate_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_gate_c_status(project_root, summary, dataset_name=dataset_name)
    _append_ledger(project_root, gpu_hours, dataset_name=dataset_name)
    return 0 if summary["status"] == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a validation candidate quality pilot for Gate C.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-name", default="MoNuSAC")
    parser.add_argument("--positive-limits", type=int, nargs="+", default=list(POSITIVE_LIMITS))
    parser.add_argument("--positive-layout", default="spread")
    parser.add_argument("--min-positive-distance", type=float, default=4.0)
    args = parser.parse_args()
    raise SystemExit(
        run_pilot(
            project_root=args.project_root,
            episodes_csv=args.episodes_csv,
            out_dir=args.out_dir,
            max_episodes=args.max_episodes,
            dry_run=args.dry_run,
            device_text=args.device,
            dataset_name=args.dataset_name,
            positive_limits=tuple(args.positive_limits),
            positive_layout=args.positive_layout,
            min_positive_distance=args.min_positive_distance,
        )
    )


if __name__ == "__main__":
    main()
