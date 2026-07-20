from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def build_episode_image_map(episodes: list[dict[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in episodes:
        episode_id = row.get("episode_id") or row.get("pilot_id")
        if not episode_id:
            raise KeyError("Episode manifest requires episode_id or pilot_id.")
        mapping[episode_id] = row["query_image_path"]
    return mapping


def group_candidates_by_query(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_case_id"]].append(row)
    return dict(grouped)


def load_point_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    coordinates = np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float32)
    labels = np.ones(len(coordinates), dtype=np.int32)
    return coordinates, labels


def main() -> None:
    import torch
    from run_monuseg_dense_prompt_experiment import load_sam_predictor, runtime_metrics

    parser = argparse.ArgumentParser(description="Materialize lightweight SAM masks for minimal-artifact candidate grids.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--sam-checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-device", default="auto")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    started_at = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    candidates = read_csv(resolve(args.candidates))
    image_by_episode = build_episode_image_map(read_csv(resolve(args.episodes)))
    grouped = group_candidates_by_query(candidates)
    predictor, _ = load_sam_predictor(resolve(args.sam_checkpoint), args.sam_model_type, args.sam_device)
    written = 0
    reused = 0
    score_errors: list[float] = []
    for query_id, rows in sorted(grouped.items()):
        episode_id = rows[0]["case_name"]
        if episode_id not in image_by_episode:
            raise KeyError(f"Missing episode {episode_id!r} in manifest.")
        image = Image.open(resolve(image_by_episode[episode_id])).convert("RGB")
        predictor.set_image(np.asarray(image))
        for row in rows:
            mask_path = resolve(row["mask_path"])
            if mask_path.exists() and not args.overwrite:
                reused += 1
                continue
            coordinates, labels = load_point_arrays(resolve(row["points_path"]))
            masks, scores, _ = predictor.predict(
                point_coords=coordinates,
                point_labels=labels,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(masks[best].astype(np.uint8) * 255, mode="L").save(mask_path)
            score_errors.append(abs(float(scores[best]) - float(row["sam_score"])))
            written += 1
        print(f"[materialize] {query_id} candidates={len(rows)} written={written} reused={reused}")

    summary: dict[str, Any] = {
        "candidate_count": len(candidates),
        "query_count": len(grouped),
        "written_masks": written,
        "reused_masks": reused,
        "mean_abs_sam_score_error": float(np.mean(score_errors)) if score_errors else 0.0,
        "max_abs_sam_score_error": float(np.max(score_errors)) if score_errors else 0.0,
    }
    summary.update(runtime_metrics(
        elapsed_seconds=time.perf_counter() - started_at,
        candidate_count=written,
        case_count=len(grouped),
        peak_gpu_mb=(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else 0.0
        ),
    ))
    summary_path = resolve(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
