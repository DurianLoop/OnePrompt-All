from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
from PIL import Image

from evaluate_negative_prompt_ablation import read_csv, resolve, score_path_from_points, select_negative_points


ROOT = Path(__file__).resolve().parents[2]


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_durations(values: list[float]) -> dict[str, Any]:
    return {
        "repeat_count": len(values),
        "mean_seconds": round(mean(values), 6),
        "median_seconds": round(median(values), 6),
        "total_seconds": round(sum(values), 6),
    }


def timed_call(function: Any, device: str) -> tuple[Any, float]:
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark two-candidate negative-response gate execution.")
    parser.add_argument("--backend", choices=("sam", "sammed2d"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--fixed-action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_rows = read_csv(resolve(args.manifest))
    candidate_rows = [
        row for row in read_csv(resolve(args.candidates))
        if row["method"] == args.fixed_action
    ]
    if not manifest_rows or not candidate_rows:
        raise ValueError("Runtime benchmark requires at least one manifest and candidate row.")
    manifest_by_id = {row["pilot_id"]: row for row in manifest_rows}
    candidate = next(row for row in candidate_rows if row["case_name"] in manifest_by_id)
    episode = manifest_by_id[candidate["case_name"]]

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    if args.backend == "sam":
        from run_monuseg_dense_prompt_experiment import load_sam_predictor

        predictor, _ = load_sam_predictor(resolve(args.checkpoint), "vit_b", args.device)
    else:
        from run_monuseg_sammed2d_from_points import load_sammed2d_predictor

        predictor = load_sammed2d_predictor(
            ROOT / "SAM-Med2D",
            resolve(args.checkpoint),
            args.device,
            256,
            True,
        )
    synchronize(args.device)
    model_load_seconds = time.perf_counter() - load_start

    image = Image.open(resolve(episode["query_image_path"])).convert("RGB")
    point_rows = read_csv(resolve(candidate["points_path"]))
    positives = np.asarray([[float(row["x"]), float(row["y"])] for row in point_rows], dtype=np.float32)
    score_map = np.load(score_path_from_points(resolve(candidate["points_path"])))
    negatives = select_negative_points(score_map, image.size, 5, positives)
    prompts = [
        (positives, np.ones(len(positives), dtype=np.int32)),
        (
            np.concatenate([positives, negatives], axis=0),
            np.concatenate([np.ones(len(positives), dtype=np.int32), np.zeros(len(negatives), dtype=np.int32)]),
        ),
    ]

    _, embedding_seconds = timed_call(lambda: predictor.set_image(np.asarray(image)), args.device)
    predictor.predict(point_coords=prompts[0][0], point_labels=prompts[0][1], multimask_output=True)
    synchronize(args.device)
    decode_durations: list[float] = []
    for _ in range(args.repeats):
        for coords, labels in prompts:
            _, duration = timed_call(
                lambda coords=coords, labels=labels: predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=True,
                ),
                args.device,
            )
            decode_durations.append(duration)

    model = getattr(predictor, "model", None)
    parameter_count = sum(parameter.numel() for parameter in model.parameters()) if model is not None else 0
    peak_mib = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else 0.0
    )
    decode = summarize_durations(decode_durations)
    result = {
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "episode_id": candidate["case_name"],
        "image_size": list(image.size),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "candidate_count": 2,
        "model_parameters": parameter_count,
        "model_parameters_million": round(parameter_count / 1_000_000, 3),
        "model_load_seconds": round(model_load_seconds, 6),
        "image_embedding_seconds": round(embedding_seconds, 6),
        "single_mask_decode": decode,
        "two_candidate_decode_seconds": round(2.0 * float(decode["median_seconds"]), 6),
        "peak_cuda_memory_mib": round(peak_mib, 3),
        "device": args.device,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
