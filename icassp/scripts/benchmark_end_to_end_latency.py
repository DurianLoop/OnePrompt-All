"""Benchmark the frozen DINOv2-L + SAM ViT-B PanNuke inference path.

The benchmark reads a support image/mask and query image only.  It deliberately
never opens a query annotation, runs the same K=40/lambda=0.75/N0/N5 pathway as
the frozen cohort, and records both full per-episode latency and the latency
when a support representation is already cached.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA = ROOT / "data" / "pannuke_hf" / "materialized_zero_shot_v1"
MANIFEST = DATA / "manifest.csv"
OUT = ROOT / "runs" / "end_to_end_latency_v1"
SURF = WORKSPACE / "surf"
SAM_CKPT = WORKSPACE / "assets" / "checkpoints" / "sam_vit_b.pth"
DINO_CKPT = WORKSPACE / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
K = 40
LAMBDA = 0.75
SEED = 20260809


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def synchronize() -> None:
    torch.cuda.synchronize()


def seconds(fn):
    synchronize()
    started = time.perf_counter()
    result = fn()
    synchronize()
    return result, time.perf_counter() - started


def summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(data.mean()), "median_s": float(np.median(data)),
        "p90_s": float(np.quantile(data, 0.90)), "min_s": float(data.min()), "max_s": float(data.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen DINOv2-L + SAM ViT-B prompt inference.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 10:
        raise ValueError("use at least one warmup and ten measured episodes")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if str(SURF) not in sys.path:
        sys.path.insert(0, str(SURF))
    from benchmarks.evaluate_negative_prompt_ablation import select_negative_points
    from benchmarks.run_monuseg_dense_prompt_experiment import (
        IMAGE_SIZE, clean_device, compute_matching_maps_from_features, dense_points_from_score,
        extract_features, load_dinov2, load_mask_image, load_sam_predictor, zscore,
    )

    all_rows = rows(MANIFEST)
    support = next(row for row in all_rows if row["role"] == "support")
    queries = [row for row in all_rows if row["role"] == "query"]
    if len(queries) < args.warmup + args.repetitions:
        raise RuntimeError("not enough pre-registered PanNuke query images")
    selected = queries[:args.warmup + args.repetitions]
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(len(selected)))
    selected = [selected[index] for index in order]

    device = clean_device("cuda")
    predictor, _ = load_sam_predictor(SAM_CKPT, "vit_b", "cuda")
    dino = load_dinov2(DINO_CKPT, "vit_large", device)
    support_path = DATA / support["image_path"]
    support_mask_path = DATA / support["mask_path"]

    records: list[dict[str, object]] = []
    for index, query in enumerate(selected):
        query_path = DATA / query["image_path"]
        total_start = time.perf_counter(); synchronize()
        (support_image, support_mask), support_io_s = seconds(lambda: (Image.open(support_path).convert("RGB"), load_mask_image(support_mask_path)))
        support_features, support_dino_s = seconds(lambda: extract_features(dino, support_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), device))
        query_image, query_io_s = seconds(lambda: Image.open(query_path).convert("RGB"))
        query_features, query_dino_s = seconds(lambda: extract_features(dino, query_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), device))

        def prompt_setup():
            s_fg, s_bg = compute_matching_maps_from_features(support_features, support_mask, query_features, device)
            score_map = zscore(s_fg) - LAMBDA * zscore(s_bg)
            points = dense_points_from_score(score_map, s_fg, s_bg, query_image.size, max_points=K)[:K]
            positive = np.asarray([[point.x, point.y] for point in points], dtype=np.float32)
            if len(positive) != K:
                raise RuntimeError("unexpected positive-point count")
            negative = select_negative_points(score_map, query_image.size, 5, positive)
            return positive, negative

        (positive, negative), matching_points_s = seconds(prompt_setup)
        _, sam_embed_s = seconds(lambda: predictor.set_image(np.asarray(query_image)))

        def decode(count: int):
            chosen_negative = negative[:count]
            coords = np.concatenate([positive, chosen_negative], axis=0)
            labels = np.concatenate([np.ones(len(positive), dtype=np.int32), np.zeros(len(chosen_negative), dtype=np.int32)])
            masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
            return int(np.argmax(scores)), masks

        _, sam_n0_decode_s = seconds(lambda: decode(0))
        _, sam_n5_decode_s = seconds(lambda: decode(5))
        synchronize(); total_s = time.perf_counter() - total_start
        records.append({
            "phase": "warmup" if index < args.warmup else "measured", "query_case_id": query["case_id"],
            "support_case_id": support["case_id"], "support_io_s": support_io_s, "support_dino_s": support_dino_s,
            "query_io_s": query_io_s, "query_dino_s": query_dino_s, "matching_points_s": matching_points_s,
            "sam_image_embedding_s": sam_embed_s, "sam_n0_decode_s": sam_n0_decode_s, "sam_n5_decode_s": sam_n5_decode_s,
            "support_cached_query_pipeline_s": query_io_s + query_dino_s + matching_points_s + sam_embed_s + sam_n0_decode_s + sam_n5_decode_s,
            "full_support_query_episode_s": total_s,
            "query_gt_opened": False,
        })
        print(f"[latency] {index + 1}/{len(selected)} {records[-1]['phase']} full_s={total_s:.4f}", flush=True)

    measured = [record for record in records if record["phase"] == "measured"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_episode.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    properties = torch.cuda.get_device_properties(0)
    report = {
        "status": "COMPLETE", "protocol": "icassp.end_to_end_latency_v1", "seed": SEED,
        "hardware": {"device_name": torch.cuda.get_device_name(0), "total_memory_gib": round(properties.total_memory / 2**30, 2), "torch": torch.__version__, "cuda": torch.version.cuda},
        "pipeline": {"support_query_input_px": [256, 256], "internal_dino_input_px": [IMAGE_SIZE, IMAGE_SIZE], "matcher": "DINOv2-L foreground-background", "segmenter": "SAM ViT-B", "positive_points": K, "negative_candidates": [0, 5], "lambda_bg": LAMBDA},
        "measurement": {"warmup_episodes": args.warmup, "measured_episodes": len(measured), "cuda_synchronized": True, "query_gt_opened": False, "full_episode_includes": ["support/query image and support-mask read", "support/query DINO features", "matching and point selection", "SAM image embedding", "N0 and N5 decode"], "support_cached_pipeline_excludes": ["support image/mask read", "support DINO feature extraction"]},
        "seconds": {key: summary([float(record[key]) for record in measured]) for key in ("full_support_query_episode_s", "support_cached_query_pipeline_s", "support_dino_s", "query_dino_s", "matching_points_s", "sam_image_embedding_s", "sam_n0_decode_s", "sam_n5_decode_s")},
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
