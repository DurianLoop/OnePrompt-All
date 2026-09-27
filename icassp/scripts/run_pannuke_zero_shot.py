"""Memory-bounded frozen PanNuke raw zero-shot evaluation.

For every non-empty fold-1 query and each of five pre-registered fold-2
supports, this runner performs real DINOv2-L foreground/background matching
and SAM ViT-B N0/N5 decoding.  It writes only model-independent raw outputs:
the saved MoNuSeg post-SAM gate is deliberately applied later in separate,
version-compatible sklearn environments and then joined deterministically.
Query masks are opened only after both action predictions, solely for metrics.
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
OUT = ROOT / "runs" / "pannuke_zero_shot_v1"
SURF = WORKSPACE / "surf"
SAM_CKPT = WORKSPACE / "assets" / "checkpoints" / "sam_vit_b.pth"
DINO_CKPT = WORKSPACE / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
SEED = 20260809
K = 40
LAMBDA = 0.75


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append(path: Path, record: dict[str, object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(record)


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = float(np.logical_and(pred, target).sum())
    return 2.0 * intersection / (float(pred.sum()) + float(target.sum()) + 1e-8)


def bootstrap_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    delta = a - b
    rng = np.random.default_rng(SEED)
    means = np.empty(10_000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = float(delta[rng.integers(0, len(delta), len(delta))].mean())
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=0, help="0 evaluates every pre-registered query")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen PanNuke run requires CUDA")
    if str(SURF) not in sys.path:
        sys.path.insert(0, str(SURF))
    from benchmarks.evaluate_negative_prompt_ablation import select_negative_points
    from benchmarks.run_monuseg_dense_prompt_experiment import (
        IMAGE_SIZE, clean_device, compute_matching_maps_from_features, dense_points_from_score,
        extract_features, load_dinov2, load_mask, load_mask_image, load_sam_predictor, zscore,
    )
    rows = read_csv(MANIFEST)
    queries = [row for row in rows if row["role"] == "query"]
    supports = [row for row in rows if row["role"] == "support"]
    if len(queries) != 2540 or len(supports) != 5:
        raise RuntimeError("unexpected PanNuke materialization manifest")
    if args.max_queries:
        queries = queries[:args.max_queries]
    OUT.mkdir(parents=True, exist_ok=True)
    progress = OUT / "per_episode.csv"
    done = {row["episode_id"] for row in read_csv(progress)} if args.resume and progress.exists() else set()
    fields = ["episode_id", "query_case_id", "support_case_id", "n0_dice", "n5_dice", "n0_sam_score", "n5_sam_score", "n0_area_pct", "n5_area_pct", "query_gt_loaded_after_prediction", "seconds"]
    device = clean_device("cuda")
    predictor, _ = load_sam_predictor(SAM_CKPT, "vit_b", "cuda")
    dino = load_dinov2(DINO_CKPT, "vit_large", device)
    support_state: list[tuple[dict[str, str], Image.Image, torch.Tensor]] = []
    for support in supports:
        image = Image.open(DATA / support["image_path"]).convert("RGB")
        with torch.inference_mode():
            features = extract_features(dino, image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), device)
        support_state.append((support, load_mask_image(DATA / support["mask_path"]), features))
    started = time.perf_counter()
    for index, query in enumerate(queries, start=1):
        query_id = query["case_id"]
        image = Image.open(DATA / query["image_path"]).convert("RGB")
        predictor.set_image(np.asarray(image))
        with torch.inference_mode():
            query_features = extract_features(dino, image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), device)
        for support, support_mask, support_features in support_state:
            episode_id = f"{support['case_id']}__to__{query_id}"
            if episode_id in done:
                continue
            pair_started = time.perf_counter()
            s_fg, s_bg = compute_matching_maps_from_features(support_features, support_mask, query_features, device)
            score_map = zscore(s_fg) - LAMBDA * zscore(s_bg)
            points = dense_points_from_score(score_map, s_fg, s_bg, image.size, max_points=K)[:K]
            positive = np.asarray([[point.x, point.y] for point in points], dtype=np.float32)
            if len(positive) != K:
                raise RuntimeError(f"expected {K} positive points for {episode_id}")
            predictions: dict[str, tuple[np.ndarray, float, float]] = {}
            for count, action in ((0, "neg0"), (5, "neg5")):
                negative = select_negative_points(score_map, image.size, count, positive)
                coords = np.concatenate([positive, negative], axis=0)
                labels = np.concatenate([np.ones(len(positive), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)])
                masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
                best = int(np.argmax(scores)); prediction = masks[best].astype(bool)
                predictions[action] = (prediction, float(scores[best]), 100.0 * float(prediction.mean()))
            # Query annotation is intentionally opened only after both action predictions.
            target = load_mask(DATA / query["mask_path"], image.size)
            record = {
                "episode_id": episode_id, "query_case_id": query_id, "support_case_id": support["case_id"],
                "n0_dice": dice(predictions["neg0"][0], target), "n5_dice": dice(predictions["neg5"][0], target),
                "n0_sam_score": predictions["neg0"][1], "n5_sam_score": predictions["neg5"][1],
                "n0_area_pct": predictions["neg0"][2], "n5_area_pct": predictions["neg5"][2], "query_gt_loaded_after_prediction": True,
                "seconds": time.perf_counter() - pair_started,
            }
            append(progress, record, fields); done.add(episode_id)
        if index % 10 == 0 or index == len(queries):
            print(f"[pannuke] queries={index}/{len(queries)} episodes={len(done)} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    all_rows = read_csv(progress)
    by_query: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        by_query.setdefault(row["query_case_id"], []).append(row)
    summaries = []
    for query_id, group in sorted(by_query.items()):
        if len(group) != 5:
            continue
        summaries.append({"query_case_id": query_id, "n0_dice": float(np.mean([float(row["n0_dice"]) for row in group])), "n5_dice": float(np.mean([float(row["n5_dice"]) for row in group]))})
    summary_path = OUT / "per_query.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_case_id", "n0_dice", "n5_dice"]); writer.writeheader(); writer.writerows(summaries)
    is_full_protocol = args.max_queries == 0 and len(queries) == 2540
    if is_full_protocol and len(summaries) == len(queries):
        n0 = np.asarray([row["n0_dice"] for row in summaries]); n5 = np.asarray([row["n5_dice"] for row in summaries])
        report = {"status": "RAW_COMPLETE", "protocol": "icassp.pannuke_zero_shot_v1", "query_count": len(summaries), "supports_per_query": 5, "means": {"n0": float(n0.mean()), "n5": float(n5.mean())}, "n5_vs_n0": dict(zip(("delta", "ci95_low", "ci95_high"), bootstrap_delta(n5, n0))), "query_gt_policy": "loaded only after both action predictions", "elapsed_seconds": time.perf_counter()-started}
    elif args.max_queries:
        report = {"status": "SMOKE_COMPLETE", "completed_queries": len(summaries), "full_protocol_queries": 2540, "note": "A max-queries run validates the runner only and must not be treated as a cohort result."}
    else:
        report = {"status": "PARTIAL", "completed_queries": len(summaries), "requested_queries": len(queries), "note": "No cohort claim may be made from a partial run."}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
