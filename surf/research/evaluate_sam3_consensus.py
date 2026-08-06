"""OnePrompt-All candidate-set consensus ablation.

The user provides exactly one positive point.  SAM3 produces an initial seed;
DINO is used *only inside that seed* to select a semantically stable virtual
point.  Independent SAM3 responses are clustered by mask overlap.  A final
candidate is accepted by repeatability, rather than a class-specific DINO
threshold or unsafe external negative prompts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demos.medical_sam_click_app import PointPrompt, Sam3TextSegmenter  # noqa: E402
from research.evaluate_sam3_multi_click import sampled_point  # noqa: E402
from research.evaluate_sam3_onepoint import iou, load_dino, metrics, nms, patch_mask, token_features  # noqa: E402


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--prompts-per-image", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--threshold", type=float, default=.45)
    parser.add_argument("--min-area-ratio", type=float, default=.002)
    parser.add_argument("--cluster-iou", type=float, default=.5)
    parser.add_argument("--min-votes", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def seed_index(masks: np.ndarray, scores: np.ndarray, point: tuple[int, int]) -> int:
    x, y = point
    matches = [i for i, mask in enumerate(masks) if mask[y, x]]
    return max(matches or list(range(len(masks))), key=lambda i: float(scores[i]))


def dino_core_point(image: np.ndarray, seed: np.ndarray, dino: torch.nn.Module, device: torch.device, avoid: tuple[int, int]) -> tuple[int, int]:
    """A virtual positive constrained to the clicked SAM3 seed itself."""
    cells = patch_mask(seed)
    if not cells.any():
        return avoid
    feats = token_features(dino, image, device)
    proto = torch.nn.functional.normalize(feats[cells].mean(0), dim=0)
    sim = (feats @ proto).detach().cpu().numpy()
    interior = distance_transform_edt(cells)
    score = sim + .10 * interior / max(float(interior.max()), 1e-6)
    score[~cells] = -np.inf
    ty, tx = np.unravel_index(int(np.argmax(score)), score.shape)
    x = min(image.shape[1] - 1, int((tx + .5) * image.shape[1] / 37))
    y = min(image.shape[0] - 1, int((ty + .5) * image.shape[0] / 37))
    if seed[y, x]:
        return x, y
    ys, xs = np.where(seed)
    index = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[index]), int(ys[index])


def distance_core_point(seed: np.ndarray, avoid: tuple[int, int]) -> tuple[int, int]:
    distance = distance_transform_edt(seed)
    # Prefer the deepest available seed location; use the original user point
    # only when the seed is too small to define a distinct interior anchor.
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    if distance[y, x] < 2:
        return avoid
    return int(x), int(y)


def consensus(masks_by_run: list[np.ndarray], scores_by_run: list[np.ndarray], min_area: float, cluster_iou: float, min_votes: int) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Cluster repeatable masks; a cluster contributes at most one instance."""
    items: list[tuple[int, np.ndarray, float]] = []
    for run, (masks, scores) in enumerate(zip(masks_by_run, scores_by_run)):
        for mask, score in zip(masks, scores):
            if float(mask.sum()) >= min_area:
                items.append((run, mask.astype(bool), float(score)))
    items.sort(key=lambda item: item[2], reverse=True)
    clusters: list[dict[str, object]] = []
    for run, mask, score in items:
        best, best_iou = -1, 0.0
        for idx, cluster in enumerate(clusters):
            overlap = iou(mask, cluster["representative"])  # type: ignore[arg-type]
            if overlap > best_iou:
                best, best_iou = idx, overlap
        if best >= 0 and best_iou >= cluster_iou:
            cluster = clusters[best]
            cluster["members"].append((run, mask, score))  # type: ignore[index]
            cluster["runs"].add(run)  # type: ignore[index]
            if score > cluster["score"]:  # type: ignore[operator]
                cluster["representative"], cluster["score"] = mask, score
        else:
            clusters.append({"representative": mask, "score": score, "members": [(run, mask, score)], "runs": {run}})
    chosen = [cluster for cluster in clusters if len(cluster["runs"]) >= min_votes]  # type: ignore[arg-type]
    final_masks = [cluster["representative"] for cluster in chosen]  # type: ignore[list-item]
    final_scores = [float(cluster["score"]) + .05 * len(cluster["runs"]) for cluster in chosen]  # type: ignore[arg-type]
    final_votes = [len(cluster["runs"]) for cluster in chosen]  # type: ignore[arg-type]
    return nms(np.asarray(final_masks, dtype=bool), np.asarray(final_scores, dtype=np.float32), .5), final_scores, final_votes


def main() -> None:
    cfg = args(); cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((cfg.dataset / "manifest.csv").open(encoding="utf-8")))[::cfg.image_stride]
    if cfg.max_images is not None:
        rows = rows[:cfg.max_images]
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"; device = torch.device(device_name)
    sam3 = Sam3TextSegmenter(cfg.checkpoint, device_name, cfg.threshold)
    dino = load_dino(cfg.dino_checkpoint, device)
    rng = np.random.default_rng(cfg.seed); output: list[dict[str, object]] = []
    for image_index, row in enumerate(rows, 1):
        image = np.asarray(Image.open(cfg.dataset / row["image"]).convert("RGB"))
        truth = [np.asarray(Image.open(path).convert("L"), dtype=bool) for path in sorted((cfg.dataset / row["instance_mask_dir"]).glob("*.png"))]
        for prompt_index in range(1, cfg.prompts_per_image + 1):
            gt_index = int(rng.integers(len(truth))); point = sampled_point(truth[gt_index], rng); started = time.perf_counter()
            masks0, scores0 = sam3.predict_interactive(image, "click", [PointPrompt(x=point[0], y=point[1], label=1)], None, "manual")
            seed = masks0[seed_index(masks0, scores0, point)]
            dino_point = dino_core_point(image, seed, dino, device, point)
            distance_point = distance_core_point(seed, point)
            runs_masks, runs_scores = [masks0], [scores0]
            for x, y in (dino_point, distance_point):
                masks, scores = sam3.predict_interactive(image, "click", [PointPrompt(x=x, y=y, label=1)], None, "manual")
                runs_masks.append(masks); runs_scores.append(scores)
            predicted, _, votes = consensus(runs_masks, runs_scores, image.shape[0] * image.shape[1] * cfg.min_area_ratio, cfg.cluster_iou, cfg.min_votes)
            precision, recall, f1, union_iou, matched = metrics(predicted, truth)
            output.append(dict(image=row["image"], class_name=row["class_name"], prompt_index=prompt_index, prompt_gt_instance=gt_index + 1, point_x=point[0], point_y=point[1], dino_x=dino_point[0], dino_y=dino_point[1], distance_x=distance_point[0], distance_y=distance_point[1], gt_instances=len(truth), predicted_instances=len(predicted), matched_instances=matched, mean_candidate_votes=float(np.mean(votes)) if votes else 0., precision=precision, recall=recall, f1=f1, union_iou=union_iou, elapsed_sec=time.perf_counter()-started))
            print(f"[{image_index}/{len(rows)} p{prompt_index}] F1={f1:.3f} K={len(predicted)}/{len(truth)}", flush=True)
            if device_name.startswith("cuda"): torch.cuda.empty_cache()
    with (cfg.out_dir / "per_prompt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in output: by_image[str(item["image"])].append(item)
    image_rows = [dict(image=name, prompts=len(group), mean_f1=float(np.mean([x["f1"] for x in group])), std_f1=float(np.std([x["f1"] for x in group])), mean_precision=float(np.mean([x["precision"] for x in group])), mean_recall=float(np.mean([x["recall"] for x in group]))) for name, group in by_image.items()]
    with (cfg.out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0])); writer.writeheader(); writer.writerows(image_rows)
    summary = dict(images=len(image_rows), prompts=len(output), virtual_runs=2, min_votes=cfg.min_votes, mean_f1=float(np.mean([x["f1"] for x in output])), std_f1=float(np.std([x["f1"] for x in output])), mean_precision=float(np.mean([x["precision"] for x in output])), mean_recall=float(np.mean([x["recall"] for x in output])), mean_per_image_std_f1=float(np.mean([x["std_f1"] for x in image_rows])), worst_prompt_f1=float(np.min([x["f1"] for x in output])), best_prompt_f1=float(np.max([x["f1"] for x in output])), seed=cfg.seed)
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
