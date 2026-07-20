from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fspl.candidates import CandidatePools, build_candidate_pools
from fspl.dataset import Episode, build_smoke_episode, discover_cases
from fspl.models import IMAGE_SIZE, PATCH_SIZE, load_binary_mask, load_dinov2, load_mask_for_features, load_rgb, load_sam_predictor, matching_maps, predict_sam


@dataclass(frozen=True)
class Stage1Config:
    dataset_root: Path = Path(r"D:\SAM\code\datasets\MonuSeg\official")
    dino_checkpoint: Path = Path(r"D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth")
    sam_checkpoint: Path = Path(r"D:\SAM\assets\checkpoints\sam_vit_b.pth")
    output_path: Path = Path("artifacts/stage1_smoke.json")
    positive_limit: int = 5
    negative_limit: int = 3
    lambda_bg: float = 1.0
    device: str = "cpu"


def write_stage1_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _scale_points(pools: CandidatePools, image_size: tuple[int, int]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    width, height = image_size
    scale_x = width / IMAGE_SIZE
    scale_y = height / IMAGE_SIZE
    positive = [(candidate.coordinate[0] * scale_x, candidate.coordinate[1] * scale_y) for candidate in pools.positives]
    negative = [(candidate.coordinate[0] * scale_x, candidate.coordinate[1] * scale_y) for candidate in pools.negatives]
    return positive, negative


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = float(np.logical_and(prediction, target).sum())
    return (2.0 * intersection) / max(1.0, float(prediction.sum() + target.sum()))


def _episode_for_smoke(config: Stage1Config) -> Episode:
    return build_smoke_episode(discover_cases(config.dataset_root, "train"))


def run_stage1(config: Stage1Config) -> dict:
    if not config.dino_checkpoint.is_file() or not config.sam_checkpoint.is_file():
        raise FileNotFoundError("Stage 1 requires both DINOv2-L and SAM ViT-B checkpoints")
    device = torch.device(config.device)
    episode = _episode_for_smoke(config)
    started = time.perf_counter()

    reference_image = load_rgb(episode.reference.image_path)
    query_image = load_rgb(episode.query.image_path)
    reference_resized = reference_image.resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
    query_resized = query_image.resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
    reference_mask = load_mask_for_features(episode.reference.mask_path)

    matcher = load_dinov2(config.dino_checkpoint, device)
    foreground_scores, background_scores = matching_maps(
        matcher, reference_resized, reference_mask, query_resized, device
    )
    pools = build_candidate_pools(
        foreground_scores,
        background_scores,
        patch_size=PATCH_SIZE,
        positive_limit=config.positive_limit,
        negative_limit=config.negative_limit,
        lambda_bg=config.lambda_bg,
    )
    positive_points, negative_points = _scale_points(pools, query_image.size)
    predictor = load_sam_predictor(config.sam_checkpoint, device)
    prediction, sam_score = predict_sam(predictor, query_image, positive_points, negative_points)
    target = load_binary_mask(episode.query.mask_path, size=query_image.size)
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "reference_case_id": episode.reference.case_id,
        "query_case_id": episode.query.case_id,
        "positive_count": len(positive_points),
        "negative_count": len(negative_points),
        "positive_points": positive_points,
        "negative_points": negative_points,
        "dice": _dice(prediction, target),
        "sam_score": sam_score,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_stage1_summary(config.output_path, summary)
    return summary


def image_resample() -> int:
    from PIL import Image

    return Image.Resampling.BICUBIC
