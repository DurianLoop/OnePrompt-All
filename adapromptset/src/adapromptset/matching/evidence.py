from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional

from adapromptset.matching.runtime import configure_external_paths, infer_dino_architecture


IMAGE_SIZE = 518
PATCH_SIZE = 14


def patch_weights_from_mask(mask: torch.Tensor, *, patch_size: int, foreground: bool) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor")
    values = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
    if not foreground:
        values = 1.0 - values
    pooled = torch_functional.avg_pool2d(values.view(1, 1, *values.shape), patch_size)
    weights = pooled.reshape(-1)
    total = weights.sum()
    if float(total.detach().cpu()) <= 0.0:
        label = "foreground" if foreground else "background"
        raise ValueError(f"reference mask has no {label} patch evidence")
    return weights / total


def _weighted_proto(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if features.shape[0] != weights.shape[0]:
        raise ValueError("feature and weight patch counts differ")
    proto = (features * weights.unsqueeze(1).to(features.device)).sum(dim=0, keepdim=True)
    return torch_functional.normalize(proto, dim=1, p=2)


def contrast_scores(s_fg: torch.Tensor, s_bg: torch.Tensor, *, lambda_bg: float = 1.0) -> torch.Tensor:
    if s_fg.shape != s_bg.shape:
        raise ValueError("s_fg and s_bg must have the same shape")
    if lambda_bg < 0.0:
        raise ValueError("lambda_bg must be non-negative")
    return s_fg - float(lambda_bg) * s_bg


def score_map_global_features(
    scores: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    high_score_quantile: float = 0.9,
    high_score_threshold: float | None = None,
) -> dict[str, float | int]:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D tensor")
    if not 0.0 <= high_score_quantile <= 1.0:
        raise ValueError("high_score_quantile must be in [0, 1]")
    if valid_mask is not None:
        if valid_mask.shape != scores.shape:
            raise ValueError("valid_mask must match scores shape")
        valid = valid_mask.to(dtype=torch.bool)
    else:
        valid = torch.ones_like(scores, dtype=torch.bool)
    valid_scores = scores[valid].to(dtype=torch.float32)
    if valid_scores.numel() == 0:
        raise ValueError("score map has no valid cells")

    flat_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)
    order = torch.argsort(valid_scores, descending=True)
    sorted_scores = valid_scores[order]
    sorted_flat_indices = flat_indices[order]
    top_flat = int(sorted_flat_indices[0].item())
    cols = scores.shape[1]
    top_row, top_col = divmod(top_flat, cols)
    top_score = float(sorted_scores[0].item())
    second_score = float(sorted_scores[1].item()) if sorted_scores.numel() > 1 else top_score
    probabilities = torch.softmax(valid_scores - valid_scores.max(), dim=0)
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1.0e-12))).sum().item())
    if valid_scores.numel() > 1:
        entropy /= float(torch.log(torch.tensor(float(valid_scores.numel()))).item())
    else:
        entropy = 0.0
    threshold = (
        float(torch.quantile(valid_scores, high_score_quantile).item())
        if high_score_threshold is None
        else float(high_score_threshold)
    )
    high_score_area = float((valid_scores >= threshold).to(dtype=torch.float32).mean().item())
    return {
        "score_count": int(valid_scores.numel()),
        "score_mean": float(valid_scores.mean().item()),
        "score_std": float(valid_scores.std(unbiased=False).item()),
        "top_score": top_score,
        "second_score": second_score,
        "top_margin": top_score - second_score,
        "top_index_row": int(top_row),
        "top_index_col": int(top_col),
        "matching_entropy": entropy,
        "high_score_threshold": threshold,
        "high_score_area_fraction": high_score_area,
    }


def matching_scores_from_features(
    *,
    support_features: torch.Tensor,
    query_features: torch.Tensor,
    foreground_weights: torch.Tensor,
    background_weights: torch.Tensor,
    grid_size: int,
    lambda_bg: float = 1.0,
) -> dict[str, torch.Tensor]:
    fg_proto = _weighted_proto(support_features, foreground_weights)
    bg_proto = _weighted_proto(support_features, background_weights)
    query = torch_functional.normalize(query_features, dim=1, p=2)
    s_fg = (fg_proto @ query.t()).reshape(grid_size, grid_size).detach().cpu()
    s_bg = (bg_proto @ query.t()).reshape(grid_size, grid_size).detach().cpu()
    return {
        "s_fg": s_fg,
        "s_bg": s_bg,
        "s_ctr": contrast_scores(s_fg, s_bg, lambda_bg=lambda_bg),
    }


def build_evidence_metadata(
    *,
    dataset: str,
    episode_id: str,
    support_case_id: str,
    query_case_id: str,
    support_image_path: str,
    support_mask_path: str,
    query_image_path: str,
    matcher_checkpoint_sha256: str,
    feature_grid_size: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "episode_id": episode_id,
        "support_case_id": support_case_id,
        "query_case_id": query_case_id,
        "support_image_path": support_image_path,
        "support_mask_path": support_mask_path,
        "query_image_path": query_image_path,
        "matcher_checkpoint_sha256": matcher_checkpoint_sha256,
        "feature_grid_size": feature_grid_size,
        "query_ground_truth_excluded_from_cache": True,
    }


def load_dinov2_model(checkpoint: Path, *, workspace_root: Path, device: torch.device) -> torch.nn.Module:
    configure_external_paths(workspace_root)
    from dinov2.models import vision_transformer as vits

    architecture = infer_dino_architecture(checkpoint)
    factory = vits.vit_large if architecture == "large" else vits.vit_base
    model = factory(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    state_dict = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def image_to_tensor(image: Any) -> torch.Tensor:
    from torchvision import transforms
    from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

    transform = transforms.Compose(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(image)


@torch.inference_mode()
def extract_dinov2_features(model: torch.nn.Module, image: Any, device: torch.device) -> torch.Tensor:
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    features = model.forward_features(tensor)["x_prenorm"][:, 1:]
    return torch_functional.normalize(features.reshape(-1, model.embed_dim), dim=1, p=2)
