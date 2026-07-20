from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from torchvision import transforms


WORKSPACE_ROOT = Path(os.environ.get("FSPL_WORKSPACE_ROOT", r"D:\SAM")).resolve()
for source_root in (WORKSPACE_ROOT / "code", WORKSPACE_ROOT / "Matcher"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform  # noqa: E402
from dinov2.models import vision_transformer as vits  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


IMAGE_SIZE = 518
PATCH_SIZE = 14


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_binary_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        values = np.asarray(image)
    return np.any(values > 0, axis=2) if values.ndim == 3 else values > 0


def load_mask_for_features(path: Path) -> Image.Image:
    return Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(image)


def dino_architecture_from_checkpoint(weights: Path) -> str:
    name = weights.name.lower()
    if "vitb" in name:
        return "base"
    if "vitl" in name:
        return "large"
    raise ValueError(f"cannot infer DINOv2 architecture from checkpoint name: {weights.name}")


def load_dinov2(weights: Path, device: torch.device) -> torch.nn.Module:
    factory = vits.vit_base if dino_architecture_from_checkpoint(weights) == "base" else vits.vit_large
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
    state_dict = torch.load(weights, map_location="cpu")
    if isinstance(state_dict, dict) and "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    state_dict = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_features(model: torch.nn.Module, image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = _image_to_tensor(image).unsqueeze(0).to(device)
    features = model.forward_features(tensor)["x_prenorm"][:, 1:]
    return torch_functional.normalize(features.reshape(-1, model.embed_dim), dim=1, p=2)


def _patch_weights(mask: Image.Image, foreground: bool, device: torch.device) -> torch.Tensor:
    values = np.asarray(mask, dtype=np.float32) / 255.0
    if not foreground:
        values = 1.0 - values
    pooled = torch_functional.avg_pool2d(torch.from_numpy(values).view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device), PATCH_SIZE)
    weights = pooled.reshape(-1)
    if float(weights.sum()) <= 0.0:
        label = "foreground" if foreground else "background"
        raise ValueError(f"reference mask has no {label} patch evidence")
    return weights / weights.sum()


@torch.inference_mode()
def reference_prototypes(
    model: torch.nn.Module, reference_image: Image.Image, reference_mask: Image.Image, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_features = extract_features(model, reference_image, device)
    foreground_weights = _patch_weights(reference_mask, foreground=True, device=device)
    background_weights = _patch_weights(reference_mask, foreground=False, device=device)
    foreground_prototype = torch_functional.normalize(
        (reference_features * foreground_weights.unsqueeze(1)).sum(dim=0, keepdim=True), dim=1, p=2
    )
    background_prototype = torch_functional.normalize(
        (reference_features * background_weights.unsqueeze(1)).sum(dim=0, keepdim=True), dim=1, p=2
    )
    return foreground_prototype, background_prototype


@torch.inference_mode()
def matching_maps_from_prototypes(
    model: torch.nn.Module,
    foreground_prototype: torch.Tensor,
    background_prototype: torch.Tensor,
    query_image: Image.Image,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    query_features = extract_features(model, query_image, device)
    grid_size = IMAGE_SIZE // PATCH_SIZE
    foreground = (foreground_prototype @ query_features.t()).reshape(grid_size, grid_size)
    background = (background_prototype @ query_features.t()).reshape(grid_size, grid_size)
    return foreground.cpu().numpy(), background.cpu().numpy()


@torch.inference_mode()
def matching_maps(
    model: torch.nn.Module,
    reference_image: Image.Image,
    reference_mask: Image.Image,
    query_image: Image.Image,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    foreground_prototype, background_prototype = reference_prototypes(
        model, reference_image, reference_mask, device
    )
    return matching_maps_from_prototypes(
        model, foreground_prototype, background_prototype, query_image, device
    )


def sam_architecture_from_checkpoint(checkpoint: Path) -> str:
    name = checkpoint.name.lower()
    for architecture in ("vit_b", "vit_l", "vit_h"):
        if architecture in name:
            return architecture
    raise ValueError(f"cannot infer SAM architecture from checkpoint name: {checkpoint.name}")


def load_sam_predictor(checkpoint: Path, device: torch.device) -> SamPredictor:
    model = sam_model_registry[sam_architecture_from_checkpoint(checkpoint)](checkpoint=str(checkpoint))
    model.eval().to(device)
    return SamPredictor(model)


def predict_sam(
    predictor: SamPredictor,
    image: Image.Image,
    positive_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    if not positive_points:
        raise ValueError("SAM requires at least one positive point")
    points = np.asarray([*positive_points, *negative_points], dtype=np.float32)
    labels = np.asarray([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
    predictor.set_image(np.asarray(image))
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
    best_index = int(np.argmax(scores))
    return masks[best_index].astype(bool), float(scores[best_index])
