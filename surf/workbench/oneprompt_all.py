"""OnePrompt-All: single rough prompt to same-class multi-instance masks.

The public interaction is exactly one positive click or one rough box.  SAM3
forms an online seed and DINOv2-B provides feature diagnostics.  Dense
same-class scenes default to zero automatic negatives: an exterior patch may
belong to another target instance, so automatically labelling it background
causes false negatives.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torchvision import transforms

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from dinov2.models import vision_transformer as vits

SIZE, PATCH = 518, 14


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def _nms(masks: np.ndarray, scores: np.ndarray, threshold: float = .5) -> tuple[np.ndarray, np.ndarray]:
    kept: list[int] = []
    for index in np.argsort(-scores):
        mask = masks[index].astype(bool)
        if mask.any() and all(_iou(mask, masks[old]) < threshold for old in kept):
            kept.append(int(index))
    return masks[kept].astype(bool), scores[kept].astype(np.float32)


class OnePromptAllSegmenter:
    def __init__(self, checkpoint: Path, device: torch.device, negative_points: int = 0) -> None:
        self.device, self.negative_points = device, negative_points
        model = vits.vit_base(img_size=SIZE, patch_size=PATCH, init_values=1e-5, ffn_layer="mlp", block_chunks=0, qkv_bias=True, proj_bias=True, ffn_bias=True)
        state = torch.load(checkpoint, map_location="cpu"); state = state.get("teacher", state)
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()
        self.transform = transforms.Compose([MaybeToTensor(), make_normalize_transform(mean=(.485, .456, .406), std=(.229, .224, .225))])

    def _features(self, image: np.ndarray) -> torch.Tensor:
        resized = Image.fromarray(image).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        with torch.inference_mode():
            tokens = self.model.forward_features(self.transform(resized).unsqueeze(0).to(self.device))["x_prenorm"][0, 1:]
        return F.normalize(tokens, dim=1).reshape(37, 37, -1)

    @staticmethod
    def _cells(mask: np.ndarray) -> np.ndarray:
        resized = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((SIZE, SIZE), Image.Resampling.NEAREST), dtype=bool)
        return resized.reshape(37, PATCH, 37, PATCH).mean((1, 3)) > .25

    def _background(self, image: np.ndarray, seed: np.ndarray, count: int) -> list[tuple[int, int]]:
        cells = self._cells(seed)
        if not cells.any() or count <= 0:
            return []
        features = self._features(image); proto = F.normalize(features[cells].mean(0), dim=0)
        similarity = (features @ proto).detach().cpu().numpy()
        distance = distance_transform_edt(~cells)
        quality = -similarity + .12 / (1. + distance); quality[cells] = -np.inf
        selected: list[tuple[int, int]] = []
        for flat in np.argsort(quality.ravel())[::-1]:
            y, x = np.unravel_index(int(flat), quality.shape)
            if not np.isfinite(quality[y, x]) or any((x - sx) ** 2 + (y - sy) ** 2 < 9 for sx, sy in selected):
                continue
            selected.append((int(x), int(y)))
            if len(selected) == count:
                break
        h, w = image.shape[:2]
        return [(min(w - 1, int((x + .5) * w / 37)), min(h - 1, int((y + .5) * h / 37))) for x, y in selected]

    def predict(self, image: np.ndarray, prompt_type: Literal["click", "box"], points: list[Any], box: list[float] | None, sam3: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return all same-class masks from one user prompt; extra points are internal."""
        if prompt_type == "click":
            positives = [point for point in points if int(point.label) == 1]
            if len(positives) != 1:
                raise ValueError("OnePrompt-All accepts exactly one foreground click")
            user_point = positives[0]
            masks, scores = sam3.predict_interactive(image, "click", [user_point], None, "manual")
            px, py = int(user_point.x), int(user_point.y)
        else:
            masks, scores = sam3.predict_interactive(image, "box", [], box, "manual")
            if len(masks) == 0:
                raise ValueError("SAM3 returned no seed for the rough box")
            seed_index = int(np.argmax(scores)); ys, xs = np.where(masks[seed_index])
            px, py = int(xs.mean()), int(ys.mean())
        if len(masks) == 0:
            raise ValueError("SAM3 returned no masks for the prompt")
        containing = [i for i, mask in enumerate(masks) if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px]]
        seed_index = max(containing or list(range(len(masks))), key=lambda i: float(scores[i]))
        negatives = self._background(image, masks[seed_index], self.negative_points)
        # One internal refinement request: positive evidence plus DINO-derived
        # background evidence, never an additional user interaction.
        prompt_points = [type(points[0] if points else object)(x=px, y=py, label=1)] if points else []
        if not prompt_points:
            from demos.medical_sam_click_app import PointPrompt
            prompt_points = [PointPrompt(x=px, y=py, label=1)]
        prompt_points.extend(type(prompt_points[0])(x=x, y=y, label=0) for x, y in negatives)
        masks, scores = sam3.predict_interactive(image, "click", prompt_points, None, "manual")
        min_area = image.shape[0] * image.shape[1] * .001
        keep = [i for i, mask in enumerate(masks) if float(mask.sum()) >= min_area]
        masks, scores = masks[keep], scores[keep]
        masks, scores = _nms(masks, scores)
        return masks, scores, {"user_prompt": prompt_type, "internal_negative_points": len(negatives), "dino": "dinov2_vitb14", "sam3": True}
