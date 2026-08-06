from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
MATCHER_ROOT = WORKSPACE_ROOT / "Matcher"
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE_ROOT / ".cache" / "matplotlib_stage1_exporter"))
for path in (str(CODE_ROOT), str(MATCHER_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform  # noqa: E402
from dinov2.models import vision_transformer as vits  # noqa: E402


IMAGE_SIZE = 518


@dataclass
class MatchPoint:
    rank: int
    patch_index: int
    patch_x: int
    patch_y: int
    x_518: int
    y_518: int
    x_original: float
    y_original: float
    similarity: float
    selected_by_threshold: bool


def parse_box(text: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--reference-box must be formatted as x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--reference-box must have x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def resized_rgb(path: Path) -> tuple[Image.Image, Image.Image]:
    original = load_rgb(path)
    resized = original.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return original, resized


def make_reference_mask(
    reference_image: Image.Image,
    reference_box: tuple[float, float, float, float] | None,
    reference_mask_path: Path | None,
) -> Image.Image:
    if reference_mask_path is not None:
        if not reference_mask_path.exists():
            raise FileNotFoundError(reference_mask_path)
        mask = Image.open(reference_mask_path).convert("L")
        return mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    if reference_box is None:
        raise ValueError("Provide either --reference-mask or --reference-box")

    orig_w, orig_h = reference_image.size
    x0, y0, x1, y1 = reference_box
    scale_x = IMAGE_SIZE / float(orig_w)
    scale_y = IMAGE_SIZE / float(orig_h)
    box_518 = [
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    ]
    box_518[0] = max(0, min(IMAGE_SIZE - 1, box_518[0]))
    box_518[1] = max(0, min(IMAGE_SIZE - 1, box_518[1]))
    box_518[2] = max(0, min(IMAGE_SIZE - 1, box_518[2]))
    box_518[3] = max(0, min(IMAGE_SIZE - 1, box_518[3]))

    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box_518, fill=255)
    return mask


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image)


def load_dinov2(weights: Path, size: str, device: torch.device) -> torch.nn.Module:
    kwargs = dict(
        img_size=IMAGE_SIZE,
        patch_size=14,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    if size not in vits.__dict__:
        raise ValueError(f"Unknown DINOv2 size: {size}")
    model = vits.__dict__[size](**kwargs)
    state_dict = torch.load(weights, map_location="cpu")
    if isinstance(state_dict, dict) and "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    state_dict = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


@torch.inference_mode()
def extract_features(model: torch.nn.Module, image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    features = model.forward_features(tensor)["x_prenorm"][:, 1:]
    features = features.reshape(-1, model.embed_dim)
    return F.normalize(features, dim=1, p=2)


def mask_to_patch_weights(mask: Image.Image, patch_size: int, device: torch.device) -> torch.Tensor:
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    pooled = F.avg_pool2d(mask_t, (patch_size, patch_size))
    weights = pooled.reshape(-1)
    if float(weights.sum().detach().cpu()) <= 0.0:
        raise ValueError("Reference mask/box produced no foreground patches")
    return weights / weights.sum()


def compute_stage1_matches(
    model: torch.nn.Module,
    reference_image: Image.Image,
    target_image: Image.Image,
    reference_mask: Image.Image,
    target_original_size: tuple[int, int],
    top_k: int,
    threshold: float,
    device: torch.device,
) -> tuple[list[MatchPoint], np.ndarray]:
    ref_features = extract_features(model, reference_image, device)
    target_features = extract_features(model, target_image, device)
    patch_size = int(model.patch_size)
    feat_size = IMAGE_SIZE // patch_size

    weights = mask_to_patch_weights(reference_mask, patch_size, device)
    ref_mask = weights > 0
    ref_proto = (ref_features[ref_mask] * weights[ref_mask].unsqueeze(1)).sum(dim=0, keepdim=True)
    ref_proto = F.normalize(ref_proto, dim=1, p=2)

    similarity = (ref_proto @ target_features.t()).squeeze(0)
    safe_top_k = min(int(top_k), int(similarity.numel()))
    top_scores, top_indices = similarity.topk(k=safe_top_k)
    heatmap = similarity.reshape(feat_size, feat_size).detach().cpu().numpy()

    target_w, target_h = target_original_size
    points: list[MatchPoint] = []
    for rank, (score_t, index_t) in enumerate(zip(top_scores, top_indices), start=1):
        patch_index = int(index_t.detach().cpu())
        patch_x = patch_index % feat_size
        patch_y = patch_index // feat_size
        x_518 = int(patch_x * patch_size + patch_size // 2)
        y_518 = int(patch_y * patch_size + patch_size // 2)
        x_original = (float(x_518) / IMAGE_SIZE) * float(target_w)
        y_original = (float(y_518) / IMAGE_SIZE) * float(target_h)
        similarity_value = float(score_t.detach().cpu())
        points.append(MatchPoint(
            rank=rank,
            patch_index=patch_index,
            patch_x=patch_x,
            patch_y=patch_y,
            x_518=x_518,
            y_518=y_518,
            x_original=round(x_original, 3),
            y_original=round(y_original, 3),
            similarity=round(similarity_value, 6),
            selected_by_threshold=similarity_value >= threshold,
        ))
    return points, heatmap


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    min_v = float(heatmap.min())
    max_v = float(heatmap.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - min_v) / (max_v - min_v)


def save_overlay(target_original: Image.Image, points: list[MatchPoint], out_path: Path, threshold: float) -> None:
    image = target_original.copy()
    draw = ImageDraw.Draw(image)
    for point in reversed(points):
        x = float(point.x_original)
        y = float(point.y_original)
        radius = 5 if point.selected_by_threshold else 3
        fill = (255, 0, 0) if point.selected_by_threshold else (255, 230, 0)
        outline = (255, 255, 255) if point.selected_by_threshold else (0, 0, 0)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    legend_y = max(4, image.height - 46)
    draw.rectangle((4, legend_y, min(image.width - 4, 360), image.height - 4), fill=(0, 0, 0))
    draw.text((12, legend_y + 8), f"top-{len(points)} points | red >= {threshold}", fill=(255, 255, 255))
    image.save(out_path)


def save_heatmap(target_original: Image.Image, heatmap: np.ndarray, out_path: Path) -> None:
    normalized = normalize_heatmap(heatmap)
    gray = Image.fromarray((normalized * 255.0).astype(np.uint8)).resize(target_original.size, Image.BICUBIC)
    color = ImageOps.colorize(gray, black=(0, 0, 40), white=(255, 60, 0), mid=(240, 200, 40))
    blended = Image.blend(target_original.convert("RGB"), color.convert("RGB"), alpha=0.55)
    blended.save(out_path)


def save_outputs(
    out_dir: Path,
    points: list[MatchPoint],
    heatmap: np.ndarray,
    reference_image_path: Path,
    target_image_path: Path,
    reference_box: tuple[float, float, float, float] | None,
    reference_mask_path: Path | None,
    target_original: Image.Image,
    top_k: int,
    threshold: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(point) for point in points]
    with (out_dir / "points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(MatchPoint.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload: dict[str, Any] = {
        "reference_image": str(reference_image_path.relative_to(WORKSPACE_ROOT)),
        "target_image": str(target_image_path.relative_to(WORKSPACE_ROOT)),
        "reference_box": list(reference_box) if reference_box is not None else None,
        "reference_mask": str(reference_mask_path.relative_to(WORKSPACE_ROOT)) if reference_mask_path is not None else None,
        "image_size": IMAGE_SIZE,
        "top_k": top_k,
        "threshold": threshold,
        "selected_count": sum(1 for point in points if point.selected_by_threshold),
        "points": rows,
        "artifacts": {
            "points_csv": str((out_dir / "points.csv").relative_to(WORKSPACE_ROOT)),
            "points_json": str((out_dir / "points.json").relative_to(WORKSPACE_ROOT)),
            "similarity_npy": str((out_dir / "similarity.npy").relative_to(WORKSPACE_ROOT)),
            "topk_overlay": str((out_dir / "topk_overlay.png").relative_to(WORKSPACE_ROOT)),
            "similarity_heatmap": str((out_dir / "similarity_heatmap.png").relative_to(WORKSPACE_ROOT)),
            "summary": str((out_dir / "summary.md").relative_to(WORKSPACE_ROOT)),
        },
    }
    (out_dir / "points.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "similarity.npy", heatmap.astype(np.float32))

    save_overlay(target_original, points, out_dir / "topk_overlay.png", threshold)
    save_heatmap(target_original, heatmap, out_dir / "similarity_heatmap.png")

    best = points[0] if points else None
    summary = [
        "# Matcher Stage 1 Export Summary",
        "",
        f"- Reference image: `{payload['reference_image']}`",
        f"- Target image: `{payload['target_image']}`",
        f"- Reference box: `{payload['reference_box']}`",
        f"- Reference mask: `{payload['reference_mask']}`",
        f"- Top-k: `{top_k}`",
        f"- Threshold: `{threshold}`",
        f"- Selected count: `{payload['selected_count']}`",
    ]
    if best is not None:
        summary.extend([
            f"- Best point: rank `{best.rank}`, original xy `({best.x_original}, {best.y_original})`, similarity `{best.similarity}`",
        ])
    summary.extend([
        "",
        "## Artifacts",
        "",
        f"- `points.csv`",
        f"- `points.json`",
        f"- `similarity.npy`",
        f"- `topk_overlay.png`",
        f"- `similarity_heatmap.png`",
    ])
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Matcher Stage 1 DINO matched points.")
    parser.add_argument("--reference-image", default="code/gradio_demo/images/elephant1.png")
    parser.add_argument("--target-image", default="code/gradio_demo/images/elephant2.png")
    parser.add_argument("--reference-box", default="0,60,295,425")
    parser.add_argument("--reference-mask", default="")
    parser.add_argument("--dinov2-weights", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"))
    parser.add_argument("--dinov2-size", default="vit_large")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--out-dir", default=str(WORKSPACE_ROOT / "runs" / "research_direction" / "stage1_matching" / "elephant_pair"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    reference_image_path = resolve_path(args.reference_image)
    target_image_path = resolve_path(args.target_image)
    reference_mask_path = resolve_path(args.reference_mask) if args.reference_mask else None
    weights_path = resolve_path(args.dinov2_weights)
    out_dir = resolve_path(args.out_dir)

    reference_original, reference_resized = resized_rgb(reference_image_path)
    target_original, target_resized = resized_rgb(target_image_path)
    reference_box = parse_box(args.reference_box) if args.reference_box else None
    reference_mask = make_reference_mask(reference_original, reference_box, reference_mask_path)

    model = load_dinov2(weights_path, args.dinov2_size, device)
    points, heatmap = compute_stage1_matches(
        model=model,
        reference_image=reference_resized,
        target_image=target_resized,
        reference_mask=reference_mask,
        target_original_size=target_original.size,
        top_k=args.top_k,
        threshold=args.threshold,
        device=device,
    )
    save_outputs(
        out_dir=out_dir,
        points=points,
        heatmap=heatmap,
        reference_image_path=reference_image_path,
        target_image_path=target_image_path,
        reference_box=reference_box,
        reference_mask_path=reference_mask_path,
        target_original=target_original,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    selected_count = sum(1 for point in points if point.selected_by_threshold)
    print(f"Exported {len(points)} top points; selected_by_threshold={selected_count}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
