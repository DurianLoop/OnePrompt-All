"""Local SAM3 image demo for this Windows workspace.

This script intentionally avoids notebooks. It expects the SAM3 Hugging Face
checkpoint access to be approved and authenticated before model loading.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DEPS = REPO_ROOT / ".deps_runtime"
if RUNTIME_DEPS.exists():
    sys.path.insert(0, str(RUNTIME_DEPS))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=str(REPO_ROOT / "assets" / "images" / "test_image.jpg"),
        help="Input image path.",
    )
    parser.add_argument("--prompt", default="child", help="Text prompt.")
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional pixel box prompt. Can be combined with text.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable CUDA bfloat16 autocast. The official SAM3 kernels expect autocast on CUDA.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional local checkpoint path. If omitted, SAM3 downloads from gated Hugging Face repo.",
    )
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def normalize_box_xyxy(box: list[float], image: Image.Image) -> list[float]:
    x0, y0, x1, y1 = box
    width, height = image.size
    x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
    y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    bw = max(1.0, x1 - x0) / width
    bh = max(1.0, y1 - y0) / height
    return [cx, cy, bw, bh]


def overlay_masks(image: Image.Image, masks: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (31, 143, 214, 120),
        (46, 160, 67, 120),
        (219, 92, 54, 120),
        (139, 92, 246, 120),
    ]

    masks_np = masks.detach().cpu().numpy()
    boxes_np = boxes.detach().float().cpu().numpy()
    scores_np = scores.detach().float().cpu().numpy()

    for idx, mask in enumerate(masks_np):
        color = colors[idx % len(colors)]
        mask_img = Image.fromarray((mask.squeeze() > 0).astype(np.uint8) * color[3], mode="L")
        color_img = Image.new("RGBA", base.size, color)
        overlay.alpha_composite(Image.composite(color_img, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask_img))
        if idx < len(boxes_np):
            x0, y0, x1, y1 = boxes_np[idx].tolist()
            draw.rectangle((x0, y0, x1, y1), outline=color[:3] + (255,), width=3)
            draw.text((x0 + 4, y0 + 4), f"{scores_np[idx]:.2f}", fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda") and not args.no_autocast
        else contextlib.nullcontext()
    )

    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=args.checkpoint,
        load_from_HF=args.checkpoint is None,
    )
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)
    with autocast_context:
        state = processor.set_image(image)

        if args.prompt:
            state = processor.set_text_prompt(prompt=args.prompt, state=state)
        if args.box is not None:
            state = processor.add_geometric_prompt(
                box=normalize_box_xyxy(args.box, image),
                label=True,
                state=state,
            )

    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sam3_{Path(args.image).stem}_{args.prompt.replace(' ', '_')}.png"
    overlay_masks(image, masks, boxes, scores).save(out_path)
    print(f"masks={len(masks)}")
    print(f"boxes={boxes.detach().float().cpu().tolist()}")
    print(f"scores={scores.detach().float().cpu().tolist()}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
