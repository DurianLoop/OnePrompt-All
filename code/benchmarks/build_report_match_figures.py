from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EXPANDED = ROOT / "runs" / "research_direction" / "matcher_medical_expanded"
OUT_DIR = ROOT / "docs" / "reports" / "7.7" / "assets"


def read_manifest() -> dict[str, dict[str, str]]:
    path = EXPANDED / "medical_matcher_pilot_manifest.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["pilot_id"]: row for row in csv.DictReader(f)}


def fit(image: Image.Image, *, width: int, height: int) -> tuple[Image.Image, float, float, int, int]:
    scale = min(width / image.width, height / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#eef3f7")
    ox = (width - new_w) // 2
    oy = (height - new_h) // 2
    canvas.paste(resized.convert("RGB"), (ox, oy))
    return canvas, scale, scale, ox, oy


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def binary_mask(mask: Image.Image) -> Image.Image:
    array = np.asarray(mask.convert("RGB"))
    binary = (array.max(axis=2) > 0).astype(np.uint8) * 255
    return Image.fromarray(binary).convert("L")


def mask_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    array = np.asarray(mask, dtype=np.uint8) > 0
    ys, xs = np.where(array)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, mask.width, mask.height)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_around_mask(image: Image.Image, mask: Image.Image, *, size: int = 620) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = mask_bbox(mask)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = size // 2
    left = max(0, min(image.width - size, cx - half))
    top = max(0, min(image.height - size, cy - half))
    right = min(image.width, left + size)
    bottom = min(image.height, top + size)
    left = max(0, right - size)
    top = max(0, bottom - size)
    box = (int(left), int(top), int(right), int(bottom))
    return image.crop(box), mask.crop(box), box


def overlay_mask(image: Image.Image, mask: Image.Image, *, alpha: int = 160, thicken: int = 7) -> Image.Image:
    base = image.convert("RGBA")
    visible = mask.convert("L")
    if thicken > 1:
        visible = visible.filter(ImageFilter.MaxFilter(thicken))
    red = Image.new("RGBA", base.size, (225, 54, 45, alpha))
    outline = Image.new("RGBA", base.size, (255, 255, 255, 210))
    halo_mask = visible.filter(ImageFilter.MaxFilter(max(3, thicken + 4)))
    base = Image.composite(outline, base, halo_mask)
    base = Image.composite(red, base, visible)
    return base.convert("RGB")


def paste_reference_with_mask(
    canvas: Image.Image,
    *,
    support: Image.Image,
    support_mask: Image.Image,
    left_x: int,
    top_y: int,
    width: int,
    height: int,
) -> None:
    overlay = overlay_mask(support, support_mask, alpha=165, thicken=13 if support.width > 1000 else 5)
    fitted, sx, sy, ox, oy = fit(overlay, width=width, height=height)
    canvas.paste(fitted, (left_x, top_y))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle([left_x, top_y, left_x + width, top_y + height], radius=12, outline="#c7d4e0", width=2)
    draw.text((left_x + 22, top_y + 22), "manual foreground mask overlay", fill="#d94841", font=font(20, bold=True))
    draw.text((left_x + 22, top_y + 50), "red = annotated foreground used for reference prototype", fill="#24384d", font=font(16))

    crop, crop_mask, box = crop_around_mask(support, support_mask, size=720 if support.width > 1000 else 120)
    crop_overlay = overlay_mask(crop, crop_mask, alpha=190, thicken=9 if support.width > 1000 else 3)
    inset, _, _, _, _ = fit(crop_overlay, width=240, height=180)
    inset_x = left_x + width - 262
    inset_y = top_y + height - 206
    draw.rounded_rectangle([inset_x - 6, inset_y - 28, inset_x + 246, inset_y + 186], radius=10, fill=(255, 255, 255, 230), outline="#b9c7d5")
    draw.text((inset_x, inset_y - 24), "mask zoom-in", fill="#17202a", font=font(16, bold=True))
    canvas.paste(inset, (inset_x, inset_y))
    draw.rounded_rectangle([inset_x, inset_y, inset_x + 240, inset_y + 180], radius=8, outline="#d94841", width=3)

    x0, y0, x1, y1 = box
    rect = [left_x + ox + x0 * sx, top_y + oy + y0 * sy, left_x + ox + x1 * sx, top_y + oy + y1 * sy]
    draw.rectangle(rect, outline="#d94841", width=3)


def draw_top_points(draw: ImageDraw.ImageDraw, points: list[dict[str, Any]], *, offset_x: int, offset_y: int) -> None:
    lines = ["top matched points"]
    for point in points[:5]:
        lines.append(
            f"#{int(point['rank'])}: ({float(point['x_original']):.1f}, {float(point['y_original']):.1f}), sim={float(point['similarity']):.3f}"
        )
    panel_w = 520
    panel_h = 34 + 28 * len(lines)
    x = offset_x + 22
    y = offset_y + 22
    draw.rounded_rectangle([x, y, x + panel_w, y + panel_h], radius=10, fill=(255, 255, 255, 232), outline="#b9c7d5")
    for i, line in enumerate(lines):
        draw.text((x + 14, y + 12 + i * 28), line, fill="#17202a", font=font(18, bold=(i == 0)))


def build_case(case_name: str, title: str, output_name: str) -> None:
    manifest = read_manifest()[case_name]
    points_payload = json.loads((EXPANDED / "run" / case_name / "stage1" / "points.json").read_text(encoding="utf-8"))
    support = Image.open(ROOT / manifest["support_image_path"]).convert("RGB")
    support_mask = binary_mask(Image.open(ROOT / manifest["support_mask_path"]))
    query_overlay = Image.open(EXPANDED / "run" / case_name / "stage1" / "topk_overlay.png").convert("RGB")

    left_w, right_w, panel_h = 760, 760, 540
    header_h, footer_h, gap = 88, 92, 58
    canvas = Image.new("RGB", (left_w + right_w + gap + 80, header_h + panel_h + footer_h), "#f6f8fb")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((40, 24), title, fill="#17202a", font=font(30, bold=True))
    draw.text(
        (40, 58),
        "Left: support image with the manual lesion mask overlay. Right: query image with Stage1 top-k matched points.",
        fill="#5f6d7e",
        font=font(18),
    )

    right, _, _, _, _ = fit(query_overlay, width=right_w, height=panel_h)
    left_x = 40
    right_x = 40 + left_w + gap
    top_y = header_h
    paste_reference_with_mask(canvas, support=support, support_mask=support_mask, left_x=left_x, top_y=top_y, width=left_w, height=panel_h)
    canvas.paste(right, (right_x, top_y))

    draw.rounded_rectangle([right_x, top_y, right_x + right_w, top_y + panel_h], radius=12, outline="#c7d4e0", width=2)

    arrow_y = top_y + panel_h // 2
    draw.line([left_x + left_w + 8, arrow_y, right_x - 8, arrow_y], fill="#245fa8", width=5)
    draw.polygon([(right_x - 8, arrow_y), (right_x - 28, arrow_y - 12), (right_x - 28, arrow_y + 12)], fill="#245fa8")
    draw.text((left_x + left_w + 5, arrow_y - 42), "DINO feature\nmatching", fill="#245fa8", font=font(18, bold=True))

    draw_top_points(draw, points=points_payload["points"], offset_x=right_x, offset_y=top_y)
    footer = (
        f"support: {manifest['support_case_id']}  ->  query: {manifest['query_case_id']}    "
        f"selected points: {points_payload['selected_count']} / top_k {points_payload['top_k']}"
    )
    draw.text((40, header_h + panel_h + 28), footer, fill="#2d4054", font=font(19, bold=True))
    draw.text(
        (40, header_h + panel_h + 56),
        f"mask source: {manifest['support_mask_path']}  |  query: {manifest['query_image_path']}",
        fill="#5f6d7e",
        font=font(15),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_DIR / output_name, quality=95)


def main() -> None:
    build_case("idrid_11", "Stage1 Match Figure - IDRID idrid_11", "idrid_11_match_composite.png")
    build_case("monuseg_04", "Stage1 Match Figure - MonuSeg monuseg_04", "monuseg_04_match_composite.png")
    print(f"Wrote match figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
