"""Build a contact sheet from existing SAM3 text probe overlays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROMPTS = ["person", "girl", "woman", "white girl", "execute"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-csv", default="runs/teacher_image_batch/sam3_text_probe/results.csv")
    parser.add_argument("--out", default="runs/teacher_image_batch/sam3_text_probe/probe_contact_sheet.jpg")
    parser.add_argument("--tile-w", type=int, default=220)
    parser.add_argument("--tile-h", type=int, default=170)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    result_csv = Path(args.result_csv)
    rows = []
    with result_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_image: dict[str, dict[str, dict[str, str]]] = {}
    image_order: list[str] = []
    for row in rows:
        image = row["image"]
        prompt = row["prompt"]
        if image not in by_image:
            by_image[image] = {}
            image_order.append(image)
        by_image[image][prompt] = row

    margin = 18
    label_h = 42
    row_title_w = 210
    cols = len(PROMPTS)
    rows_n = len(image_order)
    width = margin * 2 + row_title_w + cols * args.tile_w
    height = margin * 2 + label_h + rows_n * (args.tile_h + label_h)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)
    small = load_font(12)
    bold = load_font(18)

    draw.text((margin, margin + 6), "image", fill=(20, 29, 43), font=bold)
    for c, prompt in enumerate(PROMPTS):
        x = margin + row_title_w + c * args.tile_w
        draw.text((x + 8, margin + 6), prompt, fill=(20, 29, 43), font=bold)

    for r, image in enumerate(image_order):
        y0 = margin + label_h + r * (args.tile_h + label_h)
        stem = Path(image).stem
        draw.rectangle((margin, y0, width - margin, y0 + args.tile_h + label_h - 4), outline=(220, 226, 235))
        draw.text((margin + 8, y0 + 10), stem, fill=(20, 29, 43), font=bold)
        draw.text((margin + 8, y0 + 34), Path(image).name, fill=(71, 85, 105), font=small)

        for c, prompt in enumerate(PROMPTS):
            row = by_image[image].get(prompt)
            x0 = margin + row_title_w + c * args.tile_w
            if not row:
                continue
            overlay = Path(row["overlay"])
            if not overlay.exists():
                overlay = result_csv.parent / row["overlay"]
            if not overlay.exists():
                continue

            img = fit_image(Image.open(overlay), args.tile_w - 18, args.tile_h - 28)
            px = x0 + (args.tile_w - img.width) // 2
            py = y0 + 8
            canvas.paste(img, (px, py))

            inst = row.get("instances", "")
            area = row.get("union_area_pct", "")
            score = row.get("score_best", "")
            text = f"inst={inst} area={area}%"
            if score:
                text += f" score={float(score):.2f}"
            draw.text((x0 + 8, y0 + args.tile_h - 17), text, fill=(51, 65, 85), font=small)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)
    print(out)


if __name__ == "__main__":
    main()
