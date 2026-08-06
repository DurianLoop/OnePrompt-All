"""Create a contact sheet from OnePrompt-All per-image visualization files."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg visualizations found in {args.input_dir}")
    thumb_w, thumb_h = 440, 180
    rows = (len(files) + args.columns - 1) // args.columns
    sheet = Image.new("RGB", (args.columns * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 8, thumb_h - 26), Image.Resampling.LANCZOS)
        x = (index % args.columns) * thumb_w + 4
        y = (index // args.columns) * thumb_h + 22
        sheet.paste(image, (x, y))
        draw.text((x, y - 18), path.stem, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
