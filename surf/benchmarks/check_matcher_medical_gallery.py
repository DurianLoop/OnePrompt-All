from __future__ import annotations

import csv
import argparse
import re
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_html_src(src: str, html_dir: Path) -> Path:
    normalized = src.replace("/", "\\")
    return (html_dir / normalized).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check medical proposal review gallery image links and lightbox behavior.")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_medical_expanded")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = WORKSPACE_ROOT / out_dir
    out_dir = out_dir.resolve()
    html_path = out_dir / "review_gallery.html"
    casebook_path = out_dir / "casebook" / "medical_casebook_rows.csv"
    if not html_path.exists():
        print(f"Missing gallery: {html_path}")
        return 1
    if not casebook_path.exists():
        print(f"Missing casebook rows: {casebook_path}")
        return 1

    expected = read_csv(casebook_path)
    text = html_path.read_text(encoding="utf-8")
    proposal_img_tags = re.findall(r"<img\b[^>]*data-proposal-image=\"1\"[^>]*>", text)
    image_srcs: list[str] = []
    for tag in proposal_img_tags:
        match = re.search(r"\bsrc=\"([^\"]*)\"", tag)
        image_srcs.append(match.group(1) if match else "")

    if len(image_srcs) != len(expected):
        print(f"Expected {len(expected)} proposal images, found {len(image_srcs)}")
        return 1

    empty = [src for src in image_srcs if not src.strip()]
    if empty:
        print(f"Found {len(empty)} empty image src values")
        return 1

    missing = [src for src in image_srcs if not resolve_html_src(src, html_path.parent).exists()]
    if missing:
        print(f"Missing {len(missing)} image files; first missing: {missing[0]}")
        return 1

    if 'id="lightbox"' not in text or "openLightbox" not in text or not re.search(r"addEventListener\('click'.*openLightbox", text, re.S):
        print("Gallery does not expose proposal click-to-enlarge lightbox behavior")
        return 1

    print("Matcher medical gallery checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
