from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "reports" / "7.7" / "assets"
RUN_DIR = ROOT / "runs" / "research_direction" / "feedback_learned_matcher_mainline" / "case_split_50"
PREDICTIONS = RUN_DIR / "case_split_50_predictions.csv"
MANIFEST = ROOT / "runs" / "research_direction" / "matcher_medical_expanded_50" / "medical_matcher_pilot_manifest.csv"

PAPER = (246, 248, 251, 255)
WHITE = (255, 255, 255, 255)
BLACK = (12, 18, 26, 255)
MUTED = (66, 82, 100, 255)
LINE = (214, 224, 235, 255)
CYAN = (0, 180, 255, 255)
MAGENTA = (255, 0, 180, 255)
GREEN = (22, 135, 85, 255)
RED = (184, 52, 52, 255)
AMBER = (194, 125, 28, 255)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf", "seguiemj.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_image(path_text: str) -> Image.Image:
    return Image.open(ROOT / path_text).convert("RGB")


def load_mask(path_text: str) -> Image.Image:
    return Image.open(ROOT / path_text).convert("L")


def bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int] | None:
    bbox = mask.point(lambda v: 255 if v > 0 else 0).getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return (x0, y0, x1 - 1, y1 - 1)


def union_bbox(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> tuple[int, int, int, int]:
    if not boxes:
        return (0, 0, width - 1, height - 1)
    return (
        max(0, min(box[0] for box in boxes)),
        max(0, min(box[1] for box in boxes)),
        min(width - 1, max(box[2] for box in boxes)),
        min(height - 1, max(box[3] for box in boxes)),
    )


def pad_bbox(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    pad_ratio: float = 0.18,
    min_size: int = 260,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    half = int(max(min_size, bw, bh) * (0.5 + pad_ratio))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return (max(0, cx - half), max(0, cy - half), min(width - 1, cx + half), min(height - 1, cy + half))


def outline_overlay(base: Image.Image, mask: Image.Image, color: tuple[int, int, int, int], width: int = 6) -> Image.Image:
    out = base.convert("RGBA")
    binary = mask.point(lambda v: 255 if v > 0 else 0)
    edge = binary.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 0 else 0)
    thick = edge
    for _ in range(max(1, width // 2)):
        thick = thick.filter(ImageFilter.MaxFilter(3))
    layer = Image.new("RGBA", base.size, color)
    out.alpha_composite(Image.composite(layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), thick))
    return out


def crop_to_panel(image: Image.Image, box: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    crop = image.crop((box[0], box[1], box[2] + 1, box[3] + 1))
    scale = min(size[0] / max(1, crop.width), size[1] / max(1, crop.height))
    new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    crop = crop.resize(new_size, Image.Resampling.LANCZOS)
    panel = Image.new("RGBA", size, (232, 238, 244, 255))
    panel.alpha_composite(crop.convert("RGBA"), ((size[0] - crop.width) // 2, (size[1] - crop.height) // 2))
    return panel


def best_row(rows: list[dict[str, str]], keep_key: str | None = None) -> dict[str, str] | None:
    candidates = rows if keep_key is None else [row for row in rows if int(row[keep_key]) == 1]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["dice"]))


def manifest_by_case() -> dict[str, dict[str, str]]:
    return {row["pilot_id"]: row for row in read_csv(MANIFEST)}


def prediction_groups() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(PREDICTIONS):
        if row["setting"] == "domain_5fold":
            grouped[row["case_name"]].append(row)
    return grouped


def panel_image(
    query: Image.Image,
    gt_mask: Image.Image,
    proposal: dict[str, str] | None,
    crop_box: tuple[int, int, int, int],
    *,
    none_text: str,
) -> Image.Image:
    composite = outline_overlay(query, gt_mask, CYAN, width=8)
    if proposal is not None:
        proposal_mask = load_mask(proposal["mask_path"])
        composite = outline_overlay(composite.convert("RGB"), proposal_mask, MAGENTA, width=9)
    panel = crop_to_panel(composite, crop_box, (405, 250))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((12, 12, 180, 42), radius=6, fill=(255, 255, 255, 220), outline=LINE, width=1)
    draw.line((26, 28, 58, 28), fill=CYAN, width=5)
    draw.text((68, 18), "GT", fill=BLACK, font=font(17, bold=True))
    draw.line((112, 28, 144, 28), fill=MAGENTA, width=5)
    draw.text((154, 18), "proposal", fill=BLACK, font=font(17, bold=True))
    if proposal is None and none_text != "GT only":
        draw.rounded_rectangle((82, 205, 324, 238), radius=8, fill=(255, 255, 255, 235), outline=RED, width=2)
        draw.text((102, 212), none_text, fill=RED, font=font(17, bold=True))
    return panel


def draw_card(
    canvas: Image.Image,
    xy: tuple[int, int],
    title: str,
    subtitle: str,
    image: Image.Image,
    footer: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    w, h = 445, 365
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=WHITE, outline=LINE, width=2)
    draw.text((x + 18, y + 16), title, fill=BLACK, font=font(23, bold=True))
    draw.text((x + 18, y + 47), subtitle, fill=MUTED, font=font(17))
    canvas.alpha_composite(image, (x + 20, y + 78))
    draw.text((x + 18, y + 340), footer, fill=MUTED, font=font(16))


def draw_row(
    canvas: Image.Image,
    y: int,
    case_name: str,
    description: str,
    rows: list[dict[str, str]],
    manifest: dict[str, str],
) -> None:
    draw = ImageDraw.Draw(canvas)
    query = load_image(manifest["query_image_path"])
    gt_mask = load_mask(manifest["query_mask_path"])
    stage3 = best_row(rows, "stage3_keep")
    learned = best_row(rows, "scalar_keep")
    oracle = best_row(rows)

    boxes: list[tuple[int, int, int, int]] = []
    for row in (stage3, learned, oracle):
        if row is None:
            continue
        mask = load_mask(row["mask_path"])
        box = bbox_from_mask(mask)
        if box is not None:
            boxes.append(box)
    crop_box = pad_bbox(union_bbox(boxes, query.width, query.height), query.width, query.height, min_size=280 if query.width < 1000 else 520)

    row_head = f"{case_name} | fold {rows[0]['fold']} | {rows[0]['dataset']} | {description}"
    draw.text((42, y - 36), row_head, fill=BLACK, font=font(27, bold=True))

    cards = [
        (
            "GT only",
            "manual query mask",
            panel_image(query, gt_mask, None, crop_box, none_text="GT only"),
            f"query={Path(manifest['query_image_path']).name}",
        ),
        (
            "Stage3 selected",
            "best kept by hand rule",
            panel_image(query, gt_mask, stage3, crop_box, none_text="Stage3 selected none"),
            "none" if stage3 is None else f"group_{int(stage3['group_id']):03d}, Dice={float(stage3['dice']):.3f}, label={stage3['label']}",
        ),
        (
            "Domain learned",
            "best kept on test fold",
            panel_image(query, gt_mask, learned, crop_box, none_text="Learned selected none"),
            "none" if learned is None else f"group_{int(learned['group_id']):03d}, Dice={float(learned['dice']):.3f}, label={learned['label']}",
        ),
        (
            "Oracle proposal",
            "best available candidate",
            panel_image(query, gt_mask, oracle, crop_box, none_text="Oracle none"),
            "none" if oracle is None else f"group_{int(oracle['group_id']):03d}, Dice={float(oracle['dice']):.3f}, label={oracle['label']}",
        ),
    ]
    for idx, card in enumerate(cards):
        draw_card(canvas, (40 + idx * 468, y), *card)


def draw_summary_box(canvas: Image.Image, grouped: dict[str, list[dict[str, str]]]) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = (40, 2032, 1912, 2148)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(255, 255, 255, 255), outline=LINE, width=2)
    rows = [row for case_rows in grouped.values() for row in case_rows]
    positives = sum(int(row["label"]) for row in rows)
    stage3_kept = sum(int(row["stage3_keep"]) for row in rows)
    learned_kept = sum(int(row["scalar_keep"]) for row in rows)
    line1 = f"Held-out test-fold predictions from domain_5fold: 50 cases, 600 proposals, {positives} positive proposals."
    line2 = f"Stage3 keeps {stage3_kept} proposals; domain learned keeps {learned_kept}. Cyan = manual GT; magenta = selected SAM proposal."
    draw.text((x0 + 24, y0 + 20), "How to read this qualitative figure", fill=BLACK, font=font(25, bold=True))
    draw.text((x0 + 24, y0 + 62), line1, fill=MUTED, font=font(21))
    draw.text((x0 + 24, y0 + 90), line2, fill=MUTED, font=font(21))


def main() -> None:
    grouped = prediction_groups()
    manifest = manifest_by_case()
    examples = [
        ("monuseg_01", "success: Stage3 selected none, learned selects the best positive"),
        ("idrid_21", "success: learned keeps an extra positive that Stage3 missed"),
        ("monuseg_04", "failure: learned keeps weak false positives while oracle is strong"),
        ("idrid_11", "failure: learned is over-conservative although Stage3 keeps a good mask"),
    ]

    canvas = Image.new("RGBA", (1952, 2190), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 30), "50-case 5-fold qualitative analysis: what the learned policy selected", fill=BLACK, font=font(39, bold=True))
    draw.text(
        (42, 86),
        "Each row is a held-out test case. Panels compare GT, Stage3, domain-learned selection, and the oracle best proposal.",
        fill=MUTED,
        font=font(22),
    )
    draw.rounded_rectangle((42, 130, 670, 178), radius=10, fill=(255, 255, 255, 255), outline=LINE, width=2)
    draw.line((65, 154, 110, 154), fill=CYAN, width=7)
    draw.text((122, 142), "cyan = manual GT", fill=BLACK, font=font(20, bold=True))
    draw.line((350, 154, 395, 154), fill=MAGENTA, width=7)
    draw.text((407, 142), "magenta = selected proposal", fill=BLACK, font=font(20, bold=True))

    y = 230
    for case_name, description in examples:
        draw_row(canvas, y, case_name, description, grouped[case_name], manifest[case_name])
        y += 445
    draw_summary_box(canvas, grouped)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    output = ASSET_DIR / "case_split_50_qualitative_grid.png"
    canvas.convert("RGB").save(output, quality=94)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
