from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "reports" / "7.7" / "assets"
RUN_DIR = ROOT / "runs" / "research_direction" / "matcher_medical_expanded"
MANIFEST = RUN_DIR / "medical_matcher_pilot_manifest.csv"
CASEBOOK_20 = ROOT / "runs" / "research_direction" / "feedback_learned_matcher_mainline" / "overall_20" / "medical_casebook_rows.csv"
DOMAIN_PREDICTIONS = ROOT / "runs" / "research_direction" / "feedback_learned_matcher_mainline" / "domain_policy_transfer" / "domain_policy_predictions.csv"


CYAN = (0, 230, 255, 185)
CYAN_LINE = (0, 180, 255, 255)
MAGENTA = (255, 0, 180, 170)
MAGENTA_LINE = (255, 0, 180, 255)
YELLOW = (255, 225, 35, 255)
WHITE = (255, 255, 255, 255)
BLACK = (12, 18, 26, 255)
PAPER = (246, 248, 251, 255)
LINE = (214, 224, 235, 255)
MUTED = (66, 82, 100, 255)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_row(case_name: str) -> dict[str, str]:
    for row in read_csv(MANIFEST):
        if row["pilot_id"] == case_name:
            return row
    raise KeyError(case_name)


def proposal_row(case_name: str, group_id: int) -> dict[str, str]:
    path = RUN_DIR / "run" / case_name / "stage2_proposals" / "proposal_scores.csv"
    for row in read_csv(path):
        if int(row["group_id"]) == group_id:
            return row
    raise KeyError((case_name, group_id))


def casebook_row(case_name: str, group_id: int) -> dict[str, str]:
    for row in read_csv(CASEBOOK_20):
        if row["case_name"] == case_name and int(row["group_id"]) == group_id:
            return row
    raise KeyError((case_name, group_id))


def domain_prediction_row(case_name: str, group_id: int) -> dict[str, str] | None:
    if not DOMAIN_PREDICTIONS.exists():
        return None
    for row in read_csv(DOMAIN_PREDICTIONS):
        if row["case_name"] == case_name and int(row["group_id"]) == group_id:
            return row
    return None


def top_points(case_name: str, limit: int = 50) -> list[dict[str, float]]:
    rows = read_csv(RUN_DIR / "run" / case_name / "stage1" / "points.csv")
    points: list[dict[str, float]] = []
    for row in rows[:limit]:
        points.append(
            {
                "rank": float(row["rank"]),
                "x": float(row["x_original"]),
                "y": float(row["y_original"]),
                "similarity": float(row["similarity"]),
            }
        )
    return points


def load_image(path_text: str) -> Image.Image:
    return Image.open(ROOT / path_text).convert("RGB")


def load_mask(path_text: str) -> Image.Image:
    return Image.open(ROOT / path_text).convert("L")


def bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int]:
    bbox = mask.point(lambda v: 255 if v > 0 else 0).getbbox()
    if bbox is None:
        return (0, 0, mask.width - 1, mask.height - 1)
    x0, y0, x1, y1 = bbox
    return (x0, y0, x1 - 1, y1 - 1)


def pad_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float = 0.25,
    min_size: int = 220,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    pad = int(max(bw, bh, min_size) * pad_ratio)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = max(min_size // 2, bw // 2 + pad, bh // 2 + pad)
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(width - 1, cx + half)
    y1 = min(height - 1, cy + half)
    return (x0, y0, x1, y1)


def union_bbox(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> tuple[int, int, int, int]:
    xs0, ys0, xs1, ys1 = zip(*boxes)
    return (max(0, min(xs0)), max(0, min(ys0)), min(width - 1, max(xs1)), min(height - 1, max(ys1)))


def mask_overlay(base: Image.Image, mask: Image.Image, fill: tuple[int, int, int, int], line: tuple[int, int, int, int]) -> Image.Image:
    out = base.convert("RGBA")
    binary = mask.point(lambda v: 255 if v > 0 else 0)
    fill_layer = Image.new("RGBA", base.size, fill)
    out.alpha_composite(Image.composite(fill_layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), binary))
    edge = binary.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 0 else 0)
    thick = edge.filter(ImageFilter.MaxFilter(7))
    line_layer = Image.new("RGBA", base.size, line)
    out.alpha_composite(Image.composite(line_layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), thick))
    return out


def mask_outline_overlay(base: Image.Image, mask: Image.Image, line: tuple[int, int, int, int], width: int = 7) -> Image.Image:
    out = base.convert("RGBA")
    binary = mask.point(lambda v: 255 if v > 0 else 0)
    edge = binary.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 0 else 0)
    thick = edge
    for _ in range(max(1, width // 2)):
        thick = thick.filter(ImageFilter.MaxFilter(3))
    line_layer = Image.new("RGBA", base.size, line)
    out.alpha_composite(Image.composite(line_layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), thick))
    return out


def binary_mask_image(mask: Image.Image, color: tuple[int, int, int, int], background: tuple[int, int, int, int] = (244, 248, 251, 255)) -> Image.Image:
    binary = mask.point(lambda v: 255 if v > 0 else 0)
    image = Image.new("RGBA", mask.size, background)
    image.alpha_composite(Image.composite(Image.new("RGBA", mask.size, color), Image.new("RGBA", mask.size, (0, 0, 0, 0)), binary))
    return image


def draw_dashed_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int, int], width: int = 5) -> None:
    x0, y0, x1, y1 = box
    dash = 18
    gap = 10
    for x in range(x0, x1, dash + gap):
        draw.line((x, y0, min(x + dash, x1), y0), fill=color, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=color, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=color, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=color, width=width)


def scaled_panel(image: Image.Image, box: tuple[int, int, int, int], target_size: tuple[int, int]) -> tuple[Image.Image, float, float]:
    crop = image.crop((box[0], box[1], box[2] + 1, box[3] + 1))
    scale = min(target_size[0] / max(crop.width, 1), target_size[1] / max(crop.height, 1))
    new_size = (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale))))
    crop = crop.resize(new_size, Image.Resampling.LANCZOS)
    panel = Image.new("RGBA", target_size, (232, 238, 244, 255))
    ox = (target_size[0] - crop.width) // 2
    oy = (target_size[1] - crop.height) // 2
    panel.alpha_composite(crop.convert("RGBA"), (ox, oy))
    sx = crop.width / max(1, box[2] - box[0] + 1)
    sy = crop.height / max(1, box[3] - box[1] + 1)
    return panel, sx, sy


def map_point(x: float, y: float, box: tuple[int, int, int, int], sx: float, sy: float, target_size: tuple[int, int]) -> tuple[int, int]:
    crop_w = int((box[2] - box[0] + 1) * sx)
    crop_h = int((box[3] - box[1] + 1) * sy)
    ox = (target_size[0] - crop_w) // 2
    oy = (target_size[1] - crop_h) // 2
    return int(round(ox + (x - box[0]) * sx)), int(round(oy + (y - box[1]) * sy))


def draw_points(
    draw: ImageDraw.ImageDraw,
    points: list[dict[str, float]],
    crop_box: tuple[int, int, int, int],
    sx: float,
    sy: float,
    target_size: tuple[int, int],
    *,
    limit: int,
    radius: int,
    origin: tuple[int, int] = (0, 0),
    highlight_ranks: set[int] | None = None,
) -> None:
    highlight_ranks = highlight_ranks or set()
    x0, y0, x1, y1 = crop_box
    for point in points[:limit]:
        x = point["x"]
        y = point["y"]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        px, py = map_point(x, y, crop_box, sx, sy, target_size)
        px += origin[0]
        py += origin[1]
        rank = int(point["rank"])
        r = radius * 2 if rank in highlight_ranks else radius
        draw.ellipse((px - r, py - r, px + r, py + r), fill=YELLOW, outline=BLACK, width=max(2, radius // 2))
        if rank in highlight_ranks:
            draw.ellipse((px - r - 6, py - r - 6, px + r + 6, py + r + 6), outline=WHITE, width=4)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle((box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5), radius=8, fill=(17, 26, 36, 230))
    draw.text((x, y), text, fill=WHITE, font=font)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "seguiemj.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def crop_mask(mask: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return mask.crop((box[0], box[1], box[2] + 1, box[3] + 1))


def build_case_figure(
    *,
    case_name: str,
    group_id: int,
    output_name: str,
    title: str,
    note: str,
) -> None:
    row = manifest_row(case_name)
    proposal = proposal_row(case_name, group_id)
    points = top_points(case_name, 80)
    prompt_points = json.loads(proposal["prompt_points"])
    highlight = {int(float(p["rank"])) for p in prompt_points}

    support = load_image(row["support_image_path"])
    support_mask = load_mask(row["support_mask_path"])
    query = load_image(row["query_image_path"])
    query_mask = load_mask(row["query_mask_path"])
    proposal_mask = load_mask(proposal["mask_path"])

    support_box = tuple(int(v) for v in ast.literal_eval(row["support_box_xyxy"]))
    proposal_box = tuple(int(v) for v in ast.literal_eval(proposal["bbox_xyxy"]))
    gt_box = bbox_from_mask(query_mask)
    _ = gt_box
    zoom_box = pad_bbox(proposal_box, query.width, query.height)

    support_overlay = mask_overlay(support, support_mask, CYAN, CYAN_LINE)
    query_overlay = mask_overlay(mask_overlay(query, query_mask, CYAN, CYAN_LINE), proposal_mask, MAGENTA, MAGENTA_LINE)

    query_draw = ImageDraw.Draw(query_overlay)
    draw_dashed_box(query_draw, proposal_box, WHITE, width=max(4, query.width // 900))
    draw_dashed_box(ImageDraw.Draw(support_overlay), support_box, WHITE, width=max(4, support.width // 900))

    canvas_w = 1840
    canvas_h = 1340
    canvas = Image.new("RGBA", (canvas_w, canvas_h), PAPER)
    d = ImageDraw.Draw(canvas)
    title_font = font(38)
    head_font = font(25)
    label_font = font(19)
    small_font = font(18)
    d.text((38, 28), title, fill=BLACK, font=title_font)
    d.text((40, 82), note, fill=(66, 82, 100, 255), font=small_font)

    panel_size = (560, 420)
    support_panel, sx_s, sy_s = scaled_panel(support_overlay, (0, 0, support.width - 1, support.height - 1), panel_size)
    query_panel, sx_q, sy_q = scaled_panel(query_overlay, (0, 0, query.width - 1, query.height - 1), panel_size)

    zoom_base = query.crop((zoom_box[0], zoom_box[1], zoom_box[2] + 1, zoom_box[3] + 1))
    zoom_gt = crop_mask(query_mask, zoom_box)
    zoom_prop = crop_mask(proposal_mask, zoom_box)
    zoom_overlay = mask_overlay(mask_overlay(zoom_base, zoom_gt, CYAN, CYAN_LINE), zoom_prop, MAGENTA, MAGENTA_LINE)
    zoom_panel, sx_z, sy_z = scaled_panel(zoom_overlay, (0, 0, zoom_overlay.width - 1, zoom_overlay.height - 1), (640, 520))

    d.rounded_rectangle((30, 130, 610, 620), radius=16, fill=(255, 255, 255, 255), outline=(214, 224, 235, 255), width=2)
    d.rounded_rectangle((640, 130, 1220, 620), radius=16, fill=(255, 255, 255, 255), outline=(214, 224, 235, 255), width=2)
    d.rounded_rectangle((1250, 130, 1810, 620), radius=16, fill=(255, 255, 255, 255), outline=(214, 224, 235, 255), width=2)
    d.text((48, 148), "A. Reference image + manual mask", fill=BLACK, font=head_font)
    d.text((658, 148), "B. Query image: GT / proposal / points", fill=BLACK, font=head_font)
    d.text((1268, 148), "C. Legend", fill=BLACK, font=head_font)
    canvas.alpha_composite(support_panel, (40, 190))
    canvas.alpha_composite(query_panel, (650, 190))

    q_draw = ImageDraw.Draw(canvas)
    draw_points(
        q_draw,
        points,
        (0, 0, query.width - 1, query.height - 1),
        sx_q,
        sy_q,
        panel_size,
        limit=60,
        radius=5,
        origin=(650, 190),
        highlight_ranks=highlight,
    )
    draw_label(q_draw, (672, 202), "yellow = matched points", label_font)
    draw_label(q_draw, (62, 202), "cyan = manual reference mask", label_font)

    legend_y = 220
    for color, label in [
        (CYAN_LINE, "cyan: GT/manual mask"),
        (MAGENTA_LINE, "magenta: best SAM proposal mask"),
        (YELLOW, "yellow dots: matched/prompt points"),
        (WHITE, "white dashed box: visual bbox only"),
    ]:
        d.rounded_rectangle((1270, legend_y, 1328, legend_y + 34), radius=8, fill=color, outline=BLACK, width=2)
        d.text((1344, legend_y - 1), label, fill=BLACK, font=label_font)
        legend_y += 62
    d.text((1270, 490), "The box is drawn to make tiny regions visible.\nThe prototype comes from the manual mask.", fill=(66, 82, 100, 255), font=small_font)

    d.rounded_rectangle((30, 660, 1810, 1290), radius=16, fill=(255, 255, 255, 255), outline=(214, 224, 235, 255), width=2)
    d.text((50, 682), "D. Zoom around best proposal region: where the mask and prompt points sit", fill=BLACK, font=head_font)
    canvas.alpha_composite(zoom_panel, (50, 742))
    z_draw = ImageDraw.Draw(canvas)
    shifted_points = [{"rank": p["rank"], "x": p["x"] - zoom_box[0], "y": p["y"] - zoom_box[1], "similarity": p["similarity"]} for p in points]
    draw_points(
        z_draw,
        shifted_points,
        (0, 0, zoom_box[2] - zoom_box[0], zoom_box[3] - zoom_box[1]),
        sx_z,
        sy_z,
        (640, 520),
        limit=80,
        radius=9,
        origin=(50, 742),
        highlight_ranks=highlight,
    )
    d.text((740, 760), f"case = {case_name}", fill=BLACK, font=head_font)
    d.text((740, 812), f"best proposal = group_{group_id:03d}", fill=BLACK, font=head_font)
    d.text((740, 864), f"bbox = {proposal['bbox_xyxy']}", fill=(66, 82, 100, 255), font=label_font)
    d.text((740, 910), f"prompt ranks = {sorted(highlight)}", fill=(66, 82, 100, 255), font=label_font)
    d.text((740, 956), "Reading order:\n1) manual mask defines reference foreground\n2) DINO matching gives yellow points on query\n3) SAM makes proposal masks from prompt groups\n4) learning decides keep/reject", fill=(66, 82, 100, 255), font=small_font)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(ASSET_DIR / output_name, quality=94)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font_obj: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font_obj)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    image: Image.Image | None,
    caption: str,
    *,
    image_box: tuple[int, int, int, int] | None = None,
    points: list[dict[str, float]] | None = None,
    point_crop_box: tuple[int, int, int, int] | None = None,
    highlight_ranks: set[int] | None = None,
    point_limit: int = 80,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    head_font = font(25)
    cap_font = font(18)
    draw.rounded_rectangle(box, radius=16, fill=(255, 255, 255, 255), outline=LINE, width=2)
    draw.text((x0 + 18, y0 + 16), title, fill=BLACK, font=head_font)
    media_origin = (x0 + 18, y0 + 58)
    media_size = (x1 - x0 - 36, y1 - y0 - 160)
    if image is not None:
        source_box = image_box or (0, 0, image.width - 1, image.height - 1)
        panel, sx, sy = scaled_panel(image, source_box, media_size)
        canvas.alpha_composite(panel, media_origin)
        if points is not None and point_crop_box is not None:
            draw_points(
                draw,
                points,
                point_crop_box,
                sx,
                sy,
                media_size,
                limit=point_limit,
                radius=6 if media_size[0] > 500 else 5,
                origin=media_origin,
                highlight_ranks=highlight_ranks,
            )
    cap_y = y1 - 88
    for line in wrap_text(draw, caption, x1 - x0 - 36, cap_font)[:4]:
        draw.text((x0 + 18, cap_y), line, fill=MUTED, font=cap_font)
        cap_y += 24


def draw_metric_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    rows: list[tuple[str, str]],
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    head_font = font(25)
    label_font = font(19)
    value_font = font(22)
    draw.rounded_rectangle(box, radius=16, fill=(255, 255, 255, 255), outline=LINE, width=2)
    draw.text((x0 + 18, y0 + 16), title, fill=BLACK, font=head_font)
    y = y0 + 68
    for label, value in rows:
        draw.rounded_rectangle((x0 + 18, y, x1 - 18, y + 50), radius=10, fill=(247, 250, 253, 255), outline=(226, 234, 242, 255), width=1)
        draw.text((x0 + 34, y + 13), label, fill=MUTED, font=label_font)
        draw.text((x0 + 300, y + 11), value, fill=BLACK, font=value_font)
        y += 58


def build_case_separated_figure(
    *,
    case_name: str,
    group_id: int,
    output_name: str,
    title: str,
    note: str,
) -> None:
    row = manifest_row(case_name)
    proposal = proposal_row(case_name, group_id)
    book = casebook_row(case_name, group_id)
    domain = domain_prediction_row(case_name, group_id)
    points = top_points(case_name, 100)
    prompt_points = json.loads(proposal["prompt_points"])
    highlight = {int(float(p["rank"])) for p in prompt_points}

    support = load_image(row["support_image_path"])
    support_mask = load_mask(row["support_mask_path"])
    query = load_image(row["query_image_path"])
    query_mask = load_mask(row["query_mask_path"])
    proposal_mask = load_mask(proposal["mask_path"])

    support_box = tuple(int(v) for v in ast.literal_eval(row["support_box_xyxy"]))
    proposal_box = tuple(int(v) for v in ast.literal_eval(proposal["bbox_xyxy"]))
    zoom_box = pad_bbox(proposal_box, query.width, query.height, pad_ratio=0.18, min_size=180)

    support_ref = mask_outline_overlay(support, support_mask, CYAN_LINE, width=7)
    query_gt = mask_outline_overlay(query, query_mask, CYAN_LINE, width=7)
    query_prop = mask_outline_overlay(query, proposal_mask, MAGENTA_LINE, width=8)
    query_compare = mask_outline_overlay(mask_outline_overlay(query, query_mask, CYAN_LINE, width=7), proposal_mask, MAGENTA_LINE, width=8)
    _ = support_box

    canvas = Image.new("RGBA", (2000, 1540), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 28), title, fill=BLACK, font=font(42))
    draw.text((44, 84), note, fill=MUTED, font=font(20))

    panel_w = 610
    panel_h = 610
    gap = 35
    x_positions = [40, 40 + panel_w + gap, 40 + (panel_w + gap) * 2]
    y_positions = [130, 130 + panel_h + gap]
    cards = [
        (x_positions[0], y_positions[0], x_positions[0] + panel_w, y_positions[0] + panel_h),
        (x_positions[1], y_positions[0], x_positions[1] + panel_w, y_positions[0] + panel_h),
        (x_positions[2], y_positions[0], x_positions[2] + panel_w, y_positions[0] + panel_h),
        (x_positions[0], y_positions[1], x_positions[0] + panel_w, y_positions[1] + panel_h),
        (x_positions[1], y_positions[1], x_positions[1] + panel_w, y_positions[1] + panel_h),
        (x_positions[2], y_positions[1], x_positions[2] + panel_w, y_positions[1] + panel_h),
    ]

    draw_panel(
        canvas,
        cards[0],
        "A. Reference image + manual mask",
        support_ref,
        "Cyan outline is the manual reference mask. No box prompt is used in this panel.",
    )
    draw_panel(
        canvas,
        cards[1],
        "B. Query GT mask only",
        query_gt,
        "Only query image and GT outline are shown here. No proposal or point is drawn in this panel.",
    )
    draw_panel(
        canvas,
        cards[2],
        "C. Stage1 top-k points only",
        query,
        "Yellow dots are top-k matched points on the query image. Large circles indicate prompt ranks used by this proposal.",
        points=points,
        point_crop_box=(0, 0, query.width - 1, query.height - 1),
        highlight_ranks=highlight,
        point_limit=100,
    )
    draw_panel(
        canvas,
        cards[3],
        f"D. SAM proposal group_{group_id:03d}",
        query_prop,
        "Magenta outline is the SAM proposal. BBox is reported only as a numeric metric in panel F.",
    )
    shifted_points = [{"rank": p["rank"], "x": p["x"] - zoom_box[0], "y": p["y"] - zoom_box[1], "similarity": p["similarity"]} for p in points]
    compare_crop = query_compare.crop((zoom_box[0], zoom_box[1], zoom_box[2] + 1, zoom_box[3] + 1))
    draw_panel(
        canvas,
        cards[4],
        "E. Boundary comparison zoom",
        compare_crop,
        "Zoom view without bbox: cyan = GT boundary, magenta = proposal boundary, yellow = matched/prompt points.",
        points=shifted_points,
        point_crop_box=(0, 0, zoom_box[2] - zoom_box[0], zoom_box[3] - zoom_box[1]),
        highlight_ranks=highlight,
        point_limit=100,
    )
    metric_rows = [
        ("Dice / IoU", f"{float(book['dice']):.6f} / {float(book['iou']):.6f}"),
        ("SAM score", f"{float(proposal['sam_score']):.6f}"),
        ("Mask area", f"{float(proposal['mask_area_pct']):.4f}%"),
        ("BBox xyxy", proposal["bbox_xyxy"]),
        ("Prompt ranks", str(sorted(highlight))),
        ("Stage3 keep", str(int(float(book["stage3_keep"])))),
        ("Overall learned keep", str(int(float(book["policy_keep"])))),
        ("Domain learned keep", str(int(float(domain["domain_scalar_keep"])) if domain else "NA")),
    ]
    draw_metric_panel(canvas, cards[5], "F. Proposal metrics", metric_rows)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(ASSET_DIR / output_name, quality=94)


def main() -> None:
    build_case_figure(
        case_name="idrid_11",
        group_id=5,
        output_name="idrid_11_detail_strong.png",
        title="IDRID idrid_11: manual lesion mask matches a new retinal image",
        note="The reference is the manual lesion mask on IDRiD_01; IDRiD_12 is the query. Points and boxes have different meanings.",
    )
    build_case_figure(
        case_name="monuseg_04",
        group_id=8,
        output_name="monuseg_04_detail_strong.png",
        title="MonuSeg monuseg_04: domain-specific filtering failure",
        note="The reference is the manual nuclei mask on 01_1; 01_5 is the query. Stage2 finds usable proposals, Stage3 drops the best one.",
    )
    build_case_separated_figure(
        case_name="idrid_11",
        group_id=5,
        output_name="idrid_11_separated_evidence.png",
        title="IDRID idrid_11: separated evidence with metrics",
        note="Reference, GT, matched points, proposal, boundary comparison, and metrics are split to avoid color occlusion.",
    )
    build_case_separated_figure(
        case_name="monuseg_04",
        group_id=8,
        output_name="monuseg_04_separated_evidence.png",
        title="MonuSeg monuseg_04: separated evidence with metrics",
        note="Reference, GT, matched points, proposal, boundary comparison, and metrics are split to expose the domain filtering issue.",
    )
    print(f"Wrote detail figures to {ASSET_DIR}")


if __name__ == "__main__":
    main()
