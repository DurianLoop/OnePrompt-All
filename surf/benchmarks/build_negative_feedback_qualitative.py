from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def paired_extremes(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["query_case_id"]][int(row["negative_count"])] = row
    pairs: list[dict[str, Any]] = []
    for query_id, by_count in grouped.items():
        if 0 not in by_count or 5 not in by_count:
            continue
        pairs.append({
            "query_case_id": query_id,
            "neg0": by_count[0],
            "neg5": by_count[5],
            "delta_dice": float(by_count[5]["dice"]) - float(by_count[0]["dice"]),
        })
    if not pairs:
        raise ValueError("No paired N=0/N=5 cases found.")
    return max(pairs, key=lambda row: row["delta_dice"]), min(pairs, key=lambda row: row["delta_dice"])


def labeled_panel(row: dict[str, str], label: str, size: int = 620) -> Image.Image:
    image = Image.open(ROOT / row["overlay_path"]).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size + 58), "white")
    panel.paste(image, ((size - image.width) // 2, 58))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=20)
    draw.text((16, 15), f"{label} | Dice {float(row['dice']):.3f}", fill="#17212B", font=font)
    return panel


def main() -> None:
    rows = read_csv(ROOT / "runs/research_direction/mechanisms/negative_prompt_ablation_test14/per_case.csv")
    gain, loss = paired_extremes(rows)
    examples = [("Largest gain", gain), ("Largest loss", loss)]
    panel_size = 620
    header_height = 92
    canvas = Image.new("RGB", (panel_size * 2, header_height + (panel_size + 58) * 2), "#F4F7F9")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=24)
    legend_font = ImageFont.load_default(size=16)
    draw.text((20, 20), "Controlled negative feedback: N=0 versus N=5", fill="#17212B", font=title_font)
    draw.text((20, 56), "Green: TP | Red: FP | Blue: FN | Yellow circles: positive prompts | Cyan crosses: negative prompts", fill="#41505E", font=legend_font)
    for row_index, (name, pair) in enumerate(examples):
        y = header_height + row_index * (panel_size + 58)
        left = labeled_panel(pair["neg0"], f"{name}: N=0")
        right = labeled_panel(pair["neg5"], f"{name}: N=5, Delta {pair['delta_dice']:+.3f}")
        canvas.paste(left, (0, y))
        canvas.paste(right, (panel_size, y))
    out = ROOT / "runs/research_direction/paper_artifacts_20260715/negative_feedback_qualitative.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
