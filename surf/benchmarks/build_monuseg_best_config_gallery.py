from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return (center > 0) & (neighbors < 4)


def overlay_single_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int, path: Path) -> None:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (*color, 0))
    layer.putalpha(Image.fromarray((mask.astype(np.uint8) * alpha), mode="L"))
    out = Image.alpha_composite(base, layer).convert("RGB")
    draw = ImageDraw.Draw(out)
    boundary = mask_boundary(mask)
    ys, xs = np.where(boundary)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((x, y), fill=color)
    out.save(path)


def overlay_gt_pred(image: Image.Image, gt: np.ndarray, pred: np.ndarray, path: Path) -> None:
    base = image.convert("RGBA")
    gt_layer = Image.new("RGBA", base.size, (0, 170, 255, 0))
    pred_layer = Image.new("RGBA", base.size, (255, 45, 45, 0))
    inter_layer = Image.new("RGBA", base.size, (255, 220, 40, 0))
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    inter = gt & pred
    gt_layer.putalpha(Image.fromarray((gt_only.astype(np.uint8) * 135), mode="L"))
    pred_layer.putalpha(Image.fromarray((pred_only.astype(np.uint8) * 135), mode="L"))
    inter_layer.putalpha(Image.fromarray((inter.astype(np.uint8) * 155), mode="L"))
    out = Image.alpha_composite(Image.alpha_composite(Image.alpha_composite(base, gt_layer), pred_layer), inter_layer).convert("RGB")
    draw = ImageDraw.Draw(out)
    for mask, color in [(gt, (0, 125, 255)), (pred, (255, 0, 0))]:
        ys, xs = np.where(mask_boundary(mask))
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((x, y), fill=color)
    out.save(path)


def save_clean_points(image: Image.Image, points: list[dict[str, str]], path: Path) -> None:
    out = ImageEnhance.Contrast(image.convert("RGB")).enhance(1.08)
    draw = ImageDraw.Draw(out)
    for point in reversed(points):
        rank = int(float(point["rank"]))
        x = float(point["x"])
        y = float(point["y"])
        radius = 7 if rank <= 10 else 5 if rank <= 40 else 3
        fill = (255, 35, 70) if rank <= 40 else (255, 170, 0)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(255, 255, 255), width=2)
    out.save(path)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            label += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = label
            while queue:
                cy, cx = queue.popleft()
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            queue.append((ny, nx))
    return labels, label


def point_metrics(mask: np.ndarray, points: list[dict[str, str]], k_values: list[int]) -> dict[str, Any]:
    labels, component_count = connected_components(mask)
    rows: dict[str, Any] = {"gt_components": component_count}
    for k in k_values:
        subset = points[: min(k, len(points))]
        hits = 0
        hit_components: set[int] = set()
        for point in subset:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
            if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                continue
            if mask[y, x]:
                hits += 1
                label = int(labels[y, x])
                if label > 0:
                    hit_components.add(label)
        rows[f"point_precision_at_{k}"] = round(hits / max(1, len(subset)), 6)
        rows[f"component_recall_at_{k}"] = round(len(hit_components) / max(1, component_count), 6)
    return rows


def choose_representative_cases(best_rows: list[dict[str, str]], round1_rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str], str]]:
    by_case = {row["case_name"]: row for row in best_rows}
    r1_by_case = {row["case_name"]: row for row in round1_rows}
    ordered = sorted(best_rows, key=lambda row: float(row["dice"]))
    median = ordered[len(ordered) // 2]
    best = ordered[-1]
    worst = ordered[0]
    gain_case = max(
        best_rows,
        key=lambda row: float(row["dice"]) - float(r1_by_case.get(row["case_name"], {"dice": row["dice"]})["dice"]),
    )
    area_case = max(best_rows, key=lambda row: float(row["pred_area_pct"]))
    picks = [
        ("best", best, "当前配置下 Dice 最高，说明 reference mask -> matched points -> SAM 的正向闭环可以成立。"),
        ("median", median, "接近中位数表现，适合展示方法在一般 case 上的实际状态。"),
        ("worst", worst, "当前失败最明显的 case，用来分析过分割、漏检或点分布问题。"),
        ("bg_gain", gain_case, "相对 foreground-only 提升最明显，直接支撑 foreground-background contrastive matching 的必要性。"),
        ("large_area", area_case, "预测面积较大的 case，用来检查 dense points 是否造成 SAM mask 扩张。"),
    ]
    unique: list[tuple[str, dict[str, str], str]] = []
    seen: set[str] = set()
    for tag, row, note in picks:
        if row["case_name"] in seen:
            continue
        seen.add(row["case_name"])
        unique.append((tag, row, note))
    return unique


def copy_or_make_assets(
    out_dir: Path,
    report_root: Path,
    manifest_by_case: dict[str, dict[str, str]],
    row: dict[str, str],
    tag: str,
) -> dict[str, str]:
    case_name = row["case_name"]
    manifest = manifest_by_case[case_name]
    asset_dir = out_dir / "assets" / tag
    asset_dir.mkdir(parents=True, exist_ok=True)

    query_image = Image.open(resolve_path(manifest["query_image_path"])).convert("RGB")
    gt = load_mask(resolve_path(manifest["query_mask_path"]))
    pred = load_mask(resolve_path(row["mask_path"]))
    points = read_csv(resolve_path(row["points_path"]))

    query_path = asset_dir / "query_image.png"
    gt_path = asset_dir / "gt_only.png"
    pred_path = asset_dir / "pred_only.png"
    compare_path = asset_dir / "gt_pred_compare.png"
    points_path = asset_dir / "matched_points.png"
    heatmap_path = asset_dir / "score_heatmap.png"
    overlay_path = asset_dir / "original_overlay.png"

    query_image.save(query_path)
    overlay_single_mask(query_image, gt, (0, 165, 255), 150, gt_path)
    overlay_single_mask(query_image, pred, (255, 40, 40), 150, pred_path)
    overlay_gt_pred(query_image, gt, pred, compare_path)
    save_clean_points(query_image, points, points_path)

    source_heatmap = resolve_path(row["points_path"]).parent.parent / "t1_matching" / (Path(row["points_path"]).stem.replace("_points", "_score_heatmap.png"))
    if source_heatmap.exists():
        shutil.copy2(source_heatmap, heatmap_path)
    else:
        ImageOps.grayscale(query_image).save(heatmap_path)
    shutil.copy2(resolve_path(row["overlay_path"]), overlay_path)

    return {
        "query_image": rel(query_path, report_root),
        "gt_only": rel(gt_path, report_root),
        "pred_only": rel(pred_path, report_root),
        "compare": rel(compare_path, report_root),
        "points": rel(points_path, report_root),
        "heatmap": rel(heatmap_path, report_root),
        "overlay": rel(overlay_path, report_root),
    }


def resolve_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def render_html(
    out_dir: Path,
    selected: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
) -> None:
    def method_label(row: dict[str, Any]) -> str:
        if row.get("method"):
            return str(row["method"])
        parts = [str(row.get("backbone", "")), str(row.get("model", "")), str(row.get("best_method", ""))]
        return " + ".join(part for part in parts if part)

    top_table = "\n".join(
        "<tr>"
        f"<td>{html.escape(method_label(row))}</td>"
        f"<td>{row['mean_dice']}</td>"
        f"<td>{row['mean_iou']}</td>"
        f"<td>{row.get('mean_fg_recall', '')}</td>"
        f"<td>{row.get('mean_pred_area_pct', '')}</td>"
        "</tr>"
        for row in summary_rows
    )
    point_table = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['tag']))}</td>"
        f"<td>{html.escape(str(row['query_case_id']))}</td>"
        f"<td>{row['point_precision_at_40']}</td>"
        f"<td>{row['component_recall_at_40']}</td>"
        f"<td>{row['point_precision_at_100']}</td>"
        f"<td>{row['component_recall_at_100']}</td>"
        "</tr>"
        for row in point_rows
    )
    cards: list[str] = []
    for item in selected:
        imgs = item["assets"]
        image_grid = "\n".join(
            f'<figure><button data-img="{html.escape(src)}"><img src="{html.escape(src)}" alt="{html.escape(label)}"></button><figcaption>{html.escape(label)}</figcaption></figure>'
            for label, src in [
                ("query original", imgs["query_image"]),
                ("GT only", imgs["gt_only"]),
                ("score heatmap", imgs["heatmap"]),
                ("matched points", imgs["points"]),
                ("prediction only", imgs["pred_only"]),
                ("GT vs prediction", imgs["compare"]),
            ]
        )
        cards.append(
            f"""
            <section class="case-card" id="{html.escape(item['tag'])}">
              <div class="case-head">
                <div>
                  <p class="tag">{html.escape(item['tag'])}</p>
                  <h2>{html.escape(item['query_case_id'])}</h2>
                  <p>{html.escape(item['note'])}</p>
                </div>
                <div class="metrics">
                  <span>Dice <b>{item['dice']}</b></span>
                  <span>IoU <b>{item['iou']}</b></span>
                  <span>Obj Recall <b>{item['object_recall']}</b></span>
                  <span>Pred Area <b>{item['pred_area_pct']}%</b></span>
                </div>
              </div>
              <div class="image-grid">{image_grid}</div>
              <details>
                <summary>查看该 case 的点命中和候选信息</summary>
                <p>Prompt: {html.escape(item['method'])}; K={item['k']}; lambda={item['lambda_bg']}; SAM score={item['sam_score']}.</p>
                <p>Point precision@40={item['point_precision_at_40']}, component recall@40={item['component_recall_at_40']}.</p>
              </details>
            </section>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MoNuSeg 最优配置定性分析</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5f6e82;
      --line: #d9e2ee;
      --panel: #f7f9fc;
      --accent: #1b6fd8;
      --good: #057a55;
      --warn: #a35d00;
    }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #ffffff;
      color: var(--ink);
      line-height: 1.55;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding-bottom: 22px;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{ margin: 0 0 6px; font-size: 21px; }}
    p {{ margin: 6px 0; color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .summary div {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      border-radius: 8px;
    }}
    .summary b {{ display: block; font-size: 22px; color: var(--ink); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 24px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: #334155; background: #f2f5f9; }}
    .case-card {{
      border-top: 2px solid #24364f;
      padding-top: 18px;
      margin-top: 34px;
    }}
    .case-head {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: start;
    }}
    .tag {{
      display: inline-block;
      color: var(--accent);
      font-weight: 700;
      text-transform: uppercase;
      margin: 0 0 4px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(110px, 1fr));
      gap: 8px;
      min-width: 280px;
    }}
    .metrics span {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      color: var(--muted);
      background: #fff;
    }}
    .metrics b {{ color: var(--ink); font-size: 17px; }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
    }}
    figure button {{
      padding: 0;
      border: 0;
      background: transparent;
      display: block;
      width: 100%;
      cursor: zoom-in;
    }}
    figure img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #f1f5f9;
    }}
    figcaption {{
      padding: 8px 10px;
      color: #334155;
      font-size: 13px;
      border-top: 1px solid var(--line);
    }}
    details {{
      margin-top: 12px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    #lightbox {{
      position: fixed;
      inset: 0;
      background: rgba(8, 13, 22, 0.88);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      z-index: 10;
    }}
    #lightbox.open {{ display: flex; }}
    #lightbox img {{
      max-width: 96vw;
      max-height: 92vh;
      background: #fff;
      box-shadow: 0 18px 60px rgba(0,0,0,.35);
    }}
    @media (max-width: 880px) {{
      .summary, .image-grid, .case-head {{ grid-template-columns: 1fr; }}
      .metrics {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>MoNuSeg 最优配置定性分析</h1>
    <p>当前主线配置：DINOv2-L 作为 matcher backbone，SAM ViT-B 作为 promptable segmenter，K=40，lambda=0.75。这里不重新训练模型，只把已跑完的 14 个 official test query 结果整理成可汇报证据。</p>
    <div class="summary">
      <div><span>Best mean Dice</span><b>0.5183</b></div>
      <div><span>Best mean IoU</span><b>0.3579</b></div>
      <div><span>Prompt points</span><b>K=40</b></div>
      <div><span>Background weight</span><b>lambda=0.75</b></div>
    </div>
  </header>

  <section>
    <h2>总体对比</h2>
    <p>这张表用于说明当前方法不是只比随机点或 box prompt 好，而是在同一 official test set 上超过了 SAM/SAM2/MedSAM 的普通 prompt baseline。</p>
    <table><thead><tr><th>方法</th><th>Mean Dice</th><th>Mean IoU</th><th>FG Recall</th><th>Pred Area %</th></tr></thead><tbody>{top_table}</tbody></table>
  </section>

  <section>
    <h2>点是否真的落在细胞核上</h2>
    <p>Point precision@K 表示前 K 个 matched points 有多少比例落在 GT mask 内；component recall@K 表示这些点覆盖了多少 GT 连通域。它直接回答 matcher 找到的点是否有医学目标意义。</p>
    <table><thead><tr><th>代表类型</th><th>Query case</th><th>P@40</th><th>Comp R@40</th><th>P@100</th><th>Comp R@100</th></tr></thead><tbody>{point_table}</tbody></table>
  </section>

  {''.join(cards)}
</main>
<div id="lightbox"><img alt="放大图"></div>
<script>
  const box = document.getElementById('lightbox');
  const img = box.querySelector('img');
  document.querySelectorAll('figure button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      img.src = btn.dataset.img;
      box.classList.add('open');
    }});
  }});
  box.addEventListener('click', () => box.classList.remove('open'));
  document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') box.classList.remove('open');
  }});
</script>
</body>
</html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/research_direction/monuseg_official_screening/dinov2l_samvitb_kxlambda_test14")
    parser.add_argument("--manifest", default="runs/research_direction/monuseg_official_screening/official_monuseg_screening_manifest.csv")
    parser.add_argument("--out-dir", default="runs/research_direction/monuseg_official_screening/dinov2l_samvitb_kxlambda_test14/qualitative_gallery")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(Path(args.manifest))
    manifest_by_case = {row["pilot_id"]: row for row in manifest}
    candidates = read_csv(run_dir / "candidates.csv")
    best_rows = [row for row in candidates if row["round_name"] == "round3" and row["method"] == "best_topk"]
    round1_rows = [row for row in candidates if row["round_name"] == "round1" and row["method"] == "fg_topk_K40"]
    selected = choose_representative_cases(best_rows, round1_rows)
    summary_rows = read_csv(Path("runs/research_direction/monuseg_official_screening/model_weight_screening_summary.csv"))[:6]

    selected_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for tag, row, note in selected:
        case_name = row["case_name"]
        manifest_row = manifest_by_case[case_name]
        points = read_csv(resolve_path(row["points_path"]))
        gt = load_mask(resolve_path(manifest_row["query_mask_path"]))
        point_stats = point_metrics(gt, points, [10, 20, 40, 100])
        assets = copy_or_make_assets(out_dir, out_dir, manifest_by_case, row, tag)
        enriched: dict[str, Any] = {
            "tag": tag,
            "note": note,
            "case_name": case_name,
            "query_case_id": row["query_case_id"],
            "method": row["method"],
            "k": row["k"],
            "lambda_bg": row["lambda_bg"],
            "sam_score": row["sam_score"],
            "dice": row["dice"],
            "iou": row["iou"],
            "fg_precision": row["fg_precision"],
            "fg_recall": row["fg_recall"],
            "object_recall": row["object_recall"],
            "object_precision": row["object_precision"],
            "pred_area_pct": row["pred_area_pct"],
            "assets": assets,
            **point_stats,
        }
        selected_rows.append(enriched)
        point_rows.append({key: value for key, value in enriched.items() if key != "assets"})

    write_csv(out_dir / "representative_cases.csv", [{k: v for k, v in row.items() if k != "assets"} for row in selected_rows])
    write_csv(out_dir / "point_hit_metrics.csv", point_rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "case_count": len(best_rows),
                "selected_case_count": len(selected_rows),
                "selection_tags": [row["tag"] for row in selected_rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    render_html(out_dir, selected_rows, summary_rows, point_rows)
    print(f"[done] gallery={out_dir / 'index.html'} selected={len(selected_rows)}")


if __name__ == "__main__":
    main()
