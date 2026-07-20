from __future__ import annotations

import csv
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def mean_oracle(rows: list[dict[str, str]]) -> float:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[row["case_name"]].append(float(row["dice"]))
    return sum(max(values) for values in by_case.values()) / len(by_case)


def max_row(rows: list[dict[str, str]], metric: str = "mean_dice") -> dict[str, str]:
    return max(rows, key=lambda row: float(row[metric]))


def table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in cols)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(fmt(row.get(key, '')))}</td>" for key, _ in cols) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def md_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(title for _, title in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key, _ in cols) + " |")
    return "\n".join(lines)


def select_examples(rows: list[dict[str, str]], count: int = 4) -> list[dict[str, str]]:
    picked = sorted(rows, key=lambda row: float(row["dice"]), reverse=True)[:count]
    picked += sorted(rows, key=lambda row: float(row["dice"]))[:count]
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in picked:
        key = row["overlay_path"]
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def main() -> None:
    sam_dir = WORKSPACE_ROOT / "runs" / "research_direction" / "monuseg_dense_prompt_t1_t5"
    sam3_dir = WORKSPACE_ROOT / "runs" / "research_direction" / "monuseg_dense_prompt_sam3_grid"
    smoke_dir = WORKSPACE_ROOT / "runs" / "research_direction" / "sam3_smoke_teacher"
    out_dir = WORKSPACE_ROOT / "docs" / "reports" / "monuseg_sam3_retry"
    out_dir.mkdir(parents=True, exist_ok=True)

    sam_rows = read_csv(sam_dir / "candidates.csv")
    sam3_rows = read_csv(sam3_dir / "candidates.csv")
    sam_meta = json.loads((sam_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    sam3_meta = json.loads((sam3_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    sam_r1 = read_csv(sam_dir / "round1_prompt_budget.csv")
    sam_r2 = read_csv(sam_dir / "round2_contrastive_matching.csv")
    sam_r3 = read_csv(sam_dir / "round3_layout_clustering.csv")
    sam3_r1 = read_csv(sam3_dir / "round1_prompt_budget.csv")
    sam3_r2 = read_csv(sam3_dir / "round2_contrastive_matching.csv")
    sam3_r3 = read_csv(sam3_dir / "round3_layout_clustering.csv")
    smoke = read_csv(smoke_dir / "results.csv") if (smoke_dir / "results.csv").exists() else []

    sam_best_r3 = max_row(sam_r3)
    sam3_best_r3 = max_row(sam3_r3)
    sam3_best_r2 = max_row(sam3_r2)
    comparison = [
        {
            "model": "SAM ViT-B",
            "checkpoint": sam_meta.get("sam_checkpoint", "assets/checkpoints/sam_vit_b.pth"),
            "cases": sam_meta["case_count"],
            "candidates": sam_meta["candidate_count"],
            "best_k": sam_meta["best_k"],
            "best_lambda": sam_meta["best_lambda"],
            "best_method": sam_meta["best_round3_method"],
            "best_strategy_dice": sam_best_r3["mean_dice"],
            "oracle_dice": mean_oracle(sam_rows),
        },
        {
            "model": "SAM3",
            "checkpoint": sam3_meta.get("sam3_checkpoint", "assets/checkpoints/sam3_modelscope/sam3.pt"),
            "cases": sam3_meta["case_count"],
            "candidates": sam3_meta["candidate_count"],
            "best_k": sam3_meta["best_k"],
            "best_lambda": sam3_meta["best_lambda"],
            "best_method": sam3_meta["best_round3_method"],
            "best_strategy_dice": sam3_best_r3["mean_dice"],
            "oracle_dice": mean_oracle(sam3_rows),
        },
    ]

    compare_cols = [
        ("model", "模型"),
        ("cases", "cases"),
        ("candidates", "candidates"),
        ("best_k", "best K"),
        ("best_lambda", "best lambda"),
        ("best_method", "best layout"),
        ("best_strategy_dice", "固定策略 Dice"),
        ("oracle_dice", "候选池 oracle Dice"),
    ]
    r1_cols = [
        ("k", "K"),
        ("mean_dice", "Dice"),
        ("mean_iou", "IoU"),
        ("mean_fg_precision", "FG precision"),
        ("mean_fg_recall", "FG recall"),
        ("mean_object_recall", "Obj recall"),
        ("mean_pred_area_pct", "Pred area %"),
    ]
    r2_cols = [
        ("k", "K"),
        ("lambda_bg", "lambda"),
        ("mean_dice", "Dice"),
        ("mean_iou", "IoU"),
        ("mean_fg_precision", "FG precision"),
        ("mean_fg_recall", "FG recall"),
        ("mean_object_recall", "Obj recall"),
        ("mean_pred_area_pct", "Pred area %"),
    ]
    r3_cols = [
        ("method", "method"),
        ("layout", "layout"),
        ("clustering", "clustering"),
        ("mean_dice", "Dice"),
        ("mean_iou", "IoU"),
        ("mean_fg_precision", "FG precision"),
        ("mean_fg_recall", "FG recall"),
        ("mean_object_recall", "Obj recall"),
        ("mean_pred_area_pct", "Pred area %"),
    ]
    smoke_cols = [
        ("case_id", "case"),
        ("prompt_type", "prompt type"),
        ("prompt", "text prompt"),
        ("points", "points"),
        ("instances", "instances"),
        ("score_best", "best score"),
        ("union_area_pct", "union area %"),
        ("error", "error"),
    ]

    examples_html = []
    for row in select_examples(sam3_rows, count=3):
        overlay = WORKSPACE_ROOT / row["overlay_path"]
        points = WORKSPACE_ROOT / row["points_path"]
        heat = sam3_dir / "cases" / row["case_name"] / "t1_matching" / f"{row['round_name']}_{row['method']}_score_heatmap.png"
        examples_html.append(
            f"""
            <section class="example">
              <h3>{html.escape(row['case_name'])} / {html.escape(row['round_name'])} / {html.escape(row['method'])}</h3>
              <p>Dice={fmt(row['dice'])}, IoU={fmt(row['iou'])}, prompt_count={html.escape(row['prompt_count'])}, points=<code>{html.escape(row['points_path'])}</code></p>
              <div class="grid">
                <figure><a href="{rel(heat, out_dir)}" target="_blank"><img src="{rel(heat, out_dir)}" alt="heatmap"></a><figcaption>SAM3 使用的 matcher score heatmap。</figcaption></figure>
                <figure><a href="{rel(points, out_dir)}" target="_blank"><div class="filebox">CSV<br>点坐标表</div></a><figcaption>每个点的 rank、坐标、前景/背景相似度。</figcaption></figure>
                <figure><a href="{rel(overlay, out_dir)}" target="_blank"><img src="{rel(overlay, out_dir)}" alt="overlay"></a><figcaption>蓝色 GT、红色 SAM3 mask、绿色 positive points。</figcaption></figure>
              </div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MonuSeg SAM3 Retry</title>
  <style>
    body {{ margin:0; background:#f6f8fb; color:#182230; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
    main {{ max-width:1260px; margin:0 auto; padding:34px 24px 72px; }}
    h1 {{ margin:0 0 8px; font-size:32px; }}
    h2 {{ margin-top:34px; padding-top:16px; border-top:1px solid #d8e0ea; font-size:22px; }}
    h3 {{ font-size:18px; margin:20px 0 8px; }}
    p {{ line-height:1.72; margin:8px 0; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; margin:12px 0 22px; box-shadow:0 1px 2px rgba(16,24,40,.05); }}
    th,td {{ border:1px solid #d9e2ec; padding:8px 9px; font-size:13px; text-align:left; }}
    th {{ background:#eaf1f8; }}
    .callout {{ background:#e9f7ef; border:1px solid #b8e2c8; padding:12px 14px; border-radius:8px; }}
    .warn {{ background:#fff7e6; border-color:#f3d19c; }}
    .bad {{ background:#fdecec; border-color:#f1b7b7; }}
    code {{ background:#eef3f8; padding:2px 5px; border-radius:4px; }}
    .grid {{ display:grid; grid-template-columns:1fr 220px 1fr; gap:12px; align-items:stretch; }}
    figure {{ margin:0; background:#fff; border:1px solid #d9e2ec; border-radius:8px; padding:8px; }}
    img {{ width:100%; height:auto; display:block; border-radius:5px; }}
    figcaption {{ font-size:13px; color:#536173; line-height:1.5; margin-top:8px; }}
    .filebox {{ height:180px; display:flex; align-items:center; justify-content:center; text-align:center; border-radius:6px; background:#eef4fb; color:#234; font-weight:700; }}
    @media(max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <h1>MonuSeg Dense Point Prompt：SAM3 重测报告</h1>
  <p>这次重测专门修正上一版问题：上一版主实验用的是 <code>SAM ViT-B</code>，不是 SAM3。本版使用本地 <code>assets/checkpoints/sam3_modelscope/sam3.pt</code>，并把 K/lambda 网格补密。</p>

  <section class="callout warn">
    <p><b>模型说明：</b>上一版 340 candidates 是 SAM baseline；本版 1460 candidates 是 SAM3。两者不能混写成同一个结论。</p>
  </section>
  <section class="callout bad">
    <p><b>核心发现：</b>SAM3 在 MonuSeg dense positive point 上明显弱于 SAM baseline。SAM3 候选池 oracle Dice 只有 {mean_oracle(sam3_rows):.4f}，而 SAM baseline 是 {mean_oracle(sam_rows):.4f}。这说明“dense matched points 对 SAM 有效”不能直接推出“对 SAM3 也有效”。</p>
  </section>

  <h2>1. SAM vs SAM3 总表</h2>
  {table(comparison, compare_cols)}

  <h2>2. SAM3 text/point 冒烟验证</h2>
  <p>这一步回应老师最早截图：SAM3 text 对 <code>white girl</code>、<code>execute</code> 会失败；point prompt 能出 mask，累计点也会改变输出。说明 SAM3 本地运行没有问题，失败来自 prompt/model 行为本身。</p>
  {table(smoke, smoke_cols) if smoke else "<p>未找到 smoke results.csv。</p>"}

  <h2>3. SAM3 K 网格：K 的可能性已补全</h2>
  <p>K 覆盖 {html.escape(str(sam3_meta["k_list"]))}。结果显示 SAM3 不是“点越多越好”：K=1 的 Dice 最高之一，K 变大后 recall 和面积上升，但 Dice 并没有提升，甚至更容易扩张。</p>
  {table(sam3_r1, r1_cols)}

  <h2>4. SAM3 K-lambda 二维网格</h2>
  <p>Round2 对 K={html.escape(str(sam3_meta["round2_k_list"]))} 与 lambda={html.escape(str(sam3_meta["lambda_list"]))} 做二维搜索。最优组合是 K={sam3_meta["best_k"]}, lambda={sam3_meta["best_lambda"]}，但 mean Dice 仍只有 {fmt(sam3_best_r2["mean_dice"])}。</p>
  {table(sam3_r2, r2_cols)}

  <h2>5. SAM3 点布局/聚类对照</h2>
  <p>在 SAM3 的最优 K/lambda 下，DBSCAN medoid 略高于 top-k/local maxima，但整体 Dice 仍低。这和 SAM baseline 中 local maxima 最优的结论不同，说明 SAM3 的交互点解释方式和 SAM 不一样。</p>
  {table(sam3_r3, r3_cols)}

  <h2>6. SAM3 定性图</h2>
  <p>点击图像可以放大。SAM3 的一个直观问题是：很多 candidate 输出很小，或者在多点下突然扩张，但和 all-nuclei GT 的重叠仍不足。</p>
  {''.join(examples_html)}

  <h2>7. 现在应该怎么汇报</h2>
  <p>第一句话要先纠正：上一轮主实验是 SAM，不是 SAM3。重测后，SAM3 确实支持累计点，但在 MonuSeg all-nuclei dense prompt 上效果弱，这本身是一个重要结果。</p>
  <p>后续不应直接把 SAM3 当成 SAM 的替代模型。更合理的路线是：SAM baseline 用来证明 matcher dense point 的候选池价值；SAM3 单独研究 prompt 行为，尤其是为什么多点不能稳定合成全图细胞核 mask，以及是否需要 negative points、box/mask guidance 或不同的 instance-to-all aggregation。</p>
</main></body></html>"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")

    md_text = f"""# MonuSeg Dense Point Prompt：SAM3 重测报告

## 先纠正

上一版主实验使用的是 SAM ViT-B，不是 SAM3。本次已经用 SAM3 重新跑了 20 个 MonuSeg case，并补密 K/lambda。

## SAM vs SAM3 总表

{md_table(comparison, compare_cols)}

## SAM3 K 网格

{md_table(sam3_r1, r1_cols)}

## SAM3 K-lambda 二维网格

最优组合：K={sam3_meta["best_k"]}, lambda={sam3_meta["best_lambda"]}。

{md_table(sam3_r2, r2_cols)}

## SAM3 点布局/聚类

{md_table(sam3_r3, r3_cols)}

## 结论

- SAM3 point prompt 可以运行，也支持累计点。
- SAM3 text 对 `white girl`、`execute` 的失败已在 smoke 中复现。
- 但 SAM3 在 MonuSeg dense positive point 上明显弱于 SAM baseline。
- 因此下一步要把 SAM3 作为单独问题研究，不能把 SAM baseline 的 dense point 结论直接迁移。

网页报告：`{out_dir / "index.html"}`
"""
    (out_dir / "sam3_retry_report.md").write_text(md_text, encoding="utf-8")
    print(f"wrote {out_dir / 'index.html'}")
    print(f"wrote {out_dir / 'sam3_retry_report.md'}")


if __name__ == "__main__":
    main()
