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


def rel_from(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in cols)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(fmt(row.get(key, '')))}</td>" for key, _ in cols)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def markdown_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(title for _, title in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key, _ in cols) + " |")
    return "\n".join(lines)


def best_examples(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    picked: list[dict[str, str]] = []
    for round_name in ["round1", "round2", "round3"]:
        subset = [row for row in candidates if row["round_name"] == round_name]
        if not subset:
            continue
        picked.append(max(subset, key=lambda row: float(row["dice"])))
        picked.append(min(subset, key=lambda row: float(row["dice"])))
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in picked:
        key = row["overlay_path"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def per_case_oracle(candidates: list[dict[str, str]]) -> float:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        by_case[row["case_name"]].append(float(row["dice"]))
    return sum(max(values) for values in by_case.values()) / len(by_case)


def main() -> None:
    run_dir = WORKSPACE_ROOT / "runs" / "research_direction" / "monuseg_dense_prompt_t1_t5"
    report_dir = WORKSPACE_ROOT / "docs" / "reports" / "monuseg_dense_prompt_t1_t5"
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_csv(run_dir / "candidates.csv")
    round1 = read_csv(run_dir / "round1_prompt_budget.csv")
    round2 = read_csv(run_dir / "round2_contrastive_matching.csv")
    round3 = read_csv(run_dir / "round3_layout_clustering.csv")
    metadata = json.loads((run_dir / "experiment_metadata.json").read_text(encoding="utf-8"))

    best_k = metadata["best_k"]
    best_lambda = metadata["best_lambda"]
    best_method = metadata["best_round3_method"]
    new_oracle = per_case_oracle(candidates)

    old_stage3 = 0.051678
    old_oracle = 0.342465
    best_round3_row = max(round3, key=lambda row: float(row["mean_dice"]))
    best_round3_dice = float(best_round3_row["mean_dice"])

    r1_cols = [
        ("k", "K"),
        ("case_count", "cases"),
        ("mean_dice", "Dice"),
        ("mean_iou", "IoU"),
        ("mean_fg_recall", "FG recall"),
        ("mean_object_recall", "Obj recall"),
        ("mean_merge_error", "Merge"),
        ("mean_pred_area_pct", "Pred area %"),
    ]
    r2_cols = [
        ("lambda_bg", "lambda"),
        ("case_count", "cases"),
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
        ("mean_fg_recall", "FG recall"),
        ("mean_object_recall", "Obj recall"),
        ("mean_merge_error", "Merge"),
    ]

    example_cards = []
    for row in best_examples(candidates):
        overlay = WORKSPACE_ROOT / row["overlay_path"]
        points = WORKSPACE_ROOT / row["points_path"]
        heat = run_dir / "cases" / row["case_name"] / "t1_matching" / f"{row['round_name']}_{row['method']}_score_heatmap.png"
        points_overlay = run_dir / "cases" / row["case_name"] / "t1_matching" / f"{row['round_name']}_{row['method']}_points_overlay.png"
        example_cards.append(
            f"""
            <section class="example">
              <h3>{html.escape(row['round_name'])} / {html.escape(row['method'])} / {html.escape(row['case_name'])}</h3>
              <p>Dice={fmt(row['dice'])}, IoU={fmt(row['iou'])}, object recall={fmt(row['object_recall'])}, prompt_count={html.escape(row['prompt_count'])}。点坐标表：<code>{html.escape(row['points_path'])}</code></p>
              <div class="grid">
                <figure><a href="{rel_from(heat, report_dir)}" target="_blank"><img src="{rel_from(heat, report_dir)}" alt="score heatmap"></a><figcaption>Stage1 score heatmap：越亮代表该 patch 越像目标前景，且在 contrastive 设置下越不像背景。</figcaption></figure>
                <figure><a href="{rel_from(points_overlay, report_dir)}" target="_blank"><img src="{rel_from(points_overlay, report_dir)}" alt="points overlay"></a><figcaption>进入 SAM 的 positive points。绿色/红色点的位置来自 DINO patch score 排序或对应布局策略。</figcaption></figure>
                <figure><a href="{rel_from(overlay, report_dir)}" target="_blank"><img src="{rel_from(overlay, report_dir)}" alt="SAM overlay"></a><figcaption>SAM 输出：蓝色是人工 GT，红色是预测 mask，绿色是 prompt 点，黄色框是预测 mask 外接范围。</figcaption></figure>
              </div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MonuSeg Dense Point Prompt T1-T5</title>
  <style>
    body {{ margin:0; background:#f6f8fb; color:#17202a; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
    main {{ max-width:1240px; margin:0 auto; padding:34px 24px 70px; }}
    h1 {{ font-size:32px; margin:0 0 8px; }}
    h2 {{ margin-top:34px; padding-top:18px; border-top:1px solid #d8e0ea; font-size:22px; }}
    h3 {{ font-size:18px; margin:20px 0 8px; }}
    p {{ line-height:1.72; margin:8px 0; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0 24px; }}
    .metric {{ background:#fff; border:1px solid #d9e2ec; border-radius:8px; padding:14px; }}
    .metric strong {{ display:block; font-size:24px; margin-bottom:4px; color:#123c69; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; margin:12px 0 20px; box-shadow:0 1px 2px rgba(16,24,40,.05); }}
    th,td {{ border:1px solid #d9e2ec; padding:9px 10px; font-size:14px; text-align:left; }}
    th {{ background:#eaf1f8; }}
    code {{ background:#eef3f8; padding:2px 5px; border-radius:4px; }}
    .callout {{ background:#e9f7ef; border:1px solid #b8e2c8; padding:12px 14px; border-radius:8px; }}
    .warn {{ background:#fff7e6; border:1px solid #f3d19c; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; align-items:start; }}
    figure {{ margin:0; background:#fff; border:1px solid #d9e2ec; border-radius:8px; padding:8px; }}
    img {{ width:100%; height:auto; display:block; border-radius:5px; }}
    figcaption {{ font-size:13px; color:#52616f; line-height:1.5; margin-top:8px; }}
    .example {{ margin-top:22px; }}
    @media (max-width:900px) {{ .summary,.grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <h1>MonuSeg 单域 Dense Point Prompt T1-T5 闭环实验报告</h1>
  <p>这份报告对应老师方向中的“如何精进 matcher”：先不扩大到所有医学域，而是在 MonuSeg 细胞核单域里验证 reference-mask matching、dense point prompt、foreground/background contrastive score 和聚类对照。</p>
  <div class="summary">
    <div class="metric"><strong>{metadata['case_count']}</strong><span>MonuSeg cases</span></div>
    <div class="metric"><strong>{metadata['candidate_count']}</strong><span>SAM proposals</span></div>
    <div class="metric"><strong>K={best_k}</strong><span>Round1 最优点数</span></div>
    <div class="metric"><strong>lambda={best_lambda}</strong><span>Round2 最优背景抑制</span></div>
  </div>

  <section class="callout">
    <p><b>主结论：</b>多点 dense prompt 明显优于单点/少点；前景-背景对比 matching 明显优于 foreground-only；当前最优布局是 <code>{html.escape(str(best_method))}</code>，mean Dice={best_round3_dice:.4f}。新候选池 oracle mean Dice={new_oracle:.4f}，高于旧 MonuSeg 候选池 oracle mean Dice={old_oracle:.4f}。</p>
  </section>
  <section class="callout warn">
    <p><b>关于“空白方向”：</b>本实验先按潜在空白方向推进，但报告中只陈述本地实验事实；是否“前人完全没有做过”还需要后续系统查新和 related work 对照。</p>
  </section>

  <h2>0. 与之前 MonuSeg 基线对比</h2>
  <table><thead><tr><th>方法/池子</th><th>mean Dice</th><th>解释</th></tr></thead><tbody>
    <tr><td>旧 Stage3 手写筛选</td><td>{old_stage3:.4f}</td><td>之前规则筛出来的结果，说明 MonuSeg 上 Stage3 误杀严重。</td></tr>
    <tr><td>旧候选池 oracle</td><td>{old_oracle:.4f}</td><td>旧 12 proposals/case 中，每个 case 选 Dice 最好的候选。</td></tr>
    <tr><td>新 T1-T5 最优单策略</td><td>{best_round3_dice:.4f}</td><td>固定同一种策略，20 case 平均。</td></tr>
    <tr><td>新 T1-T5 候选池 oracle</td><td>{new_oracle:.4f}</td><td>每个 case 从本轮 17 个候选中选最优，说明候选池质量上限提高。</td></tr>
  </tbody></table>

  <h2>T1. Foreground-Background Contrastive Matching</h2>
  <p>reference 图不是只提供一个目标框，而是有人工 mask。因此我们同时利用 mask 内的 foreground prototype 和 mask 外的 background prototype：</p>
  <p><code>score(x)=z(sim(query_x, foreground_proto))-lambda*z(sim(query_x, background_proto))</code></p>
  <p>这一步回应你的想法：我们要找的是“最像细胞核前景，同时最不像参考背景”的点，而不是只按前景相似度排序。</p>

  <h2>T3. 点数预算：K 取多少</h2>
  <p>固定 foreground-only、top-k、无聚类，只看一次性送入 SAM 的点数。结果说明 K=1 几乎不够，K=10 后 recall 大幅提升，K=20 达到本轮 Dice 峰值；继续加到 50/100 会提高 recall，但预测面积和 merge 风险也上升。</p>
  {table(round1, r1_cols)}

  <h2>T4. Background 是否有用</h2>
  <p>固定 K={best_k}、top-k，不改变 SAM，只改变 matching score。lambda=0.25 把 mean Dice 从 0.3301 提到 0.3840，同时预测面积从 59.04% 降到 51.25%，说明 background prototype 对抑制背景扩张是有用的。</p>
  {table(round2, r2_cols)}

  <h2>T5. 点布局与聚类是否必要</h2>
  <p>固定 K={best_k}、lambda={best_lambda}。local maxima 略优于普通 top-k；DBSCAN/heatmap CC 的 object recall 高，但 Dice 不如 local maxima/top-k，说明 MonuSeg all-nuclei 任务里“强制聚类”暂时不是主线，下一步更应研究 dense points 如何分布。</p>
  {table(round3, r3_cols)}

  <h2>定性证据</h2>
  <p>每组图都分成三张：score heatmap、points overlay、SAM mask overlay。点击任意图片可以放大查看。</p>
  {''.join(example_cards)}

  <h2>下一步研究路线</h2>
  <p>第一优先级：沿着 MonuSeg 单域，把 <code>K</code> 和 <code>lambda</code> 从人工网格搜索改成验证集可学习策略。</p>
  <p>第二优先级：加入 negative/background points，但只在预测面积过大或 merge error 高的 case 触发，避免重新引入太多变量。</p>
  <p>第三优先级：把 local maxima 和 top-k 结合成一个可学习 point policy，输入 score map 的峰值数量、熵、面积估计和 background score，输出 K 与点布局。</p>
  <p>第四优先级：在 MonuSeg 稳定后，再迁移到 IDRID 或其他 domain；不要现在就把跨域作为主线。</p>
</main></body></html>"""
    (report_dir / "index.html").write_text(html_text, encoding="utf-8")

    md = f"""# MonuSeg 单域 Dense Point Prompt T1-T5 闭环实验报告

## 结论摘要

- MonuSeg cases：{metadata['case_count']}
- SAM proposals：{metadata['candidate_count']}
- Round1 最优 K：{best_k}
- Round2 最优 lambda：{best_lambda}
- Round3 最优策略：{best_method}
- 旧 Stage3 mean Dice：{old_stage3:.4f}
- 旧候选池 oracle mean Dice：{old_oracle:.4f}
- 新最优单策略 mean Dice：{best_round3_dice:.4f}
- 新候选池 oracle mean Dice：{new_oracle:.4f}

> 本实验先按潜在空白方向推进，但不在没有系统查新的情况下声称“前人完全没人做过”。

## T3 点数预算

{markdown_table(round1, r1_cols)}

## T4 前景-背景对比

{markdown_table(round2, r2_cols)}

## T5 点布局与聚类对照

{markdown_table(round3, r3_cols)}

## 报告网页

`{report_dir / 'index.html'}`

## 原始产物

`{run_dir}`
"""
    (report_dir / "monuseg_dense_prompt_t1_t5_report.md").write_text(md, encoding="utf-8")
    print(f"wrote {report_dir / 'index.html'}")
    print(f"wrote {report_dir / 'monuseg_dense_prompt_t1_t5_report.md'}")


if __name__ == "__main__":
    main()
