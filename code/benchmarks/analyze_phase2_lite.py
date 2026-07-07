from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "runs" / "phase2_lite"


def workspace_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: str | float | int) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def summarize(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(buckets.items()):
        ok = [r for r in group_rows if r["status"] == "ok"]
        dice = [fnum(r["dice"]) for r in ok]
        iou = [fnum(r["iou"]) for r in ok]
        runtime = [fnum(r["runtime_sec"]) for r in ok]
        record: dict[str, Any] = {k: v for k, v in zip(keys, group_key)}
        record.update(
            {
                "rows": len(group_rows),
                "ok": len(ok),
                "unsupported": sum(1 for r in group_rows if r["status"] == "unsupported"),
                "failed": sum(1 for r in group_rows if r["status"] == "failed"),
                "mean_dice": round(mean(dice), 6),
                "mean_iou": round(mean(iou), 6),
                "mean_runtime_sec": round(mean(runtime), 6),
                "low_dice_lt_0_1": sum(1 for v in dice if v < 0.1),
                "high_dice_ge_0_5": sum(1 for v in dice if v >= 0.5),
            }
        )
        out.append(record)
    return out


def add_failure_tags(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "ok":
            tags = [row["status"]]
        else:
            dice = fnum(row["dice"])
            tags = []
            if dice < 0.05:
                tags.append("near_empty_or_wrong_object")
            elif dice < 0.20:
                tags.append("low_overlap")
            if row["difficulty_bucket"] in {"tiny_target", "small_target"}:
                tags.append(row["difficulty_bucket"])
            if row["difficulty_bucket"] == "multi_instance":
                tags.append("multi_instance")
            if row["protocol"] == "perturbed_box" and dice < 0.2:
                tags.append("box_perturbation_sensitive")
            if row["protocol"] in {"centroid_point", "grid_point"} and dice < 0.2:
                tags.append("point_prompt_unstable")
            if row["protocol"] == "auto_box" and dice < 0.2:
                tags.append("auto_box_failed")
            if row["protocol"] == "auto_box" and dice >= 0.5:
                tags.append("auto_box_success")
        tagged_row = dict(row)
        tagged_row["failure_tags"] = ";".join(tags)
        tagged.append(tagged_row)
    return tagged


def load_rgb(path: Path, size: int = 220) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((size, size), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def load_mask(path: Path, size: int = 220) -> Image.Image:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    else:
        arr = arr > 0
    img = Image.fromarray(arr.astype(np.uint8) * 255, mode="L")
    img.thumbnail((size, size), Image.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def overlay_masks(image: Image.Image, gt: Image.Image, pred: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    gt_arr = np.asarray(gt) > 0
    pred_arr = np.asarray(pred) > 0
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    overlay[gt_arr] = [0, 220, 90, 90]
    overlay[pred_arr] = [255, 0, 180, 95]
    overlap = gt_arr & pred_arr
    overlay[overlap] = [255, 220, 0, 135]
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")


def case_panel(row: dict[str, str], manifest_by_case: dict[str, dict[str, str]], cell: tuple[int, int] = (240, 286)) -> Image.Image:
    width, height = cell
    panel = Image.new("RGB", cell, "white")
    draw = ImageDraw.Draw(panel)
    case = manifest_by_case[row["case_id"]]
    image = load_rgb(workspace_path(case["image_path"]), size=220)
    gt = load_mask(workspace_path(case["mask_path"]), size=220)
    if row.get("mask_path"):
        pred = load_mask(workspace_path(row["mask_path"]), size=220)
        vis = overlay_masks(image, gt, pred)
    else:
        vis = image
    panel.paste(vis, (10, 10))
    title = f"{row['model']} | {row['protocol']}"
    subtitle = f"{row['case_id']} | Dice {float(row['dice']):.3f}"
    draw.text((10, 236), title[:38], fill=(0, 0, 0))
    draw.text((10, 254), subtitle[:42], fill=(45, 45, 45))
    return panel


def make_gallery(
    path: Path,
    title: str,
    rows: list[dict[str, str]],
    manifest_by_case: dict[str, dict[str, str]],
    limit: int = 12,
    cols: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit]
    if not selected:
        return
    cell = (240, 286)
    header_h = 34
    rows_n = int(math.ceil(len(selected) / cols))
    sheet = Image.new("RGB", (cols * cell[0], rows_n * cell[1] + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill=(0, 0, 0))
    for idx, row in enumerate(selected):
        panel = case_panel(row, manifest_by_case, cell=cell)
        x = (idx % cols) * cell[0]
        y = header_h + (idx // cols) * cell[1]
        sheet.paste(panel, (x, y))
    sheet.save(path)


def make_galleries(rows: list[dict[str, str]], manifest: list[dict[str, str]], out_dir: Path) -> dict[str, str]:
    manifest_by_case = {row["case_id"]: row for row in manifest}
    ok = [r for r in rows if r["status"] == "ok" and r.get("mask_path")]
    paths: dict[str, str] = {}

    for model in sorted({r["model"] for r in ok}):
        model_rows = [r for r in ok if r["model"] == model]
        worst = sorted(model_rows, key=lambda r: fnum(r["dice"]))[:12]
        best = sorted(model_rows, key=lambda r: fnum(r["dice"]), reverse=True)[:12]
        worst_path = out_dir / "galleries" / f"worst_{model}.png"
        best_path = out_dir / "galleries" / f"best_{model}.png"
        make_gallery(worst_path, f"Worst cases: {model}", worst, manifest_by_case)
        make_gallery(best_path, f"Best cases: {model}", best, manifest_by_case)
        paths[f"worst_{model}"] = rel(worst_path)
        paths[f"best_{model}"] = rel(best_path)

    for protocol in sorted({r["protocol"] for r in ok if r["protocol"].startswith("auto")}):
        auto = [r for r in ok if r["protocol"] == protocol]
        auto_fail = sorted(auto, key=lambda r: fnum(r["dice"]))[:12]
        auto_success = sorted(auto, key=lambda r: fnum(r["dice"]), reverse=True)[:12]
        auto_fail_path = out_dir / "galleries" / f"{protocol}_failures.png"
        auto_success_path = out_dir / "galleries" / f"{protocol}_successes.png"
        make_gallery(auto_fail_path, f"{protocol} failures", auto_fail, manifest_by_case)
        make_gallery(auto_success_path, f"{protocol} successes", auto_success, manifest_by_case)
        paths[f"{protocol}_failures"] = rel(auto_fail_path)
        paths[f"{protocol}_successes"] = rel(auto_success_path)

    paths.update(make_disagreement_galleries(ok, manifest_by_case, out_dir))
    return paths


def make_disagreement_galleries(rows: list[dict[str, str]], manifest_by_case: dict[str, dict[str, str]], out_dir: Path) -> dict[str, str]:
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["case_id"], row["protocol"])][row["model"]] = row
    comparisons = [
        ("medsam_beats_sam", "medsam_vit_b", "sam_vit_b"),
        ("sam_beats_medsam", "sam_vit_b", "medsam_vit_b"),
        ("sam2_beats_sam", "sam2_tiny", "sam_vit_b"),
    ]
    paths: dict[str, str] = {}
    for name, winner, loser in comparisons:
        diffs = []
        for models in by_key.values():
            if winner in models and loser in models:
                diff = fnum(models[winner]["dice"]) - fnum(models[loser]["dice"])
                if diff > 0:
                    item = dict(models[winner])
                    item["dice_gap"] = diff
                    diffs.append(item)
        selected = sorted(diffs, key=lambda r: r["dice_gap"], reverse=True)[:12]
        path = out_dir / "galleries" / f"{name}.png"
        make_gallery(path, name.replace("_", " "), selected, manifest_by_case)
        paths[name] = rel(path)
    return paths


def write_report(out_dir: Path, summaries: dict[str, list[dict[str, Any]]], gallery_paths: dict[str, str], tagged: list[dict[str, Any]]) -> None:
    report_path = out_dir / "phase2_lite_analysis_report.md"
    model_summary = summaries["by_model"]
    dataset_summary = summaries["by_dataset"]
    protocol_summary = summaries["by_protocol"]
    hard_rows = sorted(
        [r for r in tagged if r["status"] == "ok"],
        key=lambda r: fnum(r["dice"]),
    )[:20]
    lines = [
        "# Phase2-lite Analysis Report",
        "",
        "## 关键结论",
        "",
        f"- 当前分析基于 {len(tagged)} 条评估记录，已经足够暴露两个问题：医学目标的 prompt 敏感性很强，且不同数据域的失败模式不一样。",
        "- IDRID 的 Dice 明显低于 MonuSeg，说明眼底病灶的小目标、多实例、低对比区域仍是主要难点。",
        "- `auto_box` 在均值上最高，但失败样例也很集中，下一步应把它升级成可解释、可调参的 Auto Medical Prompt Generator。",
        "- MedSAM 只支持 box 评估；点提示不能和 SAM/SAM2 直接公平比较，应保留为 `unsupported`。",
        "",
        "## Mean Dice by Model",
        "",
    ]
    for row in model_summary:
        lines.append(f"- {row['model']}: {float(row['mean_dice']):.4f} over {row['ok']} OK rows")
    lines.extend(["", "## Mean Dice by Dataset", ""])
    for row in dataset_summary:
        lines.append(f"- {row['dataset']}: {float(row['mean_dice']):.4f} over {row['ok']} OK rows")
    lines.extend(["", "## Mean Dice by Prompt Protocol", ""])
    for row in protocol_summary:
        lines.append(f"- {row['protocol']}: {float(row['mean_dice']):.4f} over {row['ok']} OK rows")
    lines.extend(["", "## Failure Galleries", ""])
    for name, path in sorted(gallery_paths.items()):
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Worst 20 Supported Runs", ""])
    lines.append("| case | dataset | model | prompt | dice | tags |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for row in hard_rows:
        lines.append(
            f"| {row['case_id']} | {row['dataset']} | {row['model']} | {row['protocol']} | "
            f"{float(row['dice']):.4f} | {row['failure_tags']} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 先人工看 `auto_box_failures.png` 和三个模型的 `worst_*.png`，确认失败是否来自 GT 合并、prompt 不合理，还是模型本身。",
            "2. 实现 `auto_medical_prompt_v1`：针对 IDRID 和 MonuSeg 分别输出候选 box、候选点、置信度和失败原因。",
            "3. 增加 prompt robustness 曲线：同一 GT box 做不同扰动强度，量化 Dice 下降斜率。",
            "4. 再接入 SAM ViT-H / Matcher / SAM3，不急着把所有模型堆进第一张主表。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage2_target_doc(path: Path) -> None:
    text = """# Phase 2 Target: Medical Prompt Robustness Benchmark

## 总目标

Phase 2 不再只是“把更多模型跑一遍”，而是建立一个面向医学图像的 promptable segmentation 诊断框架：

```text
同一数据 + 同一 prompt 协议 + 同一评价脚本 -> 找出模型在哪里失败，以及 prompt 应该如何改进。
```

这会服务后面的核心创新：结构化医学 prompt、SAM2 memory 选择和轻量 adapter。

## 研究问题

1. 通用 SAM、MedSAM、SAM2 在医学图像上的失败模式是否一致？
2. box、point、auto box 三类 prompt 哪一类最稳定？
3. IDRID 这类小目标多实例任务，与 MonuSeg 这类密集病理核任务，是否需要不同的 prompt 生成策略？
4. 我们是否能通过自动医学 prompt 生成，在不改主干模型的情况下提升准确率和交互效率？

## 当前 Phase2-lite 已完成

- 数据：IDRID 20 cases + MonuSeg 20 cases。
- Prompt：GT box、perturbed box、centroid point、grid point、auto box。
- 模型：SAM ViT-B、MedSAM ViT-B、SAM2 Tiny。
- 输出：manifest、prompt jsonl、predicted masks、metrics、失败图集。

## 阶段性观察

- `auto_box` 当前均值最高，但它既可能说明图像启发式有效，也可能说明 GT 合并任务与局部 prompt 的定义不完全一致，需要人工检查失败图集。
- IDRID 明显更难，后续创新优先围绕小病灶、多实例、低对比区域。
- MedSAM 是强 box baseline，但不应被强行纳入 point/text prompt 比较。
- SAM/SAM2 在点提示上的稳定性需要做扰动曲线，而不是只看一个 centroid/grid 点。

## Phase 2 主输出

1. `phase2_lite_analysis_report.md`：小基准结果和失败模式。
2. `summary_by_*.csv`：按模型、数据集、prompt、difficulty 的指标表。
3. `failure_cases.csv`：带失败标签的逐样本表。
4. `galleries/*.png`：最佳、最差、模型分歧、auto-box 成功/失败图集。
5. `auto_medical_prompt_v1`：下一阶段方法原型。

## Phase 3 的连接点

Phase 2 最终要给 Phase 3 一个明确结论：

```text
优先改哪种 prompt？
在哪个数据集上改？
用什么失败类型证明这个改进是必要的？
```

推荐的 Phase 3 起点是：

```text
Auto Medical Prompt Generator v1
```

它不训练大模型，先用可解释的医学图像启发式和候选选择策略生成更好的 box/point prompt，再喂给 SAM-family 模型。这个路线成本低、实验闭环短，也容易写成论文中的方法模块。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze phase2-lite metrics and create failure galleries.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    analysis_dir = run_root / "analysis"
    manifest_path = run_root / "manifests" / "phase2_lite_40_cases.csv"
    metrics_path = run_root / "metrics" / "phase2_lite_metrics.csv"

    metrics = read_csv(metrics_path)
    manifest = read_csv(manifest_path)
    tagged = add_failure_tags(metrics)

    summaries = {
        "by_model": summarize(metrics, ["model"]),
        "by_dataset": summarize(metrics, ["dataset"]),
        "by_protocol": summarize(metrics, ["protocol"]),
        "by_model_dataset": summarize(metrics, ["model", "dataset"]),
        "by_model_protocol": summarize(metrics, ["model", "protocol"]),
        "by_dataset_protocol": summarize(metrics, ["dataset", "protocol"]),
        "by_difficulty": summarize(metrics, ["difficulty_bucket"]),
    }

    write_csv(analysis_dir / "failure_cases.csv", tagged)
    for name, rows in summaries.items():
        write_csv(analysis_dir / f"summary_{name}.csv", rows)

    gallery_paths = make_galleries(tagged, manifest, analysis_dir)
    write_report(analysis_dir, summaries, gallery_paths, tagged)
    write_stage2_target_doc(WORKSPACE_ROOT / "docs" / "phase2_medical_prompt_robustness_target.md")
    print(f"Wrote analysis outputs to {analysis_dir}", flush=True)
    print(f"Wrote target doc to {WORKSPACE_ROOT / 'docs' / 'phase2_medical_prompt_robustness_target.md'}", flush=True)


if __name__ == "__main__":
    main()
