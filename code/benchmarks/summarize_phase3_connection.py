from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "runs" / "phase2_lite"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def make_comparison_rows(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in metrics:
        if row["status"] != "ok":
            continue
        by_key[(row["case_id"], row["model"])][row["protocol"]] = row

    rows: list[dict[str, Any]] = []
    for (case_id, model), protocols in sorted(by_key.items()):
        anchor = next(iter(protocols.values()))
        auto_box = fnum(protocols.get("auto_box", {}).get("dice", math.nan))
        box_v1 = fnum(protocols.get("auto_medical_box_v1", {}).get("dice", math.nan))
        centroid = fnum(protocols.get("centroid_point", {}).get("dice", math.nan))
        grid = fnum(protocols.get("grid_point", {}).get("dice", math.nan))
        point_v1 = fnum(protocols.get("auto_medical_point_v1", {}).get("dice", math.nan))
        v1_candidates = {
            "auto_medical_box_v1": box_v1,
            "auto_medical_point_v1": point_v1,
        }
        best_v1_protocol, best_v1 = max(v1_candidates.items(), key=lambda item: -math.inf if math.isnan(item[1]) else item[1])
        best_existing_candidates = {
            "auto_box": auto_box,
            "centroid_point": centroid,
            "grid_point": grid,
            "gt_box": fnum(protocols.get("gt_box", {}).get("dice", math.nan)),
        }
        best_existing_protocol, best_existing = max(
            best_existing_candidates.items(),
            key=lambda item: -math.inf if math.isnan(item[1]) else item[1],
        )
        rows.append(
            {
                "case_id": case_id,
                "dataset": anchor["dataset"],
                "model": model,
                "auto_box": round(auto_box, 6),
                "auto_medical_box_v1": round(box_v1, 6),
                "delta_box_v1_vs_auto_box": round(box_v1 - auto_box, 6) if not math.isnan(auto_box) and not math.isnan(box_v1) else "nan",
                "centroid_point": round(centroid, 6),
                "grid_point": round(grid, 6),
                "auto_medical_point_v1": round(point_v1, 6),
                "delta_point_v1_vs_centroid": round(point_v1 - centroid, 6) if not math.isnan(point_v1) and not math.isnan(centroid) else "nan",
                "best_v1_protocol": best_v1_protocol,
                "best_v1": round(best_v1, 6),
                "best_existing_protocol": best_existing_protocol,
                "best_existing": round(best_existing, 6),
                "delta_best_v1_vs_best_existing": round(best_v1 - best_existing, 6) if not math.isnan(best_v1) and not math.isnan(best_existing) else "nan",
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row[k]) for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(buckets.items()):
        record = {k: v for k, v in zip(keys, group_key)}
        for field in ("delta_box_v1_vs_auto_box", "delta_point_v1_vs_centroid", "delta_best_v1_vs_best_existing"):
            vals = [fnum(r[field]) for r in group_rows]
            record[f"mean_{field}"] = round(mean(vals), 6)
            record[f"wins_{field}"] = sum(1 for v in vals if not math.isnan(v) and v > 0)
            record[f"valid_{field}"] = sum(1 for v in vals if not math.isnan(v))
        record["rows"] = len(group_rows)
        out.append(record)
    return out


def write_connection_doc(path: Path, comparison: list[dict[str, Any]], by_dataset: list[dict[str, Any]], by_model_dataset: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 3 Connection Point",
        "",
        "## 判断",
        "",
        "Phase2-lite 已经达到 Phase3 的连接点：我们不再缺少“能跑的模型”，而是已经有证据说明下一步该研究什么。",
        "",
        "核心结论是：",
        "",
        "```text",
        "医学 prompt 不能只做一种固定策略，应该做 data-aware prompt selector。",
        "```",
        "",
        "也就是说，Phase3 的第一个方法模块不是再换一个更大的 SAM，而是：",
        "",
        "```text",
        "Auto Medical Prompt Generator + Prompt Selector",
        "```",
        "",
        "## 证据",
        "",
        "### Dataset-level delta",
        "",
        "| dataset | box_v1 vs auto_box | wins | point_v1 vs centroid | wins | best_v1 vs best_existing | wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_dataset:
        lines.append(
            f"| {row['dataset']} | {float(row['mean_delta_box_v1_vs_auto_box']):.4f} | "
            f"{row['wins_delta_box_v1_vs_auto_box']}/{row['valid_delta_box_v1_vs_auto_box']} | "
            f"{float(row['mean_delta_point_v1_vs_centroid']):.4f} | "
            f"{row['wins_delta_point_v1_vs_centroid']}/{row['valid_delta_point_v1_vs_centroid']} | "
            f"{float(row['mean_delta_best_v1_vs_best_existing']):.4f} | "
            f"{row['wins_delta_best_v1_vs_best_existing']}/{row['valid_delta_best_v1_vs_best_existing']} |"
        )
    lines.extend(
        [
            "",
            "### Model x Dataset delta",
            "",
            "| model | dataset | box_v1 vs auto_box | point_v1 vs centroid | best_v1 vs best_existing |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in by_model_dataset:
        lines.append(
            f"| {row['model']} | {row['dataset']} | "
            f"{float(row['mean_delta_box_v1_vs_auto_box']):.4f} | "
            f"{float(row['mean_delta_point_v1_vs_centroid']):.4f} | "
            f"{float(row['mean_delta_best_v1_vs_best_existing']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Phase3 方法定义",
            "",
            "Phase3 不应写成“提出一个新的 SAM”。更稳的定义是：",
            "",
            "```text",
            "给定医学图像和任务域，自动生成多个候选 prompt，预测每个 prompt 的可靠性，选择或融合最适合的 prompt，再调用 SAM-family 模型。",
            "```",
            "",
            "建议模块拆成三部分：",
            "",
            "1. Candidate generation：为 IDRID 生成红色/亮性病灶候选，为 MonuSeg 生成暗紫色细胞核候选。",
            "2. Prompt selector：在 `auto_box`、`auto_medical_box_v1`、`auto_medical_point_v1` 之间选择最可能可靠的 prompt。",
            "3. Optional refinement：只在失败类型明确时加入轻量 adapter 或后处理，不急着训练大模型。",
            "",
            "## 下一步实验",
            "",
            "1. 实现 `prompt_selector_v1`，先用无需训练的规则：dataset、candidate_area_ratio、box aspect ratio、positive/negative point count、SAM score。",
            "2. 在同一 40-case 上跑 selector，比较 `selected_prompt` 是否超过单一 prompt 均值。",
            "3. 扩到 100-case，再加入 SAM ViT-H / Matcher / SAM3 作为补充模型，而不是改变主线。",
            "",
            "## 产物位置",
            "",
            "- Comparison table: `runs/phase2_lite/analysis/auto_prompt_v1_comparison.csv`",
            "- Dataset summary: `runs/phase2_lite/analysis/auto_prompt_v1_by_dataset.csv`",
            "- Model x dataset summary: `runs/phase2_lite/analysis/auto_prompt_v1_by_model_dataset.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase2-lite auto prompt v1 results and define the Phase3 connection point.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    analysis = run_root / "analysis"
    metrics = read_csv(run_root / "metrics" / "phase2_lite_metrics.csv")
    comparison = make_comparison_rows(metrics)
    by_dataset = summarize(comparison, ["dataset"])
    by_model_dataset = summarize(comparison, ["model", "dataset"])

    write_csv(analysis / "auto_prompt_v1_comparison.csv", comparison)
    write_csv(analysis / "auto_prompt_v1_by_dataset.csv", by_dataset)
    write_csv(analysis / "auto_prompt_v1_by_model_dataset.csv", by_model_dataset)
    write_connection_doc(WORKSPACE_ROOT / "docs" / "phase3_connection_point.md", comparison, by_dataset, by_model_dataset)
    print(f"Wrote {analysis / 'auto_prompt_v1_comparison.csv'}", flush=True)
    print(f"Wrote {WORKSPACE_ROOT / 'docs' / 'phase3_connection_point.md'}", flush=True)


if __name__ == "__main__":
    main()
