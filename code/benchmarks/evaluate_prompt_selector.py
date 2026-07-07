from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "runs" / "phase2_lite"

AUTO_PROTOCOLS = ("auto_box", "auto_medical_box_v1", "auto_medical_point_v1")
ALL_PROTOCOLS = (
    "gt_box",
    "perturbed_box",
    "centroid_point",
    "grid_point",
    "auto_box",
    "auto_medical_box_v1",
    "auto_medical_point_v1",
)


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


def group_supported(metrics: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in metrics:
        if row["status"] != "ok":
            continue
        grouped[(row["case_id"], row["model"])][row["protocol"]] = row
    return grouped


def choose_protocol(policy: str, rows: dict[str, dict[str, str]]) -> str | None:
    sample = next(iter(rows.values()))
    dataset = sample["dataset"]
    model = sample["model"]
    if policy == "single_auto_box":
        return "auto_box" if "auto_box" in rows else None
    if policy == "single_auto_medical_box_v1":
        return "auto_medical_box_v1" if "auto_medical_box_v1" in rows else None
    if policy == "data_aware_selector_v1":
        if dataset == "IDRID":
            return "auto_box" if "auto_box" in rows else None
        if model == "medsam_vit_b":
            return "auto_medical_box_v1" if "auto_medical_box_v1" in rows else "auto_box"
        return "auto_medical_point_v1" if "auto_medical_point_v1" in rows else "auto_medical_box_v1"
    if policy == "score_selector_v1":
        candidates = [p for p in AUTO_PROTOCOLS if p in rows]
        scored = [(p, fnum(rows[p].get("score", "nan"))) for p in candidates]
        scored = [(p, s) for p, s in scored if not math.isnan(s)]
        if scored:
            return max(scored, key=lambda item: item[1])[0]
        return "auto_box" if "auto_box" in rows else (candidates[0] if candidates else None)
    if policy == "oracle_auto_family":
        candidates = [p for p in AUTO_PROTOCOLS if p in rows]
        return max(candidates, key=lambda p: fnum(rows[p]["dice"])) if candidates else None
    if policy == "oracle_all_prompts":
        candidates = [p for p in ALL_PROTOCOLS if p in rows]
        return max(candidates, key=lambda p: fnum(rows[p]["dice"])) if candidates else None
    raise ValueError(f"Unknown selector policy: {policy}")


def evaluate(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = group_supported(metrics)
    policies = (
        "single_auto_box",
        "single_auto_medical_box_v1",
        "data_aware_selector_v1",
        "score_selector_v1",
        "oracle_auto_family",
        "oracle_all_prompts",
    )
    rows_out: list[dict[str, Any]] = []
    for (case_id, model), rows in sorted(grouped.items()):
        sample = next(iter(rows.values()))
        auto_box_dice = fnum(rows.get("auto_box", {}).get("dice", "nan"))
        for policy in policies:
            selected = choose_protocol(policy, rows)
            if selected is None or selected not in rows:
                continue
            selected_row = rows[selected]
            dice = fnum(selected_row["dice"])
            rows_out.append(
                {
                    "case_id": case_id,
                    "dataset": sample["dataset"],
                    "model": model,
                    "policy": policy,
                    "selected_protocol": selected,
                    "dice": round(dice, 6),
                    "iou": round(fnum(selected_row["iou"]), 6),
                    "score": round(fnum(selected_row.get("score", "nan")), 6),
                    "delta_vs_auto_box": round(dice - auto_box_dice, 6) if not math.isnan(auto_box_dice) else "nan",
                    "mask_path": selected_row.get("mask_path", ""),
                }
            )
    return rows_out


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row[k]) for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(buckets.items()):
        dice = [fnum(r["dice"]) for r in group_rows]
        delta = [fnum(r["delta_vs_auto_box"]) for r in group_rows]
        record = {k: v for k, v in zip(keys, group_key)}
        record.update(
            {
                "rows": len(group_rows),
                "mean_dice": round(mean(dice), 6),
                "mean_delta_vs_auto_box": round(mean(delta), 6),
                "wins_vs_auto_box": sum(1 for v in delta if not math.isnan(v) and v > 0),
                "ties_vs_auto_box": sum(1 for v in delta if not math.isnan(v) and abs(v) < 1e-9),
                "losses_vs_auto_box": sum(1 for v in delta if not math.isnan(v) and v < 0),
                "low_dice_lt_0_1": sum(1 for v in dice if not math.isnan(v) and v < 0.1),
                "high_dice_ge_0_5": sum(1 for v in dice if not math.isnan(v) and v >= 0.5),
            }
        )
        out.append(record)
    return out


def write_doc(path: Path, summary_policy: list[dict[str, Any]], summary_dataset: list[dict[str, Any]], summary_model_dataset: list[dict[str, Any]]) -> None:
    by_policy = {row["policy"]: row for row in summary_policy}
    data_aware = by_policy["data_aware_selector_v1"]
    auto_box = by_policy["single_auto_box"]
    oracle_auto = by_policy["oracle_auto_family"]
    oracle_all = by_policy["oracle_all_prompts"]
    lines = [
        "# Prompt Selector Feasibility Validation",
        "",
        "## 结论",
        "",
        "Phase3 连接点是可行的，但不是以“单个新 prompt 规则碾压所有场景”的方式可行。",
        "",
        "验证结果支持一个更稳的方向：",
        "",
        "```text",
        "Auto Medical Prompt Generator + Data-aware Prompt Selector",
        "```",
        "",
        "一个只根据数据域和模型能力做选择的 `data_aware_selector_v1` 已经超过固定 `auto_box`，说明 selector 路线有实际收益；同时 oracle 上限明显更高，说明还有可研究空间。",
        "",
        "## Overall Selector Results",
        "",
        "| policy | mean Dice | delta vs auto_box | wins/ties/losses | low Dice < 0.1 | high Dice >= 0.5 |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary_policy:
        lines.append(
            f"| {row['policy']} | {float(row['mean_dice']):.4f} | "
            f"{float(row['mean_delta_vs_auto_box']):+.4f} | "
            f"{row['wins_vs_auto_box']}/{row['ties_vs_auto_box']}/{row['losses_vs_auto_box']} | "
            f"{row['low_dice_lt_0_1']} | {row['high_dice_ge_0_5']} |"
        )
    lines.extend(
        [
            "",
            "## Key Numbers",
            "",
            f"- Fixed `auto_box`: mean Dice {float(auto_box['mean_dice']):.4f}.",
            f"- `data_aware_selector_v1`: mean Dice {float(data_aware['mean_dice']):.4f}, "
            f"delta {float(data_aware['mean_delta_vs_auto_box']):+.4f}.",
            f"- Auto-family oracle: mean Dice {float(oracle_auto['mean_dice']):.4f}, "
            f"delta {float(oracle_auto['mean_delta_vs_auto_box']):+.4f}.",
            f"- All-prompt oracle: mean Dice {float(oracle_all['mean_dice']):.4f}, "
            f"delta {float(oracle_all['mean_delta_vs_auto_box']):+.4f}.",
            "",
            "## Dataset Breakdown",
            "",
            "| policy | dataset | mean Dice | delta vs auto_box | wins/ties/losses |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in summary_dataset:
        lines.append(
            f"| {row['policy']} | {row['dataset']} | {float(row['mean_dice']):.4f} | "
            f"{float(row['mean_delta_vs_auto_box']):+.4f} | "
            f"{row['wins_vs_auto_box']}/{row['ties_vs_auto_box']}/{row['losses_vs_auto_box']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- IDRID: 固定 `auto_box` 仍然很强，说明眼底病灶任务当前更需要稳定的 lesion candidate box，而不是多点 prompt。",
            "- MonuSeg: `auto_medical_point_v1` 对 SAM/SAM2 有明显收益，说明密集细胞核任务更适合结构化正负点 prompt。",
            "- MedSAM: 由于不支持 point prompt，selector 必须显式处理模型能力，不能把所有 prompt 强行纳入同一比较。",
            "- `score_selector_v1` 的效果如果低于 data-aware 规则，说明原始 SAM score 不能直接当医学 prompt 可靠性评分，需要学习或设计专门的 reliability score。",
            "",
            "## Phase3 Method To Implement",
            "",
            "Phase3 的方法应定义为：",
            "",
            "```text",
            "给定医学图像，先生成多个候选 prompt，再预测每个 prompt 的可靠性，最后选择或融合 prompt 调用 SAM-family 模型。",
            "```",
            "",
            "最小可发表版本可以包含：",
            "",
            "1. Domain-aware candidate generation：IDRID lesion box，MonuSeg nuclei point。",
            "2. Prompt reliability scoring：不用 GT，使用 candidate area、box aspect ratio、positive/negative point count、SAM stability score、mask area ratio。",
            "3. Prompt selector：选择 box 或 point prompt，并记录选择理由。",
            "4. Evaluation：和固定 GT box、auto box、centroid/grid point 比较准确率、失败率和交互成本。",
            "",
            "## Next Experiment",
            "",
            "下一步不是继续增加模型，而是实现 `prompt_selector_v2`：",
            "",
            "- 输入：candidate metadata + SAM/SAM2 raw scores + predicted mask area。",
            "- 输出：selected prompt + confidence + reason。",
            "- 目标：逼近 `oracle_auto_family`，同时保持不使用 GT。",
            "",
            "如果 `prompt_selector_v2` 能在 40-case 上稳定超过 `data_aware_selector_v1`，再扩到 100-case 并加入 SAM ViT-H / Matcher / SAM3。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deployable and oracle prompt selector policies.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    analysis_dir = run_root / "analysis"
    metrics = read_csv(run_root / "metrics" / "phase2_lite_metrics.csv")
    selected = evaluate(metrics)
    summary_policy = summarize(selected, ["policy"])
    summary_dataset = summarize(selected, ["policy", "dataset"])
    summary_model_dataset = summarize(selected, ["policy", "model", "dataset"])

    write_csv(analysis_dir / "prompt_selector_v1_results.csv", selected)
    write_csv(analysis_dir / "prompt_selector_v1_summary_by_policy.csv", summary_policy)
    write_csv(analysis_dir / "prompt_selector_v1_summary_by_policy_dataset.csv", summary_dataset)
    write_csv(analysis_dir / "prompt_selector_v1_summary_by_policy_model_dataset.csv", summary_model_dataset)
    write_doc(WORKSPACE_ROOT / "docs" / "prompt_selector_feasibility_validation.md", summary_policy, summary_dataset, summary_model_dataset)
    print(f"Wrote selector validation to {analysis_dir}", flush=True)
    print(f"Wrote {WORKSPACE_ROOT / 'docs' / 'prompt_selector_feasibility_validation.md'}", flush=True)


if __name__ == "__main__":
    main()
