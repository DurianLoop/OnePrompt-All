from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "runs" / "phase3_full"
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "analysis"
DEFAULT_REPORT = WORKSPACE_ROOT / "docs" / "phase3_reliability_predictor.md"

MODELS = ("sam_vit_b", "sam_vit_h", "medsam_vit_b", "sam2_tiny")
DATASETS = ("IDRID", "MonuSeg")
PROMPT_PREFIXES = ("gt_box", "perturbed_box", "centroid_point", "grid_point", "auto_box", "auto_medical_box_v1", "auto_medical_point_v1", "jitter_box", "jitter_point")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def prompt_family(protocol: str) -> str:
    if protocol.startswith("jitter_box"):
        return "jitter_box"
    if protocol.startswith("jitter_point"):
        return "jitter_point"
    return protocol


def one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def build_consensus_features(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    by_case_model: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            by_case_model[(row["case_id"], row["model"])].append(row)
    features: dict[tuple[str, str, str], dict[str, float]] = {}
    for (case_id, model), items in by_case_model.items():
        pred_areas = [fnum(r["pred_area"]) for r in items]
        scores = [fnum(r.get("score")) for r in items]
        median_area = float(np.nanmedian(pred_areas)) if pred_areas else math.nan
        mean_score = mean(scores)
        family_counts: dict[str, int] = defaultdict(int)
        for row in items:
            family_counts[prompt_family(row["protocol"])] += 1
        for row in items:
            pred_area = fnum(row["pred_area"])
            target_area_proxy = max(fnum(row.get("eval_width"), 1.0) * fnum(row.get("eval_height"), 1.0), 1.0)
            score = fnum(row.get("score"))
            features[(case_id, model, row["protocol"])] = {
                "pred_area_ratio": pred_area / target_area_proxy,
                "log_pred_area": math.log1p(max(pred_area, 0.0)),
                "area_to_median": math.log(max(pred_area, 1.0) / max(median_area, 1.0)) if not math.isnan(median_area) else 0.0,
                "score_centered": 0.0 if math.isnan(score) or math.isnan(mean_score) else score - mean_score,
                "family_count": float(family_counts[prompt_family(row["protocol"])]),
            }
    return features


def candidate_feature(row: dict[str, str], extra: dict[str, float]) -> list[float]:
    protocol = row["protocol"]
    family = prompt_family(protocol)
    score = fnum(row.get("score"), 0.0)
    runtime = fnum(row.get("runtime_sec"), 0.0)
    pred_area = fnum(row.get("pred_area"), 0.0)
    eval_area = max(fnum(row.get("eval_width"), 1.0) * fnum(row.get("eval_height"), 1.0), 1.0)
    is_box = 1.0 if row.get("prompt_type") == "box" else 0.0
    is_point = 1.0 if row.get("prompt_type") == "point" else 0.0
    return [
        1.0,
        *one_hot(row["dataset"], DATASETS),
        *one_hot(row["model"], MODELS),
        *one_hot(family, PROMPT_PREFIXES),
        is_box,
        is_point,
        0.0 if math.isnan(score) else score,
        math.log1p(max(runtime, 0.0)),
        pred_area / eval_area,
        math.log1p(max(pred_area, 0.0)) / 12.0,
        extra.get("area_to_median", 0.0),
        extra.get("score_centered", 0.0),
        extra.get("family_count", 0.0) / 8.0,
    ]


def fit_ridge(x_rows: list[list[float]], y_values: list[float], alpha: float) -> np.ndarray:
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    reg = np.eye(x.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def paired_bootstrap_delta(a: list[float], b: list[float], rounds: int = 10000, seed: int = 20260621) -> dict[str, float]:
    pairs = [(x, y) for x, y in zip(a, b) if not math.isnan(x) and not math.isnan(y)]
    if not pairs:
        return {"delta": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_value": math.nan}
    diff = np.asarray([x - y for x, y in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(rounds, dtype=np.float64)
    for idx in range(rounds):
        sample = rng.integers(0, len(diff), size=len(diff))
        boot[idx] = float(diff[sample].mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(rounds, len(diff)))
    null = (signs * diff).mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_value": float((np.abs(null) >= abs(float(diff.mean()))).mean()),
    }


def evaluate_predictor(metrics: list[dict[str, str]], alpha: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ok = [row for row in metrics if row.get("status") == "ok"]
    extra = build_consensus_features(ok)
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        if row["model"] in MODELS:
            by_case[row["case_id"]].append(row)
    all_cases = sorted(by_case)
    selected_rows: list[dict[str, Any]] = []
    feature_cache = {
        (row["case_id"], row["model"], row["protocol"]): candidate_feature(row, extra.get((row["case_id"], row["model"], row["protocol"]), {}))
        for row in ok
    }
    for case_id in all_cases:
        train_rows = [row for row in ok if row["case_id"] != case_id and row["model"] in MODELS]
        test_rows = by_case[case_id]
        weights = fit_ridge(
            [feature_cache[(r["case_id"], r["model"], r["protocol"])] for r in train_rows],
            [fnum(r["dice"]) for r in train_rows],
            alpha=alpha,
        )
        scored: list[tuple[float, dict[str, str]]] = []
        for row in test_rows:
            pred = float(np.asarray(feature_cache[(row["case_id"], row["model"], row["protocol"])], dtype=np.float64) @ weights)
            scored.append((pred, row))
        predicted, selected = max(scored, key=lambda item: item[0])
        fixed_sam = next((r for r in test_rows if r["model"] == "sam_vit_b" and r["protocol"] == "auto_box"), None)
        fixed_best_single = max(
            [r for r in test_rows if r["protocol"] == "auto_box"],
            key=lambda r: fnum(r["dice"]),
            default=None,
        )
        fixed_sam_dice = fnum(fixed_sam.get("dice")) if fixed_sam else math.nan
        best_auto_dice = fnum(fixed_best_single.get("dice")) if fixed_best_single else math.nan
        selected_rows.append(
            {
                "case_id": case_id,
                "dataset": selected["dataset"],
                "policy": "leave_one_case_out_ridge_reliability_v1",
                "selected_model": selected["model"],
                "selected_protocol": selected["protocol"],
                "predicted_reliability": round(predicted, 6),
                "dice": round(fnum(selected["dice"]), 6),
                "fixed_sam_vit_b_auto_box_dice": round(fixed_sam_dice, 6) if not math.isnan(fixed_sam_dice) else "nan",
                "best_auto_box_oracle_dice": round(best_auto_dice, 6) if not math.isnan(best_auto_dice) else "nan",
                "delta_vs_fixed_sam_vit_b_auto_box": round(fnum(selected["dice"]) - fixed_sam_dice, 6) if not math.isnan(fixed_sam_dice) else "nan",
                "delta_vs_best_auto_box_oracle": round(fnum(selected["dice"]) - best_auto_dice, 6) if not math.isnan(best_auto_dice) else "nan",
            }
        )
    summaries = summarize(selected_rows)
    return selected_rows, summaries


def evaluate_auto_box_predictor(metrics: list[dict[str, str]], alpha: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    ok = [
        row
        for row in metrics
        if row.get("status") == "ok"
        and row.get("protocol") == "auto_box"
        and row.get("model") in MODELS
    ]
    extra = build_consensus_features([row for row in metrics if row.get("status") == "ok"])
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        by_case[row["case_id"]].append(row)
    feature_cache = {
        (row["case_id"], row["model"], row["protocol"]): candidate_feature(row, extra.get((row["case_id"], row["model"], row["protocol"]), {}))
        for row in ok
    }
    selected_rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        train_rows = [row for row in ok if row["case_id"] != case_id]
        test_rows = by_case[case_id]
        weights = fit_ridge(
            [feature_cache[(r["case_id"], r["model"], r["protocol"])] for r in train_rows],
            [fnum(r["dice"]) for r in train_rows],
            alpha=alpha,
        )
        scored: list[tuple[float, dict[str, str]]] = []
        for row in test_rows:
            pred = float(np.asarray(feature_cache[(row["case_id"], row["model"], row["protocol"])], dtype=np.float64) @ weights)
            scored.append((pred, row))
        predicted, selected = max(scored, key=lambda item: item[0])
        pred_by_model = {row["model"]: pred for pred, row in scored}
        fixed_b = next((r for r in test_rows if r["model"] == "sam_vit_b"), None)
        fixed_h = next((r for r in test_rows if r["model"] == "sam_vit_h"), None)
        oracle = max(test_rows, key=lambda r: fnum(r["dice"]))
        fixed_b_dice = fnum(fixed_b.get("dice")) if fixed_b else math.nan
        fixed_h_dice = fnum(fixed_h.get("dice")) if fixed_h else math.nan
        selected_rows.append(
            {
                "case_id": case_id,
                "dataset": selected["dataset"],
                "policy": "leave_one_case_out_auto_box_reliability_v2",
                "selected_model": selected["model"],
                "selected_protocol": "auto_box",
                "predicted_reliability": round(predicted, 6),
                "predicted_sam_vit_h_reliability": round(pred_by_model.get("sam_vit_h", math.nan), 6),
                "predicted_margin_vs_sam_vit_h": round(predicted - pred_by_model.get("sam_vit_h", predicted), 6),
                "dice": round(fnum(selected["dice"]), 6),
                "fixed_sam_vit_b_auto_box_dice": round(fixed_b_dice, 6) if not math.isnan(fixed_b_dice) else "nan",
                "fixed_sam_vit_h_auto_box_dice": round(fixed_h_dice, 6) if not math.isnan(fixed_h_dice) else "nan",
                "best_auto_box_oracle_model": oracle["model"],
                "best_auto_box_oracle_dice": round(fnum(oracle["dice"]), 6),
                "delta_vs_fixed_sam_vit_b_auto_box": round(fnum(selected["dice"]) - fixed_b_dice, 6) if not math.isnan(fixed_b_dice) else "nan",
                "delta_vs_fixed_sam_vit_h_auto_box": round(fnum(selected["dice"]) - fixed_h_dice, 6) if not math.isnan(fixed_h_dice) else "nan",
                "delta_vs_best_auto_box_oracle": round(fnum(selected["dice"]) - fnum(oracle["dice"]), 6),
            }
        )
    summaries = summarize_v2(selected_rows)
    sig_h = paired_bootstrap_delta(
        [fnum(row["dice"]) for row in selected_rows],
        [fnum(row["fixed_sam_vit_h_auto_box_dice"]) for row in selected_rows],
    )
    return selected_rows, summaries, sig_h


def evaluate_hybrid_predictor(v2_rows: list[dict[str, Any]], threshold: float = 0.025) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for row in v2_rows:
        selected_model = str(row["selected_model"])
        selected_dice = fnum(row["dice"])
        fixed_h_dice = fnum(row["fixed_sam_vit_h_auto_box_dice"])
        fixed_b_dice = fnum(row["fixed_sam_vit_b_auto_box_dice"])
        # Conservative deployment rule: default to SAM ViT-H, switch only when
        # v2 picks another model and its predicted reliability is sufficiently high.
        use_selected = selected_model != "sam_vit_h" and fnum(row.get("predicted_margin_vs_sam_vit_h")) >= threshold
        final_model = selected_model if use_selected else "sam_vit_h"
        final_dice = selected_dice if use_selected else fixed_h_dice
        rows.append(
            {
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "policy": "conservative_auto_box_hybrid_v3",
                "selected_model": final_model,
                "selected_protocol": "auto_box",
                "dice": round(final_dice, 6),
                "fixed_sam_vit_b_auto_box_dice": row["fixed_sam_vit_b_auto_box_dice"],
                "fixed_sam_vit_h_auto_box_dice": row["fixed_sam_vit_h_auto_box_dice"],
                "delta_vs_fixed_sam_vit_b_auto_box": round(final_dice - fixed_b_dice, 6),
                "delta_vs_fixed_sam_vit_h_auto_box": round(final_dice - fixed_h_dice, 6),
                "decision": "switch" if use_selected else "fallback_sam_vit_h",
            }
        )
    summaries = summarize_v2(rows)
    sig_h = paired_bootstrap_delta(
        [fnum(row["dice"]) for row in rows],
        [fnum(row["fixed_sam_vit_h_auto_box_dice"]) for row in rows],
    )
    return rows, summaries, sig_h


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], "all")].append(row)
        groups[(row["policy"], row["dataset"])].append(row)
    out: list[dict[str, Any]] = []
    for (policy, dataset), items in sorted(groups.items()):
        out.append(
            {
                "policy": policy,
                "dataset": dataset,
                "rows": len(items),
                "mean_dice": round(mean([fnum(r["dice"]) for r in items]), 4),
                "delta_vs_fixed_sam_vit_b_auto_box": round(mean([fnum(r["delta_vs_fixed_sam_vit_b_auto_box"]) for r in items]), 4),
                "delta_vs_best_auto_box_oracle": round(mean([fnum(r["delta_vs_best_auto_box_oracle"]) for r in items]), 4),
            }
        )
    return out


def summarize_v2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], "all")].append(row)
        groups[(row["policy"], row["dataset"])].append(row)
    out: list[dict[str, Any]] = []
    for (policy, dataset), items in sorted(groups.items()):
        out.append(
            {
                "policy": policy,
                "dataset": dataset,
                "rows": len(items),
                "mean_dice": round(mean([fnum(r["dice"]) for r in items]), 4),
                "delta_vs_fixed_sam_vit_b_auto_box": round(mean([fnum(r["delta_vs_fixed_sam_vit_b_auto_box"]) for r in items]), 4),
                "delta_vs_fixed_sam_vit_h_auto_box": round(mean([fnum(r["delta_vs_fixed_sam_vit_h_auto_box"]) for r in items]), 4),
                "switch_rate": round(mean([1.0 if str(r.get("decision", "switch")) == "switch" else 0.0 for r in items]), 4) if "decision" in items[0] else "",
            }
        )
    return out


def fixed_rows(metrics: list[dict[str, str]], model: str, protocol: str) -> list[dict[str, Any]]:
    rows = [r for r in metrics if r.get("status") == "ok" and r.get("model") == model and r.get("protocol") == protocol]
    return [
        {
            "case_id": r["case_id"],
            "dataset": r["dataset"],
            "policy": f"fixed_{model}_{protocol}",
            "selected_model": model,
            "selected_protocol": protocol,
            "dice": round(fnum(r["dice"]), 6),
            "fixed_sam_vit_b_auto_box_dice": round(fnum(r["dice"]), 6) if model == "sam_vit_b" and protocol == "auto_box" else "",
            "delta_vs_fixed_sam_vit_b_auto_box": "",
        }
        for r in rows
    ]


def write_report(
    path: Path,
    selected: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    sig: dict[str, float],
    v2_summaries: list[dict[str, Any]],
    v2_sig_h: dict[str, float],
    v3_summaries: list[dict[str, Any]],
    v3_sig_h: dict[str, float],
) -> None:
    all_summary = next(row for row in summaries if row["dataset"] == "all")
    lines = [
        "# Phase3 Reliability Predictor v1",
        "",
        "目标：把上一版失败的简单 stability selector 升级为可训练的 reliability predictor。",
        "",
        "## 方法",
        "",
        "- 训练方式：leave-one-case-out ridge regression。",
        "- 训练标签：训练病例上的 Dice。",
        "- 测试选择：测试病例不使用 GT，只用模型名、数据集、prompt family、SAM score、预测面积、候选内部面积稳定性等可部署特征。",
        "- 候选池：SAM ViT-B / SAM ViT-H / MedSAM / SAM2 Tiny 的基础 prompt 与真实 jitter prompt。",
        "",
        "## 结果",
        "",
        "| Scope | Mean Dice | Delta vs SAM ViT-B auto_box | Delta vs best auto-box oracle |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['mean_dice']} | {row['delta_vs_fixed_sam_vit_b_auto_box']} | {row['delta_vs_best_auto_box_oracle']} |"
        )
    lines.extend(
        [
            "",
            "Paired bootstrap vs SAM ViT-B auto_box:",
            "",
            "```text",
            f"mean delta = {sig['delta']:.4f}",
            f"95% CI     = [{sig['ci_low']:.4f}, {sig['ci_high']:.4f}]",
            f"p-value    = {sig['p_value']:.4f}",
            "```",
            "",
        "## 判断",
        "",
        ]
    )
    if all_summary["delta_vs_fixed_sam_vit_b_auto_box"] > 0:
        lines.append("这个版本已经比固定 SAM ViT-B auto_box 有正向收益，可以作为 OCSAM-X 的第一版可部署 reliability predictor。")
    else:
        lines.append("这个版本仍未超过固定 SAM ViT-B auto_box，说明特征或模型选择策略还需要继续增强。")
    lines.append("")
    lines.append("注意：如果它没有超过 best auto-box oracle，不是问题；oracle 使用 GT 选模型，只是上限。真正需要超过的是固定单模型/固定 prompt 的可部署 baseline。")
    lines.extend(
        [
            "",
            "## v2/v3 保守强基线版本",
            "",
            "v1 失败后，v2 把候选限制为各模型的 auto_box；v3 进一步默认使用最强固定基线 SAM ViT-H，只在 predictor 有把握时切换。",
            "",
            "| Policy | Scope | Mean Dice | Delta vs SAM-B auto_box | Delta vs SAM-H auto_box | Switch rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in v2_summaries + v3_summaries:
        lines.append(
            f"| {row['policy']} | {row['dataset']} | {row['mean_dice']} | {row['delta_vs_fixed_sam_vit_b_auto_box']} | {row['delta_vs_fixed_sam_vit_h_auto_box']} | {row.get('switch_rate', '')} |"
        )
    lines.extend(
        [
            "",
            "v2 paired bootstrap vs SAM ViT-H auto_box:",
            "",
            "```text",
            f"mean delta = {v2_sig_h['delta']:.4f}",
            f"95% CI     = [{v2_sig_h['ci_low']:.4f}, {v2_sig_h['ci_high']:.4f}]",
            f"p-value    = {v2_sig_h['p_value']:.4f}",
            "```",
            "",
            "v3 paired bootstrap vs SAM ViT-H auto_box:",
            "",
            "```text",
            f"mean delta = {v3_sig_h['delta']:.4f}",
            f"95% CI     = [{v3_sig_h['ci_low']:.4f}, {v3_sig_h['ci_high']:.4f}]",
            f"p-value    = {v3_sig_h['p_value']:.4f}",
            "```",
            "",
            "结论：如果 v3 至少不低于 SAM ViT-H，说明我们可以用它作为安全部署策略；如果 v2/v3 仍低于 SAM ViT-H，则 Phase3 的真正创新点应转向更强的医学特征和 P0 baseline，而不是继续做浅层 selector。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate deployable Phase3 reliability predictor.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    metrics = read_csv(args.run_root / "metrics" / "phase2_lite_metrics.csv")
    selected, summaries = evaluate_predictor(metrics, alpha=args.alpha)
    v2_rows, v2_summaries, v2_sig_h = evaluate_auto_box_predictor(metrics, alpha=args.alpha)
    v3_rows, v3_summaries, v3_sig_h = evaluate_hybrid_predictor(v2_rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "reliability_predictor_v1_results.csv", selected)
    write_csv(args.output_root / "reliability_predictor_v1_summary.csv", summaries)
    write_csv(args.output_root / "reliability_predictor_v2_auto_box_results.csv", v2_rows)
    write_csv(args.output_root / "reliability_predictor_v2_auto_box_summary.csv", v2_summaries)
    write_csv(args.output_root / "reliability_predictor_v3_hybrid_results.csv", v3_rows)
    write_csv(args.output_root / "reliability_predictor_v3_hybrid_summary.csv", v3_summaries)
    fixed = {row["case_id"]: fnum(row["dice"]) for row in metrics if row.get("status") == "ok" and row["model"] == "sam_vit_b" and row["protocol"] == "auto_box"}
    sig = paired_bootstrap_delta([fnum(row["dice"]) for row in selected], [fixed.get(row["case_id"], math.nan) for row in selected])
    write_report(args.report, selected, summaries, sig, v2_summaries, v2_sig_h, v3_summaries, v3_sig_h)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.output_root / 'reliability_predictor_v1_results.csv'}")


if __name__ == "__main__":
    main()
