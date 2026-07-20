from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "research_direction" / "paper_artifacts_20260715"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(f"No row matches {criteria}")


def optional_float(row: dict[str, Any], key: str) -> float | str:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else ""


def paired_effect_tuple(label: str, summary: dict[str, Any]) -> tuple[str, float, float, float]:
    return (
        label,
        float(summary["mean_delta_dice"]),
        float(summary["ci95_low"]),
        float(summary["ci95_high"]),
    )


def five_reference_plot_payload(rows: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    method_labels = [
        ("Fixed action (5-ref mean)", "Fixed"),
        ("Anchored V7 (5-ref frozen)", "Frozen V7"),
        ("Learned N-budget (5-ref frozen)", "N-gate"),
    ]
    selected = {str(row["method"]): row for row in rows}
    labels = [short_label for method, short_label in method_labels if method in selected]
    values = [float(selected[method]["mean_dice"]) for method, _ in method_labels if method in selected]
    return labels, values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    mechanism = read_csv(ROOT / "runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/method_mechanism_summary.csv")
    negative = read_csv(ROOT / "runs/research_direction/mechanisms/negative_prompt_ablation_test14/summary.csv")
    gbmseg = read_json(ROOT / "runs/research_direction/baselines/gbmseg_style_adapted_test14/summary.json")
    v7 = read_csv(ROOT / "runs/research_direction/prompt_policy/adaptive_policy_models_v7_multiref5/summary.csv")
    v8 = read_csv(ROOT / "runs/research_direction/prompt_policy/adaptive_policy_models_v8_pairwise_multiref5/summary.csv")
    negative_policy = read_csv(ROOT / "runs/research_direction/prompt_policy/adaptive_negative_budget_v1/summary.csv")
    negative_policy_eval5 = read_csv(ROOT / "runs/research_direction/prompt_policy/adaptive_negative_budget_v1_eval5/summary.csv")
    regret = read_json(ROOT / "runs/research_direction/prompt_policy/adaptive_policy_models_v7_multiref5/regret_analysis/regret_summary.json")
    background_effect = read_json(ROOT / "runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/background_suppression_paired.json")["metrics"]["dice"]
    gbmseg_effect = read_json(ROOT / "runs/research_direction/baselines/gbmseg_style_adapted_test14/paired_vs_fixed_fg_bg/paired_summary.json")
    negative_effect = read_json(ROOT / "runs/research_direction/mechanisms/negative_prompt_ablation_test14/paired_neg5_vs_neg0/paired_summary.json")
    v7_effect = read_json(ROOT / "runs/research_direction/prompt_policy/adaptive_policy_models_v7_multiref5/paired_analysis/paired_summary.json")
    v8_effect = read_json(ROOT / "runs/research_direction/prompt_policy/adaptive_policy_models_v8_pairwise_multiref5/paired_analysis/paired_summary.json")
    negative_policy_effect = read_json(ROOT / "runs/research_direction/prompt_policy/adaptive_negative_budget_v1/paired_analysis/paired_summary.json")
    negative_policy_eval5_effect = read_json(ROOT / "runs/research_direction/prompt_policy/adaptive_negative_budget_v1_eval5/paired_analysis/paired_summary.json")
    eval5 = read_json(ROOT / "runs/research_direction/prompt_policy/adaptive_policy_models_v7_multiref5_eval5/test/summary.json")
    eval5_effect = read_json(ROOT / "runs/research_direction/prompt_policy/adaptive_policy_models_v7_multiref5_eval5/test/paired_analysis/paired_summary.json")

    fg = find_row(mechanism, method="fg_topk_K40")
    fg_bg = find_row(mechanism, method="contrast_topk_K40_lam0.75")
    neg5 = find_row(negative, negative_count="5")
    v7_row = find_row(v7, split="test", method="learned_anchored_top3")
    v8_row = find_row(v8, split="test", method="learned_pairwise_anchored_top3")
    negative_policy_row = find_row(negative_policy, split="test", method="learned_negative_budget_top2")
    negative_policy_eval5_row = find_row(negative_policy_eval5, split="test", method="learned_negative_budget_top2")
    rows = [
        {"method": "GBMSeg-style adapted", "protocol": "single-reference controlled", "family": "reference baseline", "candidates_per_query": 1, "mean_dice": gbmseg["mean_dice"], "mean_object_recall": gbmseg["mean_object_recall"], "mean_merge_error": gbmseg["mean_merge_error"], "mean_pred_area_pct": gbmseg["mean_pred_area_pct"]},
        {"method": "FG-only K40", "protocol": "single-reference controlled", "family": "fixed matcher", "candidates_per_query": 1, "mean_dice": fg["mean_dice"], "mean_object_recall": fg["mean_object_recall"], "mean_merge_error": optional_float(fg, "mean_merge_error"), "mean_pred_area_pct": fg["mean_pred_area_pct"]},
        {"method": "FG-BG K40 lambda0.75", "protocol": "single-reference controlled", "family": "fixed matcher", "candidates_per_query": 1, "mean_dice": fg_bg["mean_dice"], "mean_object_recall": fg_bg["mean_object_recall"], "mean_merge_error": optional_float(fg_bg, "mean_merge_error"), "mean_pred_area_pct": fg_bg["mean_pred_area_pct"]},
        {"method": "FG-BG + 5 negatives (ablation)", "protocol": "single-reference controlled", "family": "feedback mechanism", "candidates_per_query": 1, "mean_dice": neg5["mean_dice"], "mean_object_recall": neg5["mean_object_recall"], "mean_merge_error": neg5["mean_merge_error"], "mean_pred_area_pct": neg5["mean_pred_area_pct"]},
        {"method": "Anchored policy V7", "protocol": "single-reference controlled", "family": "learned policy", "candidates_per_query": 3, "mean_dice": v7_row["mean_dice"], "mean_object_recall": v7_row["mean_object_recall"], "mean_merge_error": v7_row["mean_merge_error"], "mean_pred_area_pct": ""},
        {"method": "Pairwise policy V8", "protocol": "single-reference controlled", "family": "learned policy", "candidates_per_query": 3, "mean_dice": v8_row["mean_dice"], "mean_object_recall": v8_row["mean_object_recall"], "mean_merge_error": v8_row["mean_merge_error"], "mean_pred_area_pct": ""},
        {"method": "Learned negative budget", "protocol": "single-reference controlled", "family": "learned feedback policy", "candidates_per_query": 2, "mean_dice": negative_policy_row["mean_dice"], "mean_object_recall": negative_policy_row["mean_object_recall"], "mean_merge_error": negative_policy_row["mean_merge_error"], "mean_pred_area_pct": ""},
        {"method": "Fixed action (5-ref mean)", "protocol": "five-reference frozen evaluation", "family": "fixed matcher", "candidates_per_query": 1, "mean_dice": eval5["fixed"]["mean_dice"], "mean_object_recall": eval5["fixed"]["mean_object_recall"], "mean_merge_error": eval5["fixed"]["mean_merge_error"], "mean_pred_area_pct": ""},
        {"method": "Anchored V7 (5-ref frozen)", "protocol": "five-reference frozen evaluation", "family": "learned policy", "candidates_per_query": 3, "mean_dice": eval5["learned"]["mean_dice"], "mean_object_recall": eval5["learned"]["mean_object_recall"], "mean_merge_error": eval5["learned"]["mean_merge_error"], "mean_pred_area_pct": ""},
        {"method": "Learned N-budget (5-ref frozen)", "protocol": "five-reference frozen evaluation", "family": "learned feedback policy", "candidates_per_query": 2, "mean_dice": negative_policy_eval5_row["mean_dice"], "mean_object_recall": negative_policy_eval5_row["mean_object_recall"], "mean_merge_error": negative_policy_eval5_row["mean_merge_error"], "mean_pred_area_pct": ""},
        {"method": "Full action oracle", "protocol": "single-reference diagnostic", "family": "upper bound", "candidates_per_query": 15, "mean_dice": regret["mean_full_oracle_dice"], "mean_object_recall": "", "mean_merge_error": "", "mean_pred_area_pct": ""},
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "main_result_table.csv", rows)

    plot_rows = rows[:7]
    labels = [row["method"] for row in plot_rows]
    values = [float(row["mean_dice"]) for row in plot_rows]
    colors = ["#7A8A99", "#9AA6B2", "#167D8D", "#D99A2B", "#2D6A9F", "#9C4F4F", "#3F7D4F"]
    fig, (ax, ax_eval5) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [2.2, 1]})
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(float(regret["mean_full_oracle_dice"]), color="#20272E", linestyle="--", linewidth=1.5, label="Full oracle")
    ax.set_xlim(0.40, 0.60)
    ax.set_xlabel("Dice on 14 held-out MoNuSeg cases")
    ax.set_title("A. Single-reference controlled protocol")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    eval5_rows = rows[7:10]
    eval5_labels, eval5_values = five_reference_plot_payload(eval5_rows)
    eval5_bars = ax_eval5.bar(
        eval5_labels, eval5_values, color=["#9AA6B2", "#2D6A9F", "#3F7D4F"], width=0.62
    )
    ax_eval5.set_ylim(0.47, 0.528)
    ax_eval5.set_ylabel("Dice")
    ax_eval5.set_title("B. Five-reference frozen evaluation")
    ax_eval5.grid(axis="y", alpha=0.2)
    ax_eval5.text(
        0.98,
        0.97,
        "V7: +0.00875 [0.00376, 0.01535]\nN-gate: +0.02615 [0.01687, 0.03605]",
        transform=ax_eval5.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    for bar, value in zip(eval5_bars, eval5_values):
        ax_eval5.text(bar.get_x() + bar.get_width() / 2, value + 0.0008, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "main_method_comparison.png", dpi=220)
    plt.close(fig)

    neg_x = [int(row["negative_count"]) for row in negative]
    neg_dice = [float(row["mean_dice"]) for row in negative]
    neg_area = [float(row["mean_pred_area_pct"]) for row in negative]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(neg_x, neg_dice, marker="o", linewidth=2.2, color="#167D8D", label="Dice")
    ax.set_xlabel("Number of negative points")
    ax.set_ylabel("Dice", color="#167D8D")
    ax.tick_params(axis="y", labelcolor="#167D8D")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(neg_x, neg_area, marker="s", linewidth=1.8, color="#D46B45", label="Predicted area")
    ax2.set_ylabel("Predicted area (%)", color="#D46B45")
    ax2.tick_params(axis="y", labelcolor="#D46B45")
    ax.set_title("Negative feedback helps only under a controlled budget")
    fig.tight_layout()
    fig.savefig(OUT / "negative_feedback_curve.png", dpi=220)
    plt.close(fig)

    components = [regret["mean_prefilter_regret"], regret["mean_selector_regret"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.bar(["Prefilter regret", "Selector regret"], components, color=["#D46B45", "#2D6A9F"])
    ax.set_ylabel("Mean Dice regret")
    ax.set_title("V7 failure is dominated by candidate prefiltering")
    ax.grid(axis="y", alpha=0.2)
    for index, value in enumerate(components):
        ax.text(index, value + 0.001, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "v7_regret_decomposition.png", dpi=220)
    plt.close(fig)

    effects = [
        ("FG-BG vs FG-only", background_effect["mean_delta"], background_effect["ci95_low"], background_effect["ci95_high"]),
        ("FG-BG vs GBMSeg-style", gbmseg_effect["mean_delta_dice"], gbmseg_effect["ci95_low"], gbmseg_effect["ci95_high"]),
        ("5 negatives vs no negatives", negative_effect["mean_delta_dice"], negative_effect["ci95_low"], negative_effect["ci95_high"]),
        paired_effect_tuple("Anchored V7 vs fixed (1-ref)", v7_effect),
        paired_effect_tuple("Anchored V7 vs fixed (5-ref)", eval5_effect),
        paired_effect_tuple("Learned N-budget vs N0 (1-ref)", negative_policy_effect),
        paired_effect_tuple("Learned N-budget vs N0 (5-ref)", negative_policy_eval5_effect),
        paired_effect_tuple("Pairwise V8 vs fixed", v8_effect),
    ]
    effect_rows = [
        {"comparison": label, "delta_dice": delta, "ci95_low": low, "ci95_high": high}
        for label, delta, low, high in effects
    ]
    write_csv(OUT / "paired_effects.csv", effect_rows)
    labels = [item[0] for item in effects]
    deltas = np.asarray([item[1] for item in effects], dtype=float)
    lows = np.asarray([item[2] for item in effects], dtype=float)
    highs = np.asarray([item[3] for item in effects], dtype=float)
    y = np.arange(len(effects))[::-1]
    significant = np.logical_or(lows > 0, highs < 0)
    point_colors = np.where(significant, "#167D8D", "#8A949E")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.axvline(0.0, color="#20272E", linewidth=1.2)
    for index in range(len(effects)):
        ax.errorbar(
            deltas[index], y[index],
            xerr=[[deltas[index] - lows[index]], [highs[index] - deltas[index]]],
            fmt="o", color=point_colors[index], ecolor=point_colors[index],
            capsize=4, markersize=7, linewidth=2,
        )
        ax.text(highs[index] + 0.003, y[index], f"{deltas[index]:+.3f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired Delta Dice (95% bootstrap CI)")
    ax.set_title("Paired effects under controlled evaluation protocols")
    ax.set_xlim(float(lows.min()) - 0.008, float(highs.max()) + 0.025)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "paired_effect_forest.png", dpi=220)
    plt.close(fig)

    metadata = {
        "held_out_query_count": 14,
        "primary_metric": "Dice",
        "inference_unit": "query case",
        "warnings": [
            "GBMSeg-style is an adapted predecessor baseline, not an official SAT reproduction.",
            "The five-negative result is a pre-specified mechanism ablation but its test-best count is post-hoc; it is not a validation-frozen method result.",
            "Single-reference and five-reference rows use different evaluation protocols and are shown in separate panels; only within-protocol paired effects are interpretable.",
        ],
        "artifacts": ["main_result_table.csv", "main_method_comparison.png", "negative_feedback_curve.png", "negative_policy_qualitative.png", "v7_regret_decomposition.png", "paired_effects.csv", "paired_effect_forest.png"],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
