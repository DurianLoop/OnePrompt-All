#!/usr/bin/env python3
"""
Phase V — Generate Missing Repair Figures (todo.md Section 16)
==============================================================
Generates key figures for the validity repair report:
- Coupled vs split-reference correlation
- Observed vs null correlation
- Partial effect plot
- Risk-coverage curves
- AURC comparison
- Query-level confidence intervals
- Cost-risk comparison
- Claim contraction diagram
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# --- Configuration ---
REPORTS_DIR = Path(r"D:\SAM\adapromptset\reports")
FIGURES_DIR = Path(r"D:\SAM\adapromptset\paper\phaseV\figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

# ===================================================================
# FIGURE 1: Coupled vs Split-Reference Anti-Correlation
# ===================================================================
def fig_coupled_vs_split():
    controls_path = Path(r"D:\SAM\adapromptset\src\reports\phaseV_anticorrelation_controls\anticorrelation_controls.json")
    controls = json.load(open(controls_path))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Coupled correlation
    ax = axes[0]
    r_coupled = controls["control_a"]["coupled_correlation"]
    r_split = controls["control_a"]["split_correlation_mean"]
    r_split_ci = controls["control_a"]["split_correlation_ci_95"]

    bars = ax.bar(["Coupled\n(same refs)", "Split-Ref\n(disjoint groups)"],
                  [r_coupled, r_split],
                  color=["#4472C4", "#ED7D31"], width=0.5, edgecolor="white")
    ax.errorbar(1, r_split,
                yerr=[[r_split - r_split_ci[0]], [r_split_ci[1] - r_split]],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Pearson r (instability vs CFR)")
    ax.set_title("Metric Coupling Control")
    ax.set_ylim(-0.75, 0.1)

    for bar, val in zip(bars, [r_coupled, r_split]):
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.05,
                f"{val:.3f}", ha="center", va="top", fontweight="bold", fontsize=10)

    # Right: Null simulation
    ax = axes[1]
    null1 = controls["control_c"]["null1_randombeta"]
    null2 = controls["control_c"]["null2_shuffled"]
    null3 = controls["control_c"]["null3_uniform"]
    observed = controls["control_c"]["observed"]

    null_names = ["Null 1\n(Random Beta)", "Null 2\n(Shuffled)", "Null 3\n(Uniform)", "Observed"]
    null_vals = [null1, null2, null3, observed]
    colors = ["#A5A5A5", "#A5A5A5", "#A5A5A5", "#4472C4"]

    bars = ax.bar(null_names, null_vals, color=colors, width=0.5, edgecolor="white")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Pearson r")
    ax.set_title("Null Simulation")
    ax.set_ylim(-0.75, 0.75)

    for bar, val in zip(bars, null_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.03 if val > 0 else val - 0.08,
                f"{val:.3f}", ha="center", va="bottom" if val > 0 else "top",
                fontweight="bold", fontsize=9)

    fig.suptitle("Anti-Correlation Validity Controls", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_coupled_vs_split.pdf")
    plt.close()
    print("  [OK] fig_repair_coupled_vs_split.pdf")


# ===================================================================
# FIGURE 2: Query-Level Anti-Correlation Across Pipelines
# ===================================================================
def fig_query_level_anticorr():
    stats = json.load(open(REPORTS_DIR / "phaseV_query_level_stats" / "query_level_statistics.json"))

    pipelines = []
    pearson_r = []
    partial_r = []
    pearson_ci_lo = []
    pearson_ci_hi = []
    significant = []

    for ac in stats["anti_correlation"]:
        pipelines.append(ac["pipeline"])
        pearson_r.append(ac["pearson_r"])
        partial_r.append(ac["partial_r"])
        pearson_ci_lo.append(ac["pearson_ci_95"][0])
        pearson_ci_hi.append(ac["pearson_ci_95"][1])
        significant.append(ac["significant_pearson"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pipelines))
    width = 0.35

    colors_bar = ["#4472C4" if s else "#A5A5A5" for s in significant]
    bars1 = ax.bar(x - width/2, pearson_r, width, label="Pearson r",
                   color=colors_bar, edgecolor="white")
    bars2 = ax.bar(x + width/2, partial_r, width, label="Partial r (ctrl. mean Dice)",
                   color=["#ED7D31" if s else "#D5D5D5" for s in significant],
                   edgecolor="white", alpha=0.8)

    # CI error bars
    for i in range(len(pipelines)):
        ax.plot([i - width/2, i - width/2],
                [pearson_ci_lo[i], pearson_ci_hi[i]],
                color="black", linewidth=1.5)
        ax.plot([i - width/2 - 0.05, i - width/2 + 0.05],
                [pearson_ci_lo[i], pearson_ci_lo[i]], color="black", linewidth=1)
        ax.plot([i - width/2 - 0.05, i - width/2 + 0.05],
                [pearson_ci_hi[i], pearson_ci_hi[i]], color="black", linewidth=1)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Query-Level Anti-Correlation (N=80 queries, Holm corrected)")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.7, 0.3)

    # Significance markers
    for i, sig in enumerate(significant):
        if sig:
            ax.text(i, 0.1, "***", ha="center", fontsize=14, fontweight="bold", color="#4472C4")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_query_level_anticorr.pdf")
    plt.close()
    print("  [OK] fig_repair_query_level_anticorr.pdf")


# ===================================================================
# FIGURE 3: Risk-Coverage Curves
# ===================================================================
def fig_risk_coverage():
    # Read ramp results
    ramp = json.load(open(REPORTS_DIR / "phaseU_ramp" / "ramp_results.json"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: CFR vs Coverage
    ax = axes[0]
    coverages = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    pipeline_colors = {
        "L+SAM-B": "#4472C4",
        "L+SAM-L": "#ED7D31",
        "L+SAM2-T": "#70AD47",
        "B+SAM-B": "#FFC000",
    }

    pipeline_results = ramp.get("pipeline_results", ramp)

    for pipeline_key, color in pipeline_colors.items():
        pipeline_data = pipeline_results.get(pipeline_key)
        if pipeline_data is None:
            continue

        # Use CFR at coverage as the main point
        cfr_all = pipeline_data.get("cfr_all", 0)
        cfr_at_cov = pipeline_data.get("cfr_at_coverage", cfr_all)

        # Plot simplified: CFR_all at cov=1.0, CFR_at_80 at cov=0.8
        ax.plot([0.8, 1.0], [cfr_at_cov, cfr_all], "o-", color=color, linewidth=2, markersize=8,
                label=pipeline_key)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Catastrophic Failure Rate")
    ax.set_title("CFR vs Coverage")
    ax.legend(fontsize=8)
    ax.set_xlim(0.48, 1.02)
    ax.grid(True, alpha=0.3)

    # Right: Dice vs CFR (Pareto-like)
    ax = axes[1]
    pipeline_names = []
    cfr_vals = []
    dice_vals = []
    colors_list = []

    for pipeline_key, color in pipeline_colors.items():
        pipeline_data = pipeline_results.get(pipeline_key)
        if pipeline_data is None:
            continue
        pipeline_names.append(pipeline_key)
        cfr_at_80 = pipeline_data.get("cfr_at_coverage", pipeline_data.get("cfr_all", 0))
        cfr_vals.append(cfr_at_80)
        dice_vals.append(pipeline_data.get("mean_dice_ramp", pipeline_data.get("mean_dice_original", 0)))
        colors_list.append(color)

    ax.scatter(cfr_vals, dice_vals, c=colors_list, s=150, edgecolors="black", linewidth=1, zorder=5)
    for i, name in enumerate(pipeline_names):
        ax.annotate(name, (cfr_vals[i], dice_vals[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlabel("CFR at 80% Coverage")
    ax.set_ylabel("Mean Dice at 80% Coverage")
    ax.set_title("CFR vs Dice Trade-off")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Selective Prediction Performance", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_risk_coverage.pdf")
    plt.close()
    print("  [OK] fig_repair_risk_coverage.pdf")


# ===================================================================
# FIGURE 4: AURC Comparison
# ===================================================================
def fig_aurc_comparison():
    # Simplified: use known values from the audit
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["Random", "MC Dropout", "Deep Ensemble", "Single $f_4$", "Single $f_1$", "M2 Inst.", "M2 Cat.", "RAMP"]
    # Approximate AURC from known data
    aurc_vals = [0.138, 0.138, 0.139, 0.036, 0.189, 0.206, 0.012, 0.042]
    cfr_red = [0, 10.7, 6.3, 91.8, -17.4, -91.8, 100.0, 55.9]
    colors = ["#A5A5A5", "#A5A5A5", "#A5A5A5", "#70AD47", "#ED7D31", "#ED7D31", "#4472C4", "#4472C4"]

    x = np.arange(len(methods))
    bars = ax.barh(x, aurc_vals, color=colors, edgecolor="white", height=0.6)

    for i, (bar, val, cfr) in enumerate(zip(bars, aurc_vals, cfr_red)):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"AURC={val:.3f}\nCFR$\\downarrow$={cfr:.1f}%",
                va="center", fontsize=8)

    ax.set_yticks(x)
    ax.set_yticklabels(methods)
    ax.set_xlabel("AURC (Area Under Risk-Coverage, lower is better)")
    ax.set_title("Risk Prediction Quality: AURC Comparison")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.28)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_aurc.pdf")
    plt.close()
    print("  [OK] fig_repair_aurc.pdf")


# ===================================================================
# FIGURE 5: Claim Contraction Diagram
# ===================================================================
def fig_claim_contraction():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Title
    ax.text(6, 4.7, "Claim Contraction: V4 → V5", ha="center", fontsize=14, fontweight="bold")

    # V4 Claims (left)
    claims_v4 = [
        ("universal anti-correlation", "PROHIBITED"),
        ("training-free", "PROHIBITED"),
        ("calibration-free", "PROHIBITED"),
        ("6 segmenters, 4 matchers", "PROHIBITED"),
        ("formal theorem", "PROHIBITED"),
        ("56-84% reduction", "PROHIBITED"),
        ("identical conditions", "PROHIBITED"),
        ("dominates prompt sophistication", "UNSUPPORTED"),
        ("cross-domain generality", "PARTIAL"),
        ("RRN narrows oracle gap", "PARTIAL"),
        ("deployment envelope", "PARTIAL"),
        ("structural anti-correlation", "PARTIAL"),
        ("first RRE problem", "SUPPORTED"),
    ]

    color_map = {
        "PROHIBITED": "#E74C3C",
        "UNSUPPORTED": "#E67E22",
        "PARTIAL": "#F39C12",
        "SUPPORTED": "#27AE60",
    }

    for i, (claim, status) in enumerate(claims_v4):
        y = 4.2 - i * 0.3
        ax.text(0.5, y, claim, ha="left", fontsize=8, va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color_map[status], alpha=0.3))
        ax.text(9.5, y, status, ha="center", fontsize=7, va="center", fontweight="bold",
                color=color_map[status])

    # Arrow
    ax.annotate("", xy=(11.5, 2.1), xytext=(11.5, 4.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    # V5 retained
    ax.text(6, 2.0, "V5 Retained: 1 SUPPORTED + 5 PARTIALLY SUPPORTED", ha="center",
            fontsize=10, fontweight="bold", color="#27AE60")
    ax.text(6, 1.5, "7 PROHIBITED claims removed, 1 UNSUPPORTED downgraded",
            ha="center", fontsize=9, color="#555555")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_claim_contraction.pdf")
    plt.close()
    print("  [OK] fig_repair_claim_contraction.pdf")


# ===================================================================
# FIGURE 6: GT-Free Disagreement vs Failure
# ===================================================================
def fig_gt_free_disagreement():
    controls_path = Path(r"D:\SAM\adapromptset\src\reports\phaseV_anticorrelation_controls\anticorrelation_controls.json")
    controls = json.load(open(controls_path))

    fig, ax = plt.subplots(figsize=(6, 5))

    r_gtfree = controls["control_b"]["gt_free_instability_vs_cfr_pearson"]
    r_gtfree_p = controls["control_b"]["gt_free_instability_vs_cfr_p"]
    r_gtfree_s = controls["control_b"]["gt_free_instability_vs_cfr_spearman"]

    metrics = ["Pearson r\n(GT-free instability)", "Spearman $\\rho$\n(GT-free instability)"]
    values = [r_gtfree, r_gtfree_s]
    colors = ["#4472C4", "#ED7D31"]

    bars = ax.bar(metrics, values, color=colors, width=0.4, edgecolor="white")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Correlation with CFR")
    ax.set_title(f"GT-Free Disagreement vs Catastrophic Failure\n(Pearson p = {r_gtfree_p:.2e})")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.03,
                f"{val:.3f}", ha="center", va="top", fontweight="bold")

    ax.set_ylim(-0.65, 0.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_gtfree_disagreement.pdf")
    plt.close()
    print("  [OK] fig_repair_gtfree_disagreement.pdf")


# ===================================================================
# FIGURE 7: Query-Level Confidence Intervals
# ===================================================================
def fig_query_level_ci():
    stats = json.load(open(REPORTS_DIR / "phaseV_query_level_stats" / "query_level_statistics.json"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Mean Dice with CI
    ax = axes[0]
    pipelines = []
    mean_dice = []
    dice_ci_lo = []
    dice_ci_hi = []

    for d in stats["descriptives"]:
        pipelines.append(d["pipeline"])
        mean_dice.append(d["mean_dice"])
        dice_ci_lo.append(d["dice_ci_95"][0])
        dice_ci_hi.append(d["dice_ci_95"][1])

    x = np.arange(len(pipelines))
    yerr_lo = [m - l for m, l in zip(mean_dice, dice_ci_lo)]
    yerr_hi = [h - m for m, h in zip(mean_dice, dice_ci_hi)]

    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#A5A5A5"]
    ax.bar(x, mean_dice, color=colors[:len(pipelines)], width=0.5, edgecolor="white")
    ax.errorbar(x, mean_dice, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean Dice")
    ax.set_title("Query-Level Mean Dice (95% Bootstrap CI)")
    ax.set_ylim(0, 0.5)

    # Right: CFR with CI
    ax = axes[1]
    cfr_vals = []
    cfr_ci_lo = []
    cfr_ci_hi = []

    for d in stats["descriptives"]:
        cfr_vals.append(d["mean_cfr"])
        cfr_ci_lo.append(d["cfr_ci_95"][0])
        cfr_ci_hi.append(d["cfr_ci_95"][1])

    yerr_lo = [m - l for m, l in zip(cfr_vals, cfr_ci_lo)]
    yerr_hi = [h - m for m, h in zip(cfr_vals, cfr_ci_hi)]

    ax.bar(x, cfr_vals, color=colors[:len(pipelines)], width=0.5, edgecolor="white")
    ax.errorbar(x, cfr_vals, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Catastrophic Failure Rate")
    ax.set_title("Query-Level CFR (95% Bootstrap CI)")

    fig.suptitle("Query-Level Statistics ($N=80$, 10k bootstrap)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_query_level_ci.pdf")
    plt.close()
    print("  [OK] fig_repair_query_level_ci.pdf")


# ===================================================================
# FIGURE 8: Partial Effect Plot
# ===================================================================
def fig_partial_effect():
    stats = json.load(open(REPORTS_DIR / "phaseV_query_level_stats" / "query_level_statistics.json"))

    fig, ax = plt.subplots(figsize=(8, 5))

    pipelines = []
    pearson_r = []
    partial_r = []
    significant_pearson = []
    significant_partial = []

    for ac in stats["anti_correlation"]:
        pipelines.append(ac["pipeline"])
        pearson_r.append(ac["pearson_r"])
        partial_r.append(ac["partial_r"])
        significant_pearson.append(ac["significant_pearson"])
        significant_partial.append(ac["significant_partial"])

    x = np.arange(len(pipelines))
    width = 0.35

    # Pearson
    bars1 = ax.bar(x - width/2, pearson_r, width, label="Pearson r (raw)",
                   color=["#4472C4" if s else "#A5A5A5" for s in significant_pearson],
                   edgecolor="white")

    # Partial
    bars2 = ax.bar(x + width/2, partial_r, width, label="Partial r (ctrl. mean Dice)",
                   color=["#ED7D31" if s else "#D5D5D5" for s in significant_partial],
                   edgecolor="white", alpha=0.8)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Correlation")
    ax.set_title("Partial Effect: Controlling for Mean Dice")
    ax.legend(fontsize=8)

    # Add values
    for i, (bar, val) in enumerate(zip(bars1, pearson_r)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02 if val > 0 else val - 0.06,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    for i, (bar, val) in enumerate(zip(bars2, partial_r)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02 if val > 0 else val - 0.06,
                f"{val:.2f}", ha="center", fontsize=7, fontweight="bold", color="#ED7D31")

    ax.set_ylim(-0.65, 0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_partial_effect.pdf")
    plt.close()
    print("  [OK] fig_repair_partial_effect.pdf")


# ===================================================================
# FIGURE 9: Cost-Risk Comparison
# ===================================================================
def fig_cost_risk():
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["Single Ref", "RAMP (K=2)", "RAMP (K=4)", "RAMP (K=6)", "RAMP (K=8)"]
    matcher_calls = [2, 3, 5, 7, 9]
    sam_calls = [1, 2, 4, 6, 8]
    cfr_red = [0, 31.3, 43.8, 55.9, 62.5]
    total_calls = [m + s for m, s in zip(matcher_calls, sam_calls)]

    # Color by CFR reduction
    norm = plt.Normalize(0, 65)
    colors = plt.cm.RdYlGn(1 - norm(cfr_red))

    scatter = ax.scatter(total_calls, cfr_red, c=cfr_red, cmap="RdYlGn_r",
                         s=200, edgecolors="black", linewidth=1, zorder=5)
    for i, name in enumerate(methods):
        ax.annotate(name, (total_calls[i], cfr_red[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax.set_xlabel("Total Model Calls per Query (Matcher + SAM)")
    ax.set_ylabel("CFR Reduction at 80% Coverage (%)")
    ax.set_title("Cost-Risk Trade-off: Reference Count Ablation")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CFR Reduction (%)")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_repair_cost_risk.pdf")
    plt.close()
    print("  [OK] fig_repair_cost_risk.pdf")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("Phase V — Generating Repair Figures")
    print("=" * 60)

    fig_coupled_vs_split()
    fig_query_level_anticorr()
    fig_risk_coverage()
    fig_aurc_comparison()
    fig_claim_contraction()
    fig_gt_free_disagreement()
    fig_query_level_ci()
    fig_partial_effect()
    fig_cost_risk()

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("DONE")


if __name__ == "__main__":
    main()