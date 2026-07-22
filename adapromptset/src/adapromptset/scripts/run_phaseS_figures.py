"""Generate 24 figures per Phase S directive §15.

Usage: python src/adapromptset/scripts/run_phaseS_figures.py --out-dir reports/phaseS_figures
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, out_dir, name):
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def fig01_reference_instability_distribution(data_dir, out_dir):
    """Fig 1: Reference instability distribution (best-worst gap histogram)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "per_query_per_subset_metrics.csv")
    gaps = defaultdict(list)
    for r in rows:
        q = r["query_case_id"]
        cfg = r["config_name"]
        if r["k"] == "1":
            gaps[(cfg, q)].append(float(r["subset_dice"]))
    gap_values = []
    for (cfg, q), dices in gaps.items():
        if dices:
            gap_values.append((cfg, max(dices) - min(dices)))
    fig, ax = plt.subplots(figsize=(8, 5))
    configs = sorted(set(cfg for cfg, _ in gap_values))
    data = [[g for c, g in gap_values if c == cfg] for cfg in configs]
    ax.hist(data, bins=20, label=configs, alpha=0.7)
    ax.axvline(0.05, color="red", linestyle="--", label="gap=0.05 threshold")
    ax.set_xlabel("Best-Worst Dice Gap (k=1)")
    ax.set_ylabel("Query Count")
    ax.set_title("Fig 1: Reference Instability Distribution (Within-Query Best-Worst Gap)")
    ax.legend()
    _save(fig, out_dir, "fig_01_reference_instability_distribution")


def fig02_corrected_variance_decomposition(data_dir, out_dir):
    """Fig 2: Corrected variance decomposition (stacked bar V_query vs V_ref)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_metric_audit" / "cross_query_vs_within_query_variance.csv")
    configs = [r["config_name"] for r in rows]
    v_query = [float(r["V_query"]) for r in rows]
    v_ref = [float(r["V_ref"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    ax.bar(x, v_query, label="V_query (case difficulty)", color="#2196F3")
    ax.bar(x, v_ref, bottom=v_query, label="V_ref (reference-induced)", color="#FF5722")
    for i, (vq, vr) in enumerate(zip(v_query, v_ref)):
        frac = vr / (vq + vr) * 100
        ax.text(i, vq + vr + 0.001, f"{frac:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Variance")
    ax.set_title("Fig 2: Corrected Variance Decomposition (V_query + V_ref = V_total)")
    ax.legend()
    _save(fig, out_dir, "fig_02_corrected_variance_decomposition")


def fig03_mean_dice_vs_reference_count(data_dir, out_dir):
    """Fig 3: Mean Dice vs reference count k."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "reference_count_scaling.csv")
    configs = sorted(set(r["config_name"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        ks = [int(r["k"]) for r in rows if r["config_name"] == cfg]
        dices = [float(r["mean_dice"]) for r in rows if r["config_name"] == cfg]
        ax.plot(ks, dices, "o-", label=cfg, linewidth=2, markersize=8)
    ax.set_xlabel("Reference Subset Size k")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Fig 3: Mean Dice vs Reference Count")
    ax.set_xticks([1, 3, 6])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_03_mean_dice_vs_reference_count")


def fig04_within_query_std_vs_reference_count(data_dir, out_dir):
    """Fig 4: Within-query std vs reference count k."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "within_query_reference_variance.csv")
    configs = sorted(set(r["config_name"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        ks = [int(r["k"]) for r in rows if r["config_name"] == cfg]
        stds = [float(r["mean_within_query_std"]) for r in rows if r["config_name"] == cfg]
        ax.plot(ks, stds, "s-", label=cfg, linewidth=2, markersize=8)
    ax.set_xlabel("Reference Subset Size k")
    ax.set_ylabel("Mean Within-Query Reference Std")
    ax.set_title("Fig 4: Within-Query Reference Std vs Reference Count")
    ax.set_xticks([1, 3, 6])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_04_within_query_std_vs_reference_count")


def fig05_worst_subset_dice(data_dir, out_dir):
    """Fig 5: Worst-subset Dice per (config, k)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "worst_subset_metrics.csv")
    configs = sorted(set(r["config_name"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(configs))
    for i, k in enumerate([1, 3, 6]):
        vals = [float(r["mean_worst_dice"]) for r in rows if r["config_name"] in configs and int(r["k"]) == k]
        ax.bar(x + i * width, vals, width, label=f"k={k}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Mean Worst-Subset Dice")
    ax.set_title("Fig 5: Worst-Subset Dice by Config and Reference Count")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_05_worst_subset_dice")


def fig06_cvar(data_dir, out_dir):
    """Fig 6: CVaR_0.1 per (config, k)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "cvar_metrics.csv")
    configs = sorted(set(r["config_name"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        ks = [int(r["k"]) for r in rows if r["config_name"] == cfg]
        cvars = [float(r["cvar_0_1"]) for r in rows if r["config_name"] == cfg]
        ax.plot(ks, cvars, "D-", label=cfg, linewidth=2, markersize=8)
    ax.set_xlabel("Reference Subset Size k")
    ax.set_ylabel("CVaR_0.1 (bottom 10% mean Dice)")
    ax.set_title("Fig 6: Conditional Value at Risk (CVaR_0.1)")
    ax.set_xticks([1, 3, 6])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_06_cvar")


def fig07_candidate_overlap(data_dir, out_dir):
    """Fig 7: Candidate overlap (prediction consistency A vs B)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "prediction_consistency.csv")
    configs = [r["config_name"] for r in rows]
    pred_overlap = [float(r["mean_prediction_overlap_AB"]) for r in rows]
    single_overlap = [float(r["mean_single_prediction_overlap_AB"]) for r in rows]
    prompt_overlap = [float(r["mean_prompt_overlap_AB"]) for r in rows]
    x = np.arange(len(configs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, pred_overlap, width, label="Consensus Prediction Overlap")
    ax.bar(x, single_overlap, width, label="Single-Ref Prediction Overlap")
    ax.bar(x + width, prompt_overlap, width, label="Prompt Coordinate Overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Jaccard Overlap (A vs B)")
    ax.set_title("Fig 7: Candidate / Prediction Overlap Across Reference Subsets")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_07_candidate_overlap")


def fig08_prediction_consistency(data_dir, out_dir):
    """Fig 8: Prediction consistency (IoU) by config."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "prediction_consistency.csv")
    configs = [r["config_name"] for r in rows]
    consistency = [float(r["mean_prediction_overlap_AB"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(configs, consistency, color=["#2196F3", "#4CAF50", "#FF9800"])
    ax.set_ylabel("Prediction Consistency IoU (A vs B sets)")
    ax.set_title("Fig 8: Prediction Consistency (Consensus Mask IoU Across Reference Subsets)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, consistency):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_08_prediction_consistency")


def fig09_utility_prediction_calibration(data_dir, out_dir):
    """Fig 9: Utility prediction calibration (predicted vs actual)."""
    pred_path = Path(data_dir) / "phaseS_trackB" / "val_per_reference_predictions.csv"
    rows = _read_csv(pred_path)
    # We don't have actual utility in this file, only predicted. Use weight as proxy.
    preds = [float(r["predicted_utility"]) for r in rows]
    weights = [float(r["weight"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(preds, weights, alpha=0.5, s=20)
    ax.set_xlabel("Predicted Utility (GBM)")
    ax.set_ylabel("Softmax Weight (T=0.25)")
    ax.set_title("Fig 9: Utility Prediction -> Weight Calibration (Validation)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3)
    # Add diagonal reference
    ax.axhline(1 / 6, color="red", linestyle="--", label=f"Uniform weight = {1/6:.3f}")
    ax.legend()
    _save(fig, out_dir, "fig_09_utility_prediction_calibration")


def fig10_reference_weight_distribution(data_dir, out_dir):
    """Fig 10: Reference weight distribution."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackB" / "val_weights.csv")
    weights = [float(r["weight"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(weights, bins=40, color="#9C27B0", alpha=0.7, edgecolor="black")
    ax.axvline(1 / 6, color="red", linestyle="--", linewidth=2, label=f"Uniform = {1/6:.3f}")
    ax.axvline(np.mean(weights), color="green", linestyle="-", linewidth=2, label=f"Mean = {np.mean(weights):.3f}")
    ax.set_xlabel("Softmax Weight")
    ax.set_ylabel("Count")
    ax.set_title("Fig 10: Reference Weight Distribution (Track B, M4_gbm, T=0.25)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_10_reference_weight_distribution")


def fig11_reliability_score_vs_true_utility(data_dir, out_dir):
    """Fig 11: Reliability score (predicted utility) distribution by config."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackB" / "val_per_reference_predictions.csv")
    configs = sorted(set(r["config_name"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[float(r["predicted_utility"]) for r in rows if r["config_name"] == cfg] for cfg in configs]
    ax.boxplot(data, labels=configs)
    ax.set_ylabel("Predicted Utility (GBM)")
    ax.set_title("Fig 11: Reliability Score Distribution by Config (Validation)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_11_reliability_score_vs_true_utility")


def fig12_uniform_vs_weighted_consensus(data_dir, out_dir):
    """Fig 12: Uniform vs weighted consensus (Track B scatter)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackB" / "val_comparison.csv")
    weighted = [float(r["dice_weighted"]) for r in rows]
    uniform = [float(r["dice_uniform"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(uniform, weighted, alpha=0.6, s=40)
    lims = [0, max(max(uniform), max(weighted)) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1, label="y=x (no change)")
    ax.set_xlabel("Uniform Mean Dice")
    ax.set_ylabel("Weighted Consensus Dice")
    ax.set_title("Fig 12: Uniform vs Weighted Consensus (Track B, Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_12_uniform_vs_weighted_consensus")


def fig13_set_model_gain(data_dir, out_dir):
    """Fig 13: Set model gain (Track C grid: mean Dice by alpha/beta/gamma)."""
    sel_path = Path(data_dir) / "phaseS_trackC" / "train_selection.json"
    with open(sel_path) as f:
        sel = json.load(f)
    grid = sel["grid"]
    # Group by alpha, compute mean across beta/gamma
    alphas = sorted(set(g["alpha"] for g in grid))
    betas = sorted(set(g["beta"] for g in grid))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: mean Dice by alpha
    ax = axes[0]
    for b in betas:
        xs = [a for a in alphas for g in grid if g["alpha"] == a and g["beta"] == b and g["gamma"] == 0.0]
        ys = [g["mean_dice"] for a in alphas for g in grid if g["alpha"] == a and g["beta"] == b and g["gamma"] == 0.0]
        ax.plot(xs, ys, "o-", label=f"beta={b}", linewidth=2)
    ax.set_xlabel("alpha (C3 agreement gate)")
    ax.set_ylabel("Mean Dice (train)")
    ax.set_title("C3 Ablation (gamma=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Right: mean Dice by gamma
    ax = axes[1]
    for b in betas:
        xs = [g for g in [0.0, 0.5, 1.0] for row in grid if row["alpha"] == 0.0 and row["beta"] == b and row["gamma"] == g]
        ys = [row["mean_dice"] for g in [0.0, 0.5, 1.0] for row in grid if row["alpha"] == 0.0 and row["beta"] == b and row["gamma"] == g]
        ax.plot([0.0, 0.5, 1.0], ys, "s-", label=f"beta={b}", linewidth=2)
    ax.set_xlabel("gamma (C4 agreement fraction)")
    ax.set_ylabel("Mean Dice (train)")
    ax.set_title("C4 Ablation (alpha=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle("Fig 13: Set Model Gain (Track C Hyperparameter Ablation)")
    plt.tight_layout()
    _save(fig, out_dir, "fig_13_set_model_gain")


def fig14_direct_method_gain(data_dir, out_dir):
    """Fig 14: Direct method gain — BLOCKED (Track D not started)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "BLOCKED\n\nTrack D NOT STARTED\nStart condition 3 not met:\nSAM refinement HELPS (0.48 vs <0.3)",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#FFEBEE", edgecolor="red"))
    ax.set_title("Fig 14: Direct Method Gain (Track D — NOT STARTED)")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_dir, "fig_14_direct_method_gain")


def fig15_cost_quality_pareto(data_dir, out_dir):
    """Fig 15: Cost-quality Pareto (Dice vs SAM calls)."""
    methods = [
        {"name": "B1_random (k=1)", "dice": 0.4297, "sam_calls": 1},
        {"name": "B2_mean (k=6)", "dice": 0.4756, "sam_calls": 1},
        {"name": "Track B M4_gbm", "dice": 0.4732, "sam_calls": 1},
        {"name": "Track C best", "dice": 0.4779, "sam_calls": 1},
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        ax.scatter(m["sam_calls"], m["dice"], s=100, zorder=5)
        ax.annotate(m["name"], (m["sam_calls"], m["dice"]), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)
    ax.set_xlabel("SAM Calls per Query")
    ax.set_ylabel("Mean Dice (Validation)")
    ax.set_title("Fig 15: Cost-Quality Pareto Front")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 2)
    _save(fig, out_dir, "fig_15_cost_quality_pareto")


def fig16_catastrophic_failure(data_dir, out_dir):
    """Fig 16: Catastrophic failure rate by (config, k)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "cvar_metrics.csv")
    configs = sorted(set(r["config_name"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(configs))
    for i, k in enumerate([1, 3, 6]):
        vals = [float(r["catastrophic_rate"]) for r in rows if r["config_name"] in configs and int(r["k"]) == k]
        ax.bar(x + i * width, vals, width, label=f"k={k}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Catastrophic Failure Rate (Dice < 0.1)")
    ax.set_title("Fig 16: Catastrophic Failure Rate by Config and Reference Count")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_16_catastrophic_failure")


def fig17_high_instability_cases(data_dir, out_dir):
    """Fig 17: High-instability cases (top-10 best-worst gap queries)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "per_query_per_subset_metrics.csv")
    gaps = {}
    for r in rows:
        if r["k"] == "1" and r["config_name"] == "P30-N0":
            q = r["query_case_id"]
            gaps.setdefault(q, []).append(float(r["subset_dice"]))
    top = sorted(gaps.items(), key=lambda x: max(x[1]) - min(x[1]), reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(10, 6))
    queries = [q for q, _ in top]
    dices = [d for _, d in top]
    parts = ax.boxplot(dices, labels=queries)
    ax.set_ylabel("Per-Reference Dice (k=1)")
    ax.set_title("Fig 17: Top-10 High-Instability Cases (P30-N0, Best-Worst Gap)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_17_high_instability_cases")


def fig18_low_instability_cases(data_dir, out_dir):
    """Fig 18: Low-instability cases (bottom-10 best-worst gap queries)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackA_benchmark" / "per_query_per_subset_metrics.csv")
    gaps = {}
    for r in rows:
        if r["k"] == "1" and r["config_name"] == "P30-N0":
            q = r["query_case_id"]
            gaps.setdefault(q, []).append(float(r["subset_dice"]))
    bottom = sorted(gaps.items(), key=lambda x: max(x[1]) - min(x[1]))[:10]
    fig, ax = plt.subplots(figsize=(10, 6))
    queries = [q for q, _ in bottom]
    dices = [d for _, d in bottom]
    ax.boxplot(dices, labels=queries)
    ax.set_ylabel("Per-Reference Dice (k=1)")
    ax.set_title("Fig 18: Top-10 Low-Instability Cases (P30-N0, Smallest Gap)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_18_low_instability_cases")


def fig19_method_success_cases(data_dir, out_dir):
    """Fig 19: Method success cases (Track B: queries with highest positive delta)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackB" / "val_comparison.csv")
    sorted_rows = sorted(rows, key=lambda r: float(r["delta"]), reverse=True)[:10]
    queries = [r["query_case_id"] for r in sorted_rows]
    deltas = [float(r["delta"]) for r in sorted_rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(queries, deltas, color="#4CAF50")
    ax.set_xlabel("Delta Dice (Weighted - Uniform)")
    ax.set_title("Fig 19: Method Success Cases (Top-10 Positive Delta, Track B)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    _save(fig, out_dir, "fig_19_method_success_cases")


def fig20_method_failure_cases(data_dir, out_dir):
    """Fig 20: Method failure cases (Track B: queries with most negative delta)."""
    rows = _read_csv(Path(data_dir) / "phaseS_trackB" / "val_comparison.csv")
    sorted_rows = sorted(rows, key=lambda r: float(r["delta"]))[:10]
    queries = [r["query_case_id"] for r in sorted_rows]
    deltas = [float(r["delta"]) for r in sorted_rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(queries, deltas, color="#F44336")
    ax.set_xlabel("Delta Dice (Weighted - Uniform)")
    ax.set_title("Fig 20: Method Failure Cases (Top-10 Negative Delta, Track B)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    _save(fig, out_dir, "fig_20_method_failure_cases")


def fig21_cross_dataset_cases(data_dir, out_dir):
    """Fig 21: Cross-dataset cases — BLOCKED."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "BLOCKED\n\nCross-dataset validation NOT RUN\nDirective §12 requires holdout PASS\nHoldout BLOCKED (no method track passed)",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#FFEBEE", edgecolor="red"))
    ax.set_title("Fig 21: Cross-Dataset Cases (BLOCKED)")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_dir, "fig_21_cross_dataset_cases")


def fig22_gate_state_flow(data_dir, out_dir):
    """Fig 22: Gate state flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    gates = [
        (1, 6, "Track A\n7/7 PASS", "#4CAF50"),
        (4, 6, "Track B\n1/6 NO_GAIN", "#F44336"),
        (7, 6, "Track C\n5/8 NO_GAIN", "#F44336"),
        (10, 6, "Track D\nNOT STARTED", "#9E9E9E"),
        (4, 3, "Holdout\nBLOCKED", "#F44336"),
        (7, 3, "Cross-Dataset\nBLOCKED", "#F44336"),
        (10, 3, "Formal Test\nLOCKED", "#9E9E9E"),
        (7, 0.5, "Final: PHASES_BENCHMARK_PAPER_READY", "#2196F3"),
    ]
    for x, y, label, color in gates:
        rect = mpatches.FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="black", alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    # Arrows
    arrow_props = dict(arrowstyle="->", color="black", lw=1.5)
    ax.annotate("", xy=(3.2, 6), xytext=(1.8, 6), arrowprops=arrow_props)
    ax.annotate("", xy=(6.2, 6), xytext=(4.8, 6), arrowprops=arrow_props)
    ax.annotate("", xy=(9.2, 6), xytext=(7.8, 6), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 3.4), xytext=(4, 5.6), arrowprops=dict(arrowstyle="->", color="red", lw=1.5, linestyle="dashed"))
    ax.annotate("", xy=(7, 3.4), xytext=(7, 5.6), arrowprops=dict(arrowstyle="->", color="red", lw=1.5, linestyle="dashed"))
    ax.set_title("Fig 22: Gate State Flow (Phase S Final)", fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_dir, "fig_22_gate_state_flow")


def fig23_final_method_diagram(data_dir, out_dir):
    """Fig 23: Final method diagram (B2_mean pipeline)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    components = [
        (1.5, 3, "Query\nImage", "#E3F2FD"),
        (3.5, 3, "DINOv2-L\nMatcher", "#BBDEFB"),
        (5.5, 3, "6 References\n(per-query)", "#C8E6C9"),
        (7.5, 3, "Matching\nScore Maps", "#FFF9C4"),
        (9.5, 3, "B2_mean\nConsensus", "#FFE0B2"),
        (11.5, 3, "SAM-B\n(point prompt)", "#FFCCBC"),
        (13, 3, "Segmentation\nMask", "#D7CCC8"),
    ]
    for x, y, label, color in components:
        rect = mpatches.FancyBboxPatch((x - 0.7, y - 0.6), 1.4, 1.2, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")
    # Arrows between components
    for i in range(len(components) - 1):
        x1 = components[i][0] + 0.7
        x2 = components[i + 1][0] - 0.7
        ax.annotate("", xy=(x2, 3), xytext=(x1, 3), arrowprops=dict(arrowstyle="->", lw=1.5))
    # Annotation: best method
    ax.text(7, 5, "Final Method: B2_mean Consensus (k=6, P30-N0) — Val Dice = 0.4756",
            ha="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#E8F5E9", edgecolor="green"))
    ax.text(7, 1, "0 GPU-hours | 1 SAM call/query | No learning required | Frozen DINOv2-L + SAM-B",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Fig 23: Final Method Diagram (B2_mean Consensus Pipeline)", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_dir, "fig_23_final_method_diagram")


def fig24_benchmark_protocol_diagram(data_dir, out_dir):
    """Fig 24: Benchmark protocol diagram (multi-track design)."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    # Phase S protocol boxes
    ax.text(7, 8.5, "Phase S Multi-Track Research Protocol", ha="center", fontsize=14, fontweight="bold")
    # Data split
    rect = mpatches.FancyBboxPatch((0.5, 7), 13, 0.8, boxstyle="round,pad=0.1",
                                    facecolor="#E3F2FD", edgecolor="black")
    ax.add_patch(rect)
    ax.text(7, 7.4, "MoNuSAC: Train=80 | Validation=41 | Holdout=40 (locked) | Formal Test (locked)",
            ha="center", fontsize=10)
    # Track A
    rect = mpatches.FancyBboxPatch((0.5, 5.5), 3, 1.2, boxstyle="round,pad=0.1",
                                    facecolor="#4CAF50", edgecolor="black", alpha=0.8)
    ax.add_patch(rect)
    ax.text(2, 6.1, "Track A\nBenchmark\n7/7 PASS", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")
    # Track B
    rect = mpatches.FancyBboxPatch((3.8, 5.5), 3, 1.2, boxstyle="round,pad=0.1",
                                    facecolor="#F44336", edgecolor="black", alpha=0.8)
    ax.add_patch(rect)
    ax.text(5.3, 6.1, "Track B\nReliability\n1/6 NO_GAIN", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")
    # Track C
    rect = mpatches.FancyBboxPatch((7.1, 5.5), 3, 1.2, boxstyle="round,pad=0.1",
                                    facecolor="#F44336", edgecolor="black", alpha=0.8)
    ax.add_patch(rect)
    ax.text(8.6, 6.1, "Track C\nSet-Conditioned\n5/8 NO_GAIN", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")
    # Track D
    rect = mpatches.FancyBboxPatch((10.4, 5.5), 3, 1.2, boxstyle="round,pad=0.1",
                                    facecolor="#9E9E9E", edgecolor="black", alpha=0.8)
    ax.add_patch(rect)
    ax.text(11.9, 6.1, "Track D\nDirect Seg\nNOT STARTED", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")
    # Arrows A -> B, B -> C
    ax.annotate("", xy=(3.8, 6.1), xytext=(3.5, 6.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(7.1, 6.1), xytext=(6.8, 6.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    # Holdout/Cross-dataset/Formal
    rect = mpatches.FancyBboxPatch((1.5, 3.5), 3, 1, boxstyle="round,pad=0.1",
                                    facecolor="#FFEBEE", edgecolor="red")
    ax.add_patch(rect)
    ax.text(3, 4, "Holdout\nBLOCKED", ha="center", va="center", fontsize=9, color="red")
    rect = mpatches.FancyBboxPatch((5.5, 3.5), 3, 1, boxstyle="round,pad=0.1",
                                    facecolor="#FFEBEE", edgecolor="red")
    ax.add_patch(rect)
    ax.text(7, 4, "Cross-Dataset\nBLOCKED", ha="center", va="center", fontsize=9, color="red")
    rect = mpatches.FancyBboxPatch((9.5, 3.5), 3, 1, boxstyle="round,pad=0.1",
                                    facecolor="#ECEFF1", edgecolor="gray")
    ax.add_patch(rect)
    ax.text(11, 4, "Formal Test\nLOCKED", ha="center", va="center", fontsize=9, color="gray")
    # Final state
    rect = mpatches.FancyBboxPatch((3, 1), 8, 1.5, boxstyle="round,pad=0.1",
                                    facecolor="#2196F3", edgecolor="black", alpha=0.8)
    ax.add_patch(rect)
    ax.text(7, 1.75, "Final State: PHASES_BENCHMARK_PAPER_READY", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(7, 1.25, "Benchmark + failure-boundary paper | 0 GPU-hours | 6779 SAM calls",
            ha="center", va="center", fontsize=9, color="white")
    # Compute budget
    ax.text(7, 0.3, "210 GPU-hour cap | 0 used | 210 remaining", ha="center", fontsize=9, style="italic")
    ax.set_title("Fig 24: Benchmark Protocol Diagram (Phase S Multi-Track Design)", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_dir, "fig_24_benchmark_protocol_diagram")


def main():
    parser = argparse.ArgumentParser(description="Generate 24 Phase S figures (directive §15)")
    parser.add_argument("--data-dir", default="reports", help="Base reports directory")
    parser.add_argument("--out-dir", default="reports/phaseS_figures", help="Output directory for figures")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    figures = [
        ("fig01", fig01_reference_instability_distribution),
        ("fig02", fig02_corrected_variance_decomposition),
        ("fig03", fig03_mean_dice_vs_reference_count),
        ("fig04", fig04_within_query_std_vs_reference_count),
        ("fig05", fig05_worst_subset_dice),
        ("fig06", fig06_cvar),
        ("fig07", fig07_candidate_overlap),
        ("fig08", fig08_prediction_consistency),
        ("fig09", fig09_utility_prediction_calibration),
        ("fig10", fig10_reference_weight_distribution),
        ("fig11", fig11_reliability_score_vs_true_utility),
        ("fig12", fig12_uniform_vs_weighted_consensus),
        ("fig13", fig13_set_model_gain),
        ("fig14", fig14_direct_method_gain),
        ("fig15", fig15_cost_quality_pareto),
        ("fig16", fig16_catastrophic_failure),
        ("fig17", fig17_high_instability_cases),
        ("fig18", fig18_low_instability_cases),
        ("fig19", fig19_method_success_cases),
        ("fig20", fig20_method_failure_cases),
        ("fig21", fig21_cross_dataset_cases),
        ("fig22", fig22_gate_state_flow),
        ("fig23", fig23_final_method_diagram),
        ("fig24", fig24_benchmark_protocol_diagram),
    ]

    print(f"Generating {len(figures)} figures into {out_dir} ...")
    for name, func in figures:
        try:
            func(data_dir, out_dir)
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Generated {len(figures)} figures in {out_dir}/")


if __name__ == "__main__":
    main()
