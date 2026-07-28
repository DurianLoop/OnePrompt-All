"""Phase U Theory Verification: Generate empirical evidence figures.

Creates:
  Figure 1: Reference failure landscape (entropy vs instability/CFR scatter)
  Figure 2: Entropy vs prompt dispersion scatter
  Figure 3: Entropy threshold estimation (CFR by entropy bin)
  Figure 4: Feature correlation heatmap

Uses episode-level data from phaseU experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr, spearmanr

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "paper" / "phaseU" / "figures"

# Style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SEGMENTER_COLORS = {
    "SAM-B": "#E63946",
    "SAM-L": "#457B9D",
    "SAM-H": "#2A9D8F",
    "SAM2-Tiny": "#E9C46A",
    "SAM2-Small": "#F4A261",
    "MedSAM": "#264653",
}

MATCHER_MARKERS = {
    "DINOv2-L": "o",
    "DINOv2-B": "s",
    "DINOv2-G": "D",
    "CLIP": "^",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_episode_data() -> pd.DataFrame:
    """Load episode data from all phaseU working points."""
    wp_dirs = {
        "SAM-B": REPORTS_DIR / "phaseU_wp1" / "sam_b",
        "SAM-L": REPORTS_DIR / "phaseU_wp1" / "sam_l",
        "SAM2-Tiny": REPORTS_DIR / "phaseU_wp1" / "sam2_t",
        "SAM2-Small": REPORTS_DIR / "phaseU_wp1" / "sam2_s",
        "DINOv2-B+SAM-B": REPORTS_DIR / "phaseU_wp4",
    }

    dfs = []
    for name, wp_dir in wp_dirs.items():
        csv_path = wp_dir / "phase2_episode_instability.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["segmenter"] = name
            df["wp_dir"] = str(wp_dir)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_ramp_results() -> dict:
    """Load RAMP results."""
    path = REPORTS_DIR / "phaseU_ramp" / "ramp_results.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Figure 1: Reference failure landscape
# ---------------------------------------------------------------------------

def generate_figure1_failure_landscape(df: pd.DataFrame, out_dir: Path) -> dict:
    """Generate the reference failure landscape scatter plot.

    x-axis: matching_map_disagreement (proxy for entropy)
    y-axis left: best_worst_gap_k1 (instability)
    y-axis right: catastrophic_failure_rate_k1

    Shows the anti-correlation clearly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Filter to P20-N1 config for consistency
    plot_df = df[df["config_name"] == "P20-N1"].copy()

    # Compute per-query aggregation (mean across seeds)
    query_agg = plot_df.groupby("query_case_id").agg(
        matching_map_disagreement=("matching_map_disagreement", "mean"),
        best_worst_gap_k1=("best_worst_gap_k1", "mean"),
        catastrophic_failure_rate_k1=("catastrophic_failure_rate_k1", "mean"),
        mean_dice_k1=("mean_dice_k1", "mean"),
        std_dice_k1=("std_dice_k1", "mean"),
        segmenter=("segmenter", "first"),
    ).reset_index()

    # Plot 1: Instability vs entropy
    ax1 = axes[0]
    segs = query_agg["segmenter"].unique()
    for seg in segs:
        subset = query_agg[query_agg["segmenter"] == seg]
        ax1.scatter(
            subset["matching_map_disagreement"],
            subset["best_worst_gap_k1"],
            alpha=0.6, s=25, label=seg,
            color=SEGMENTER_COLORS.get(seg, "gray"),
        )

    # Fit and plot trend line
    mask = ~(query_agg["matching_map_disagreement"].isna() | query_agg["best_worst_gap_k1"].isna())
    x = query_agg.loc[mask, "matching_map_disagreement"].values
    y = query_agg.loc[mask, "best_worst_gap_k1"].values
    r_inst, p_inst = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax1.plot(x_line, p_line(x_line), "k--", linewidth=1.5, alpha=0.7)

    ax1.set_xlabel("Matching Map Disagreement (entropy proxy)")
    ax1.set_ylabel("Best-Worst Dice Gap (instability)")
    ax1.set_title(f"Instability vs Entropy\nPearson r = {r_inst:.3f}, p = {p_inst:.4f}")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: CFR vs entropy
    ax2 = axes[1]
    for seg in segs:
        subset = query_agg[query_agg["segmenter"] == seg]
        ax2.scatter(
            subset["matching_map_disagreement"],
            subset["catastrophic_failure_rate_k1"],
            alpha=0.6, s=25, label=seg,
            color=SEGMENTER_COLORS.get(seg, "gray"),
        )

    mask = ~(query_agg["matching_map_disagreement"].isna() | query_agg["catastrophic_failure_rate_k1"].isna())
    x = query_agg.loc[mask, "matching_map_disagreement"].values
    y = query_agg.loc[mask, "catastrophic_failure_rate_k1"].values
    r_cfr, p_cfr = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax2.plot(x_line, p_line(x_line), "k--", linewidth=1.5, alpha=0.7)

    ax2.set_xlabel("Matching Map Disagreement (entropy proxy)")
    ax2.set_ylabel("Catastrophic Failure Rate")
    ax2.set_title(f"CFR vs Entropy\nPearson r = {r_cfr:.3f}, p = {p_cfr:.4f}")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "fig_theory_failure_landscape.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Theory] Figure 1 saved to {out_path}")

    return {
        "instability_entropy_pearson_r": float(r_inst),
        "instability_entropy_pearson_p": float(p_inst),
        "cfr_entropy_pearson_r": float(r_cfr),
        "cfr_entropy_pearson_p": float(p_cfr),
    }


# ---------------------------------------------------------------------------
# Figure 2: Entropy threshold estimation
# ---------------------------------------------------------------------------

def generate_figure2_entropy_bins(df: pd.DataFrame, out_dir: Path) -> dict:
    """Generate entropy binned analysis.

    Bin queries by matching_map_disagreement into 5 equal-width bins.
    Compute mean instability and CFR per bin.
    Estimate θ_L and θ_H thresholds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_df = df[df["config_name"] == "P20-N1"].copy()
    query_agg = plot_df.groupby("query_case_id").agg(
        matching_map_disagreement=("matching_map_disagreement", "mean"),
        best_worst_gap_k1=("best_worst_gap_k1", "mean"),
        catastrophic_failure_rate_k1=("catastrophic_failure_rate_k1", "mean"),
        mean_dice_k1=("mean_dice_k1", "mean"),
    ).reset_index()

    # Bin by entropy
    n_bins = 5
    query_agg["entropy_bin"] = pd.qcut(
        query_agg["matching_map_disagreement"], n_bins, labels=False,
        duplicates="drop",
    )

    bin_stats = query_agg.groupby("entropy_bin").agg(
        entropy_mean=("matching_map_disagreement", "mean"),
        entropy_std=("matching_map_disagreement", "std"),
        instability_mean=("best_worst_gap_k1", "mean"),
        instability_std=("best_worst_gap_k1", "std"),
        cfr_mean=("catastrophic_failure_rate_k1", "mean"),
        cfr_std=("catastrophic_failure_rate_k1", "std"),
        dice_mean=("mean_dice_k1", "mean"),
        n_queries=("query_case_id", "count"),
    ).reset_index()

    # Plot 1: Instability by entropy bin
    ax1 = axes[0]
    x = np.arange(len(bin_stats))
    width = 0.35
    bars1 = ax1.bar(x, bin_stats["instability_mean"].values, width,
                    yerr=bin_stats["instability_std"].values,
                    color="#E63946", alpha=0.7, capsize=3,
                    label="Instability")
    ax1.set_xlabel("Entropy Bin (low → high)")
    ax1.set_ylabel("Mean Best-Worst Dice Gap")
    ax1.set_title("Instability by Entropy Bin")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Bin {i}" for i in range(len(bin_stats))])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Add entropy range labels
    for i, row in bin_stats.iterrows():
        ax1.annotate(f"E={row['entropy_mean']:.3f}", (x[i], row["instability_mean"]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7, rotation=0)

    # Plot 2: CFR by entropy bin
    ax2 = axes[1]
    bars2 = ax2.bar(x, bin_stats["cfr_mean"].values, width,
                    yerr=bin_stats["cfr_std"].values,
                    color="#457B9D", alpha=0.7, capsize=3,
                    label="CFR")
    ax2.set_xlabel("Entropy Bin (low → high)")
    ax2.set_ylabel("Mean Catastrophic Failure Rate")
    ax2.set_title("Catastrophic Failure Rate by Entropy Bin")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Bin {i}" for i in range(len(bin_stats))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    for i, row in bin_stats.iterrows():
        ax2.annotate(f"E={row['entropy_mean']:.3f}", (x[i], row["cfr_mean"]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7, rotation=0)

    plt.tight_layout()
    out_path = out_dir / "fig_theory_entropy_bins.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Theory] Figure 2 saved to {out_path}")

    # Estimate thresholds
    # θ_L: entropy below which instability is high and CFR is low
    # θ_H: entropy above which CFR is high and instability is low
    bin_stats_sorted = bin_stats.sort_values("entropy_mean")
    if len(bin_stats_sorted) >= 3:
        theta_L = float(bin_stats_sorted.iloc[1]["entropy_mean"])
        theta_H = float(bin_stats_sorted.iloc[-2]["entropy_mean"])
    else:
        theta_L = 0.0
        theta_H = 1.0

    return {
        "n_bins": n_bins,
        "theta_L": theta_L,
        "theta_H": theta_H,
        "bin_stats": bin_stats.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Figure 3: Entropy vs dispersion (Lemma 1 verification)
# ---------------------------------------------------------------------------

def generate_figure3_dispersion(df: pd.DataFrame, out_dir: Path) -> dict:
    """Generate entropy vs prompt dispersion scatter.

    Verifies Lemma 1: Entropy determines dispersion.
    prompt_overlap_mean_k1 is used as proxy: low overlap = high dispersion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_df = df[df["config_name"] == "P20-N1"].copy()
    query_agg = plot_df.groupby("query_case_id").agg(
        matching_map_disagreement=("matching_map_disagreement", "mean"),
        prompt_overlap_mean_k1=("prompt_overlap_mean_k1", "mean"),
        candidate_jaccard_mean_k1=("candidate_jaccard_mean_k1", "mean"),
        prediction_jaccard_k1_within=("prediction_jaccard_k1_within", "mean"),
        best_worst_gap_k1=("best_worst_gap_k1", "mean"),
        mean_dice_k1=("mean_dice_k1", "mean"),
        segmenter=("segmenter", "first"),
    ).reset_index()

    # Plot 1: Entropy vs prompt overlap (inverse of dispersion)
    ax1 = axes[0]
    ax1.scatter(
        query_agg["matching_map_disagreement"],
        query_agg["prompt_overlap_mean_k1"],
        alpha=0.5, s=20, c="#E63946",
    )
    mask = ~(query_agg["matching_map_disagreement"].isna() | query_agg["prompt_overlap_mean_k1"].isna())
    x = query_agg.loc[mask, "matching_map_disagreement"].values
    y = query_agg.loc[mask, "prompt_overlap_mean_k1"].values
    r, p = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax1.plot(x_line, p_line(x_line), "k--", linewidth=1.5, alpha=0.7)

    ax1.set_xlabel("Matching Map Disagreement")
    ax1.set_ylabel("Prompt Overlap (lower = more dispersion)")
    ax1.set_title(f"Entropy vs Prompt Dispersion\nPearson r = {r:.3f}, p = {p:.3f}")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Entropy vs prediction Jaccard (mask consistency)
    ax2 = axes[1]
    ax2.scatter(
        query_agg["matching_map_disagreement"],
        query_agg["prediction_jaccard_k1_within"],
        alpha=0.5, s=20, c="#457B9D",
    )
    mask = ~(query_agg["matching_map_disagreement"].isna() | query_agg["prediction_jaccard_k1_within"].isna())
    x = query_agg.loc[mask, "matching_map_disagreement"].values
    y = query_agg.loc[mask, "prediction_jaccard_k1_within"].values
    r, p = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax2.plot(x_line, p_line(x_line), "k--", linewidth=1.5, alpha=0.7)

    ax2.set_xlabel("Matching Map Disagreement")
    ax2.set_ylabel("Prediction Jaccard (within-ref)")
    ax2.set_title(f"Entropy vs Mask Consistency\nPearson r = {r:.3f}, p = {p:.3f}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "fig_theory_dispersion.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Theory] Figure 3 saved to {out_path}")

    return {
        "entropy_overlap_pearson_r": float(r),
        "entropy_overlap_pearson_p": float(p),
    }


# ---------------------------------------------------------------------------
# Figure 4: Anti-correlation matrix
# ---------------------------------------------------------------------------

def generate_figure4_anticorrelation_matrix(df: pd.DataFrame, out_dir: Path) -> dict:
    """Generate anti-correlation matrix across segmenters and configs.

    For each (segmenter, config) pair, compute the Pearson r between
    instability and CFR.
    """
    plot_df = df[df["config_name"].isin(["P20-N1", "P30-N0", "P50-N0"])].copy()

    results = []
    for (seg, cfg), group in plot_df.groupby(["segmenter", "config_name"]):
        query_agg = group.groupby("query_case_id").agg(
            instability=("best_worst_gap_k1", "mean"),
            cfr=("catastrophic_failure_rate_k1", "mean"),
        ).reset_index()

        mask = ~(query_agg["instability"].isna() | query_agg["cfr"].isna())
        if mask.sum() >= 5:
            r, p = pearsonr(
                query_agg.loc[mask, "instability"].values,
                query_agg.loc[mask, "cfr"].values,
            )
            results.append({
                "segmenter": seg,
                "config": cfg,
                "pearson_r": float(r),
                "pearson_p": float(p),
                "n_queries": int(mask.sum()),
            })

    # Create heatmap
    result_df = pd.DataFrame(results)
    pivoted = result_df.pivot_table(
        index="segmenter", columns="config", values="pearson_r", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivoted.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(pivoted.columns)))
    ax.set_yticks(np.arange(len(pivoted.index)))
    ax.set_xticklabels(pivoted.columns)
    ax.set_yticklabels(pivoted.index)

    # Add text annotations
    for i in range(len(pivoted.index)):
        for j in range(len(pivoted.columns)):
            val = pivoted.values[i, j]
            text = ax.text(j, i, f"{val:.2f}",
                           ha="center", va="center",
                           color="white" if abs(val) > 0.5 else "black",
                           fontsize=10, fontweight="bold")

    ax.set_title("Anti-Correlation (Instability vs CFR)\nPearson r by Segmenter x Config")
    plt.colorbar(im, ax=ax, label="Pearson r")

    plt.tight_layout()
    out_path = out_dir / "fig_theory_anticorrelation_matrix.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Theory] Figure 4 saved to {out_path}")

    return {"correlations": results}


# ---------------------------------------------------------------------------
# Figure 5: Feature correlation heatmap
# ---------------------------------------------------------------------------

def generate_figure5_feature_correlation(df: pd.DataFrame, out_dir: Path) -> dict:
    """Generate feature correlation heatmap for all pipeline features."""
    plot_df = df[df["config_name"] == "P20-N1"].copy()

    feature_cols = [
        "matching_map_disagreement",
        "candidate_jaccard_mean_k1",
        "prompt_overlap_mean_k1",
        "prediction_jaccard_k1_within",
        "mean_dice_k1",
        "std_dice_k1",
        "best_worst_gap_k1",
        "catastrophic_failure_rate_k1",
    ]
    feature_names = [
        "Match Disagreement",
        "Candidate Jaccard",
        "Prompt Overlap",
        "Prediction Jaccard",
        "Mean Dice",
        "Std Dice",
        "Instability",
        "CFR",
    ]

    corr_matrix = plot_df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)

    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            val = corr_matrix.values[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black",
                    fontsize=8)

    ax.set_title("Feature Correlation Matrix")
    plt.colorbar(im, ax=ax, label="Pearson r")

    plt.tight_layout()
    out_path = out_dir / "fig_theory_feature_correlation.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Theory] Figure 5 saved to {out_path}")

    return {"correlation_matrix": corr_matrix.to_dict()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase U Theory Verification: Generate empirical evidence figures",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=FIGURES_DIR,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[Theory] Loading episode data...")
    df = load_all_episode_data()
    print(f"[Theory] Loaded {len(df)} rows, {df['segmenter'].nunique()} segmenters, "
          f"{df['query_case_id'].nunique()} queries")

    results = {}

    print("\n[Theory] Generating Figure 1: Failure landscape...")
    results["figure1"] = generate_figure1_failure_landscape(df, args.out_dir)

    print("\n[Theory] Generating Figure 2: Entropy bins...")
    results["figure2"] = generate_figure2_entropy_bins(df, args.out_dir)

    print("\n[Theory] Generating Figure 3: Entropy vs dispersion...")
    results["figure3"] = generate_figure3_dispersion(df, args.out_dir)

    print("\n[Theory] Generating Figure 4: Anti-correlation matrix...")
    results["figure4"] = generate_figure4_anticorrelation_matrix(df, args.out_dir)

    print("\n[Theory] Generating Figure 5: Feature correlation...")
    results["figure5"] = generate_figure5_feature_correlation(df, args.out_dir)

    # Save summary
    summary_path = args.out_dir / "theory_verification_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Theory] All results saved to {summary_path}")


if __name__ == "__main__":
    main()