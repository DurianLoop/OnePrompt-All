"""Generate the 18 Phase T core paper figures (Section 15).

Reads the existing Phase T data artifacts (WP1 holdout, WP2 cross-dataset,
WP3 M1 / M2) and writes 18 PNG figure files into
``reports/phaseT_figures/``::

    fig_01_benchmark_overview.png
    fig_02_reference_resampling_protocol.png
    fig_03_variance_decomposition.png
    fig_04_best_worst_gap_distribution.png
    fig_05_reference_count_scaling.png
    fig_06_dice_vs_vref.png
    fig_07_cvar_cfr.png
    fig_08_prediction_consistency.png
    fig_09_method_instability_comparison.png
    fig_10_risk_coverage_curve.png
    fig_11_dynamic_budget_policy.png
    fig_12_cost_quality_pareto.png
    fig_13_cross_dataset_summary.png
    fig_14_high_instability_examples.png
    fig_15_mean_consensus_success_cases.png
    fig_16_mean_consensus_failure_cases.png
    fig_17_selective_prediction_failures.png
    fig_18_final_method_diagram.png

All figures: 8x6 inches, DPI=150, tight_layout, sans-serif font,
colorblind-friendly (Okabe-Ito) palette.

Run from the repo root as::

    python src/adapromptset/scripts/generate_phaseT_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]  # d:\SAM\adapromptset
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "phaseT_figures"

WP1_DIR = REPORTS_DIR / "phaseT_wp1_holdout"
WP2_MONUSEG_DIR = REPORTS_DIR / "phaseT_wp2_monuseg"
WP2_IDRID_DIR = REPORTS_DIR / "phaseT_wp2_idrid"
WP3_M1_DIR = REPORTS_DIR / "phaseT_wp3_m1"
WP3_M2_DIR = REPORTS_DIR / "phaseT_wp3_m2"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-friendly palette
OI_ORANGE = "#E69F00"
OI_SKYBLUE = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_BLACK = "#000000"

CONFIG_ORDER = ["P20-N1", "P30-N0", "P50-N0"]
CONFIG_LABEL = {"P20-N1": "P20", "P30-N0": "P30", "P50-N0": "P50"}
CONFIG_COLOR = {"P20-N1": OI_VERMILLION, "P30-N0": OI_BLUE, "P50-N0": OI_GREEN}

DATASET_ORDER = ["MoNuSAC", "MoNuSeg", "IDRID"]
DATASET_DIR = {
    "MoNuSAC": WP1_DIR,
    "MoNuSeg": WP2_MONUSEG_DIR,
    "IDRID": WP2_IDRID_DIR,
}
DATASET_COLOR = {"MoNuSAC": OI_VERMILLION, "MoNuSeg": OI_BLUE, "IDRID": OI_GREEN}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _save(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def _to_float(val: str, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val: str, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _filter(rows: Sequence[Dict[str, str]], **cond: str) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in cond.items()):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Figure 1: benchmark overview (V_ref_frac across datasets, k=1)
# ---------------------------------------------------------------------------


def fig_01_benchmark_overview() -> None:
    """Bar chart: V_ref_frac across MoNuSAC/MoNuSeg/IDRID, 3 configs at k=1."""
    fig, ax = plt.subplots()
    x = np.arange(len(DATASET_ORDER))
    width = 0.25
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = []
        for ds in DATASET_ORDER:
            rows = _read_csv(DATASET_DIR[ds] / "within_query_reference_variance.csv")
            row = _filter(rows, config_name=cfg, k="1")[0]
            vals.append(_to_float(row["V_ref_frac"]))
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=CONFIG_LABEL[cfg],
            color=CONFIG_COLOR[cfg],
            edgecolor="white",
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("V_ref_frac (reference-induced variance fraction)")
    ax.set_title("Fig 1: Benchmark Overview — V_ref_frac by Dataset (k=1)")
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.1))
    ax.legend(title="Config")
    _save(fig, "fig_01_benchmark_overview")


# ---------------------------------------------------------------------------
# Figure 2: reference resampling protocol (schematic)
# ---------------------------------------------------------------------------


def _draw_box(ax, xy, w, h, text, facecolor="#EEEEEE", edgecolor="#333333", fontsize=9):
    box = mpatches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def _draw_arrow(ax, start, end, text=""):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
    )
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.05, text, ha="center", va="bottom", fontsize=8, color="#555555")


def fig_02_reference_resampling_protocol() -> None:
    """Schematic: query -> reference pool -> resampled subsets -> SAM -> consensus."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Fig 2: Reference Resampling Protocol", pad=12)

    # Stage 1: query image
    _draw_box(ax, (0.2, 2.6), 1.6, 0.9, "Query\nImage", facecolor=OI_SKYBLUE)

    # Stage 2: reference pool
    _draw_box(ax, (2.4, 2.6), 1.8, 0.9, "Reference Pool\n(6 refs)", facecolor=OI_ORANGE)
    _draw_arrow(ax, (1.8, 3.05), (2.4, 3.05))

    # Stage 3: resampled subsets (3 rows)
    _draw_box(ax, (4.8, 4.6), 2.4, 0.8, "k=1: 6 individual\n(single-ref subsets)", facecolor="#FFFFFF")
    _draw_box(ax, (4.8, 2.7), 2.4, 0.8, "k=3: first3\n(consensus of 3)", facecolor="#FFFFFF")
    _draw_box(ax, (4.8, 0.8), 2.4, 0.8, "k=6: all6\n(consensus of 6)", facecolor="#FFFFFF")
    for y in (5.0, 3.1, 1.2):
        _draw_arrow(ax, (4.2, 3.05), (4.8, y))

    # Stage 4: SAM predictions
    _draw_box(ax, (7.8, 2.6), 1.8, 0.9, "SAM\nPredictions", facecolor=OI_GREEN)
    for y in (5.0, 3.1, 1.2):
        _draw_arrow(ax, (7.2, y), (7.8, 3.05))

    # Stage 5: consensus
    _draw_box(ax, (7.8, 0.5), 1.8, 0.8, "Consensus\nDecision", facecolor=OI_PURPLE)
    _draw_arrow(ax, (8.7, 2.6), (8.7, 1.3))

    ax.text(
        5.0,
        5.7,
        "Each query is evaluated under reference subsets of increasing size k.\n"
        "Variance across subsets quantifies reference-induced instability.",
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#555555",
    )
    _save(fig, "fig_02_reference_resampling_protocol")


# ---------------------------------------------------------------------------
# Figure 3: variance decomposition (stacked bar, MoNuSAC holdout k=1)
# ---------------------------------------------------------------------------


def fig_03_variance_decomposition() -> None:
    """Stacked bar of V_ref vs V_query for MoNuSAC holdout, 3 configs at k=1."""
    rows = _read_csv(WP1_DIR / "within_query_reference_variance.csv")
    cfgs = [c for c in CONFIG_ORDER]
    v_ref = [_to_float(_filter(rows, config_name=c, k="1")[0]["V_ref"]) for c in cfgs]
    v_query = [_to_float(_filter(rows, config_name=c, k="1")[0]["V_query"]) for c in cfgs]
    frac = [r / (r + q) for r, q in zip(v_ref, v_query)]

    fig, ax = plt.subplots()
    x = np.arange(len(cfgs))
    ax.bar(x, v_query, label="V_query (case difficulty)", color=OI_BLUE, edgecolor="white")
    ax.bar(x, v_ref, bottom=v_query, label="V_ref (reference-induced)", color=OI_VERMILLION, edgecolor="white")
    for i, (q, r, f) in enumerate(zip(v_query, v_ref, frac)):
        ax.text(i, q + r + 0.0015, f"V_ref_frac={f:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABEL[c] for c in cfgs])
    ax.set_ylabel("Variance")
    ax.set_title("Fig 3: Variance Decomposition — MoNuSAC Holdout (k=1)")
    ax.legend()
    _save(fig, "fig_03_variance_decomposition")


# ---------------------------------------------------------------------------
# Figure 4: best-worst gap distribution (histogram, MoNuSAC holdout k=1)
# ---------------------------------------------------------------------------


def fig_04_best_worst_gap_distribution() -> None:
    """Histogram of per-query best-worst gap (k=1) for MoNuSAC holdout."""
    rows = _read_csv(WP1_DIR / "phase2_episode_instability.csv")
    # Per-query (averaged over seeds) gap, per config
    gap_by = defaultdict(list)
    for r in rows:
        gap_by[(r["config_name"], r["query_case_id"])].append(_to_float(r["best_worst_gap_k1"]))
    fig, ax = plt.subplots()
    for cfg in CONFIG_ORDER:
        vals = [np.mean(v) for (c, _), v in gap_by.items() if c == cfg]
        if vals:
            ax.hist(vals, bins=15, alpha=0.6, label=CONFIG_LABEL[cfg], color=CONFIG_COLOR[cfg])
    ax.axvline(0.05, color=OI_BLACK, linestyle="--", linewidth=2, label="gap=0.05 threshold")
    ax.set_xlabel("Per-Query Best-Worst Dice Gap (k=1)")
    ax.set_ylabel("Query Count")
    ax.set_title("Fig 4: Best-Worst Gap Distribution — MoNuSAC Holdout (k=1)")
    ax.legend()
    _save(fig, "fig_04_best_worst_gap_distribution")


# ---------------------------------------------------------------------------
# Figure 5: reference count scaling (mean Dice vs k, error bars = std)
# ---------------------------------------------------------------------------


def _per_query_k_dice(rows: Sequence[Dict[str, str]], config: str, k: int) -> Tuple[float, float, List[float]]:
    """Return (mean, std, per_query_values) of Dice for (config, k) on holdout."""
    sub = [r for r in rows if r["config_name"] == config and _to_int(r["k"]) == k]
    by_query: Dict[str, List[float]] = defaultdict(list)
    for r in sub:
        by_query[r["query_case_id"]].append(_to_float(r["subset_dice"]))
    per_query = [np.mean(v) for v in by_query.values() if v]
    if not per_query:
        return (float("nan"), 0.0, [])
    return (float(np.mean(per_query)), float(np.std(per_query, ddof=1)), per_query)


def fig_05_reference_count_scaling() -> None:
    """Line plot: mean Dice vs k (1,3,6) for 3 configs, error bars = std."""
    rows = _read_csv(WP1_DIR / "per_query_per_subset_metrics.csv")
    ks = [1, 3, 6]
    fig, ax = plt.subplots()
    for cfg in CONFIG_ORDER:
        means, stds = [], []
        for k in ks:
            m, s, _ = _per_query_k_dice(rows, cfg, k)
            means.append(m)
            stds.append(s)
        ax.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            capsize=5,
            label=CONFIG_LABEL[cfg],
            color=CONFIG_COLOR[cfg],
        )
    ax.set_xticks(ks)
    ax.set_xlabel("Reference Subset Size k")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Fig 5: Reference Count Scaling — MoNuSAC Holdout")
    ax.legend(title="Config")
    _save(fig, "fig_05_reference_count_scaling")


# ---------------------------------------------------------------------------
# Figure 6: dice vs V_ref scatter (per-query, MoNuSAC holdout)
# ---------------------------------------------------------------------------


def fig_06_dice_vs_vref() -> None:
    """Scatter: per-query V_ref_frac (x) vs mean Dice (y), colored by config."""
    rows = _read_csv(WP1_DIR / "per_query_per_subset_metrics.csv")
    k1 = [r for r in rows if _to_int(r["k"]) == 1]
    fig, ax = plt.subplots()
    for cfg in CONFIG_ORDER:
        # per (query, seed): 6 subset dices
        per_query_dice: Dict[str, List[float]] = defaultdict(list)
        per_query_vref: Dict[str, List[float]] = defaultdict(list)
        for r in k1:
            if r["config_name"] != cfg:
                continue
            per_query_dice[r["query_case_id"]].append(_to_float(r["subset_dice"]))
        # per-seed within-query variance
        grouped = defaultdict(list)
        for r in k1:
            if r["config_name"] == cfg:
                grouped[(r["query_case_id"], r["seed"])].append(_to_float(r["subset_dice"]))
        seed_vref: Dict[str, List[float]] = defaultdict(list)
        for (q, _s), dices in grouped.items():
            if len(dices) > 1:
                seed_vref[q].append(float(np.var(dices, ddof=1)))
        # cross-query variance of per-query mean dice
        qmeans = {q: np.mean(v) for q, v in per_query_dice.items()}
        v_query_global = float(np.var(list(qmeans.values()), ddof=1)) if len(qmeans) > 1 else 1.0
        xs, ys = [], []
        for q in qmeans:
            v_ref_q = float(np.mean(seed_vref.get(q, [0.0])))
            frac = v_ref_q / (v_ref_q + v_query_global) if (v_ref_q + v_query_global) > 0 else 0.0
            xs.append(frac)
            ys.append(qmeans[q])
        ax.scatter(xs, ys, label=CONFIG_LABEL[cfg], color=CONFIG_COLOR[cfg], alpha=0.75, edgecolor="white", s=45)
    ax.set_xlabel("Per-Query V_ref_frac")
    ax.set_ylabel("Per-Query Mean Dice (k=1)")
    ax.set_title("Fig 6: Per-Query Mean Dice vs V_ref_frac — MoNuSAC Holdout")
    ax.legend(title="Config")
    _save(fig, "fig_06_dice_vs_vref")


# ---------------------------------------------------------------------------
# Figure 7: CVaR / CFR two-panel (MoNuSAC holdout)
# ---------------------------------------------------------------------------


def fig_07_cvar_cfr() -> None:
    """Two-panel: CVaR_0.1 by k (left), CFR by k (right), 3 configs."""
    rows = _read_csv(WP1_DIR / "cvar_metrics.csv")
    ks = [1, 3, 6]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for cfg in CONFIG_ORDER:
        cvar = [_to_float(_filter(rows, config_name=cfg, k=str(k))[0]["cvar_0_1"]) for k in ks]
        cfr = [_to_float(_filter(rows, config_name=cfg, k=str(k))[0]["catastrophic_rate"]) for k in ks]
        ax1.plot(ks, cvar, "o-", linewidth=2, markersize=8, label=CONFIG_LABEL[cfg], color=CONFIG_COLOR[cfg])
        ax2.plot(ks, cfr, "s-", linewidth=2, markersize=8, label=CONFIG_LABEL[cfg], color=CONFIG_COLOR[cfg])
    for ax, ylabel, title in [
        (ax1, "CVaR_0.1 (lower-bound Dice, 10th pct)", "Fig 7a: CVaR_0.1 vs k"),
        (ax2, "CFR (catastrophic failure rate)", "Fig 7b: CFR vs k"),
    ]:
        ax.set_xticks(ks)
        ax.set_xlabel("Reference Subset Size k")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Config")
    fig.suptitle("Fig 7: Tail-Risk Metrics — MoNuSAC Holdout", y=1.02)
    _save(fig, "fig_07_cvar_cfr")


# ---------------------------------------------------------------------------
# Figure 8: prediction consistency (J_pred at k=1 vs k=6)
# ---------------------------------------------------------------------------


def fig_08_prediction_consistency() -> None:
    """Bar chart: J_pred at k=1 vs k=6 for 3 configs (MoNuSAC holdout)."""
    rows = _read_csv(WP1_DIR / "prediction_consistency.csv")
    # k=1 -> cross_seed_b1_single ; k=6 -> cross_seed_consensus_k6
    k1 = {r["config_name"]: _to_float(r["mean_j_pred"]) for r in rows if r["j_pred_type"] == "cross_seed_b1_single"}
    k6 = {r["config_name"]: _to_float(r["mean_j_pred"]) for r in rows if r["j_pred_type"] == "cross_seed_consensus_k6"}
    x = np.arange(len(CONFIG_ORDER))
    width = 0.35
    fig, ax = plt.subplots()
    b1 = ax.bar(x - width / 2, [k1[c] for c in CONFIG_ORDER], width, label="k=1 (single)", color=OI_SKYBLUE, edgecolor="white")
    b6 = ax.bar(x + width / 2, [k6[c] for c in CONFIG_ORDER], width, label="k=6 (consensus)", color=OI_PURPLE, edgecolor="white")
    for bars, vals in [(b1, [k1[c] for c in CONFIG_ORDER]), (b6, [k6[c] for c in CONFIG_ORDER])]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABEL[c] for c in CONFIG_ORDER])
    ax.set_ylabel("Mean J_pred (prediction consistency)")
    ax.set_title("Fig 8: Prediction Consistency — MoNuSAC Holdout (k=1 vs k=6)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    _save(fig, "fig_08_prediction_consistency")


# ---------------------------------------------------------------------------
# Figure 9: method instability comparison (grouped bar, P20 k=1 across datasets)
# ---------------------------------------------------------------------------


def fig_09_method_instability_comparison() -> None:
    """Grouped bar: best-worst gap for MoNuSAC vs MoNuSeg vs IDRID (k=1, P20)."""
    ds_vals = []
    for ds in DATASET_ORDER:
        rows = _read_csv(DATASET_DIR[ds] / "within_query_reference_variance.csv")
        row = _filter(rows, config_name="P20-N1", k="1")[0]
        ds_vals.append(_to_float(row["mean_best_worst_gap"]))
    fig, ax = plt.subplots()
    bars = ax.bar(DATASET_ORDER, ds_vals, color=[DATASET_COLOR[d] for d in DATASET_ORDER], edgecolor="white", width=0.55)
    for b, v in zip(bars, ds_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Best-Worst Dice Gap")
    ax.set_title("Fig 9: Method Instability Comparison — P20 Config, k=1")
    ax.set_ylim(0, max(ds_vals) * 1.18)
    _save(fig, "fig_09_method_instability_comparison")


# ---------------------------------------------------------------------------
# Figure 10: risk-coverage curve (M2)
# ---------------------------------------------------------------------------


def fig_10_risk_coverage_curve() -> None:
    """Risk-coverage for M2: coverage vs selective_dice (primary) and selective_cfr (secondary)."""
    rows = _read_csv(WP3_M2_DIR / "m2_risk_coverage.csv")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    for level, marker, color in [("episode", "o", OI_BLUE), ("query", "s", OI_VERMILLION)]:
        sub = [r for r in rows if r["level"] == level]
        cov = [_to_float(r["coverage"]) for r in sub]
        dice = [_to_float(r["selective_dice"]) for r in sub]
        cfr = [_to_float(r["selective_cfr"]) for r in sub]
        ax1.plot(cov, dice, marker + "-", linewidth=2, markersize=6, label=f"{level} selective_dice", color=color)
        ax2.plot(cov, cfr, marker + "--", linewidth=1.5, markersize=6, label=f"{level} selective_cfr", color=color, alpha=0.7)
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Selective Dice (primary)", color=OI_BLACK)
    ax2.set_ylabel("Selective CFR (secondary)", color=OI_BLACK)
    ax1.set_title("Fig 10: Risk-Coverage Curve — M2 Catastrophic Model")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    _save(fig, "fig_10_risk_coverage_curve")


# ---------------------------------------------------------------------------
# Figure 11: dynamic budget policy (M1-0..M1-5)
# ---------------------------------------------------------------------------


def fig_11_dynamic_budget_policy() -> None:
    """Bar chart: avg_k and mean_dice for M1-0..M1-5."""
    rows = _read_csv(WP3_M1_DIR / "m1_model_comparison.csv")
    names = [r["model_name"].split("_")[0] for r in rows]
    avg_k = [_to_float(r["avg_k"]) for r in rows]
    dice = [_to_float(r["mean_dice"]) for r in rows]
    x = np.arange(len(names))
    width = 0.35
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    b1 = ax1.bar(x - width / 2, avg_k, width, label="avg_k", color=OI_BLUE, edgecolor="white")
    b2 = ax1.bar(x + width / 2, dice, width, label="mean_dice", color=OI_VERMILLION, edgecolor="white")
    for b, v in zip(b1, avg_k):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(b2, dice):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=0)
    ax1.set_ylabel("avg_k  /  mean_dice")
    ax1.set_title("Fig 11: Dynamic Budget Policy — M1-0 .. M1-5")
    ax1.set_ylim(0, max(max(avg_k), max(dice)) * 1.25)
    ax1.legend(loc="upper left")
    _save(fig, "fig_11_dynamic_budget_policy")


# ---------------------------------------------------------------------------
# Figure 12: cost-quality Pareto (M1)
# ---------------------------------------------------------------------------


def _pareto_front(points: Sequence[Tuple[float, float]]) -> List[int]:
    """Return indices of Pareto-optimal points (minimize cost, maximize dice)."""
    front = []
    for i, (c, d) in enumerate(points):
        dominated = False
        for j, (c2, d2) in enumerate(points):
            if i == j:
                continue
            if c2 <= c and d2 >= d and (c2 < c or d2 > d):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def fig_12_cost_quality_pareto() -> None:
    """Scatter: cost (x) vs mean_dice (y) for M1-0..M1-5, Pareto highlighted."""
    rows = _read_csv(WP3_M1_DIR / "m1_model_comparison.csv")
    names = [r["model_name"].split("_")[0] for r in rows]
    cost = [_to_float(r["cost"]) for r in rows]
    dice = [_to_float(r["mean_dice"]) for r in rows]
    pts = list(zip(cost, dice))
    front = set(_pareto_front(pts))

    fig, ax = plt.subplots()
    for i, (c, d) in enumerate(pts):
        is_front = i in front
        ax.scatter(
            c,
            d,
            s=160 if is_front else 80,
            color=OI_GREEN if is_front else OI_SKYBLUE,
            edgecolor=OI_BLACK if is_front else "white",
            linewidth=1.5 if is_front else 1.0,
            zorder=3 if is_front else 2,
            marker=("*" if is_front else "o"),
        )
        ax.annotate(names[i], (c, d), textcoords="offset points", xytext=(8, 6), fontsize=9, fontweight="bold" if is_front else "normal")
    # baselines annotation: k=1/3/6
    baseline_k = {"M1-0": "k=1", "M1-1": "k=3", "M1-2": "k=6"}
    for i, nm in enumerate(names):
        if nm in baseline_k:
            ax.annotate(f"baseline {baseline_k[nm]}", (cost[i], dice[i]), textcoords="offset points", xytext=(8, -12), fontsize=7, color="#666666", style="italic")
    ax.set_xlabel("Cost (normalized SAM calls)")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Fig 12: Cost-Quality Pareto — M1 Models (★ = Pareto frontier)")
    _save(fig, "fig_12_cost_quality_pareto")


# ---------------------------------------------------------------------------
# Figure 13: cross-dataset summary (3-panel)
# ---------------------------------------------------------------------------


def fig_13_cross_dataset_summary() -> None:
    """3-panel: V_ref_frac, consensus gain, best-worst gap across datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    x = np.arange(len(DATASET_ORDER))
    width = 0.25

    # Panel 1: V_ref_frac (k=1)
    ax = axes[0]
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = []
        for ds in DATASET_ORDER:
            rows = _read_csv(DATASET_DIR[ds] / "within_query_reference_variance.csv")
            vals.append(_to_float(_filter(rows, config_name=cfg, k="1")[0]["V_ref_frac"]))
        ax.bar(x + (i - 1) * width, vals, width, label=CONFIG_LABEL[cfg], color=CONFIG_COLOR[cfg], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("V_ref_frac (k=1)")
    ax.set_title("(a) V_ref_frac")
    ax.legend(title="Config", fontsize=8)

    # Panel 2: consensus gain = mean_dice(k=6) - mean_dice(k=1)
    ax = axes[1]
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = []
        for ds in DATASET_ORDER:
            rows = _read_csv(DATASET_DIR[ds] / "reference_count_scaling.csv")
            d6 = _to_float(_filter(rows, config_name=cfg, k="6")[0]["mean_dice"])
            d1 = _to_float(_filter(rows, config_name=cfg, k="1")[0]["mean_dice"])
            vals.append(d6 - d1)
        ax.bar(x + (i - 1) * width, vals, width, label=CONFIG_LABEL[cfg], color=CONFIG_COLOR[cfg], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Consensus Gain (Dice@k6 − Dice@k1)")
    ax.set_title("(b) Consensus Gain")
    ax.axhline(0, color=OI_BLACK, linewidth=0.8)
    ax.legend(title="Config", fontsize=8)

    # Panel 3: best-worst gap (k=1)
    ax = axes[2]
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = []
        for ds in DATASET_ORDER:
            rows = _read_csv(DATASET_DIR[ds] / "within_query_reference_variance.csv")
            vals.append(_to_float(_filter(rows, config_name=cfg, k="1")[0]["mean_best_worst_gap"]))
        ax.bar(x + (i - 1) * width, vals, width, label=CONFIG_LABEL[cfg], color=CONFIG_COLOR[cfg], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Mean Best-Worst Gap (k=1)")
    ax.set_title("(c) Best-Worst Gap")
    ax.legend(title="Config", fontsize=8)

    fig.suptitle("Fig 13: Cross-Dataset Summary — MoNuSAC vs MoNuSeg vs IDRID", y=1.02)
    _save(fig, "fig_13_cross_dataset_summary")


# ---------------------------------------------------------------------------
# Figure 14: high-instability examples (top-3 by best-worst gap)
# ---------------------------------------------------------------------------


def _agg_episode_per_query() -> Dict[str, Dict[str, float]]:
    """Aggregate phase2_episode_instability per query_case_id (averaged over seeds & configs)."""
    rows = _read_csv(WP1_DIR / "phase2_episode_instability.csv")
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        q = r["query_case_id"]
        agg[q]["mean_dice_k1"].append(_to_float(r["mean_dice_k1"]))
        agg[q]["std_dice_k1"].append(_to_float(r["std_dice_k1"]))
        agg[q]["best_worst_gap_k1"].append(_to_float(r["best_worst_gap_k1"]))
    out = {}
    for q, d in agg.items():
        out[q] = {k: float(np.mean(v)) for k, v in d.items()}
    return out


def fig_14_high_instability_examples() -> None:
    """Top-3 highest best-worst gap queries: mean_dice ± std as error bars, with text."""
    agg = _agg_episode_per_query()
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["best_worst_gap_k1"], reverse=True)[:3]
    fig, ax = plt.subplots()
    names = [q for q, _ in ranked]
    means = [d["mean_dice_k1"] for _, d in ranked]
    stds = [d["std_dice_k1"] for _, d in ranked]
    gaps = [d["best_worst_gap_k1"] for _, d in ranked]
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=OI_VERMILLION, edgecolor="white", alpha=0.85)
    for i, (b, m, g) in enumerate(zip(bars, means, gaps)):
        ax.text(b.get_x() + b.get_width() / 2, m + stds[i] + 0.02, f"gap={g:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Mean Dice (k=1) ± std")
    ax.set_title("Fig 14: Top-3 High-Instability Queries — MoNuSAC Holdout")
    info = "\n".join([f"{q}: gap={d['best_worst_gap_k1']:.3f}, dice={d['mean_dice_k1']:.3f}±{d['std_dice_k1']:.3f}" for q, d in ranked])
    ax.text(
        0.98,
        0.02,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999", alpha=0.9),
    )
    _save(fig, "fig_14_high_instability_examples")


# ---------------------------------------------------------------------------
# Figure 15/16: consensus success / failure cases (k=1 vs k=6 Dice)
# ---------------------------------------------------------------------------


def _per_query_k1_k6_dice(config: str = "P20-N1") -> List[Tuple[str, float, float]]:
    """Per-query (k=1 mean dice, k=6 consensus dice) for a config on holdout."""
    rows = _read_csv(WP1_DIR / "per_query_per_subset_metrics.csv")
    k1_by_q: Dict[str, List[float]] = defaultdict(list)
    k6_by_q: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["config_name"] != config:
            continue
        k = _to_int(r["k"])
        if k == 1:
            k1_by_q[r["query_case_id"]].append(_to_float(r["subset_dice"]))
        elif k == 6:
            k6_by_q[r["query_case_id"]].append(_to_float(r["subset_dice"]))
    out = []
    for q in sorted(set(k1_by_q) & set(k6_by_q)):
        d1 = float(np.mean(k1_by_q[q]))
        d6 = float(np.mean(k6_by_q[q]))
        out.append((q, d1, d6))
    return out


def _consensus_case_chart(name: str, title: str, selector) -> None:
    pairs = _per_query_k1_k6_dice("P20-N1")
    ranked = sorted(pairs, key=selector, reverse=True)[:3]
    fig, ax = plt.subplots()
    qnames = [q for q, _, _ in ranked]
    d1 = [a for _, a, _ in ranked]
    d6 = [b for _, _, b in ranked]
    x = np.arange(len(qnames))
    width = 0.35
    b1 = ax.bar(x - width / 2, d1, width, label="k=1 (single ref)", color=OI_SKYBLUE, edgecolor="white")
    b6 = ax.bar(x + width / 2, d6, width, label="k=6 (consensus)", color=OI_GREEN, edgecolor="white")
    for bars, vals in [(b1, d1), (b6, d6)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(qnames, rotation=15)
    ax.set_ylabel("Mean Dice")
    ax.set_title(title)
    ax.legend()
    gains = [b - a for _, a, b in ranked]
    info = "\n".join([f"{q}: gain={g:+.3f}" for q, g in zip(qnames, gains)])
    ax.text(
        0.98,
        0.02,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999", alpha=0.9),
    )
    _save(fig, name)


def fig_15_mean_consensus_success_cases() -> None:
    """Top-3 queries where k=6 Dice >> k=1 Dice (biggest consensus gain)."""
    _consensus_case_chart(
        "fig_15_mean_consensus_success_cases",
        "Fig 15: Top-3 Consensus Success Cases — k=6 >> k=1 (P20)",
        lambda p: p[2] - p[1],
    )


def fig_16_mean_consensus_failure_cases() -> None:
    """Top-3 queries where k=6 Dice < k=1 Dice (consensus hurt)."""
    _consensus_case_chart(
        "fig_16_mean_consensus_failure_cases",
        "Fig 16: Top-3 Consensus Failure Cases — k=6 < k=1 (P20)",
        lambda p: p[1] - p[2],
    )


# ---------------------------------------------------------------------------
# Figure 17: selective prediction failures (M2 false negatives table)
# ---------------------------------------------------------------------------


def fig_17_selective_prediction_failures() -> None:
    """M2 val predictions: false negatives (low predicted_risk, catastrophic_label=1) as a table."""
    rows = _read_csv(WP3_M2_DIR / "m2_val_predictions.csv")
    # False negatives: catastrophic_label == 1 and predicted_risk below a low threshold (0.2)
    fns = [
        r
        for r in rows
        if _to_int(r.get("catastrophic_label", "0")) == 1 and _to_float(r.get("predicted_risk", "1")) < 0.2
    ]
    # If none, relax threshold to median of positive predictions
    if not fns:
        fns = [r for r in rows if _to_int(r.get("catastrophic_label", "0")) == 1]
        fns = sorted(fns, key=lambda r: _to_float(r.get("predicted_risk", "1")))[:8]

    # Limit to 8 rows for display
    fns = sorted(fns, key=lambda r: _to_float(r.get("predicted_risk", "1")))[:8]

    cols = [
        ("query_case_id", "query"),
        ("seed", "seed"),
        ("config_name", "cfg"),
        ("best_worst_gap_k1", "gap"),
        ("mean_dice_k1", "dice"),
        ("prediction_jaccard_k1_within", "jacc"),
        ("predicted_risk", "risk"),
        ("predicted_risk_instability", "risk_inst"),
    ]
    # round numeric cols
    cell_text = []
    for r in fns:
        row = []
        for key, _ in cols:
            v = r.get(key, "")
            if key in ("best_worst_gap_k1", "mean_dice_k1", "prediction_jaccard_k1_within", "predicted_risk", "predicted_risk_instability"):
                row.append(f"{_to_float(v):.3f}")
            else:
                row.append(str(v))
        cell_text.append(row)

    n_rows = max(len(cell_text), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    if cell_text:
        table = ax.table(
            cellText=cell_text,
            colLabels=[c[1] for c in cols],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        # header styling
        for j in range(len(cols)):
            cell = table[0, j]
            cell.set_facecolor(OI_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No false-negative cases found", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(
        f"Fig 17: Selective Prediction Failures — M2 False Negatives (catastrophic_label=1, low predicted_risk); n={len(cell_text)}",
        pad=20,
    )
    _save(fig, "fig_17_selective_prediction_failures")


# ---------------------------------------------------------------------------
# Figure 18: final method diagram (dual-model M2 schematic)
# ---------------------------------------------------------------------------


def fig_18_final_method_diagram() -> None:
    """Schematic: 4 features -> [instability model] + [catastrophic model] -> selective decision."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Fig 18: Final Method — Dual-Model Selective Prediction (M2)", pad=12)

    # 4 features (left column)
    features = [
        ("F1: matching_map_disagreement", 5.7),
        ("F2: candidate_jaccard (k=1)", 4.8),
        ("F3: prediction_jaccard (k=1)", 3.9),
        ("F4: best_worst_gap (k=1)", 3.0),
    ]
    for text, y in features:
        _draw_box(ax, (0.2, y - 0.35), 2.6, 0.7, text, facecolor=OI_SKYBLUE, fontsize=8)

    # Instability model (cond1)
    _draw_box(ax, (3.6, 5.2), 2.6, 0.9, "Instability Model\n(cond 1: reference instability)", facecolor=OI_ORANGE, fontsize=9)
    # Catastrophic model (cond 2-5)
    _draw_box(ax, (3.6, 2.9), 2.6, 0.9, "Catastrophic Model\n(cond 2-5: tail-risk gate)", facecolor=OI_VERMILLION, fontsize=9)

    # arrows features -> models
    for _, fy in features:
        _draw_arrow(ax, (2.8, fy), (3.6, 5.65))
        _draw_arrow(ax, (2.8, fy), (3.6, 3.35))

    # Selective prediction decision
    _draw_box(ax, (7.0, 4.0), 2.6, 1.0, "Selective Prediction\nDecision", facecolor=OI_GREEN, fontsize=10)
    _draw_arrow(ax, (6.2, 5.65), (7.0, 4.7))
    _draw_arrow(ax, (6.2, 3.35), (7.0, 4.4))

    # Outputs
    _draw_box(ax, (7.0, 5.6), 2.6, 0.7, "Abstain\n(high risk)", facecolor=OI_PURPLE, fontsize=9)
    _draw_box(ax, (7.0, 2.4), 2.6, 0.7, "Predict\n(low risk)", facecolor=OI_GREEN, fontsize=9)
    _draw_arrow(ax, (8.3, 5.0), (8.3, 5.6))
    _draw_arrow(ax, (8.3, 4.0), (8.3, 3.1))

    ax.text(
        5.0,
        1.3,
        "Dual-model gating: instability model detects reference-induced variance;\n"
        "catastrophic model flags tail-risk failures. Abstain when either fires.",
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#555555",
    )
    _save(fig, "fig_18_final_method_diagram")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


FIGURES = [
    fig_01_benchmark_overview,
    fig_02_reference_resampling_protocol,
    fig_03_variance_decomposition,
    fig_04_best_worst_gap_distribution,
    fig_05_reference_count_scaling,
    fig_06_dice_vs_vref,
    fig_07_cvar_cfr,
    fig_08_prediction_consistency,
    fig_09_method_instability_comparison,
    fig_10_risk_coverage_curve,
    fig_11_dynamic_budget_policy,
    fig_12_cost_quality_pareto,
    fig_13_cross_dataset_summary,
    fig_14_high_instability_examples,
    fig_15_mean_consensus_success_cases,
    fig_16_mean_consensus_failure_cases,
    fig_17_selective_prediction_failures,
    fig_18_final_method_diagram,
]


def main() -> None:
    _apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(FIGURES)} Phase T figures into {OUTPUT_DIR}")
    for fn in FIGURES:
        print(f"-> {fn.__name__}")
        fn()
    print("Done.")


if __name__ == "__main__":
    main()
