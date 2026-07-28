#!/usr/bin/env python3
"""
Phase V — Query-Level Statistics Reanalysis (Gate V4/P0-6)
============================================================
Fixes the statistical unit problem: previous analysis used episode-level
(80 queries × 5 seeds × 3 configs = 1200 rows) as independent samples,
but the correct unit is query (80 queries).

This script:
1. Reads per-episode instability data from all segmenter pipelines
2. Aggregates by query (query_case_id) — mean across seeds and configs
3. Recomputes anti-correlation (Pearson/Spearman) at query level
4. Recomputes Wilcoxon signed-rank tests for paired comparisons
5. Recomputes bootstrap CIs for key metrics
6. Outputs corrected statistics to reports/phaseV_query_level_stats/
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# --- Configuration ---
REPORTS_DIR = Path(r"D:\SAM\adapromptset\reports")
OUTPUT_DIR = REPORTS_DIR / "phaseV_query_level_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline definitions: (matcher, segmenter, data_dir)
PIPELINES = {
    "L+SAM-B": {
        "data_dir": REPORTS_DIR / "phaseU_wp1" / "sam_b",
        "configs": ["P20-N1", "P30-N0", "P50-N0"],
        "n_seeds": 5,
        "matcher": "DINOv2-L",
        "segmenter": "SAM-B"
    },
    "L+SAM-L": {
        "data_dir": REPORTS_DIR / "phaseU_wp1" / "sam_l",
        "configs": ["P20-N1", "P30-N0", "P50-N0"],
        "n_seeds": 5,
        "matcher": "DINOv2-L",
        "segmenter": "SAM-L"
    },
    "L+SAM2-T": {
        "data_dir": REPORTS_DIR / "phaseU_wp1" / "sam2_t",
        "configs": ["P20-N1", "P30-N0", "P50-N0"],
        "n_seeds": 5,
        "matcher": "DINOv2-L",
        "segmenter": "SAM2-Tiny"
    },
    "L+SAM2-S": {
        "data_dir": REPORTS_DIR / "phaseU_wp1" / "sam2_s",
        "configs": ["P20-N1", "P30-N0", "P50-N0"],
        "n_seeds": 5,
        "matcher": "DINOv2-L",
        "segmenter": "SAM2-Small"
    },
    "B+SAM-B": {
        "data_dir": REPORTS_DIR / "phaseU_wp2",  # DINOv2-B matcher
        "configs": ["P20-N1", "P30-N0", "P50-N0"],
        "n_seeds": 5,
        "matcher": "DINOv2-B",
        "segmenter": "SAM-B"
    },
}

def load_episode_data(data_dir, configs, n_seeds):
    """Load episode instability data and aggregate by query."""
    csv_path = data_dir / "phase2_episode_instability.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter to relevant configs
    df = df[df["config_name"].isin(configs)]
    
    # Aggregate by query: mean across seeds and configs
    query_agg = df.groupby("query_case_id").agg(
        mean_dice=("mean_dice_k1", "mean"),
        std_dice=("std_dice_k1", "mean"),
        min_dice=("min_dice_k1", "min"),
        max_dice=("max_dice_k1", "max"),
        best_worst_gap=("best_worst_gap_k1", "mean"),
        cfr=("catastrophic_failure_rate_k1", "mean"),
        n_episodes=("mean_dice_k1", "count"),
        matching_map_disagreement=("matching_map_disagreement", "mean"),
        candidate_jaccard=("candidate_jaccard_mean_k1", "mean"),
        prompt_overlap=("prompt_overlap_mean_k1", "mean"),
        prediction_jaccard=("prediction_jaccard_k1_within", "mean"),
    ).reset_index()
    
    # Also compute per-config aggregates for config-level analysis
    config_agg = df.groupby(["query_case_id", "config_name"]).agg(
        mean_dice=("mean_dice_k1", "mean"),
        cfr=("catastrophic_failure_rate_k1", "mean"),
        best_worst_gap=("best_worst_gap_k1", "mean"),
    ).reset_index()
    
    return query_agg, config_agg


def compute_anti_correlation(query_df, pipeline_name):
    """Compute anti-correlation between instability and CFR at query level."""
    instability = query_df["best_worst_gap"].values
    cfr = query_df["cfr"].values
    mean_dice = query_df["mean_dice"].values
    
    # Remove NaN
    mask = ~(np.isnan(instability) | np.isnan(cfr))
    instability = instability[mask]
    cfr = cfr[mask]
    mean_dice = mean_dice[mask]
    
    n_queries = len(instability)
    
    # Pearson
    r_pearson, p_pearson = stats.pearsonr(instability, cfr)
    
    # Spearman
    r_spearman, p_spearman = stats.spearmanr(instability, cfr)
    
    # Bootstrap CI for Pearson
    n_bootstrap = 10000
    np.random.seed(42)
    r_boot = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_queries, size=n_queries, replace=True)
        r_b, _ = stats.pearsonr(instability[idx], cfr[idx])
        r_boot.append(r_b)
    r_ci = np.percentile(r_boot, [2.5, 97.5])
    
    # Partial correlation controlling for mean Dice
    # Compute residuals
    from scipy.stats import pearsonr
    # Regress instability on mean_dice, take residuals
    slope_inst, intercept_inst, _, _, _ = stats.linregress(mean_dice, instability)
    resid_inst = instability - (slope_inst * mean_dice + intercept_inst)
    
    slope_cfr, intercept_cfr, _, _, _ = stats.linregress(mean_dice, cfr)
    resid_cfr = cfr - (slope_cfr * mean_dice + intercept_cfr)
    
    r_partial, p_partial = stats.pearsonr(resid_inst, resid_cfr)
    
    # Bootstrap CI for partial
    r_partial_boot = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_queries, size=n_queries, replace=True)
        mi = mean_dice[idx]
        ii = instability[idx]
        ci = cfr[idx]
        si, _, _, _, _ = stats.linregress(mi, ii)
        sc, _, _, _, _ = stats.linregress(mi, ci)
        ri = ii - (si * mi + np.mean(ii) - si * np.mean(mi))
        rc = ci - (sc * mi + np.mean(ci) - sc * np.mean(mi))
        rp, _ = stats.pearsonr(ri, rc)
        r_partial_boot.append(rp)
    r_partial_ci = np.percentile(r_partial_boot, [2.5, 97.5])
    
    return {
        "pipeline": pipeline_name,
        "n_queries": n_queries,
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "pearson_ci_95": [float(r_ci[0]), float(r_ci[1])],
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "partial_r": float(r_partial),
        "partial_p": float(p_partial),
        "partial_ci_95": [float(r_partial_ci[0]), float(r_partial_ci[1])],
        "significant_pearson": bool(p_pearson < 0.05),
        "significant_spearman": bool(p_spearman < 0.05),
        "significant_partial": bool(p_partial < 0.05),
    }


def compute_query_level_descriptives(query_df, pipeline_name):
    """Compute query-level descriptive statistics."""
    dice = query_df["mean_dice"].values
    instability = query_df["best_worst_gap"].values
    cfr = query_df["cfr"].values
    
    mask = ~np.isnan(dice)
    dice = dice[mask]
    instability = instability[mask]
    cfr = cfr[mask]
    
    n_queries = len(dice)
    
    # Bootstrap CI for mean Dice
    np.random.seed(42)
    n_boot = 10000
    dice_boot_means = []
    cfr_boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(n_queries, size=n_queries, replace=True)
        dice_boot_means.append(np.mean(dice[idx]))
        cfr_boot_means.append(np.mean(cfr[idx]))
    
    return {
        "pipeline": pipeline_name,
        "n_queries": n_queries,
        "mean_dice": float(np.mean(dice)),
        "dice_ci_95": [float(np.percentile(dice_boot_means, 2.5)), float(np.percentile(dice_boot_means, 97.5))],
        "dice_std": float(np.std(dice, ddof=1)),
        "mean_instability": float(np.mean(instability)),
        "instability_std": float(np.std(instability, ddof=1)),
        "mean_cfr": float(np.mean(cfr)),
        "cfr_ci_95": [float(np.percentile(cfr_boot_means, 2.5)), float(np.percentile(cfr_boot_means, 97.5))],
        "prevalence_catastrophic": float(np.mean(cfr > 0.5)),
        "prevalence_high_instability": float(np.mean(instability > 0.3)),
    }


def compute_paired_comparison(query_dfs, pipeline_a, pipeline_b):
    """Compute paired Wilcoxon test between two pipelines at query level."""
    df_a = query_dfs[pipeline_a]
    df_b = query_dfs[pipeline_b]
    
    # Merge on query_case_id
    merged = df_a.merge(df_b, on="query_case_id", suffixes=("_a", "_b"))
    
    dice_diff = merged["mean_dice_a"].values - merged["mean_dice_b"].values
    cfr_diff = merged["cfr_a"].values - merged["cfr_b"].values
    
    mask = ~(np.isnan(dice_diff) | np.isnan(cfr_diff))
    dice_diff = dice_diff[mask]
    cfr_diff = cfr_diff[mask]
    
    n = len(dice_diff)
    
    # Wilcoxon signed-rank
    w_dice, p_dice = stats.wilcoxon(dice_diff, alternative="two-sided") if np.any(dice_diff != 0) else (0, 1.0)
    w_cfr, p_cfr = stats.wilcoxon(cfr_diff, alternative="two-sided") if np.any(cfr_diff != 0) else (0, 1.0)
    
    # Bootstrap CI for mean difference
    np.random.seed(42)
    n_boot = 10000
    dice_diff_boot = []
    cfr_diff_boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        dice_diff_boot.append(np.mean(dice_diff[idx]))
        cfr_diff_boot.append(np.mean(cfr_diff[idx]))
    
    # Win/tie/loss
    wins_dice = np.sum(dice_diff > 0.01)
    ties_dice = np.sum(np.abs(dice_diff) <= 0.01)
    losses_dice = np.sum(dice_diff < -0.01)
    
    wins_cfr = np.sum(cfr_diff < -0.05)  # lower CFR is better
    ties_cfr = np.sum(np.abs(cfr_diff) <= 0.05)
    losses_cfr = np.sum(cfr_diff > 0.05)
    
    return {
        "pipeline_a": pipeline_a,
        "pipeline_b": pipeline_b,
        "n_paired_queries": n,
        "dice": {
            "mean_diff": float(np.mean(dice_diff)),
            "diff_ci_95": [float(np.percentile(dice_diff_boot, 2.5)), float(np.percentile(dice_diff_boot, 97.5))],
            "wilcoxon_stat": float(w_dice),
            "p_value": float(p_dice),
            "win_tie_loss": f"{wins_dice}/{ties_dice}/{losses_dice}",
        },
        "cfr": {
            "mean_diff": float(np.mean(cfr_diff)),
            "diff_ci_95": [float(np.percentile(cfr_diff_boot, 2.5)), float(np.percentile(cfr_diff_boot, 97.5))],
            "wilcoxon_stat": float(w_cfr),
            "p_value": float(p_cfr),
            "win_tie_loss": f"{wins_cfr}/{ties_cfr}/{losses_cfr}",
        },
    }


def main():
    print("=" * 70)
    print("Phase V — Query-Level Statistics Reanalysis")
    print("=" * 70)
    
    # Load all pipeline data
    query_dfs = {}
    all_anti_corr = []
    all_descriptives = []
    
    for pipeline_name, pipeline_info in PIPELINES.items():
        print(f"\nLoading {pipeline_name}...")
        result = load_episode_data(
            pipeline_info["data_dir"],
            pipeline_info["configs"],
            pipeline_info["n_seeds"]
        )
        if result is None:
            continue
        query_df, config_df = result
        query_dfs[pipeline_name] = query_df
        
        print(f"  Episodes: {len(config_df)}, Unique queries: {len(query_df)}")
        print(f"  Mean Dice: {query_df['mean_dice'].mean():.4f}")
        print(f"  Mean CFR: {query_df['cfr'].mean():.4f}")
        
        # Anti-correlation
        ac = compute_anti_correlation(query_df, pipeline_name)
        all_anti_corr.append(ac)
        print(f"  Pearson r (instability vs CFR): {ac['pearson_r']:.4f} (p={ac['pearson_p']:.4f})")
        print(f"  Partial r (controlling mean Dice): {ac['partial_r']:.4f} (p={ac['partial_p']:.4f})")
        
        # Descriptives
        desc = compute_query_level_descriptives(query_df, pipeline_name)
        all_descriptives.append(desc)
    
    # Paired comparisons
    print("\n" + "=" * 70)
    print("Paired Pipeline Comparisons (Query-Level)")
    print("=" * 70)
    
    paired_results = []
    pipeline_names = list(query_dfs.keys())
    for i in range(len(pipeline_names)):
        for j in range(i + 1, len(pipeline_names)):
            a, b = pipeline_names[i], pipeline_names[j]
            result = compute_paired_comparison(query_dfs, a, b)
            paired_results.append(result)
            print(f"\n{a} vs {b}:")
            print(f"  Dice diff: {result['dice']['mean_diff']:.4f} [{result['dice']['diff_ci_95'][0]:.4f}, {result['dice']['diff_ci_95'][1]:.4f}]")
            print(f"  Dice p: {result['dice']['p_value']:.4f} (W/T/L: {result['dice']['win_tie_loss']})")
            print(f"  CFR diff: {result['cfr']['mean_diff']:.4f} [{result['cfr']['diff_ci_95'][0]:.4f}, {result['cfr']['diff_ci_95'][1]:.4f}]")
            print(f"  CFR p: {result['cfr']['p_value']:.4f} (W/T/L: {result['cfr']['win_tie_loss']})")
    
    # Holm correction for anti-correlation p-values
    print("\n" + "=" * 70)
    print("Holm-Bonferroni Correction (Anti-Correlation)")
    print("=" * 70)
    
    p_values = [(ac["pipeline"], ac["pearson_p"]) for ac in all_anti_corr]
    p_values_sorted = sorted(p_values, key=lambda x: x[1])
    n_tests = len(p_values_sorted)
    
    holm_results = []
    for rank, (name, p) in enumerate(p_values_sorted):
        holm_threshold = 0.05 / (n_tests - rank)
        holm_significant = p < holm_threshold
        holm_results.append({
            "pipeline": name,
            "p_raw": float(p),
            "holm_threshold": float(holm_threshold),
            "holm_significant": bool(holm_significant)
        })
        print(f"  {name}: p={p:.4f}, Holm threshold={holm_threshold:.4f}, significant={holm_significant}")
    
    # Save results
    output = {
        "phase": "PhaseV",
        "task": "Query-Level Statistics Reanalysis",
        "date": pd.Timestamp.now().isoformat(),
        "corrected_statistical_unit": "query (n=80 per dataset)",
        "n_pipelines": len(query_dfs),
        "anti_correlation": all_anti_corr,
        "descriptives": all_descriptives,
        "paired_comparisons": paired_results,
        "holm_correction": holm_results,
        "summary": {
            "anti_correlation_query_level": {
                ac["pipeline"]: {
                    "pearson_r": ac["pearson_r"],
                    "pearson_p": ac["pearson_p"],
                    "significant": ac["significant_pearson"],
                    "partial_r": ac["partial_r"],
                    "partial_p": ac["partial_p"],
                }
                for ac in all_anti_corr
            },
            "key_findings": [
                "All statistics now use query (n=80) as the primary unit, not episode (n=1200)",
                f"Anti-correlation range at query level: {min(ac['pearson_r'] for ac in all_anti_corr):.3f} to {max(ac['pearson_r'] for ac in all_anti_corr):.3f}",
                f"Partial correlation (controlling mean Dice) range: {min(ac['partial_r'] for ac in all_anti_corr):.3f} to {max(ac['partial_r'] for ac in all_anti_corr):.3f}",
                "Paired comparisons use query-level aggregation with Wilcoxon signed-rank test",
                "Bootstrap CIs (10,000 resamples) provided for all key metrics",
                "Holm-Bonferroni correction applied for multiple anti-correlation tests",
            ]
        }
    }
    
    output_path = OUTPUT_DIR / "query_level_statistics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    
    # Save summary table
    summary_df = pd.DataFrame([
        {
            "Pipeline": ac["pipeline"],
            "N_Queries": ac["n_queries"],
            "Pearson_r": f"{ac['pearson_r']:.4f}",
            "Pearson_p": f"{ac['pearson_p']:.4f}",
            "Pearson_CI": f"[{ac['pearson_ci_95'][0]:.3f}, {ac['pearson_ci_95'][1]:.3f}]",
            "Significant": "YES" if ac["significant_pearson"] else "NO",
            "Partial_r": f"{ac['partial_r']:.4f}",
            "Partial_p": f"{ac['partial_p']:.4f}",
            "Partial_CI": f"[{ac['partial_ci_95'][0]:.3f}, {ac['partial_ci_95'][1]:.3f}]",
            "Partial_Significant": "YES" if ac["significant_partial"] else "NO",
        }
        for ac in all_anti_corr
    ])
    summary_path = OUTPUT_DIR / "anti_correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to {summary_path}")
    
    # Save descriptives table
    desc_df = pd.DataFrame([
        {
            "Pipeline": d["pipeline"],
            "N_Queries": d["n_queries"],
            "Mean_Dice": f"{d['mean_dice']:.4f}",
            "Dice_CI_95": f"[{d['dice_ci_95'][0]:.4f}, {d['dice_ci_95'][1]:.4f}]",
            "Mean_Instability": f"{d['mean_instability']:.4f}",
            "Mean_CFR": f"{d['mean_cfr']:.4f}",
            "CFR_CI_95": f"[{d['cfr_ci_95'][0]:.4f}, {d['cfr_ci_95'][1]:.4f}]",
            "Prevalence_Catastrophic": f"{d['prevalence_catastrophic']:.3f}",
        }
        for d in all_descriptives
    ])
    desc_path = OUTPUT_DIR / "descriptives_summary.csv"
    desc_df.to_csv(desc_path, index=False)
    print(f"Descriptives table saved to {desc_path}")
    
    print("\n" + "=" * 70)
    print("DONE — Query-Level Statistics Reanalysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()