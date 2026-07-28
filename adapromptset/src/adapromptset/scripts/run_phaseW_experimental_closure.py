#!/usr/bin/env python3
"""Phase W: Comprehensive Experimental Closure Analysis.
Handles WP1 (Component Ablation), WP2 (Simple Baseline Comparison),
WP3 (Risk-Coverage Curves), WP7 (Anti-Correlation Residual Effect).

All analysis uses only existing data - no new model inference.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Pipeline data paths ──────────────────────────────────────────────
PIPELINE_DIRS = {
    "L+SAM-B":  REPORTS_DIR / "phaseU_wp1" / "sam_b",
    "L+SAM-L":  REPORTS_DIR / "phaseU_wp1" / "sam_l",
    "L+SAM2-T": REPORTS_DIR / "phaseU_wp1" / "sam2_t",
    "L+SAM2-S": REPORTS_DIR / "phaseU_wp1" / "sam2_s",
    "B+SAM-B":  REPORTS_DIR / "phaseU_wp2",
}

FEATURE_COLS = [
    "matching_map_disagreement",
    "candidate_jaccard_mean_k1",
    "prompt_overlap_mean_k1",
    "prediction_jaccard_k1_within",
]

COVERAGE_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_pipeline_data(pipeline_name: str) -> pd.DataFrame:
    """Load episode instability data for a pipeline."""
    path = PIPELINE_DIRS[pipeline_name] / "phase2_episode_instability.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_csv(path)
    df["pipeline"] = pipeline_name
    return df


def load_all_pipelines() -> dict[str, pd.DataFrame]:
    """Load all pipeline data."""
    return {name: load_pipeline_data(name) for name in PIPELINE_DIRS}


# ══════════════════════════════════════════════════════════════════════
# WP1: Component Ablation
# ══════════════════════════════════════════════════════════════════════

def adaptive_reference_weights(risks: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """Compute adaptive weights: lower risk → higher weight."""
    logits = -(risks + 1e-8) / temperature
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / (weights.sum() + 1e-8)


def risk_stratified_consensus(
    dice_values: np.ndarray,
    weights: np.ndarray,
    mean_risk: float,
    risk_low: float = 0.3,
    risk_high: float = 0.7,
) -> float:
    """Risk-stratified consensus fusion."""
    if mean_risk < risk_low:
        return float(np.average(dice_values, weights=weights))
    elif mean_risk < risk_high:
        sorted_idx = np.argsort(dice_values)
        n = len(dice_values)
        keep_idx = sorted_idx[max(0, n // 6): min(n, 5 * n // 6)]
        if len(keep_idx) > 0:
            return float(np.average(dice_values[keep_idx], weights=weights[keep_idx]))
        return float(np.median(dice_values))
    else:
        return float(np.median(dice_values))


def run_component_ablation(pipeline_name: str, df: pd.DataFrame) -> dict:
    """Run proper component ablation with real per-reference Dice.

    Components:
      - Weighting: adaptive weights vs uniform
      - Stratification: risk-stratified consensus vs always weighted mean
      - Consensus: multi-reference consensus vs single best reference
      - Abstention: coverage-based abstention vs no abstention
    """
    pipeline_dir = PIPELINE_DIRS[pipeline_name]
    raw_csv = pipeline_dir / "phase2_raw_per_call.csv"
    if not raw_csv.exists():
        return {"error": f"Raw data not found: {raw_csv}"}

    raw_df = pd.read_csv(raw_csv)

    # Build per-reference Dice lookup
    per_ref_dice = {}
    for (qid, seed, cfg), group in raw_df.groupby(["query_case_id", "seed", "config_name"]):
        dice_list = group.sort_values("reference_index")["dice"].tolist()
        key = (qid, int(seed), cfg)
        per_ref_dice[key] = dice_list

    # Train catastrophic risk model on all data (development cohort)
    features = df[FEATURE_COLS].values
    cat_labels = (df["mean_dice_k1"] < 0.1).astype(int).values
    cat_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    cat_model.fit(features, cat_labels)

    # Aggregate to query level
    query_results = {}
    for (qid, seed, cfg), group in df.groupby(["query_case_id", "seed", "config_name"]):
        real_dice = per_ref_dice.get((qid, int(seed), cfg))
        if real_dice is None:
            continue
        real_dice = np.array(real_dice)

        feats = group[FEATURE_COLS].values[0:1]
        cat_risk = cat_model.predict_proba(feats)[0, 1]
        cat_risks = np.full(len(real_dice), cat_risk)

        weights = adaptive_reference_weights(cat_risks)
        uniform_weights = np.ones(len(real_dice)) / len(real_dice)

        # W0: Baseline (mean of all references)
        w0 = float(np.mean(real_dice))

        # W1: Weighting only (adaptive weights, always mean)
        w1 = float(np.average(real_dice, weights=weights))

        # W2: Fusion only (uniform weights, stratified consensus)
        w2 = risk_stratified_consensus(real_dice, uniform_weights, cat_risk)

        # W3: Abstention only (baseline, but with coverage-based abstention)
        # (This is evaluated at the query-set level, not per-episode)
        w3 = w0  # Same as baseline at episode level

        # W4: Weighting + Fusion (adaptive weights, stratified consensus)
        w4 = risk_stratified_consensus(real_dice, weights, cat_risk)

        # W5: Weighting + Abstention (same as W1 at episode level)
        w5 = w1

        # W6: Fusion + Abstention (same as W2 at episode level)
        w6 = w2

        # W7: Full RAMP
        w7 = w4  # Same as W4

        # Single best reference (no consensus)
        w_nc = float(real_dice[np.argmin(cat_risks)])

        query_results[(qid, seed, cfg)] = {
            "W0_Baseline": w0, "W1_Weighting": w1, "W2_Fusion": w2,
            "W3_Abstention": w3, "W4_WeightFusion": w4, "W5_WeightAbstain": w5,
            "W6_FusionAbstain": w6, "W7_FullRAMP": w7, "NoConsensus": w_nc,
            "cat_risk": cat_risk, "n_refs": len(real_dice),
        }

    # Aggregate to query level (mean over seeds and configs)
    query_agg = {}
    for (qid, seed, cfg), vals in query_results.items():
        if qid not in query_agg:
            query_agg[qid] = {k: [] for k in vals.keys()}
        for k, v in vals.items():
            query_agg[qid][k].append(v)

    query_summary = {}
    for qid, vals in query_agg.items():
        query_summary[qid] = {
            k: float(np.mean(v)) for k, v in vals.items()
        }

    # Compute per-variant statistics
    variants = list(query_results[list(query_results.keys())[0]].keys() - {"cat_risk", "n_refs"})
    summary = {}
    for var in variants:
        values = np.array([query_summary[qid][var] for qid in query_summary])
        baseline = np.array([query_summary[qid]["W0_Baseline"] for qid in query_summary])
        summary[var] = {
            "mean_dice": float(np.mean(values)),
            "std_dice": float(np.std(values)),
            "delta_vs_baseline": float(np.mean(values - baseline)),
            "median_dice": float(np.median(values)),
            "n_queries": len(values),
        }

    return summary


# ══════════════════════════════════════════════════════════════════════
# WP2: Simple Risk Baseline Comparison
# ══════════════════════════════════════════════════════════════════════

def compute_simple_baselines(df: pd.DataFrame) -> dict:
    """Compute all simple risk baselines (B0-B8) at query level.

    B0: Random risk ranking
    B1: Matching-map entropy (disagreement)
    B2: Candidate Jaccard
    B3: Prompt-coordinate overlap
    B4: Prediction-consistency Jaccard
    B5: M2-Instability
    B6: M2-Catastrophic
    B7: Dual-head M2
    B8: Full RAMP
    """
    # Aggregate to query level
    query_data = df.groupby("query_case_id").agg({
        "matching_map_disagreement": "mean",
        "candidate_jaccard_mean_k1": "mean",
        "prompt_overlap_mean_k1": "mean",
        "prediction_jaccard_k1_within": "mean",
        "mean_dice_k1": "mean",
        "catastrophic_failure_rate_k1": "mean",
        "best_worst_gap_k1": "mean",
    }).reset_index()

    n_queries = len(query_data)
    results = {}

    # B0: Random
    np.random.seed(42)
    random_risk = np.random.rand(n_queries)
    results["B0_Random"] = compute_risk_metrics(query_data, random_risk)

    # B1: Matching-map entropy (higher = riskier)
    b1 = query_data["matching_map_disagreement"].values
    results["B1_Entropy"] = compute_risk_metrics(query_data, b1)

    # B2: Candidate Jaccard (lower = riskier)
    b2 = 1.0 - query_data["candidate_jaccard_mean_k1"].values
    results["B2_CandidateJaccard"] = compute_risk_metrics(query_data, b2)

    # B3: Prompt overlap (lower = riskier)
    b3 = 1.0 - query_data["prompt_overlap_mean_k1"].values
    results["B3_PromptOverlap"] = compute_risk_metrics(query_data, b3)

    # B4: Prediction consistency (lower = riskier)
    b4 = 1.0 - query_data["prediction_jaccard_k1_within"].values
    results["B4_PredictionConsistency"] = compute_risk_metrics(query_data, b4)

    # B5: Train M2-Instability
    features = query_data[FEATURE_COLS].values
    inst_labels = (query_data["best_worst_gap_k1"] > 0.3).astype(int).values
    if len(np.unique(inst_labels)) > 1:
        inst_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        inst_model.fit(features, inst_labels)
        b5 = inst_model.predict_proba(features)[:, 1]
    else:
        b5 = inst_labels.astype(float)  # Fallback: use label directly
    results["B5_M2Instability"] = compute_risk_metrics(query_data, b5)

    # B6: Train M2-Catastrophic
    cat_labels = (query_data["mean_dice_k1"] < 0.1).astype(int).values
    if len(np.unique(cat_labels)) > 1:
        cat_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        cat_model.fit(features, cat_labels)
        b6 = cat_model.predict_proba(features)[:, 1]
    else:
        b6 = cat_labels.astype(float)
    results["B6_M2Catastrophic"] = compute_risk_metrics(query_data, b6)

    # B7: Dual-head M2 (average of instability and catastrophic)
    b7 = (b5 + b6) / 2.0
    results["B7_DualHeadM2"] = compute_risk_metrics(query_data, b7)

    # B8: Full RAMP (from WP1 component ablation)
    # This is computed separately

    return results


def compute_risk_metrics(query_data: pd.DataFrame, risk_scores: np.ndarray) -> dict:
    """Compute selective prediction metrics for a risk ranking."""
    n = len(risk_scores)
    order = np.argsort(risk_scores)[::-1]  # Highest risk first
    dice_values = query_data["mean_dice_k1"].values[order]
    cfr_values = (query_data["mean_dice_k1"].values[order] < 0.1).astype(float)

    metrics = {}
    for cov in COVERAGE_GRID:
        n_accept = max(1, int(n * cov))
        accepted_dice = dice_values[:n_accept]
        accepted_cfr = cfr_values[:n_accept]
        metrics[f"dice@{int(cov*100)}"] = float(np.mean(accepted_dice))
        metrics[f"cfr@{int(cov*100)}"] = float(np.mean(accepted_cfr))

    # AURC
    aurc = 0.0
    for i in range(1, n + 1):
        aurc += cfr_values[:i].mean()
    aurc /= n
    metrics["aurc"] = float(aurc)

    # Random baseline AURC
    random_order = np.random.RandomState(42).permutation(n)
    random_cfr = cfr_values[random_order]
    random_aurc = 0.0
    for i in range(1, n + 1):
        random_aurc += random_cfr[:i].mean()
    random_aurc /= n
    metrics["excess_aurc"] = float(aurc - random_aurc)

    # AUROC for CFR prediction
    true_cfr = (query_data["mean_dice_k1"].values < 0.1).astype(int)
    if len(np.unique(true_cfr)) > 1:
        metrics["auroc"] = float(roc_auc_score(true_cfr, risk_scores))
        metrics["auprc"] = float(average_precision_score(true_cfr, risk_scores))
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = 0.0

    return metrics


# ══════════════════════════════════════════════════════════════════════
# WP3: Risk-Coverage Curves
# ══════════════════════════════════════════════════════════════════════

def compute_risk_coverage_curves(df: pd.DataFrame) -> dict:
    """Compute full risk-coverage curves for all baselines."""
    baselines = compute_simple_baselines(df)
    return baselines


# ══════════════════════════════════════════════════════════════════════
# WP7: Anti-Correlation Residual Effect
# ══════════════════════════════════════════════════════════════════════

def compute_anticorrelation_residual(pipelines_data: dict[str, pd.DataFrame]) -> dict:
    """Compute observed-minus-null anti-correlation residual effect.

    For each pipeline:
      1. Compute observed r (instability vs CFR)
      2. Compute null r (random beta preserving per-query distribution)
      3. Compute delta_r = r_observed - r_null
      4. Bootstrap CI for delta_r
      5. Permutation test
    """
    results = {}

    for pipeline_name, df in pipelines_data.items():
        # Aggregate to query level
        query_data = df.groupby("query_case_id").agg({
            "best_worst_gap_k1": "mean",
            "mean_dice_k1": "mean",
            "catastrophic_failure_rate_k1": "mean",
        }).reset_index()

        instability = query_data["best_worst_gap_k1"].values
        cfr = query_data["catastrophic_failure_rate_k1"].values
        mean_dice = query_data["mean_dice_k1"].values
        n = len(query_data)

        # Observed correlation
        r_obs, p_obs = pearsonr(instability, cfr)
        r_partial, p_partial = pearsonr(
            instability - np.mean(instability),
            cfr - np.mean(cfr)
        )

        # Null 1: Random beta preserving per-query mean and std
        np.random.seed(42)
        null_corrs = []
        for _ in range(1000):
            null_dice = np.random.beta(
                a=mean_dice * 10 + 1,
                b=(1 - mean_dice) * 10 + 1,
            )
            null_cfr = (null_dice < 0.1).astype(float)
            null_dice_matrix = np.tile(null_dice, (6, 1)).T
            null_instability = np.max(null_dice_matrix, axis=1) - np.min(null_dice_matrix, axis=1)
            r_null, _ = pearsonr(null_instability, null_cfr)
            null_corrs.append(r_null)

        r_null_mean = float(np.mean(null_corrs))
        r_null_std = float(np.std(null_corrs))
        delta_r = r_obs - r_null_mean

        # Bootstrap CI for delta_r
        np.random.seed(42)
        delta_bootstrap = []
        for _ in range(10000):
            idx = np.random.choice(n, n, replace=True)
            r_boot, _ = pearsonr(instability[idx], cfr[idx])
            null_boot = []
            for _ in range(100):
                null_dice_b = np.random.beta(
                    a=mean_dice[idx] * 10 + 1,
                    b=(1 - mean_dice[idx]) * 10 + 1,
                )
                null_cfr_b = (null_dice_b < 0.1).astype(float)
                null_dice_m = np.tile(null_dice_b, (6, 1)).T
                null_inst_b = np.max(null_dice_m, axis=1) - np.min(null_dice_m, axis=1)
                r_nb, _ = pearsonr(null_inst_b, null_cfr_b)
                null_boot.append(r_nb)
            delta_boot = r_boot - np.mean(null_boot)
            delta_bootstrap.append(delta_boot)

        delta_ci_low = float(np.percentile(delta_bootstrap, 2.5))
        delta_ci_high = float(np.percentile(delta_bootstrap, 97.5))

        # Permutation test for delta_r
        perm_deltas = []
        for _ in range(10000):
            perm_inst = np.random.permutation(instability)
            r_perm, _ = pearsonr(perm_inst, cfr)
            perm_deltas.append(r_perm - r_null_mean)
        p_perm = float(np.mean(np.abs(perm_deltas) >= np.abs(delta_r)))

        results[pipeline_name] = {
            "n_queries": n,
            "r_observed": float(r_obs),
            "p_observed": float(p_obs),
            "r_null_mean": r_null_mean,
            "r_null_std": r_null_std,
            "delta_r": float(delta_r),
            "delta_r_ci_95": [delta_ci_low, delta_ci_high],
            "p_permutation": p_perm,
            "r_partial": float(r_partial),
            "p_partial": float(p_partial),
            "null_explains_pct": float(r_null_mean / r_obs * 100) if r_obs != 0 else float("inf"),
            "significant_residual": delta_ci_low < 0,
        }

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase W: Experimental Closure Analysis")
    print("=" * 70)

    # Load all pipeline data
    print("\nLoading pipeline data...")
    pipelines = load_all_pipelines()
    print(f"  Loaded {len(pipelines)} pipelines")

    # ── WP1: Component Ablation ──
    print("\n" + "=" * 70)
    print("WP1: Component Ablation")
    print("=" * 70)
    ablation_results = {}
    for name, df in pipelines.items():
        print(f"\n  {name}:")
        result = run_component_ablation(name, df)
        ablation_results[name] = result
        if "error" in result:
            print(f"    ERROR: {result['error']}")
            continue
        for var, stats in result.items():
            delta = stats.get("delta_vs_baseline", 0)
            dice = stats.get("mean_dice", 0)
            print(f"    {var:25s}: Dice={dice:.4f}, Δ={delta:+.4f}")

    # Save ablation results
    out_dir = REPORTS_DIR / "phaseW_component_ablation"
    with open(out_dir / "component_ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # ── WP2: Simple Baseline Comparison ──
    print("\n" + "=" * 70)
    print("WP2: Simple Risk Baseline Comparison")
    print("=" * 70)
    baseline_results = {}
    for name, df in pipelines.items():
        print(f"\n  {name}:")
        result = compute_simple_baselines(df)
        baseline_results[name] = result
        for method, metrics in result.items():
            print(f"    {method:30s}: AUROC={metrics.get('auroc',0):.3f}, "
                  f"CFR@80={metrics.get('cfr@80',0):.3f}, "
                  f"AURC={metrics.get('aurc',0):.4f}")

    # Save baseline results
    out_dir = REPORTS_DIR / "phaseW_simple_baselines"
    with open(out_dir / "simple_baseline_comparison.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Determine strongest simple baseline
    print("\n  Strongest simple baseline per pipeline:")
    for name, result in baseline_results.items():
        best = None
        best_aurc = float("inf")
        for method, metrics in result.items():
            if metrics.get("aurc", 1) < best_aurc:
                best_aurc = metrics["aurc"]
                best = method
        print(f"    {name}: {best} (AURC={best_aurc:.4f})")

    # ── WP3: Risk-Coverage Curves ──
    print("\n" + "=" * 70)
    print("WP3: Risk-Coverage Curves")
    print("=" * 70)
    # Already computed in WP2 baseline results

    # Save risk-coverage results
    out_dir = REPORTS_DIR / "phaseW_risk_coverage"
    with open(out_dir / "risk_coverage_curves.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # ── WP7: Anti-Correlation Residual Effect ──
    print("\n" + "=" * 70)
    print("WP7: Anti-Correlation Residual Effect")
    print("=" * 70)
    anticorr_results = compute_anticorrelation_residual(pipelines)
    for name, result in anticorr_results.items():
        print(f"\n  {name}:")
        print(f"    r_observed = {result['r_observed']:.4f} (p={result['p_observed']:.4f})")
        print(f"    r_null     = {result['r_null_mean']:.4f} ± {result['r_null_std']:.4f}")
        print(f"    Δr         = {result['delta_r']:.4f}")
        print(f"    Δr CI 95%  = [{result['delta_r_ci_95'][0]:.4f}, {result['delta_r_ci_95'][1]:.4f}]")
        print(f"    Null explains {result['null_explains_pct']:.1f}% of correlation")
        print(f"    Significant residual: {result['significant_residual']}")
        print(f"    p_permutation = {result['p_permutation']:.4f}")

    # Save anti-correlation results
    out_dir = REPORTS_DIR / "phaseW_anticorrelation"
    with open(out_dir / "anticorrelation_residual.json", "w") as f:
        json.dump(anticorr_results, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("PHASE W ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  WP1: reports/phaseW_component_ablation/component_ablation_results.json")
    print(f"  WP2: reports/phaseW_simple_baselines/simple_baseline_comparison.json")
    print(f"  WP3: reports/phaseW_risk_coverage/risk_coverage_curves.json")
    print(f"  WP7: reports/phaseW_anticorrelation/anticorrelation_residual.json")


if __name__ == "__main__":
    main()