#!/usr/bin/env python3
"""Phase W WP7: Anti-Correlation Residual Effect (optimized).
Uses moderate bootstrap/permutation iterations for speed.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

PIPELINE_DIRS = {
    "L+SAM-B":  REPORTS_DIR / "phaseU_wp1" / "sam_b",
    "L+SAM-L":  REPORTS_DIR / "phaseU_wp1" / "sam_l",
    "L+SAM2-T": REPORTS_DIR / "phaseU_wp1" / "sam2_t",
    "L+SAM2-S": REPORTS_DIR / "phaseU_wp1" / "sam2_s",
    "B+SAM-B":  REPORTS_DIR / "phaseU_wp2",
}

N_BOOTSTRAP = 5000
N_NULL_INNER = 50
N_PERM = 5000
SEED = 42


def compute_null_correlation(mean_dice: np.ndarray, rng: np.random.RandomState) -> float:
    """Compute null correlation for one sample."""
    null_dice = rng.beta(a=mean_dice * 10 + 1, b=(1 - mean_dice) * 10 + 1)
    null_cfr = (null_dice < 0.1).astype(float)
    # Simulate instability as max-min across 6 references
    null_dice_matrix = np.tile(null_dice, (6, 1)).T
    null_instability = np.max(null_dice_matrix, axis=1) - np.min(null_dice_matrix, axis=1)
    r, _ = pearsonr(null_instability, null_cfr)
    return r if not np.isnan(r) else 0.0


def compute_anticorrelation_residual(
    pipeline_name: str, df: pd.DataFrame, rng: np.random.RandomState
) -> dict:
    """Compute anti-correlation residual for one pipeline."""
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
    if np.isnan(r_obs):
        r_obs, p_obs = 0.0, 1.0

    # Null model: random beta with same mean Dice
    null_corrs = np.array([
        compute_null_correlation(mean_dice, rng) for _ in range(1000)
    ])
    r_null_mean = float(np.mean(null_corrs))
    r_null_std = float(np.std(null_corrs))
    delta_r = r_obs - r_null_mean

    # Bootstrap CI for delta_r (vectorized where possible)
    delta_bootstrap = np.zeros(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        r_boot, _ = pearsonr(instability[idx], cfr[idx])
        if np.isnan(r_boot):
            r_boot = 0.0
        null_boot = np.array([
            compute_null_correlation(mean_dice[idx], rng) for _ in range(N_NULL_INNER)
        ])
        delta_bootstrap[i] = r_boot - np.mean(null_boot)

    delta_ci_low = float(np.percentile(delta_bootstrap, 2.5))
    delta_ci_high = float(np.percentile(delta_bootstrap, 97.5))

    # Permutation test
    perm_deltas = np.zeros(N_PERM)
    for i in range(N_PERM):
        perm_inst = rng.permutation(instability)
        r_perm, _ = pearsonr(perm_inst, cfr)
        if np.isnan(r_perm):
            r_perm = 0.0
        perm_deltas[i] = r_perm - r_null_mean
    p_perm = float(np.mean(np.abs(perm_deltas) >= np.abs(delta_r)))

    return {
        "n_queries": n,
        "r_observed": float(r_obs),
        "p_observed": float(p_obs),
        "r_null_mean": r_null_mean,
        "r_null_std": r_null_std,
        "delta_r": float(delta_r),
        "delta_r_ci_95": [delta_ci_low, delta_ci_high],
        "p_permutation": p_perm,
        "null_explains_pct": float(r_null_mean / r_obs * 100) if r_obs != 0 else float("inf"),
        "significant_residual": delta_ci_low < 0,
    }


def main():
    print("=" * 70)
    print("Phase W WP7: Anti-Correlation Residual Effect")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    results = {}

    for pipeline_name, data_dir in PIPELINE_DIRS.items():
        csv_path = data_dir / "phase2_episode_instability.csv"
        if not csv_path.exists():
            print(f"  {pipeline_name}: SKIP (no data)")
            continue
        df = pd.read_csv(csv_path)
        print(f"\n  {pipeline_name}: {len(df)} episodes, {df['query_case_id'].nunique()} queries")
        result = compute_anticorrelation_residual(pipeline_name, df, rng)
        results[pipeline_name] = result
        print(f"    r_observed = {result['r_observed']:.4f} (p={result['p_observed']:.4f})")
        print(f"    r_null     = {result['r_null_mean']:.4f} ± {result['r_null_std']:.4f}")
        print(f"    Δr         = {result['delta_r']:.4f}")
        print(f"    Δr CI 95%  = [{result['delta_r_ci_95'][0]:.4f}, {result['delta_r_ci_95'][1]:.4f}]")
        print(f"    Null explains {result['null_explains_pct']:.1f}% of correlation")
        print(f"    Significant residual: {result['significant_residual']}")
        print(f"    p_permutation = {result['p_permutation']:.4f}")

    out_dir = REPORTS_DIR / "phaseW_anticorrelation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anticorrelation_residual.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("=" * 70)
    print("WP7 COMPLETE")


if __name__ == "__main__":
    main()