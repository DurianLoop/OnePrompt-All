#!/usr/bin/env python3
"""WP1b: Power Analysis for Confirmatory Experiment.

Uses Development Cohort query-level paired results to simulate
required sample size for detecting AURC improvement.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
SAMPLE_SIZES = [20, 30, 40, 50, 75, 100, 150]

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "phaseAB_power"

# Load Development Cohort data
PIPE_DIR = REPORTS_DIR / "phaseU_wp1" / "sam_b"
CSV_PATH = PIPE_DIR / "phase2_episode_instability.csv"

def main():
    if not CSV_PATH.exists():
        print(f"Data not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # Aggregate to query level
    qdf = df.groupby("query_case_id").agg({
        "prediction_jaccard_k1_within": "mean",
        "matching_map_disagreement": "mean",
        "mean_dice_k1": "mean",
        "catastrophic_failure_rate_k1": "mean",
    }).reset_index()

    n_queries_total = len(qdf)
    print(f"Development Cohort: {n_queries_total} queries")

    # Compute B4 risk and CFR
    b4_risk = 1.0 - qdf["prediction_jaccard_k1_within"].values
    dice = qdf["mean_dice_k1"].values
    cfr = (dice < 0.1).astype(float)

    # Compute baseline AURC
    def compute_aurc(risk, cfr_vals):
        order = np.argsort(risk)[::-1]
        sorted_cfr = cfr_vals[order]
        n = len(sorted_cfr)
        aurc = sum(sorted_cfr[:i].mean() for i in range(1, n + 1)) / n
        return aurc

    b4_aurc_full = compute_aurc(b4_risk, cfr)
    print(f"Full cohort B4 AURC: {b4_aurc_full:.4f}")

    # Effect distribution: per-query AURC contribution
    order = np.argsort(b4_risk)[::-1]
    sorted_cfr = cfr[order]
    n = len(sorted_cfr)
    per_query_aurc_contrib = np.array([
        sorted_cfr[:i].mean() for i in range(1, n + 1)
    ])

    # Power simulation
    rng = np.random.RandomState(RANDOM_SEED)
    power_results = {}

    for N in SAMPLE_SIZES:
        if N > n_queries_total:
            power_results[N] = {"error": f"N={N} > available queries ({n_queries_total})"}
            continue

        aurc_diffs = []
        cfr80_diffs = []
        event_counts = []
        cfr80_ci_widths = []

        for _ in range(N_BOOTSTRAP):
            # Bootstrap sample
            idx = rng.choice(n_queries_total, size=N, replace=True)
            sample_risk = b4_risk[idx]
            sample_cfr = cfr[idx]

            # B4 AURC
            b4_aurc = compute_aurc(sample_risk, sample_cfr)

            # Random AURC
            random_risk = rng.rand(N)
            random_aurc = compute_aurc(random_risk, sample_cfr)

            aurc_diff = random_aurc - b4_aurc  # positive = B4 better
            aurc_diffs.append(aurc_diff)

            # CFR@80
            order_b4 = np.argsort(sample_risk)[::-1]
            n_accept = max(1, int(N * 0.8))
            cfr80 = sample_cfr[order_b4[:n_accept]].mean()
            cfr80_diffs.append(cfr80)

            # Event count
            event_counts.append(int(sample_cfr.sum()))

        aurc_diffs = np.array(aurc_diffs)
        cfr80_diffs = np.array(cfr80_diffs)
        event_counts = np.array(event_counts)

        # Power: fraction where AURC diff > 0 (B4 better than random)
        power = float((aurc_diffs > 0).mean())

        # Median effect size
        median_effect = float(np.median(aurc_diffs))

        # CFR@80 CI
        cfr80_ci = np.percentile(cfr80_diffs, [2.5, 97.5])

        # Event count distribution
        event_median = float(np.median(event_counts))
        event_p10 = float(np.percentile(event_counts, 10))
        event_p90 = float(np.percentile(event_counts, 90))

        # Minimum detectable effect (80% power)
        sorted_diffs = np.sort(aurc_diffs)
        mde_80 = float(sorted_diffs[int(N_BOOTSTRAP * 0.2)])  # 20th percentile

        power_results[N] = {
            "n": N,
            "power": power,
            "median_effect": median_effect,
            "cfr80_median": float(np.median(cfr80_diffs)),
            "cfr80_ci_lower": float(cfr80_ci[0]),
            "cfr80_ci_upper": float(cfr80_ci[1]),
            "event_count_median": event_median,
            "event_count_p10": event_p10,
            "event_count_p90": event_p90,
            "mde_80pct_power": mde_80,
            "meets_power_80": power >= 0.80,
            "meets_event_min": event_median >= 10,
        }

        print(f"  N={N}: power={power:.3f}, median_effect={median_effect:.4f}, "
              f"events_median={event_median:.0f}, CFR80_median={cfr80_diffs.mean():.4f}")

    # Recommendation
    min_n_power = min(
        (N for N, r in power_results.items()
         if isinstance(r, dict) and r.get("meets_power_80")),
        default=None
    )
    min_n_events = min(
        (N for N, r in power_results.items()
         if isinstance(r, dict) and r.get("meets_event_min")),
        default=None
    )
    recommended_n = max(min_n_power or 0, min_n_events or 0)

    print(f"\n  Recommended N: {recommended_n} (power >= 0.80: N={min_n_power}, events >= 10: N={min_n_events})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Power curve CSV
    power_df = pd.DataFrame([
        {**{"N": N}, **{k: v for k, v in r.items() if k != "n"}}
        for N, r in power_results.items() if isinstance(r, dict) and "error" not in r
    ])
    power_df.to_csv(OUTPUT_DIR / "power_curve.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'power_curve.csv'}")

    # Effect distribution
    effect_df = pd.DataFrame({
        "aurc_diff": aurc_diffs if 'aurc_diffs' in dir() else [],
    })
    # Save full cohort effect
    full_effect = {
        "n_queries": n_queries_total,
        "b4_aurc": b4_aurc_full,
        "cfr_baseline": float(cfr.mean()),
        "catastrophic_events": int(cfr.sum()),
        "catastrophic_prevalence": float(cfr.mean()),
    }
    with open(OUTPUT_DIR / "development_effect_distribution.json", "w") as f:
        json.dump(full_effect, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'development_effect_distribution.json'}")

    # Event count simulation
    valid_Ns = [N for N in SAMPLE_SIZES if isinstance(power_results[N], dict) and "error" not in power_results[N]]
    event_df = pd.DataFrame({
        "N": valid_Ns,
        "median_events": [power_results[N]["event_count_median"] for N in valid_Ns],
        "p10_events": [power_results[N]["event_count_p10"] for N in valid_Ns],
        "p90_events": [power_results[N]["event_count_p90"] for N in valid_Ns],
    })
    event_df.to_csv(OUTPUT_DIR / "event_count_simulation.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'event_count_simulation.csv'}")

    # Recommendation
    rec = {
        "recommended_n": recommended_n,
        "min_n_for_power_80": min_n_power,
        "min_n_for_events_10": min_n_events,
        "current_n": n_queries_total,
        "can_confirm": recommended_n <= n_queries_total,
        "note": "No fresh cohort available — confirmatory experiment blocked",
    }
    with open(OUTPUT_DIR / "recommended_sample_size.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'recommended_sample_size.json'}")

    # Primary endpoint freeze
    with open(OUTPUT_DIR / "primary_endpoint_freeze.md", "w") as f:
        f.write("# Primary Endpoint Freeze\n\n")
        f.write(f"**Date**: 2026-07-27\n\n")
        f.write(f"## Primary Endpoint\n\n")
        f.write(f"AURC difference vs Random Ranking\n\n")
        f.write(f"## Development Cohort Effect\n\n")
        f.write(f"- N queries: {n_queries_total}\n")
        f.write(f"- B4 AURC: {b4_aurc_full:.4f}\n")
        f.write(f"- CFR baseline: {cfr.mean():.4f}\n")
        f.write(f"- Catastrophic events: {int(cfr.sum())}\n\n")
        f.write(f"## Power Analysis\n\n")
        f.write(f"- Recommended N for 80% power: {min_n_power}\n")
        f.write(f"- Recommended N for >= 10 events: {min_n_events}\n")
        f.write(f"- Recommended minimum N: {recommended_n}\n\n")
        f.write(f"## Status\n\n")
        f.write(f"BLOCKED — No fresh confirmatory cohort available.\n")
    print(f"Saved: {OUTPUT_DIR / 'primary_endpoint_freeze.md'}")

    print("\nWP1b Power Analysis Complete")


if __name__ == "__main__":
    main()