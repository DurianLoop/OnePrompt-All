"""Phase U Selective Prediction Baselines.

Compares M2 rank-based selective prediction against:
    1. Random abstention (lower bound)
    2. Oracle abstention (upper bound - uses true Dice)
    3. Single-feature baselines (matching_map_disagreement only)
    4. M2-3 gradient boosting (our method)

All baselines are evaluated on the same coverage-risk curve across all
WP1 models (SAM-B/L/SAM2-T) and WP2 matcher (DINOv2-B/SAM-B).

Output:
    - Coverage-risk curves for each baseline
    - CFR reduction comparison table
    - Coverage-Dice trade-off curves

CLI::

    python src/adapromptset/scripts/run_phaseU_selective_prediction_baselines.py \
        --episode-dirs reports/phaseU_wp1/sam_b,reports/phaseU_wp1/sam_l,reports/phaseU_wp1/sam2_t,reports/phaseU_wp2 \
        --labels "DINOv2-L+SAM-B","DINOv2-L+SAM-L","DINOv2-L+SAM2-T","DINOv2-B+SAM-B" \
        --out-dir reports/phaseU_baselines
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_phaseT_wp3_m2_selective as m2  # noqa: E402

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def baseline_random(n: int, seed: int = 42) -> np.ndarray:
    """Random risk scores in [0, 1]."""
    rng = np.random.RandomState(seed)
    return rng.rand(n).astype(np.float64)


def baseline_oracle(dice: np.ndarray) -> np.ndarray:
    """Oracle: use 1 - Dice as risk score (upper bound)."""
    return 1.0 - np.clip(dice, 0.0, 1.0)


def baseline_single_feature(X: np.ndarray, feature_idx: int) -> np.ndarray:
    """Use a single feature as risk score (min-max normalized)."""
    feat = X[:, feature_idx].astype(np.float64)
    fmin, fmax = feat.min(), feat.max()
    if fmax <= fmin:
        return np.zeros_like(feat)
    return (feat - fmin) / (fmax - fmin)


def compute_coverage_risk_curve(
    risk_scores: np.ndarray,
    y_cat: np.ndarray,
    dice: np.ndarray,
    n_points: int = 10,
) -> dict[str, Any]:
    """Compute coverage-risk and coverage-Dice curves.

    For each coverage level (50% .. 90%):
    1. Rank by risk score (ascending = less risky)
    2. Accept top coverage% of episodes
    3. Compute CFR (catastrophic failure rate) and mean Dice among accepted
    """
    n = len(risk_scores)
    baseline_cfr = float(np.mean(y_cat))
    baseline_dice = float(np.mean(dice))

    coverages = np.linspace(0.5, 0.9, n_points)
    cfr_curve: list[float] = []
    dice_curve: list[float] = []
    cfr_reduction: list[float] = []

    order = np.argsort(risk_scores)  # ascending risk
    for cov in coverages:
        n_accept = max(1, int(n * cov))
        accepted = order[:n_accept]
        cfr = float(np.mean(y_cat[accepted]))
        mean_dice = float(np.mean(dice[accepted]))
        cfr_curve.append(cfr)
        dice_curve.append(mean_dice)
        cfr_reduction.append(float((baseline_cfr - cfr) / baseline_cfr) if baseline_cfr > 0 else 0.0)

    return {
        "coverages": coverages.tolist(),
        "cfr_curve": cfr_curve,
        "dice_curve": dice_curve,
        "cfr_reduction": cfr_reduction,
        "baseline_cfr": baseline_cfr,
        "baseline_dice": baseline_dice,
    }


def train_m2_model(X: np.ndarray, y: np.ndarray, label_type: str) -> Any:
    """Train M2-3 gradient boosting on the given label."""
    model = m2.M2SklearnModel(
        f"M2-3_{label_type}",
        m2.GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=m2.RANDOM_STATE,
        ),
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase U Selective Prediction Baselines")
    parser.add_argument(
        "--episode-dirs", type=str, required=True,
        help="Comma-separated episode directories",
    )
    parser.add_argument(
        "--labels", type=str, required=True,
        help="Comma-separated labels for each pipeline",
    )
    parser.add_argument(
        "--out-dir", type=str, default="reports/phaseU_baselines",
    )
    args = parser.parse_args()

    dirs = [Path(d.strip()) for d in args.episode_dirs.split(",")]
    labels = [l.strip() for l in args.labels.split(",")]

    assert len(dirs) == len(labels), f"Got {len(dirs)} dirs but {len(labels)} labels"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    for pipeline_dir, pipeline_label in zip(dirs, labels):
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline_label}")
        print(f"Dir: {pipeline_dir}")

        csv_path = pipeline_dir / "phase2_episode_instability.csv"
        if not csv_path.is_file():
            print(f"  SKIP: no episode CSV")
            continue

        rows = m2.load_episodes(csv_path)
        X = m2.episodes_to_matrix(rows)
        labels_data = m2.construct_labels(rows)
        y_cat = labels_data["catastrophic"]
        y_inst = labels_data["instability"]
        dice = np.asarray([r["mean_dice_k1"] for r in rows], dtype=np.float64)
        n = len(rows)

        print(f"  Episodes: {n}")
        print(f"  Baseline CFR: {np.mean(y_cat):.4f}")
        print(f"  Baseline Dice: {np.mean(dice):.4f}")

        # --- Train M2 models ---
        print(f"  Training M2 instability model...")
        inst_model = train_m2_model(X, y_inst, "instability")
        inst_risk = inst_model.predict_risk(X)

        print(f"  Training M2 catastrophic model...")
        cat_model = train_m2_model(X, y_cat, "catastrophic")
        cat_risk = cat_model.predict_risk(X)

        # --- Compute all baselines ---
        baselines: dict[str, dict[str, Any]] = {}

        # Random
        print(f"  Computing random baseline...")
        random_risk = baseline_random(n)
        baselines["random"] = compute_coverage_risk_curve(random_risk, y_cat, dice)

        # Oracle
        print(f"  Computing oracle baseline...")
        oracle_risk = baseline_oracle(dice)
        baselines["oracle"] = compute_coverage_risk_curve(oracle_risk, y_cat, dice)

        # Single feature: matching_map_disagreement (index 0)
        print(f"  Computing single-feature baseline (matching_map_disagreement)...")
        feat_risk = baseline_single_feature(X, 0)
        baselines["single_feat_disagreement"] = compute_coverage_risk_curve(feat_risk, y_cat, dice)

        # Single feature: prediction_jaccard_k1_within (index 3)
        pred_risk = baseline_single_feature(X, 3)
        baselines["single_feat_jaccard"] = compute_coverage_risk_curve(pred_risk, y_cat, dice)

        # M2 instability model
        print(f"  Computing M2 instability baseline...")
        baselines["m2_instability"] = compute_coverage_risk_curve(inst_risk, y_cat, dice)

        # M2 catastrophic model
        print(f"  Computing M2 catastrophic baseline...")
        baselines["m2_catastrophic"] = compute_coverage_risk_curve(cat_risk, y_cat, dice)

        # --- Print comparison table ---
        print(f"\n  {'Baseline':<30} {'CFR@80%':>8} {'CFR_red@80%':>12} {'Dice@80%':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*8}")
        for bname, bdata in baselines.items():
            cfr80 = bdata["cfr_curve"][-2]  # 80% coverage = index -2
            cfr_red80 = bdata["cfr_reduction"][-2]
            dice80 = bdata["dice_curve"][-2]
            print(f"  {bname:<30} {cfr80:>8.4f} {cfr_red80:>12.4f} {dice80:>8.4f}")

        all_results[pipeline_label] = {
            "n_episodes": n,
            "baseline_cfr": float(np.mean(y_cat)),
            "baseline_dice": float(np.mean(dice)),
            "baselines": {
                bname: {
                    "coverages": bdata["coverages"],
                    "cfr_curve": bdata["cfr_curve"],
                    "dice_curve": bdata["dice_curve"],
                    "cfr_reduction": bdata["cfr_reduction"],
                }
                for bname, bdata in baselines.items()
            },
        }

    # --- Write summary ---
    summary = {
        "phase": "PhaseU_Baselines",
        "timestamp": datetime.now().isoformat(),
        "pipelines": all_results,
    }

    with (out_dir / "baselines_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary written to {out_dir / 'baselines_summary.json'}")

    # --- Print final comparison ---
    print(f"\n{'='*120}")
    print(f"FINAL COMPARISON: CFR Reduction @80% Coverage")
    print(f"{'Pipeline':<30}", end="")
    for bname in ["random", "single_feat_disagreement", "single_feat_jaccard", "m2_instability", "m2_catastrophic", "oracle"]:
        print(f" {bname:>20}", end="")
    print()

    for pipeline_label, pdata in all_results.items():
        print(f"{pipeline_label:<30}", end="")
        for bname in ["random", "single_feat_disagreement", "single_feat_jaccard", "m2_instability", "m2_catastrophic", "oracle"]:
            if bname in pdata["baselines"]:
                val = pdata["baselines"][bname]["cfr_reduction"][-2]
                print(f" {val:>20.4f}", end="")
            else:
                print(f" {'N/A':>20}", end="")
        print()


if __name__ == "__main__":
    main()