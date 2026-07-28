"""Phase U WP5 Cross-Pipeline M2 Transfer.

Trains M2 dual-model on one source pipeline (e.g. SAM-B + DINOv2-L) and
evaluates rank-based selective prediction on target pipelines (SAM-L, SAM2-T)
without any target-domain recalibration. This tests whether the anti-correlation
is pipeline-agnostic and whether M2 risk scores can transfer cross-model.

The key innovation: rank-based abstention (not threshold-based) means we only
need risk ordering to be consistent, not risk calibration. This eliminates the
calibration drift problem that Phase T found on IDRID.

CLI::

    python src/adapromptset/scripts/run_phaseU_wp5_cross_pipeline_transfer.py \
        --source sam_b --targets sam_l,sam2_t \
        --episode-dir reports/phaseU_wp1 \
        --out-dir reports/phaseU_wp5

Output metrics per (source -> target) pair:
    - AUROC (instability model on target)
    - AUPRC (catastrophic model on target)
    - Coverage-risk curve (CFR @ 50%, 60%, 70%, 80%, 90% coverage)
    - CFR reduction relative to no-abstention baseline
    - Spearman rank correlation between source and target risk scores
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
from scipy.stats import spearmanr  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_KEYS = ["sam_b", "sam_l", "sam2_t"]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_episodes_for_model(episode_dir: Path, model_key: str) -> list[dict[str, Any]]:
    """Load episode CSV for a model from the WP1 output directory."""
    csv_path = episode_dir / model_key / "phase2_episode_instability.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Episode CSV not found: {csv_path}")
    return m2.load_episodes(csv_path)


def train_m2_dual(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Train M2 dual-model (instability + catastrophic) on episode data."""
    X = m2.episodes_to_matrix(rows)
    labels = m2.construct_labels(rows)

    y_inst = labels["instability"]
    y_cat = labels["catastrophic"]

    # M2-3 Gradient Boosting (best model from Phase T)
    inst_model = m2.M2SklearnModel(
        "M2-3_instability",
        m2.GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=m2.RANDOM_STATE,
        ),
    )
    inst_model.fit(X, y_inst)

    cat_model = m2.M2SklearnModel(
        "M2-3_catastrophic",
        m2.GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=m2.RANDOM_STATE,
        ),
    )
    cat_model.fit(X, y_cat)

    inst_risk = inst_model.predict_risk(X)
    cat_risk = cat_model.predict_risk(X)

    inst_auroc = float(m2.roc_auc_score(y_inst, inst_risk))
    cat_auprc = float(m2.average_precision_score(y_cat, cat_risk))

    prevalence_inst = float(np.mean(y_inst))
    prevalence_cat = float(np.mean(y_cat))

    return {
        "model": {
            "instability": inst_model,
            "catastrophic": cat_model,
        },
        "train_metrics": {
            "inst_auroc": inst_auroc,
            "cat_auprc": cat_auprc,
            "prevalence_inst": prevalence_inst,
            "prevalence_cat": prevalence_cat,
            "n_episodes": len(rows),
        },
    }


def evaluate_transfer(
    trained: dict[str, Any],
    target_rows: list[dict[str, Any]],
    target_name: str,
    n_coverage_points: int = 10,
) -> dict[str, Any]:
    """Evaluate rank-based selective prediction on a target pipeline.

    For each coverage level (50% .. 90%), we:
    1. Rank episodes by risk score (from trained model)
    2. Accept the top coverage% of episodes
    3. Compute CFR (catastrophic failure rate) among accepted episodes
    4. Compare to no-abstention CFR
    5. Also compute per-episode Dice (mean of k=1 references)
    """
    X_target = m2.episodes_to_matrix(target_rows)
    labels_target = m2.construct_val_labels(target_rows)
    y_cat_target = labels_target["catastrophic"]
    y_inst_target = labels_target["instability"]

    inst_model = trained["model"]["instability"]
    cat_model = trained["model"]["catastrophic"]

    inst_risk = inst_model.predict_risk(X_target)
    cat_risk = cat_model.predict_risk(X_target)

    # AUROC / AUPRC on target (rank-based, no calibration)
    inst_auroc = float(m2.roc_auc_score(y_inst_target, inst_risk))
    cat_auprc = float(m2.average_precision_score(y_cat_target, cat_risk))

    # Spearman rank correlation between source risk and target DICE
    target_dice = np.asarray([r["mean_dice_k1"] for r in target_rows], dtype=np.float64)
    spearman_cat = float(spearmanr(cat_risk, target_dice)[0])
    spearman_inst = float(spearmanr(inst_risk, target_dice)[0])

    # Coverage-risk curve
    baseline_cfr = float(np.mean(y_cat_target))
    n = len(target_rows)
    coverages = np.linspace(0.5, 0.9, n_coverage_points)
    cfr_curve: dict[str, list[float]] = {
        "coverage": [],
        "cfr_inst": [],  # CFR when using instability model ranking
        "cfr_cat": [],   # CFR when using catastrophic model ranking
        "cfr_reduction_inst": [],
        "cfr_reduction_cat": [],
        "mean_dice_inst": [],
        "mean_dice_cat": [],
    }

    for cov in coverages:
        n_accept = max(1, int(n * cov))

        # By instability risk ranking (accept low risk)
        inst_order = np.argsort(inst_risk)
        accepted_inst = inst_order[:n_accept]
        cfr_inst = float(np.mean(y_cat_target[accepted_inst]))
        mean_dice_inst = float(np.mean(target_dice[accepted_inst]))

        # By catastrophic risk ranking (accept low risk)
        cat_order = np.argsort(cat_risk)
        accepted_cat = cat_order[:n_accept]
        cfr_cat = float(np.mean(y_cat_target[accepted_cat]))
        mean_dice_cat = float(np.mean(target_dice[accepted_cat]))

        cfr_curve["coverage"].append(float(cov))
        cfr_curve["cfr_inst"].append(cfr_inst)
        cfr_curve["cfr_cat"].append(cfr_cat)
        cfr_curve["cfr_reduction_inst"].append(
            float((baseline_cfr - cfr_inst) / baseline_cfr) if baseline_cfr > 0 else 0.0
        )
        cfr_curve["cfr_reduction_cat"].append(
            float((baseline_cfr - cfr_cat) / baseline_cfr) if baseline_cfr > 0 else 0.0
        )
        cfr_curve["mean_dice_inst"].append(mean_dice_inst)
        cfr_curve["mean_dice_cat"].append(mean_dice_cat)

    prevalence_inst = float(np.mean(y_inst_target))
    prevalence_cat = float(np.mean(y_cat_target))

    return {
        "target": target_name,
        "n_episodes": n,
        "baseline_cfr": baseline_cfr,
        "inst_auroc": inst_auroc,
        "cat_auprc": cat_auprc,
        "spearman_cat": spearman_cat,
        "spearman_inst": spearman_inst,
        "prevalence_inst": prevalence_inst,
        "prevalence_cat": prevalence_cat,
        "cfr_curve": cfr_curve,
        "gate1_inst": inst_auroc >= m2.GATE_AUROC_THRESHOLD,
        "gate2_cat": cat_auprc > prevalence_cat,
        "cfr80_reduction_cat": cfr_curve["cfr_reduction_cat"][-2],  # 80% coverage
        "gate3_cfr80": cfr_curve["cfr_reduction_cat"][-2] >= m2.GATE_CFR_REDUCTION_THRESHOLD,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase U WP5 Cross-Pipeline M2 Transfer")
    parser.add_argument(
        "--source", type=str, default="sam_b",
        choices=MODEL_KEYS,
        help="Source model for M2 training",
    )
    parser.add_argument(
        "--targets", type=str, default="sam_l,sam2_t",
        help="Comma-separated target model keys",
    )
    parser.add_argument(
        "--episode-dir", type=str, default="reports/phaseU_wp1",
        help="Directory containing WP1 episode CSVs",
    )
    parser.add_argument(
        "--out-dir", type=str, default="reports/phaseU_wp5",
        help="Output directory",
    )
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_keys = [t.strip() for t in args.targets.split(",")]

    print(f"=== Phase U WP5: Cross-Pipeline M2 Transfer ===")
    print(f"Source: {args.source}")
    print(f"Targets: {target_keys}")
    print(f"Episode dir: {episode_dir}")
    print()

    # --- Load source episodes ---
    print(f"[1/3] Loading source episodes: {args.source}")
    source_rows = load_episodes_for_model(episode_dir, args.source)
    print(f"  Loaded {len(source_rows)} episodes")

    # --- Train M2 dual-model on source ---
    print(f"[2/3] Training M2 dual-model on source")
    trained = train_m2_dual(source_rows)
    tm = trained["train_metrics"]
    print(f"  Instability model: AUROC={tm['inst_auroc']:.4f}, prevalence={tm['prevalence_inst']:.4f}")
    print(f"  Catastrophic model: AUPRC={tm['cat_auprc']:.4f}, prevalence={tm['prevalence_cat']:.4f}")

    # --- Evaluate transfer to each target ---
    print(f"[3/3] Evaluating transfer to targets")
    all_results: list[dict[str, Any]] = []
    for target_key in target_keys:
        print(f"  --- {args.source} -> {target_key} ---")
        target_rows = load_episodes_for_model(episode_dir, target_key)
        result = evaluate_transfer(trained, target_rows, target_key)
        all_results.append(result)

        print(f"    Target episodes: {result['n_episodes']}")
        print(f"    Baseline CFR: {result['baseline_cfr']:.4f}")
        print(f"    Inst AURAC: {result['inst_auroc']:.4f} (gate: {result['gate1_inst']})")
        print(f"    Cat AUPRC: {result['cat_auprc']:.4f} (gate: {result['gate2_cat']})")
        print(f"    Spearman (cat vs dice): {result['spearman_cat']:.4f}")
        print(f"    Spearman (inst vs dice): {result['spearman_inst']:.4f}")
        print(f"    CFR reduction @80%: {result['cfr80_reduction_cat']:.4f} ({result['cfr80_reduction_cat']*100:.1f}%)")
        print(f"    Gate3 CFR80: {result['gate3_cfr80']}")

        # Print coverage-risk curve
        cc = result["cfr_curve"]
        print(f"    Coverage-risk curve (catastrophic model):")
        for i in range(len(cc["coverage"])):
            print(
                f"      cov={cc['coverage'][i]:.1f}  CFR={cc['cfr_cat'][i]:.4f}  "
                f"CFR_red={cc['cfr_reduction_cat'][i]:.4f}  Dice={cc['mean_dice_cat'][i]:.4f}"
            )
        print()

    # --- Write summary ---
    summary = {
        "phase": "PhaseU_WP5",
        "timestamp": datetime.now().isoformat(),
        "source": args.source,
        "train_metrics": tm,
        "targets": all_results,
    }

    with (out_dir / "wp5_cross_pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Summary written to {out_dir / 'wp5_cross_pipeline_summary.json'}")

    # --- Final verdict ---
    gates_pass = all(
        r["gate1_inst"] and r["gate2_cat"] and r["gate3_cfr80"]
        for r in all_results
    )
    if gates_pass:
        print("\n=== WP5 RESULT: CROSS_PIPELINE_TRANSFER_SUCCESS ===")
        print("M2 rank-based selective prediction transfers without recalibration!")
    else:
        failed = [
            r["target"] for r in all_results
            if not (r["gate1_inst"] and r["gate2_cat"] and r["gate3_cfr80"])
        ]
        print(f"\n=== WP5 RESULT: CROSS_PIPELINE_TRANSFER_PARTIAL ===")
        print(f"Failed targets: {failed}")


if __name__ == "__main__":
    main()