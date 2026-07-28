"""Phase U WP4 Factorial Decomposition: DINOv2-B + SAM2-T cell.

Completes the 2×2 matcher×segmenter matrix:
    DINOv2-L+SAM-B   (WP1)     DINOv2-L+SAM2-T  (WP1)
    DINOv2-B+SAM-B   (WP2)     DINOv2-B+SAM2-T  (this script)

CLI::

    python src/adapromptset/scripts/run_phaseU_wp4_dinov2b_sam2t.py \
        --device cuda --out-dir reports/phaseU_wp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_phaseT_wp1_holdout as wp1
import run_phaseT_wp3_m2_selective as m2
from run_phaseU_wp1_segmenter_audit import _load_sam2_predictor, _Sam2PredictorWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CKPT = WORKSPACE_ROOT / "assets" / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="reports/phaseU_wp4")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase U WP4: DINOv2-B + SAM2-T ===")

    seeds = list(range(5))
    wp1.DINOV2_CHECKPOINT = CKPT / "dinov2_vitb14_pretrain.pth"
    train_manifest = PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv"
    wp1.HOLDOUT_QUERY_MANIFEST = train_manifest
    wp1.PHASET_MANIFEST_DIR = out_dir / "manifests"
    wp1.PHASET_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    wp1.SEED_COUNT = len(seeds)

    # Phase 1
    print("  Phase 1: reference resampling")
    wp1.run_phase1(seeds=seeds)

    # Phase 2: DINOv2-B + SAM2-T
    print("  Phase 2: DINOv2-B + SAM2-T")
    t0 = time.perf_counter()

    # Load SAM2-T predictor
    sam2_predictor = _load_sam2_predictor(
        str(device),
        "sam2.1/sam2.1_hiera_t",
        CKPT / "sam2.1_hiera_tiny.pt",
    )
    import contextlib
    wrapper = _Sam2PredictorWrapper(sam2_predictor, contextlib.nullcontext())
    wp1.load_sam_predictor = lambda d: wrapper

    feature_cache = wp1.FeatureCache(device)
    wp1.run_phase2(
        feature_cache=feature_cache,
        predictor=wrapper,
        out_dir=out_dir,
        seeds=range(len(seeds)),
    )

    elapsed = time.perf_counter() - t0
    print(f"  Phase 2 done in {elapsed:.0f}s")

    # M2 training
    episode_csv = out_dir / "phase2_episode_instability.csv"
    rows = m2.load_episodes(episode_csv)
    print(f"  Episodes: {len(rows)}")

    X = m2.episodes_to_matrix(rows)
    labels = m2.construct_labels(rows)

    # Instability model
    inst_model = m2.M2SklearnModel(
        "M2-3_inst",
        m2.GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=m2.RANDOM_STATE),
    )
    inst_model.fit(X, labels["instability"])
    inst_risk = inst_model.predict_risk(X)
    inst_auroc = float(m2.roc_auc_score(labels["instability"], inst_risk))

    # Catastrophic model
    cat_model = m2.M2SklearnModel(
        "M2-3_cat",
        m2.GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=m2.RANDOM_STATE),
    )
    cat_model.fit(X, labels["catastrophic"])
    cat_risk = cat_model.predict_risk(X)
    cat_auprc = float(m2.average_precision_score(labels["catastrophic"], cat_risk))

    # CFR
    n = len(rows)
    n_accept = max(1, int(n * 0.8))
    cat_order = np.argsort(cat_risk)
    cfr_baseline = float(np.mean(labels["catastrophic"]))
    cfr_accepted = float(np.mean(labels["catastrophic"][cat_order[:n_accept]]))
    cfr_reduction = float((cfr_baseline - cfr_accepted) / cfr_baseline) if cfr_baseline > 0 else 0.0

    result = {
        "pipeline": "DINOv2-B+SAM2-T",
        "n_episodes": len(rows),
        "inst_auroc": inst_auroc,
        "cat_auprc": cat_auprc,
        "prevalence_inst": float(np.mean(labels["instability"])),
        "prevalence_cat": float(np.mean(labels["catastrophic"])),
        "cfr_baseline": cfr_baseline,
        "cfr_accepted": cfr_accepted,
        "cfr_reduction": cfr_reduction,
        "anti_correlation": inst_auroc >= 0.70 and cat_auprc > cfr_baseline,
        "phase2_elapsed_s": elapsed,
    }

    print(f"  Result: {json.dumps(result, indent=2)}")
    with (out_dir / "wp4_dinov2b_sam2t.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()