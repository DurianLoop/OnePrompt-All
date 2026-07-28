"""Phase U WP2 Matcher Sensitivity Audit.

Runs the WP1 pipeline (Phase 1+2+M2) with DINOv2-B instead of DINOv2-L
on SAM-B, to test whether the matcher is the dominant source of instability.

CLI::

    python src/adapromptset/scripts/run_phaseU_wp2_matcher_audit.py \
        --matcher dinov2_b --segmenter sam_b --device cuda \
        --out-dir reports/phaseU_wp2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CKPT = WORKSPACE_ROOT / "assets" / "checkpoints"


def run_phase2_dinov2_b(
    out_dir: Path,
    device: torch.device,
    smoke: bool = False,
) -> Path:
    """Run Phase 2 with DINOv2-B matcher + SAM-B segmenter."""
    seeds = list(range(2 if smoke else 5))
    max_queries = 2 if smoke else None

    wp1.DINOV2_CHECKPOINT = CKPT / "dinov2_vitb14_pretrain.pth"
    train_manifest = PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv"
    wp1.HOLDOUT_QUERY_MANIFEST = train_manifest
    wp1.PHASET_MANIFEST_DIR = out_dir / "manifests"
    wp1.PHASET_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    wp1.SEED_COUNT = len(seeds)

    # Phase 1: reference resampling
    print("  [WP2] Phase 1: reference resampling")
    wp1.run_phase1(seeds=seeds, max_queries=max_queries)

    # Phase 2: DINOv2-B matching + SAM-B segmentation
    print("  [WP2] Phase 2: DINOv2-B + SAM-B")
    t0 = time.perf_counter()

    # Load SAM-B predictor
    wp1.configure_external_paths(WORKSPACE_ROOT)
    from segment_anything import SamPredictor, sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=str(CKPT / "sam_vit_b.pth"))
    model.to(device)
    model.eval()
    predictor = SamPredictor(model)

    feature_cache = wp1.FeatureCache(device)

    wp1.run_phase2(
        feature_cache=feature_cache,
        predictor=predictor,
        out_dir=out_dir,
        seeds=range(len(seeds)),
        max_queries=max_queries,
    )

    elapsed = time.perf_counter() - t0
    episode_csv = out_dir / "phase2_episode_instability.csv"
    print(f"  [WP2] Phase 2 done in {elapsed:.0f}s")

    return episode_csv


def train_and_evaluate_m2(episode_csv: Path, out_dir: Path) -> dict[str, Any]:
    """Train M2 dual-model and evaluate."""
    rows = m2.load_episodes(episode_csv)
    if len(rows) < 50:
        return {"status": "INSUFFICIENT_DATA", "n_rows": len(rows)}

    X = m2.episodes_to_matrix(rows)
    labels = m2.construct_labels(rows)
    y_inst = labels["instability"]
    y_cat = labels["catastrophic"]

    # Train instability model
    inst_model = m2.M2SklearnModel(
        "M2-3_inst",
        m2.GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=m2.RANDOM_STATE,
        ),
    )
    inst_model.fit(X, y_inst)
    inst_risk = inst_model.predict_risk(X)
    inst_auroc = float(m2.roc_auc_score(y_inst, inst_risk))

    # Train catastrophic model
    cat_model = m2.M2SklearnModel(
        "M2-3_cat",
        m2.GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=m2.RANDOM_STATE,
        ),
    )
    cat_model.fit(X, y_cat)
    cat_risk = cat_model.predict_risk(X)
    cat_auprc = float(m2.average_precision_score(y_cat, cat_risk))

    # CFR reduction
    n = len(rows)
    n_accept = max(1, int(n * 0.8))
    cat_order = np.argsort(cat_risk)
    accepted = cat_order[:n_accept]
    cfr_baseline = float(np.mean(y_cat))
    cfr_accepted = float(np.mean(y_cat[accepted]))
    cfr_reduction = float((cfr_baseline - cfr_accepted) / cfr_baseline) if cfr_baseline > 0 else 0.0

    return {
        "status": "OK",
        "n_rows": len(rows),
        "inst_auroc": inst_auroc,
        "cat_auprc": cat_auprc,
        "prevalence_inst": float(np.mean(y_inst)),
        "prevalence_cat": float(np.mean(y_cat)),
        "cfr_baseline": cfr_baseline,
        "cfr_accepted": cfr_accepted,
        "cfr_reduction": cfr_reduction,
        "anti_correlation": inst_auroc >= 0.70 and cat_auprc > float(np.mean(y_cat)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase U WP2 Matcher Sensitivity Audit")
    parser.add_argument("--matcher", type=str, default="dinov2_b")
    parser.add_argument("--segmenter", type=str, default="sam_b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="reports/phaseU_wp2")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase U WP2: Matcher Sensitivity Audit ===")
    print(f"Matcher: {args.matcher} | Segmenter: {args.segmenter} | Device: {device}")
    print()

    # Run Phase 2 with DINOv2-B
    episode_csv = run_phase2_dinov2_b(out_dir, device, smoke=args.smoke)

    if args.smoke:
        print("Smoke test complete, skipping M2")
        return

    # Train M2
    print("  [WP2] M2 dual-model training")
    result = train_and_evaluate_m2(episode_csv, out_dir)
    print(f"  Result: {json.dumps(result, indent=2)}")

    with (out_dir / "wp2_matcher_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Summary written to {out_dir / 'wp2_matcher_summary.json'}")


if __name__ == "__main__":
    main()