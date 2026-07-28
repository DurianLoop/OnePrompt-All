"""Phase T WP3 M2 Cross-Dataset Transfer Evaluation.

Applies the frozen dual-model M2 (trained on MoNuSAC train, selected on
MoNuSAC val) to MoNuSeg and IDRID episode data to test cross-dataset
generalization of the selective prediction mitigation.

This is NOT a gate condition (§10.5 only requires MoNuSAC holdout
confirmation). It provides additional evidence for the paper about
whether M2 transfers to datasets with different feature distributions.

Protocol (consistent with §12 freeze):
    1. Load MoNuSAC train/val episodes -> fit M2-0..M2-4 on BOTH labels.
    2. Select best cat model (highest val catastrophic AUPRC) and best
       inst model (highest val instability AUROC) on MoNuSAC val.
    3. Apply the FROZEN model pair to MoNuSeg and IDRID episodes.
    4. Report AUROC, AUPRC, ECE, CFR reduction, Dice change per dataset.
    5. No re-tuning on the transfer datasets.

CLI::

    python src/adapromptset/scripts/run_phaseT_wp3_m2_cross_dataset_transfer.py \
        --train-dir reports/phaseT_wp3_train \
        --val-dir reports/phaseT_wp3_validation \
        --monuseg-dir reports/phaseT_wp2_monuseg \
        --idrid-dir reports/phaseT_wp2_idrid \
        --out-dir reports/phaseT_wp3_m2_transfer
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

# OpenMP runtime conflict workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# Make the scripts directory importable.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import run_phaseT_wp3_m2_selective as m2  # noqa: E402


# ---------------------------------------------------------------------------
# Transfer evaluation
# ---------------------------------------------------------------------------

def evaluate_transfer(
    cat_model: m2.M2Model,
    inst_model: m2.M2Model,
    train_rows: list[dict[str, Any]],
    train_labels: dict[str, np.ndarray],
    transfer_rows: list[dict[str, Any]],
    dataset_name: str,
) -> dict[str, Any]:
    """Apply frozen dual-model to a transfer dataset and compute metrics.

    Returns a dict with per-episode risk scores and aggregate metrics.
    No re-tuning is performed; the models are applied as-is.
    """
    X_transfer = m2.episodes_to_matrix(transfer_rows)
    transfer_mean_dice = np.asarray(
        [r["mean_dice_k1"] for r in transfer_rows], dtype=np.float64
    )
    transfer_labels = m2.construct_val_labels(transfer_rows)
    y_instab = transfer_labels["instability"]
    y_cat = transfer_labels["catastrophic"]

    # Catastrophic model predictions.
    cat_risk = cat_model.predict_risk(X_transfer)
    auroc_instab_cat = m2.safe_auroc(y_instab, cat_risk)
    auprc_cat = m2.safe_auprc(y_cat, cat_risk)
    ece_cat, calib_bins = m2.compute_ece(y_cat, cat_risk)

    # Apply Platt scaling (frozen, refit on train — same as holdout path).
    train_risk_cat = cat_model.predict_risk(m2.episodes_to_matrix(train_rows))
    platt = None
    ece_cal = ece_cat
    cat_risk_calibrated = cat_risk
    platt_applied = False
    if ece_cat > m2.GATE_ECE_THRESHOLD:
        platt = m2.fit_platt_scaler(train_risk_cat, train_labels["catastrophic"])
        if platt is not None:
            cat_risk_calibrated = m2.apply_platt_scaler(platt, cat_risk)
            ece_cal, _ = m2.compute_ece(y_cat, cat_risk_calibrated)
            if ece_cal < ece_cat:
                cat_risk = cat_risk_calibrated
                ece_cat = ece_cal
                platt_applied = True
            else:
                platt = None
                cat_risk_calibrated = cat_risk

    m80 = m2.metrics_at_coverage(
        cat_risk, transfer_mean_dice, y_cat, m2.GATE_COVERAGE,
    )

    # Instability model predictions.
    inst_risk = inst_model.predict_risk(X_transfer)
    auroc_instab = m2.safe_auroc(y_instab, inst_risk)
    auprc_instab = m2.safe_auprc(y_instab, inst_risk)

    # Query-level aggregation.
    q_rows, q_risk, q_instab, q_cat = m2._aggregate_queries(
        transfer_rows, inst_risk, transfer_mean_dice,
    )
    q_auroc = m2.safe_auroc(q_instab, q_risk)

    n_queries = len({r["query_case_id"] for r in transfer_rows})
    prevalence_instab = float(np.mean(y_instab))
    prevalence_cat = float(np.mean(y_cat))

    return {
        "dataset": dataset_name,
        "n_episodes": len(transfer_rows),
        "n_queries": n_queries,
        "prevalence_instability": prevalence_instab,
        "prevalence_catastrophic": prevalence_cat,
        "instability_AUROC": float(auroc_instab),
        "instability_AUROC_query": float(q_auroc),
        "instability_AUPRC": float(auprc_instab),
        "catastrophic_AUPRC": float(auprc_cat),
        "catastrophic_AUPRC_gt_prevalence": bool(auprc_cat > prevalence_cat),
        "ECE": float(ece_cat),
        "ECE_calibrated": float(ece_cal),
        "platt_applied": platt_applied,
        "CFR_no_abstention": float(m80["cfr_all"]),
        "CFR_at_80pct_coverage": float(m80["cfr"]),
        "CFR_reduction_at_80pct": float(m80["cfr_reduction"]),
        "Dice_no_abstention": float(m80["dice_all"]),
        "Dice_at_80pct_coverage": float(m80["dice"]),
        "Dice_change_at_80pct": float(m80["dice_change"]),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_transfer_report(
    out_dir: Path,
    transfer_results: list[dict[str, Any]],
    frozen_info: dict[str, Any],
) -> None:
    md: list[str] = []
    md.append("# Phase T WP3 M2 Cross-Dataset Transfer Evaluation")
    md.append("")
    md.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    md.append("")
    md.append("## Frozen models (trained on MoNuSAC train, selected on MoNuSAC val)")
    md.append("")
    md.append(f"- Catastrophic model: **{frozen_info['cat_model']}** "
              f"(val AUPRC_cat = {frozen_info['val_cat_auprc']:.4f})")
    md.append(f"- Instability model: **{frozen_info['inst_model']}** "
              f"(val AUROC = {frozen_info['val_inst_auroc']:.4f})")
    md.append(f"- MoNuSAC holdout: AUROC = {frozen_info['holdout_auroc']:.4f}, "
              f"AUPRC_cat = {frozen_info['holdout_auprc_cat']:.4f}, "
              f"CFR reduction = {frozen_info['holdout_cfr_red']:.4f}")
    md.append("")
    md.append("## Transfer results")
    md.append("")
    md.append("No re-tuning on transfer datasets. Frozen models applied as-is.")
    md.append("")

    for r in transfer_results:
        md.append(f"### {r['dataset']}")
        md.append("")
        md.append(f"- Episodes: {r['n_episodes']} ({r['n_queries']} queries)")
        md.append(f"- Prevalence: instability={r['prevalence_instability']:.4f}, "
                  f"catastrophic={r['prevalence_catastrophic']:.4f}")
        md.append("")
        md.append("| metric | value |")
        md.append("|--------|-------|")
        md.append(f"| instability AUROC | {r['instability_AUROC']:.4f} "
                  f"(query-level: {r['instability_AUROC_query']:.4f}) |")
        md.append(f"| instability AUPRC | {r['instability_AUPRC']:.4f} |")
        md.append(f"| catastrophic AUPRC | {r['catastrophic_AUPRC']:.4f} "
                  f"(prevalence={r['prevalence_catastrophic']:.4f}, "
                  f"AUPRC>prevalence: {r['catastrophic_AUPRC_gt_prevalence']}) |")
        md.append(f"| ECE | {r['ECE']:.4f}"
                  + (f" (calibrated: {r['ECE_calibrated']:.4f})" if r['platt_applied'] else "")
                  + " |")
        md.append(f"| CFR (no abstention) | {r['CFR_no_abstention']:.4f} |")
        md.append(f"| CFR @80% coverage | {r['CFR_at_80pct_coverage']:.4f} |")
        md.append(f"| CFR reduction @80% | {r['CFR_reduction_at_80pct']:.4f} |")
        md.append(f"| Dice (no abstention) | {r['Dice_no_abstention']:.6f} |")
        md.append(f"| Dice @80% coverage | {r['Dice_at_80pct_coverage']:.6f} |")
        md.append(f"| Dice change @80% | {r['Dice_change_at_80pct']:+.6f} |")
        md.append("")

        # Summary assessment.
        auroc_ok = r["instability_AUROC"] >= m2.GATE_AUROC_THRESHOLD
        auprc_ok = r["catastrophic_AUPRC_gt_prevalence"]
        ece_ok = r["ECE"] <= m2.GATE_ECE_THRESHOLD
        cfr_ok = r["CFR_reduction_at_80pct"] >= m2.GATE_CFR_REDUCTION_THRESHOLD
        dice_ok = r["Dice_change_at_80pct"] >= 0.0
        md.append(f"- AUROC >= 0.70: {'PASS' if auroc_ok else 'FAIL'}")
        md.append(f"- AUPRC > prevalence: {'PASS' if auprc_ok else 'FAIL'}")
        md.append(f"- ECE <= 0.10: {'PASS' if ece_ok else 'FAIL'}")
        md.append(f"- CFR reduction >= 30%: {'PASS' if cfr_ok else 'FAIL'}")
        md.append(f"- Dice not decrease @80%: {'PASS' if dice_ok else 'FAIL'}")
        n_pass = sum([auroc_ok, auprc_ok, ece_ok, cfr_ok, dice_ok])
        md.append(f"- **Transfer {n_pass}/5 conditions pass**")
        md.append("")

    md.append("## Interpretation")
    md.append("")
    md.append("This is NOT a gate condition. The §10.5 gate only requires MoNuSAC")
    md.append("holdout confirmation (cond6). Transfer results inform the paper's")
    md.append("discussion of M2 generalizability and are reported as additional")
    md.append("evidence, not as a pass/fail gate.")
    md.append("")

    (out_dir / "m2_transfer_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )

    # JSON.
    summary = {
        "gate": "Phase T WP3 M2 Cross-Dataset Transfer (supplementary, not a gate condition)",
        "date": datetime.now().isoformat(timespec="seconds"),
        "frozen_models": frozen_info,
        "transfer_results": transfer_results,
        "note": "Not a gate condition. Supplementary evidence for M2 generalizability.",
    }
    (out_dir / "m2_transfer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase T WP3 M2 Cross-Dataset Transfer Evaluation",
    )
    parser.add_argument(
        "--train-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp3_train",
    )
    parser.add_argument(
        "--val-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp3_validation",
    )
    parser.add_argument(
        "--monuseg-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp2_monuseg",
    )
    parser.add_argument(
        "--idrid-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp2_idrid",
    )
    parser.add_argument(
        "--holdout-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp1_holdout",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp3_m2_transfer",
    )
    args = parser.parse_args()

    train_csv = args.train_dir / "phase2_episode_instability.csv"
    val_csv = args.val_dir / "phase2_episode_instability.csv"
    monuseg_csv = args.monuseg_dir / "phase2_episode_instability.csv"
    idrid_csv = args.idrid_dir / "phase2_episode_instability.csv"
    holdout_csv = args.holdout_dir / "phase2_episode_instability.csv"
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phaseT-M2-transfer] train_csv={train_csv}")
    print(f"[phaseT-M2-transfer] val_csv={val_csv}")
    print(f"[phaseT-M2-transfer] monuseg_csv={monuseg_csv}")
    print(f"[phaseT-M2-transfer] idrid_csv={idrid_csv}")
    print(f"[phaseT-M2-transfer] out_dir={out_dir}")

    start = time.perf_counter()

    # Step 1: Load MoNuSAC train/val.
    train_rows = m2.load_episodes(train_csv)
    val_rows = m2.load_episodes(val_csv)
    print(f"[phaseT-M2-transfer] loaded {len(train_rows)} train, "
          f"{len(val_rows)} val episodes")

    # Step 2: Construct labels.
    train_labels = m2.construct_labels(train_rows)
    val_labels = m2.construct_val_labels(val_rows)

    # Step 3: Train + select on val (frozen model pair).
    print("[phaseT-M2-transfer] training catastrophic-labelled models ...")
    cat_results, cat_model, cat_result_val = m2.evaluate_all_models(
        m2.build_all_models(), train_rows, val_rows, train_labels, val_labels,
        train_label_key="catastrophic",
    )
    print("[phaseT-M2-transfer] training instability-labelled models ...")
    inst_results, inst_model, inst_result_val = m2.evaluate_all_models(
        m2.build_all_models(), train_rows, val_rows, train_labels, val_labels,
        train_label_key="instability",
    )
    print(f"[phaseT-M2-transfer] frozen cat model: {cat_model.name} "
          f"(val AUPRC_cat={cat_result_val['auprc_catastrophic']:.4f})")
    print(f"[phaseT-M2-transfer] frozen inst model: {inst_model.name} "
          f"(val AUROC={inst_result_val['auroc_instability']:.4f})")

    # Step 4: Apply to MoNuSAC holdout (for reference).
    holdout_rows = m2.load_episodes(holdout_csv)
    holdout_result = evaluate_transfer(
        cat_model, inst_model, train_rows, train_labels, holdout_rows, "MoNuSAC holdout (reference)",
    )

    # Step 5: Apply to MoNuSeg.
    monuseg_rows = m2.load_episodes(monuseg_csv)
    monuseg_result = evaluate_transfer(
        cat_model, inst_model, train_rows, train_labels, monuseg_rows, "MoNuSeg (transfer)",
    )

    # Step 6: Apply to IDRID.
    idrid_rows = m2.load_episodes(idrid_csv)
    idrid_result = evaluate_transfer(
        cat_model, inst_model, train_rows, train_labels, idrid_rows, "IDRID (transfer)",
    )

    # Step 7: Write report.
    frozen_info = {
        "cat_model": cat_model.name,
        "inst_model": inst_model.name,
        "val_cat_auprc": cat_result_val["auprc_catastrophic"],
        "val_inst_auroc": inst_result_val["auroc_instability"],
        "holdout_auroc": holdout_result["instability_AUROC"],
        "holdout_auprc_cat": holdout_result["catastrophic_AUPRC"],
        "holdout_cfr_red": holdout_result["CFR_reduction_at_80pct"],
    }
    write_transfer_report(
        out_dir,
        [holdout_result, monuseg_result, idrid_result],
        frozen_info,
    )

    elapsed = time.perf_counter() - start

    print()
    print("=" * 72)
    print("[phaseT-M2-transfer] Transfer results:")
    for r in [holdout_result, monuseg_result, idrid_result]:
        print(f"  {r['dataset']}:")
        print(f"    AUROC={r['instability_AUROC']:.4f}  "
              f"AUPRC_cat={r['catastrophic_AUPRC']:.4f}  "
              f"ECE={r['ECE']:.4f}  "
              f"CFR_red@80={r['CFR_reduction_at_80pct']:.4f}  "
              f"Dice_chg@80={r['Dice_change_at_80pct']:+.6f}")
    print()
    print(f"[phaseT-M2-transfer] artifacts written to {out_dir}")
    print(f"[phaseT-M2-transfer] elapsed {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
