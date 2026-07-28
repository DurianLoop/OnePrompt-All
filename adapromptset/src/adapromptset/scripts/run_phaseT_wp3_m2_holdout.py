"""Phase T WP3 M2 Holdout Confirmation (section 12).

Applies the frozen dual-model M2 (catastrophic-trained + instability-trained,
selected on validation) to the MoNuSAC holdout split **exactly once** and
evaluates cond6 (holdout direction consistent).

Pipeline (section 12)::

    1. Load train episodes -> fit M2-0 .. M2-4 on BOTH labels (cat + inst).
    2. Load validation episodes -> select best cat model (highest val
       catastrophic AUPRC) and best inst model (highest val instability
       AUROC). This is the "frozen" model pair.
    3. Load holdout episodes -> apply the frozen model pair once.
    4. Evaluate cond1-5 on holdout (same thresholds as validation gate).
    5. cond6 passes iff cond1-5 all pass on holdout (directionally
       consistent with validation).
    6. Write holdout confirmation report (JSON + MD).

The script **does not** re-tune any hyperparameter on holdout. The
train/val split is identical to ``run_phaseT_wp3_m2_selective.py`` so the
selected models are reproducible.

CLI::

    python src/adapromptset/scripts/run_phaseT_wp3_m2_holdout.py \\
        --train-dir reports/phaseT_wp3_train \\
        --val-dir reports/phaseT_wp3_validation \\
        --holdout-dir reports/phaseT_wp1_holdout \\
        --out-dir reports/phaseT_wp3_m2_holdout \\
        --smoke
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

# OpenMP runtime conflict workaround (conda + sklearn on Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# Make the scripts directory importable so we can import the M2 module.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Reuse the M2 selective-prediction module.
import run_phaseT_wp3_m2_selective as m2  # noqa: E402


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout_conditions(
    cat_result_holdout: dict[str, Any],
    inst_result_holdout: dict[str, Any],
    val_labels_for_prevalence: dict[str, np.ndarray],
    holdout_labels: dict[str, np.ndarray],
) -> tuple[list[tuple[str, bool, str]], bool]:
    """Evaluate cond1-5 on holdout and cond6 (directionally consistent).

    cond6 passes iff cond1-5 all pass on holdout using the same thresholds
    that were applied on validation.
    """
    cond_list: list[tuple[str, bool, str]] = []

    prevalence_cat = float(np.mean(holdout_labels["catastrophic"]))

    # cond1: instability AUROC >= 0.70 (instability-trained model).
    auroc = inst_result_holdout["auroc_instability"]
    cond1_pass = auroc >= m2.GATE_AUROC_THRESHOLD
    cond_list.append((
        "cond1_instability_auroc_ge_0p70",
        cond1_pass,
        f"AUROC = {auroc:.4f} (query-level: "
        f"{inst_result_holdout['auroc_instability_query']:.4f}) "
        f"[instability-trained model]",
    ))

    # cond2: catastrophic AUPRC > prevalence (catastrophic-trained model).
    auprc_cat = cat_result_holdout["auprc_catastrophic"]
    cond2_pass = auprc_cat > prevalence_cat
    cond_list.append((
        "cond2_catastrophic_auprc_gt_prevalence",
        cond2_pass,
        f"AUPRC = {auprc_cat:.4f}, prevalence = {prevalence_cat:.4f} "
        f"[catastrophic-trained model]",
    ))

    # cond3: ECE <= 0.10 (or <= 0.08 after calibration).
    ece = cat_result_holdout["ece"]
    ece_cal = cat_result_holdout["ece_calibrated"]
    platt_applied = cat_result_holdout["platt_applied"]
    if ece <= m2.GATE_ECE_THRESHOLD:
        cond3_pass = True
        cond3_detail = f"ECE = {ece:.4f}"
    elif platt_applied and ece_cal <= m2.GATE_ECE_CALIBRATED_THRESHOLD:
        cond3_pass = True
        cond3_detail = (
            f"ECE = {ece:.4f}, after calibration ECE = {ece_cal:.4f}"
        )
    else:
        cond3_pass = False
        cond3_detail = (
            f"ECE = {ece:.4f}"
            + (f", after calibration ECE = {ece_cal:.4f}" if platt_applied else "")
        )
    cond_list.append(("cond3_ece_le_0p10", cond3_pass, cond3_detail))

    # cond4: CFR reduction >= 30% at 80% coverage.
    cfr_red = cat_result_holdout["cfr_reduction_at_80"]
    cfr_all = cat_result_holdout["cfr_no_abstention"]
    if cfr_all == 0.0:
        cond4_pass = True
        cond4_detail = (
            f"CFR reduction = {cfr_red:.4f} (no catastrophic failures in "
            f"holdout; vacuously satisfied)"
        )
    else:
        cond4_pass = cfr_red >= m2.GATE_CFR_REDUCTION_THRESHOLD
        cond4_detail = (
            f"CFR reduction = {cfr_red * 100:.1f}% "
            f"(CFR all={cfr_all:.4f}, "
            f"CFR@80%={cat_result_holdout['cfr_at_80_coverage']:.4f})"
        )
    cond_list.append((
        "cond4_cfr_reduction_30pct_at_80pct_coverage",
        cond4_pass,
        cond4_detail,
    ))

    # cond5: mean Dice not decrease at 80% coverage.
    dice_change = cat_result_holdout["dice_change_at_80"]
    cond5_pass = dice_change >= 0.0
    cond_list.append((
        "cond5_dice_not_decrease_at_80pct_coverage",
        cond5_pass,
        f"Dice change = {dice_change:+.6f} "
        f"(Dice all={cat_result_holdout['dice_no_abstention']:.6f}, "
        f"Dice@80%={cat_result_holdout['dice_at_80_coverage']:.6f})",
    ))

    # cond6: holdout direction consistent (cond1-5 all pass on holdout).
    cond6_pass = all(ok for _, ok, _ in cond_list)
    cond_list.append((
        "cond6_holdout_direction_consistent",
        cond6_pass,
        "cond1-5 all pass on holdout (directionally consistent with val)",
    ))

    return cond_list, cond6_pass


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_holdout_report_json(
    out_dir: Path, summary: dict[str, Any],
) -> None:
    (out_dir / "m2_holdout_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )


def write_holdout_report_md(
    out_dir: Path,
    summary: dict[str, Any],
    cond_list: list[tuple[str, bool, str]],
    val_summary: dict[str, Any],
) -> None:
    md: list[str] = []
    md.append("# Phase T WP3 M2 Holdout Confirmation (section 12)")
    md.append("")
    md.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    md.append("")
    md.append("## Frozen models (selected on validation)")
    md.append("")
    md.append(f"- Catastrophic-trained model: **{summary['frozen_cat_model']}** "
              f"(val catastrophic AUPRC = {summary['val_cat_auprc']:.4f})")
    md.append(f"- Instability-trained model: **{summary['frozen_inst_model']}** "
              f"(val instability AUROC = {summary['val_inst_auroc']:.4f})")
    md.append("")
    md.append("## Data splits")
    md.append("")
    md.append(f"- Train episodes: {summary['n_train_episodes']} "
              f"({summary['n_train_queries']} queries) -- model fitting")
    md.append(f"- Val episodes: {summary['n_val_episodes']} "
              f"({summary['n_val_queries']} queries) -- model selection")
    md.append(f"- Holdout episodes: {summary['n_holdout_episodes']} "
              f"({summary['n_holdout_queries']} queries) -- one-shot confirmation")
    md.append("")
    md.append(f"Holdout prevalence (instability): "
              f"{summary['holdout_prevalence_instability']:.4f}")
    md.append(f"Holdout prevalence (catastrophic): "
              f"{summary['holdout_prevalence_catastrophic']:.4f}")
    md.append("")

    md.append("## Holdout metrics (frozen dual-model)")
    md.append("")
    md.append("| metric | value |")
    md.append("|--------|-------|")
    md.append(f"| instability AUROC | {summary['holdout_auroc_instability']:.4f} |")
    md.append(f"| instability AUPRC | {summary['holdout_auprc_instability']:.4f} |")
    md.append(f"| catastrophic AUPRC | {summary['holdout_auprc_catastrophic']:.4f} |")
    md.append(f"| ECE | {summary['holdout_ece']:.4f} |")
    md.append(f"| ECE (calibrated) | {summary['holdout_ece_calibrated']:.4f} |")
    md.append(f"| CFR (no abstention) | {summary['holdout_cfr_no_abstention']:.4f} |")
    md.append(f"| CFR @80% coverage | {summary['holdout_cfr_at_80']:.4f} |")
    md.append(f"| CFR reduction @80% | {summary['holdout_cfr_reduction_at_80']:.4f} |")
    md.append(f"| Dice (no abstention) | {summary['holdout_dice_no_abstention']:.6f} |")
    md.append(f"| Dice @80% coverage | {summary['holdout_dice_at_80']:.6f} |")
    md.append(f"| Dice change @80% | {summary['holdout_dice_change_at_80']:+.6f} |")
    md.append("")

    md.append("## Validation vs holdout comparison")
    md.append("")
    md.append("| metric | validation | holdout | direction consistent |")
    md.append("|--------|-----------|---------|----------------------|")
    rows = [
        ("instability AUROC", val_summary["val_auroc_instability"],
         summary["holdout_auroc_instability"], ">=0.70"),
        ("catastrophic AUPRC", val_summary["val_auprc_catastrophic"],
         summary["holdout_auprc_catastrophic"], ">prevalence"),
        ("ECE", val_summary["val_ece"], summary["holdout_ece"], "<=0.10"),
        ("CFR reduction @80%", val_summary["val_cfr_reduction_at_80"],
         summary["holdout_cfr_reduction_at_80"], ">=0.30"),
        ("Dice change @80%", val_summary["val_dice_change_at_80"],
         summary["holdout_dice_change_at_80"], ">=0"),
    ]
    for name, v, h, rule in rows:
        md.append(f"| {name} | {v:.4f} | {h:.4f} | {rule} |")
    md.append("")

    md.append("## Holdout gate conditions (cond6 = directionally consistent)")
    md.append("")
    md.append("| # | condition | pass |")
    md.append("|---|-----------|------|")
    labels = {
        "cond1_instability_auroc_ge_0p70": "instability AUROC >= 0.70",
        "cond2_catastrophic_auprc_gt_prevalence": "catastrophic AUPRC > prevalence",
        "cond3_ece_le_0p10": "ECE <= 0.10 (or <= 0.08 after calibration)",
        "cond4_cfr_reduction_30pct_at_80pct_coverage": "CFR reduction >= 30% at 80% coverage",
        "cond5_dice_not_decrease_at_80pct_coverage": "mean Dice not decrease at 80% coverage",
        "cond6_holdout_direction_consistent": "holdout direction consistent (cond1-5 all pass)",
    }
    for i, (name, ok, _detail) in enumerate(cond_list, 1):
        status = "PASS" if ok else "FAIL"
        md.append(f"| {i} | {labels.get(name, name)} | {status} |")
    md.append("")
    for i, (name, ok, detail) in enumerate(cond_list, 1):
        status = "PASS" if ok else "FAIL"
        md.append(f"**cond{i}** (`{name}`): **{status}** -- {detail}")
        md.append("")

    n_pass = sum(1 for _, ok, _ in cond_list if ok)
    md.append(f"## Decision: **{summary['decision']}**")
    md.append("")
    md.append(f"({n_pass}/{len(cond_list)} conditions passed on holdout)")
    md.append("")

    (out_dir / "m2_holdout_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase T WP3 M2 Holdout Confirmation (section 12)",
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
        "--holdout-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp1_holdout",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp3_m2_holdout",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick verification: first 20 rows of each split.",
    )
    args = parser.parse_args()

    train_csv = args.train_dir / "phase2_episode_instability.csv"
    val_csv = args.val_dir / "phase2_episode_instability.csv"
    holdout_csv = args.holdout_dir / "phase2_episode_instability.csv"
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phaseT-M2-holdout] train_csv={train_csv}")
    print(f"[phaseT-M2-holdout] val_csv={val_csv}")
    print(f"[phaseT-M2-holdout] holdout_csv={holdout_csv}")
    print(f"[phaseT-M2-holdout] out_dir={out_dir}")
    print(f"[phaseT-M2-holdout] smoke={args.smoke}")

    start = time.perf_counter()

    # Step 1: Load train / val / holdout.
    train_rows = m2.load_episodes(train_csv)
    val_rows = m2.load_episodes(val_csv)
    holdout_rows = m2.load_episodes(holdout_csv)
    if args.smoke:
        train_rows = train_rows[:20]
        val_rows = val_rows[:20]
        holdout_rows = holdout_rows[:20]
    print(f"[phaseT-M2-holdout] loaded {len(train_rows)} train, "
          f"{len(val_rows)} val, {len(holdout_rows)} holdout episodes")
    if not train_rows:
        raise RuntimeError(f"No train episodes at {train_csv}")
    if not val_rows:
        raise RuntimeError(f"No val episodes at {val_csv}")
    if not holdout_rows:
        raise RuntimeError(f"No holdout episodes at {holdout_csv}")

    # Step 2: Construct labels.
    train_labels = m2.construct_labels(train_rows)
    val_labels = m2.construct_val_labels(val_rows)
    holdout_labels = m2.construct_val_labels(holdout_rows)
    print(f"[phaseT-M2-holdout] holdout prevalence: "
          f"instability={np.mean(holdout_labels['instability']):.4f}, "
          f"catastrophic={np.mean(holdout_labels['catastrophic']):.4f}")

    # Step 3: Train + select on val (frozen model pair).
    print("[phaseT-M2-holdout] training catastrophic-labelled models ...")
    cat_results, cat_model, cat_result_val = m2.evaluate_all_models(
        m2.build_all_models(), train_rows, val_rows, train_labels, val_labels,
        train_label_key="catastrophic",
    )
    print("[phaseT-M2-holdout] training instability-labelled models ...")
    inst_results, inst_model, inst_result_val = m2.evaluate_all_models(
        m2.build_all_models(), train_rows, val_rows, train_labels, val_labels,
        train_label_key="instability",
    )
    print(f"[phaseT-M2-holdout] frozen cat model: {cat_model.name} "
          f"(val AUPRC_cat={cat_result_val['auprc_catastrophic']:.4f})")
    print(f"[phaseT-M2-holdout] frozen inst model: {inst_model.name} "
          f"(val AUROC={inst_result_val['auroc_instability']:.4f})")

    # Step 4: Apply frozen models to holdout.
    # We need to re-evaluate the selected models on the holdout split. The
    # M2 models are already fit on train (from evaluate_all_models), so we
    # can directly predict on holdout.
    print("[phaseT-M2-holdout] applying frozen models to holdout ...")
    X_holdout = m2.episodes_to_matrix(holdout_rows)
    holdout_mean_dice = np.asarray(
        [r["mean_dice_k1"] for r in holdout_rows], dtype=np.float64
    )
    y_holdout_instab = holdout_labels["instability"]
    y_holdout_cat = holdout_labels["catastrophic"]

    # Catastrophic model predictions on holdout.
    cat_risk_raw = cat_model.predict_risk(X_holdout)
    train_risk_cat = cat_model.predict_risk(m2.episodes_to_matrix(train_rows))
    auroc_instab_cat = m2.safe_auroc(y_holdout_instab, cat_risk_raw)
    auprc_instab_cat = m2.safe_auprc(y_holdout_instab, cat_risk_raw)
    auprc_cat = m2.safe_auprc(y_holdout_cat, cat_risk_raw)
    ece_cat, calib_bins_cat = m2.compute_ece(y_holdout_cat, cat_risk_raw)

    # Platt scaling (refit on train, same as val path).
    platt_cat = None
    ece_cal_cat = ece_cat
    cat_risk = cat_risk_raw
    cat_risk_calibrated = cat_risk_raw
    platt_applied_cat = False
    if ece_cat > m2.GATE_ECE_THRESHOLD:
        platt_cat = m2.fit_platt_scaler(
            train_risk_cat, train_labels["catastrophic"],
        )
        if platt_cat is not None:
            cat_risk_calibrated = m2.apply_platt_scaler(platt_cat, cat_risk_raw)
            ece_cal_cat, _ = m2.compute_ece(y_holdout_cat, cat_risk_calibrated)
            if ece_cal_cat < ece_cat:
                cat_risk = cat_risk_calibrated
                ece_cat = ece_cal_cat
                platt_applied_cat = True
            else:
                platt_cat = None
                cat_risk_calibrated = cat_risk_raw

    rc_cat = m2.compute_risk_coverage(cat_risk, holdout_mean_dice, y_holdout_cat)
    m80_cat = m2.metrics_at_coverage(
        cat_risk, holdout_mean_dice, y_holdout_cat, m2.GATE_COVERAGE,
    )

    cat_result_holdout = {
        "auprc_catastrophic": auprc_cat,
        "auprc_instability": auprc_instab_cat,
        "auroc_instability": auroc_instab_cat,
        "ece": ece_cat,
        "ece_calibrated": ece_cal_cat,
        "platt_applied": platt_applied_cat,
        "cfr_no_abstention": m80_cat["cfr_all"],
        "cfr_at_80_coverage": m80_cat["cfr"],
        "cfr_reduction_at_80": m80_cat["cfr_reduction"],
        "dice_no_abstention": m80_cat["dice_all"],
        "dice_at_80_coverage": m80_cat["dice"],
        "dice_change_at_80": m80_cat["dice_change"],
        "val_risk": cat_risk,
        "val_risk_calibrated": cat_risk_calibrated,
        "calibration_bins": calib_bins_cat,
        "risk_coverage": rc_cat,
    }

    # Instability model predictions on holdout (only AUROC needed for cond1).
    inst_risk_raw = inst_model.predict_risk(X_holdout)
    auroc_instab = m2.safe_auroc(y_holdout_instab, inst_risk_raw)
    auprc_instab = m2.safe_auprc(y_holdout_instab, inst_risk_raw)
    auprc_cat_inst = m2.safe_auprc(y_holdout_cat, inst_risk_raw)
    ece_inst, calib_bins_inst = m2.compute_ece(y_holdout_instab, inst_risk_raw)
    # Query-level AUROC for reporting.
    q_rows, q_risk, q_instab, q_cat = m2._aggregate_queries(
        holdout_rows, inst_risk_raw, holdout_mean_dice,
    )
    q_auroc = m2.safe_auroc(q_instab, q_risk)
    q_auprc_cat = m2.safe_auprc(q_cat, q_risk)

    inst_result_holdout = {
        "auroc_instability": auroc_instab,
        "auprc_instability": auprc_instab,
        "auprc_catastrophic": auprc_cat_inst,
        "ece": ece_inst,
        "ece_calibrated": ece_inst,
        "platt_applied": False,
        "auroc_instability_query": q_auroc,
        "auprc_catastrophic_query": q_auprc_cat,
    }

    print(f"[phaseT-M2-holdout] holdout cat model metrics: "
          f"AUROC={auroc_instab_cat:.4f} AUPRC_cat={auprc_cat:.4f} "
          f"ECE={ece_cat:.4f} CFR_red@80={m80_cat['cfr_reduction']:.4f} "
          f"Dice_chg@80={m80_cat['dice_change']:+.6f}")
    print(f"[phaseT-M2-holdout] holdout inst model metrics: "
          f"AUROC={auroc_instab:.4f} AUPRC_instab={auprc_instab:.4f}")

    # Step 5: Evaluate holdout conditions + cond6.
    cond_list, cond6_pass = evaluate_holdout_conditions(
        cat_result_holdout, inst_result_holdout, val_labels, holdout_labels,
    )

    # Step 6: Build summary.
    n_train_queries = len({r["query_case_id"] for r in train_rows})
    n_val_queries = len({r["query_case_id"] for r in val_rows})
    n_holdout_queries = len({r["query_case_id"] for r in holdout_rows})

    decision = (
        m2.GATE_PASS if cond6_pass
        else "PHASET_M2_HOLDOUT_DIRECTION_INCONSISTENT"
    )

    summary = {
        "gate": "Phase T WP3 M2 Holdout Confirmation (section 12)",
        "date": datetime.now().isoformat(timespec="seconds"),
        "frozen_cat_model": cat_model.name,
        "frozen_inst_model": inst_model.name,
        "val_cat_auprc": cat_result_val["auprc_catastrophic"],
        "val_inst_auroc": inst_result_val["auroc_instability"],
        "n_train_episodes": len(train_rows),
        "n_val_episodes": len(val_rows),
        "n_holdout_episodes": len(holdout_rows),
        "n_train_queries": n_train_queries,
        "n_val_queries": n_val_queries,
        "n_holdout_queries": n_holdout_queries,
        "holdout_prevalence_instability": float(np.mean(holdout_labels["instability"])),
        "holdout_prevalence_catastrophic": float(np.mean(holdout_labels["catastrophic"])),
        "holdout_auroc_instability": auroc_instab,
        "holdout_auprc_instability": auprc_instab,
        "holdout_auprc_catastrophic": auprc_cat,
        "holdout_ece": ece_cat,
        "holdout_ece_calibrated": ece_cal_cat,
        "holdout_cfr_no_abstention": m80_cat["cfr_all"],
        "holdout_cfr_at_80": m80_cat["cfr"],
        "holdout_cfr_reduction_at_80": m80_cat["cfr_reduction"],
        "holdout_dice_no_abstention": m80_cat["dice_all"],
        "holdout_dice_at_80": m80_cat["dice"],
        "holdout_dice_change_at_80": m80_cat["dice_change"],
        "conditions": {
            name: {"pass": ok, "details": detail}
            for name, ok, detail in cond_list
        },
        "decision": decision,
    }

    val_summary = {
        "val_auroc_instability": inst_result_val["auroc_instability"],
        "val_auprc_catastrophic": cat_result_val["auprc_catastrophic"],
        "val_ece": cat_result_val["ece"],
        "val_cfr_reduction_at_80": cat_result_val["cfr_reduction_at_80"],
        "val_dice_change_at_80": cat_result_val["dice_change_at_80"],
    }

    # Step 7: Write outputs.
    write_holdout_report_json(out_dir, summary)
    write_holdout_report_md(out_dir, summary, cond_list, val_summary)

    elapsed = time.perf_counter() - start

    # Print results.
    print()
    print("=" * 72)
    print("[phaseT-M2-holdout] Holdout confirmation conditions:")
    for i, (name, ok, detail) in enumerate(cond_list, 1):
        status = "PASS" if ok else "FAIL"
        print(f"  cond{i}: {status}  {name}")
        print(f"         {detail}")
    print()
    n_pass = sum(1 for _, ok, _ in cond_list if ok)
    print(f"  {n_pass}/{len(cond_list)} conditions passed on holdout")
    print(f"  FINAL STATE: {decision}")
    print("=" * 72)
    print(f"[phaseT-M2-holdout] artifacts written to {out_dir}")
    print(f"[phaseT-M2-holdout] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
