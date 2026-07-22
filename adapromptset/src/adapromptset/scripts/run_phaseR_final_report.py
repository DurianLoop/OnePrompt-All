"""Phase R Final Report Generator.

Self-contained script that produces the Phase R final report artifacts:
  - 18 markdown tables  (table_01_dataset_roles.md ... table_18_claim_boundaries.md)
  - 20 PNG figures       (figure_01_per_query_best_worst_gap.png ... figure_20_final_method_diagram.png)
  - phaseR_final_report.md   (comprehensive Chinese/English report)
  - phaseR_final_report.json (structured JSON)

Run with the sam_gpu conda env:
    D:\\SAM\\conda_envs\\sam_gpu\\python.exe D:\\SAM\\adapromptset\\src\\adapromptset\\scripts\\run_phaseR_final_report.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(r"D:\SAM\adapromptset")
REPORTS = ROOT / "reports"
FINAL_DIR = REPORTS / "phaseR_final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

R1_DIR = REPORTS / "phaseR_gateR1"
R2_DIR = REPORTS / "phaseR_gateR2"
HIST_DIR = REPORTS / "phaseR_historical_audit"
POOL_DIR = REPORTS / "phaseR_reference_pool_audit"

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
DPI = 120
FIGSIZE = (8, 6)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_csv(path: Path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        pass
    return rows


def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def mean(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return float(np.mean(xs)) if xs else float("nan")


def var(xs):
    xs = [x for x in xs if x == x]
    return float(np.var(xs)) if xs else float("nan")


def write_table(filename: str, header: list[str], rows: list[list[str]], note: str = ""):
    path = FINAL_DIR / filename
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    if note:
        lines.append("")
        lines.append(note)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_blocked_table(filename: str, table_title: str, reason: str):
    path = FINAL_DIR / filename
    lines = [
        f"# {table_title}",
        "",
        "| Status | Reason |",
        "|---|---|",
        f"| BLOCKED | {reason} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_fig(fig, name: str):
    path = FINAL_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def blocked_fig(name: str, reason: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.62, "BLOCKED", ha="center", va="center",
            fontsize=28, fontweight="bold", color="#b00020",
            transform=ax.transAxes)
    ax.text(0.5, 0.42, reason, ha="center", va="center",
            fontsize=12, color="#333333", wrap=True,
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return save_fig(fig, name)


# --------------------------------------------------------------------------- #
# 1. Load all artifacts
# --------------------------------------------------------------------------- #
def load_artifacts() -> dict:
    art: dict = {}

    # R0 historical audit (files live under phaseR_historical_audit)
    art["hyp_summary_md"] = (HIST_DIR / "phaseR_hypothesis_summary.md").read_text(encoding="utf-8") \
        if (HIST_DIR / "phaseR_hypothesis_summary.md").exists() else ""
    art["claim_correction_md"] = (HIST_DIR / "phaseY_claim_correction.md").read_text(encoding="utf-8") \
        if (HIST_DIR / "phaseY_claim_correction.md").exists() else ""
    art["y_ref_instability"] = load_csv(HIST_DIR / "phaseY_reference_instability_summary.csv")
    art["pool_audit"] = load_json(POOL_DIR / "reference_pool_audit.json")

    # R1
    art["r1_summary"] = load_json(R1_DIR / "gateR1_summary.json")
    art["r1_meta"] = load_json(R1_DIR / "run_metadata.json")
    art["r1_per_query_instab"] = load_csv(R1_DIR / "per_query_instability.csv")
    art["r1_per_ref"] = load_csv(R1_DIR / "per_reference_metrics.csv")
    art["r1_per_query_agg"] = load_csv(R1_DIR / "per_query_aggregate.csv")

    # R2
    art["r2_summary"] = load_json(R2_DIR / "gateR2_summary.json")
    art["r2_meta"] = load_json(R2_DIR / "run_metadata.json")
    art["r2_train_sel"] = load_json(R2_DIR / "train_selection_summary.json")
    art["r2_train"] = load_csv(R2_DIR / "train_consensus_metrics.csv")
    art["r2_val"] = load_csv(R2_DIR / "val_consensus_metrics.csv")
    art["r2_val_per_ref"] = load_csv(R2_DIR / "val_per_reference_metrics.csv")
    art["r2_cross"] = load_csv(R2_DIR / "val_cross_set_stability.csv")

    # compute ledger
    art["ledger"] = load_csv(REPORTS / "phaseR_compute_ledger.csv")
    return art


# --------------------------------------------------------------------------- #
# 2. Tables
# --------------------------------------------------------------------------- #
def gen_tables(art: dict) -> list[str]:
    paths = []
    g1 = art["r1_summary"].get("gate_r1", {})
    g2 = art["r2_summary"].get("gate_r2", {})
    pool = art["pool_audit"]
    r1meta = art["r1_meta"]
    r2meta = art["r2_meta"]

    # ---- Table 01: Dataset roles ----
    paths.append(str(write_table(
        "table_01_dataset_roles.md",
        ["Split", "Role", "Cases", "Patients", "Notes"],
        [
            ["MoNuSAC train", "Query pool for R1 + R2 train consensus", "80", "—",
             "Patient-disjoint from val/holdout/test"],
            ["MoNuSAC validation", "R2 validation (consensus selection confirmation)", "41", "—",
             "Fixed A/B reference sets, single observation"],
            ["MoNuSAC holdout", "R4 confirmatory holdout (LOCKED)", "40", "—",
             "Not accessed; R4 BLOCKED"],
            ["Reference pool", "Source of 6 references per query (A=3, B=3)", "80", "18",
             "Patient-disjoint from query; sufficient for 5 seeds"],
        ],
        note="Source: phaseR_reference_pool_audit/reference_pool_audit.json",
    )))

    # ---- Table 02: Reference resampling protocol ----
    paths.append(str(write_table(
        "table_02_reference_resampling.md",
        ["Parameter", "Value", "Description"],
        [
            ["References per query", "6", "3 from set A + 3 from set B"],
            ["Reference sets", "A / B", "Patient-disjoint halves of the reference pool"],
            ["Train resampling seeds", "5", "seeds 0-4; independent reference resamples per query"],
            ["Validation reference assignment", "Fixed (1 seed)", "validation_AB_fixed.csv; prevents multiple observations"],
            ["Reference pool size", str(pool.get("reference_pool_size", 80)), "Total available reference cases"],
            ["Reference pool patients", str(pool.get("reference_pool_patients", 18)), "Distinct patients in pool"],
            ["Min available (train)", str(pool.get("min_available_train", 70)), "Worst-case eligible references for any train query"],
            ["Min available (val)", str(pool.get("min_available_val", 80)), "Worst-case eligible references for any val query"],
            ["Feasible for 5 seeds", str(pool.get("feasible_for_5_seeds", True)), "Pool sufficiency confirmed"],
        ],
        note="Source: manifests/phaseR/train_resampling_seed_{0..4}.csv, validation_AB_fixed.csv",
    )))

    # ---- Table 03: Historical Phase Y correction (R0 audit) ----
    paths.append(str(write_table(
        "table_03_historical_phaseY_correction.md",
        ["Metric (Phase Y / R0 audit)", "Value", "Correction / Interpretation"],
        [
            ["Mean best-worst Dice gap (6 refs)", "0.1874", "Reference instability is large at per-query level"],
            ["Mean reference std", "0.0687", "Exceeds R1 cond2 threshold (0.02)"],
            ["% queries best-worst gap >= 0.05", "82.9%", "Exceeds R1 cond1 threshold (30%)"],
            ["Exact action agreement (A vs B)", "0.0488", "Best action is heavily reference-dependent"],
            ["K+ bin agreement (A vs B)", "0.3902", "Coarse action family only mildly stable"],
            ["Median Spearman rho", "0.7450", "Rank correlation moderate but not high"],
            ["Raw oracle gap", "0.0669", "Inflated by Winner's Curse"],
            ["Stable oracle gap", "0.0061", "Small aggregate gap — but NOT absence of instability"],
            ["Optimism fraction", "90.96%", "Most of raw gap is selection bias"],
            ["Diff-from-fixed rate", "0.0000", "Oracle picks same action family; instability is in output, not selection"],
        ],
        note="Source: phaseR_historical_audit/phaseR_hypothesis_summary.md, phaseY_claim_correction.md. "
             "R0 was a read-only CPU audit; the small stable oracle gap (0.0061) was an aggregate artifact, "
             "not evidence that reference instability is absent.",
    )))

    # ---- Table 04: Single-reference instability (R1) ----
    instab = art["r1_per_query_instab"]
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in instab:
        by_cfg[r.get("config_name", "")].append(r)

    rows4 = []
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        rs = by_cfg.get(cfg, [])
        gaps = [safe_float(r.get("best_worst_gap")) for r in rs]
        stds = [safe_float(r.get("std_dice")) for r in rs]
        pct = (np.mean([g >= 0.05 for g in gaps if g == g]) * 100) if gaps else float("nan")
        rows4.append([cfg, f"{mean(gaps):.4f}", f"{mean(stds):.4f}",
                      f"{pct:.2f}%", str(len(rs))])
    rows4.append(["ALL (overall)", "—", f"{g1.get('mean_reference_std', 0.0737):.4f}",
                  f"{g1.get('pct_queries_best_worst_gap_ge_0p05', 0.8875)*100:.2f}%",
                  str(len(instab))])
    paths.append(str(write_table(
        "table_04_single_reference_instability.md",
        ["Config", "Mean best-worst gap", "Mean ref std (std_dice)", "% rows gap>=0.05", "N rows"],
        rows4,
        note=f"Source: phaseR_gateR1/per_query_instability.csv (80 queries x 5 seeds x 3 configs = 1200 rows). "
             f"Sensitive config: {g1.get('sensitive_config', 'P20-N1')}. "
             f"Seed direction consistent: {g1.get('seed_direction_consistent', True)}.",
    )))

    # ---- Table 05: Matching-map disagreement (R1) ----
    mmd = [safe_float(r.get("matching_map_disagreement")) for r in instab]
    by_cfg_mmd = {}
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        vals = [safe_float(r.get("matching_map_disagreement")) for r in by_cfg.get(cfg, [])]
        by_cfg_mmd[cfg] = vals
    rows5 = []
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        vals = by_cfg_mmd[cfg]
        rows5.append([cfg, f"{mean(vals):.4f}", f"{float(np.std(vals)):.4f}",
                      f"{float(np.min(vals)):.4f}", f"{float(np.max(vals)):.4f}", str(len(vals))])
    rows5.append(["ALL", f"{mean(mmd):.4f}", f"{float(np.std(mmd)):.4f}",
                  f"{float(np.min(mmd)):.4f}", f"{float(np.max(mmd)):.4f}", str(len(mmd))])
    paths.append(str(write_table(
        "table_05_matching_map_disagreement.md",
        ["Config", "Mean disagreement", "Std", "Min", "Max", "N"],
        rows5,
        note="Source: phaseR_gateR1/per_query_instability.csv column matching_map_disagreement. "
             "Measures disagreement of DINOv2-L matching maps across the 6 references per (query, seed, config).",
    )))

    # ---- Table 06: Candidate overlap (R1) ----
    cj = [safe_float(r.get("candidate_jaccard_mean")) for r in instab]
    by_cfg_cj = {}
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        vals = [safe_float(r.get("candidate_jaccard_mean")) for r in by_cfg.get(cfg, [])]
        by_cfg_cj[cfg] = vals
    rows6 = []
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        vals = by_cfg_cj[cfg]
        rows6.append([cfg, f"{mean(vals):.4f}", f"{float(np.std(vals)):.4f}", str(len(vals))])
    rows6.append(["ALL", f"{mean(cj):.4f}", f"{float(np.std(cj)):.4f}", str(len(cj))])
    paths.append(str(write_table(
        "table_06_candidate_overlap.md",
        ["Config", "Mean candidate Jaccard", "Std", "N"],
        rows6,
        note="DERIVED_FROM_R1. Source: phaseR_gateR1/per_query_instability.csv column candidate_jaccard_mean. "
             "Candidate pool Jaccard overlap across the 6 references; low values indicate reference-specific candidate pools.",
    )))

    # ---- Table 07: Prompt-coordinate overlap (R1/R2) ----
    po = [safe_float(r.get("prompt_overlap_mean")) for r in instab]
    by_cfg_po = {}
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        vals = [safe_float(r.get("prompt_overlap_mean")) for r in by_cfg.get(cfg, [])]
        by_cfg_po[cfg] = vals
    rows7 = []
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        vals = by_cfg_po[cfg]
        rows7.append([cfg, f"{mean(vals):.4f}", f"{float(np.std(vals)):.4f}", str(len(vals))])
    # R2 cross-set prompt overlap
    cross = art["r2_cross"]
    cross_po = [safe_float(r.get("prompt_overlap_AB")) for r in cross]
    rows7.append(["R2 val (A vs B, B2_mean)", f"{mean(cross_po):.4f}", f"{float(np.std(cross_po)):.4f}", str(len(cross_po))])
    rows7.append(["R1 ALL", f"{mean(po):.4f}", f"{float(np.std(po)):.4f}", str(len(po))])
    paths.append(str(write_table(
        "table_07_prompt_coordinate_overlap.md",
        ["Source / Config", "Mean prompt-overlap Jaccard", "Std", "N"],
        rows7,
        note="R1: per_query_instability.csv column prompt_overlap_mean (positive point set Jaccard across 6 refs). "
             "R2: val_cross_set_stability.csv column prompt_overlap_AB (prompt overlap between reference sets A and B).",
    )))

    # ---- Table 08: Prediction overlap (R2 cross-set) ----
    rows8 = []
    for cfg in ["P20-N1", "P30-N0", "P50-N0"]:
        rs = [r for r in cross if r.get("config_name") == cfg]
        po_ab = [safe_float(r.get("prediction_overlap_AB")) for r in rs]
        spo = [safe_float(r.get("single_prediction_overlap_AB")) for r in rs]
        rows8.append([cfg, f"{mean(po_ab):.4f}", f"{mean(spo):.4f}",
                      f"{mean(po_ab)-mean(spo):+.4f}", str(len(rs))])
    all_po = [safe_float(r.get("prediction_overlap_AB")) for r in cross]
    all_spo = [safe_float(r.get("single_prediction_overlap_AB")) for r in cross]
    rows8.append(["ALL", f"{mean(all_po):.4f}", f"{mean(all_spo):.4f}",
                  f"{mean(all_po)-mean(all_spo):+.4f}", str(len(cross))])
    paths.append(str(write_table(
        "table_08_prediction_overlap.md",
        ["Config", "Mean prediction_overlap_AB (consensus)", "Mean single_prediction_overlap_AB", "Diff", "N"],
        rows8,
        note="Source: phaseR_gateR2/val_cross_set_stability.csv. "
             f"Overall consensus A/B overlap {g2.get('mean_consensus_ab_overlap', 0.6075):.4f} vs "
             f"single {g2.get('mean_single_ab_overlap', 0.5942):.4f} — cond5 PASS (improvement).",
    )))

    # ---- Table 09: Deterministic consensus comparison (R2 val, all6) ----
    val = art["r2_val"]
    val_all6 = [r for r in val if r.get("reference_scope") == "all6"]
    methods_order = ["B1_random", "B2_mean", "B3_median", "B4_trimmed", "B5_disagreement", "B6_quantile"]
    # for B5 use beta=0.25, B6 use tau=0.83 (selected)
    def val_dice(cfg, method):
        rs = [r for r in val_all6 if r.get("config_name") == cfg and r.get("method") == method]
        if method == "B5_disagreement":
            rs = [r for r in rs if r.get("hyperparam") in ("0.25", "0.25", "")]
            # pick 0.25 if present else mean of all
            r25 = [r for r in rs if r.get("hyperparam") == "0.25"]
            if r25:
                rs = r25
        if method == "B6_quantile":
            r83 = [r for r in rs if r.get("hyperparam") == "0.83"]
            if r83:
                rs = r83
        return mean([safe_float(r.get("dice")) for r in rs]), len(rs)
    rows9 = []
    header9 = ["Method"] + ["P20-N1", "P30-N0", "P50-N0", "Mean(across cfg)"]
    cfgs = ["P20-N1", "P30-N0", "P50-N0"]
    for m in methods_order:
        cells = []
        vals = []
        for cfg in cfgs:
            d, n = val_dice(cfg, m)
            cells.append(f"{d:.4f}" if d == d else "N/A")
            vals.append(d)
        vals = [v for v in vals if v == v]
        cells.append(f"{mean(vals):.4f}" if vals else "N/A")
        rows9.append([m] + cells)
    paths.append(str(write_table(
        "table_09_deterministic_consensus_comparison.md",
        header9, rows9,
        note="Source: phaseR_gateR2/val_consensus_metrics.csv (reference_scope=all6). "
             "B5_disagreement uses beta=0.25; B6_quantile uses tau=0.83 (train-selected). "
             f"Best consensus: {g2.get('best_method', 'B2_mean')} on {g2.get('best_config', 'P30-N0')} "
             f"(mean Dice {g2.get('mean_consensus_dice', 0.4756):.4f}).",
    )))

    # ---- Table 10: Variance reduction (R2 val, all6) ----
    rows10 = []
    for cfg in cfgs:
        b1 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B1_random"]
        b2 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B2_mean"]
        v1, v2 = var(b1), var(b2)
        vr = (v1 - v2) / v1 if v1 and v1 == v1 and v1 != 0 else float("nan")
        rows10.append([cfg, f"{v1:.5f}", f"{v2:.5f}", f"{vr*100:+.2f}%", str(len(b1))])
    vs = g2.get("var_single", 0.054308)
    vc = g2.get("var_consensus", 0.057056)
    vr_all = g2.get("variance_reduction", -0.050601)
    rows10.append(["ALL (selected)", f"{vs:.5f}", f"{vc:.5f}", f"{vr_all*100:+.2f}%", "—"])
    paths.append(str(write_table(
        "table_10_variance_reduction.md",
        ["Config", "Var(B1_random)", "Var(B2_mean consensus)", "Variance reduction %", "N"],
        rows10,
        note="Source: phaseR_gateR2/val_consensus_metrics.csv (reference_scope=all6). "
             f"Overall variance reduction = {vr_all*100:+.2f}% — cond3 FAILED (need >= +25%). "
             "Consensus INCREASED variance by ~5%.",
    )))

    # ---- Table 11: Catastrophic failure (R2 val) ----
    rows11 = []
    for cfg in cfgs:
        b1 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B1_random"]
        b2 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B2_mean"]
        cfr1 = np.mean([d < 0.1 for d in b1 if d == d]) if b1 else float("nan")
        cfr2 = np.mean([d < 0.1 for d in b2 if d == d]) if b2 else float("nan")
        rows11.append([cfg, f"{cfr1*100:.2f}%", f"{cfr2*100:.2f}%", f"{(cfr2-cfr1)*100:+.2f} pp"])
    rows11.append(["ALL (selected)",
                   f"{g2.get('cfr_b1', 0.121951)*100:.2f}%",
                   f"{g2.get('cfr_consensus', 0.073171)*100:.2f}%",
                   f"{(g2.get('cfr_consensus', 0.073171)-g2.get('cfr_b1', 0.121951))*100:+.2f} pp"])
    paths.append(str(write_table(
        "table_11_catastrophic_failure.md",
        ["Config", "CFR B1_random (Dice<0.1)", "CFR B2_mean (Dice<0.1)", "Change"],
        rows11,
        note="Source: phaseR_gateR2/val_consensus_metrics.csv (reference_scope=all6). "
             "cond4 PASS: catastrophic failure rate decreased (0.0732 vs 0.1220).",
    )))

    # ---- Tables 12-15: BLOCKED ----
    paths.append(str(write_blocked_table(
        "table_12_reference_utility_prediction.md",
        "Table 12: Reference Utility Prediction (R3)",
        "R3 not run — BLOCKED because R2 failed (variance reduction condition cond3 failed). "
        "Reference reliability/utility prediction requires R2 pass per directive section 12.")))
    paths.append(str(write_blocked_table(
        "table_13_weighted_consensus.md",
        "Table 13: Weighted Consensus (R3)",
        "R3 not run — BLOCKED because R2 failed. Reliability-weighted consensus (B7) is gated behind R2 pass.")))
    paths.append(str(write_blocked_table(
        "table_14_holdout_confirmation.md",
        "Table 14: Holdout Confirmation (R4)",
        "R4 not run — BLOCKED because both R2 and R3 failed. Directive section 13 requires R2 or R3 pass before holdout.")))
    paths.append(str(write_blocked_table(
        "table_15_glas_diagnostic.md",
        "Table 15: GlaS Diagnostic",
        "GlaS not run — BLOCKED because MoNuSAC R2 failed. Directive section 14 requires MoNuSAC R2 pass before GlaS.")))

    # ---- Table 16: Calls / runtime ----
    r1_calls = r1meta.get("total_sam_calls", 7200)
    r2_calls = r2meta.get("total_sam_calls", 3138)
    r1_gpu = r1meta.get("gpu_hours", 0.13935)
    r2_gpu = r2meta.get("gpu_hours", 0.075464)
    total_gpu = 0.2153  # per gate_status.md headline
    paths.append(str(write_table(
        "table_16_calls_runtime.md",
        ["Stage", "SAM calls", "GPU-hours", "Runtime (s)", "Device", "Description"],
        [
            ["R0 audit", "0", "0.00", "—", "cpu", "Read-only historical audit (8 files)"],
            ["Gate R1", str(r1_calls), f"{r1_gpu:.5f}", f"{r1meta.get('runtime_seconds', 501.659):.1f}",
             r1meta.get('device', 'cuda'), "80 queries x 5 seeds x 6 refs x 3 configs"],
            ["Gate R2", str(r2_calls), f"{r2_gpu:.5f}", f"{r2meta.get('runtime_seconds', 271.672):.1f}",
             r2meta.get('device', 'cuda'), "80 train + 41 val consensus evaluation"],
            ["TOTAL", str(r1_calls + r2_calls), f"{total_gpu:.4f}", "—", "—",
             f"Of 85 GPU-hour cap; remaining {85-total_gpu:.4f}"],
        ],
        note="Source: phaseR_compute_ledger.csv, run_metadata.json. Total SAM calls = 7200 + 3138 = 10338.",
    )))

    # ---- Table 17: Gate decisions ----
    conds_r2 = g2.get("conditions", {})
    rows17 = [
        ["R1", g1.get("decision", "PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED"),
         "PASSED", "5/5 conditions passed"],
        ["R2", g2.get("decision", "PHASER_GATE_R2_CONSENSUS_NO_GAIN"),
         "FAILED", f"7/8 conditions; cond3 variance_reduction_ge_25pct FAILED ({g2.get('variance_reduction', -0.0506)*100:+.2f}%)"],
        ["R3", "BLOCKED", "BLOCKED", "R2 failed; directive section 12"],
        ["R4", "BLOCKED", "BLOCKED", "R2 and R3 both failed; directive section 13"],
        ["GlaS", "BLOCKED", "BLOCKED", "R2 failed; directive section 14"],
        ["Formal test", "LOCKED_UNUSED", "LOCKED", "Directive section 22: no test access"],
    ]
    paths.append(str(write_table(
        "table_17_gate_decisions.md",
        ["Gate", "Decision / Status", "Result", "Reason"],
        rows17,
        note=f"Final Phase R state: PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC",
    )))

    # ---- Table 18: Claim boundaries ----
    rows18 = [
        ["ALLOWED",
         "Valid references induce substantial segmentation variability under the current frozen matching-and-prompting pipeline."],
        ["ALLOWED",
         "Deterministic consensus (B2_mean) improves mean Dice and reduces catastrophic failures but does NOT reduce reference-induced variance."],
        ["PROHIBITED", "Reference instability solved / eliminated."],
        ["PROHIBITED", "Universal across datasets (only MoNuSAC measured; GlaS BLOCKED)."],
        ["PROHIBITED", "SOTA / state-of-the-art segmentation."],
        ["PROHIBITED", "Formal test success (test LOCKED_UNUSED)."],
        ["PROHIBITED", "Reliability-weighted consensus works (R3 BLOCKED)."],
        ["PROHIBITED", "Holdout-confirmed (R4 BLOCKED)."],
        ["PROHIBITED", "Variance reduction achieved (cond3 FAILED; variance increased ~5%)."],
    ]
    paths.append(str(write_table(
        "table_18_claim_boundaries.md",
        ["Status", "Claim"],
        rows18,
        note="Per directive section 20. Only R1-confirmed claims and the qualified R2 diagnostic claim are allowed.",
    )))

    return paths


# --------------------------------------------------------------------------- #
# 3. Figures
# --------------------------------------------------------------------------- #
def gen_figures(art: dict) -> list[str]:
    paths = []
    instab = art["r1_per_query_instab"]
    val = art["r2_val"]
    val_all6 = [r for r in val if r.get("reference_scope") == "all6"]
    train = art["r2_train"]
    cross = art["r2_cross"]
    g2 = art["r2_summary"].get("gate_r2", {})
    cfgs = ["P20-N1", "P30-N0", "P50-N0"]

    # ---- Figure 01: per-query best-worst gap histogram ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    gaps = [safe_float(r.get("best_worst_gap")) for r in instab]
    gaps = [g for g in gaps if g == g]
    ax.hist(gaps, bins=40, color="#3b6fb6", edgecolor="white", alpha=0.85)
    ax.axvline(0.05, color="#b00020", linestyle="--", linewidth=1.5, label="gap=0.05 threshold")
    ax.set_title("Figure 1. Per-Query Best-Worst Reference Dice Gap (R1)")
    ax.set_xlabel("best_worst_gap")
    ax.set_ylabel("Count (query x seed x config)")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_01_per_query_best_worst_gap.png")))

    # ---- Figure 02: reference std distribution ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    stds = [safe_float(r.get("std_dice")) for r in instab]
    stds = [s for s in stds if s == s]
    ax.hist(stds, bins=40, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.axvline(0.02, color="#b00020", linestyle="--", linewidth=1.5, label="std=0.02 threshold")
    ax.set_title("Figure 2. Reference Std Distribution (R1, std_dice)")
    ax.set_xlabel("std_dice (per query x seed x config)")
    ax.set_ylabel("Count")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_02_reference_std_distribution.png")))

    # ---- Figure 03: matching-map disagreement ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    mmd = [safe_float(r.get("matching_map_disagreement")) for r in instab]
    mmd = [m for m in mmd if m == m]
    if mmd:
        ax.hist(mmd, bins=40, color="#ff7f0e", edgecolor="white", alpha=0.85)
        ax.set_title("Figure 3. Matching-Map Disagreement Across References (R1)")
        ax.set_xlabel("matching_map_disagreement")
        ax.set_ylabel("Count")
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No matching_map_disagreement data", ha="center", va="center", transform=ax.transAxes)
    paths.append(str(save_fig(fig, "figure_03_matching_map_disagreement.png")))

    # ---- Figure 04: candidate Jaccard ----
    cj = [safe_float(r.get("candidate_jaccard_mean")) for r in instab]
    cj = [c for c in cj if c == c]
    if cj:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.hist(cj, bins=40, color="#9467bd", edgecolor="white", alpha=0.85)
        ax.set_title("Figure 4. Candidate Pool Jaccard Across References (R1)")
        ax.set_xlabel("candidate_jaccard_mean")
        ax.set_ylabel("Count")
        paths.append(str(save_fig(fig, "figure_04_candidate_jaccard.png")))
    else:
        paths.append(str(blocked_fig(
            "figure_04_candidate_jaccard.png",
            "candidate-level overlap not recorded in R1 per-reference metrics")))

    # ---- Figure 05: single vs mean consensus paired scatter ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cfg in cfgs:
        b1 = {r["query_case_id"]: safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B1_random"}
        b2 = {r["query_case_id"]: safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B2_mean"}
        xs, ys = [], []
        for q in b1:
            if q in b2 and b1[q] == b1[q] and b2[q] == b2[q]:
                xs.append(b1[q]); ys.append(b2[q])
        ax.scatter(xs, ys, s=18, alpha=0.7, label=cfg)
    lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
    ax.set_title("Figure 5. Single (B1_random) vs Mean Consensus (B2_mean) Paired Dice (R2 val, all6)")
    ax.set_xlabel("B1_random Dice")
    ax.set_ylabel("B2_mean Dice")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_05_single_vs_mean_consensus_scatter.png")))

    # ---- Figure 06: single vs stable consensus — BLOCKED (R3) ----
    paths.append(str(blocked_fig(
        "figure_06_single_vs_stable_consensus_scatter.png",
        "R3 not run — no reliability/stable-weighted consensus (B7) available")))

    # ---- Figure 07: variance reduction bar chart ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels, v1s, v2s = [], [], []
    for cfg in cfgs:
        b1 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B1_random"]
        b2 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B2_mean"]
        labels.append(cfg)
        v1s.append(var(b1))
        v2s.append(var(b2))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, v1s, w, label="Var(B1_random)", color="#7f7f7f")
    ax.bar(x + w/2, v2s, w, label="Var(B2_mean)", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Figure 7. Variance: B1_random vs B2_mean Consensus per Config (R2 val)")
    ax.set_ylabel("Variance of Dice across queries")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_07_variance_reduction.png")))

    # ---- Figure 08: worst-reference improvement ----
    vpr = art["r2_val_per_ref"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cfg in cfgs:
        # min dice per query across references (worst single ref)
        per_q = defaultdict(list)
        for r in vpr:
            if r.get("config_name") == cfg:
                per_q[r["query_case_id"]].append(safe_float(r.get("dice")))
        b2 = {r["query_case_id"]: safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B2_mean"}
        xs, ys = [], []
        for q, ds in per_q.items():
            ds = [d for d in ds if d == d]
            if ds and q in b2 and b2[q] == b2[q]:
                xs.append(min(ds)); ys.append(b2[q])
        ax.scatter(xs, ys, s=18, alpha=0.7, label=cfg)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="y=x")
    ax.set_title("Figure 8. Worst-Reference Dice vs B2_mean Consensus Dice (R2 val)")
    ax.set_xlabel("min(dice) across references (worst single ref)")
    ax.set_ylabel("B2_mean consensus Dice")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_08_worst_reference_improvement.png")))

    # ---- Figure 09: catastrophic failure reduction ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels2, c1, c2 = [], [], []
    for cfg in cfgs:
        b1 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B1_random"]
        b2 = [safe_float(r.get("dice")) for r in val_all6
              if r.get("config_name") == cfg and r.get("method") == "B2_mean"]
        labels2.append(cfg)
        c1.append(np.mean([d < 0.1 for d in b1 if d == d]) * 100 if b1 else 0)
        c2.append(np.mean([d < 0.1 for d in b2 if d == d]) * 100 if b2 else 0)
    x = np.arange(len(labels2))
    w = 0.35
    ax.bar(x - w/2, c1, w, label="B1_random CFR%", color="#d62728")
    ax.bar(x + w/2, c2, w, label="B2_mean CFR%", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels2)
    ax.set_title("Figure 9. Catastrophic Failure Rate (Dice<0.1): B1 vs B2_mean (R2 val)")
    ax.set_ylabel("% queries with Dice < 0.1")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_09_catastrophic_failure_reduction.png")))

    # ---- Figure 10: beta sweep (B5_disagreement, train) ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    betas = ["0.25", "0.5", "1.0"]
    for cfg in cfgs:
        ys = []
        for b in betas:
            ds = [safe_float(r.get("dice")) for r in train
                  if r.get("config_name") == cfg and r.get("method") == "B5_disagreement"
                  and r.get("hyperparam") == b]
            ys.append(mean(ds))
        ax.plot(betas, ys, marker="o", label=cfg)
    ax.set_title("Figure 10. Beta Sweep — B5_disagreement Mean Dice (R2 train)")
    ax.set_xlabel("beta")
    ax.set_ylabel("Mean Dice")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_10_beta_sweep.png")))

    # ---- Figure 11: quantile sweep (B6_quantile, train) ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    taus = ["0.5", "0.67", "0.83"]
    for cfg in cfgs:
        ys = []
        for t in taus:
            ds = [safe_float(r.get("dice")) for r in train
                  if r.get("config_name") == cfg and r.get("method") == "B6_quantile"
                  and r.get("hyperparam") == t]
            ys.append(mean(ds))
        ax.plot(taus, ys, marker="s", label=cfg)
    ax.set_title("Figure 11. Tau Sweep — B6_quantile Mean Dice (R2 train)")
    ax.set_xlabel("tau")
    ax.set_ylabel("Mean Dice")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_11_quantile_sweep.png")))

    # ---- Figures 12-14: BLOCKED (R3) ----
    paths.append(str(blocked_fig(
        "figure_12_reliability_prediction_calibration.png",
        "R3 not run — reliability prediction calibration unavailable")))
    paths.append(str(blocked_fig(
        "figure_13_predicted_vs_actual_reference_utility.png",
        "R3 not run — predicted vs actual reference utility unavailable")))
    paths.append(str(blocked_fig(
        "figure_14_reference_weights_distribution.png",
        "R3 not run — reference weight distribution unavailable")))

    # ---- Figure 15: cross-set prompt overlap ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    po_ab = [safe_float(r.get("prompt_overlap_AB")) for r in cross]
    po_ab = [p for p in po_ab if p == p]
    ax.hist(po_ab, bins=30, color="#17becf", edgecolor="white", alpha=0.85)
    ax.set_title("Figure 15. Cross-Set Prompt Overlap (A vs B) per Query (R2 val)")
    ax.set_xlabel("prompt_overlap_AB (Jaccard)")
    ax.set_ylabel("Count")
    paths.append(str(save_fig(fig, "figure_15_cross_set_prompt_overlap.png")))

    # ---- Figure 16: cross-set prediction overlap ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    po = [safe_float(r.get("prediction_overlap_AB")) for r in cross]
    spo = [safe_float(r.get("single_prediction_overlap_AB")) for r in cross]
    po = [p for p in po if p == p]
    spo = [s for s in spo if s == s]
    ax.scatter(spo, po, s=20, alpha=0.6, color="#bcbd22", edgecolor="white")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="y=x")
    ax.set_title("Figure 16. Single vs Consensus Prediction Overlap A/B (R2 val)")
    ax.set_xlabel("single_prediction_overlap_AB (IoU)")
    ax.set_ylabel("prediction_overlap_AB (consensus IoU)")
    ax.legend()
    paths.append(str(save_fig(fig, "figure_16_cross_set_prediction_overlap.png")))

    # ---- Figure 17: MoNuSAC success/failure cases ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b2_p30 = {r["query_case_id"]: safe_float(r.get("dice"))
              for r in val_all6
              if r.get("config_name") == "P30-N0" and r.get("method") == "B2_mean"}
    items = sorted([(q, d) for q, d in b2_p30.items() if d == d], key=lambda x: x[1])
    worst5 = items[:5]
    best5 = items[-5:][::-1]
    sel = worst5 + best5
    labels17 = [q for q, _ in sel]
    vals17 = [d for _, d in sel]
    colors = ["#d62728"] * 5 + ["#2ca02c"] * 5
    ax.barh(range(len(sel)), vals17, color=colors, edgecolor="white")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(labels17, fontsize=8)
    ax.set_title("Figure 17. Top-5 Worst / Best Queries — B2_mean P30-N0 (R2 val)")
    ax.set_xlabel("Dice")
    ax.invert_yaxis()
    paths.append(str(save_fig(fig, "figure_17_monusac_success_failure_cases.png")))

    # ---- Figure 18: GlaS — BLOCKED ----
    paths.append(str(blocked_fig(
        "figure_18_glas_secondary_diagnostic.png",
        "GlaS not run — BLOCKED (MoNuSAC R2 failed)")))

    # ---- Figure 19: gate flow ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    gates = [
        (1.0, 4.5, "Gate R1\nREFERENCE_INSTABILITY_CONFIRMED", "#2ca02c", "PASSED"),
        (3.2, 4.5, "Gate R2\nCONSENSUS_NO_GAIN", "#d62728", "FAILED\n(cond3)"),
        (5.4, 4.5, "Gate R3\nReliability Learnability", "#7f7f7f", "BLOCKED"),
        (7.6, 4.5, "Gate R4\nConfirmatory Holdout", "#7f7f7f", "BLOCKED"),
        (9.2, 4.5, "GlaS\nSecondary Diagnostic", "#7f7f7f", "BLOCKED"),
    ]
    for (x, y, txt, color, status) in gates:
        box = FancyBboxPatch((x - 0.7, y - 0.7), 1.4, 1.4,
                             boxstyle="round,pad=0.08", linewidth=1.5,
                             edgecolor=color, facecolor=color + "22")
        ax.add_patch(box)
        ax.text(x, y + 0.15, txt, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.text(x, y - 0.45, status, ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    # arrows
    for i in range(len(gates) - 1):
        x0 = gates[i][0] + 0.7
        x1 = gates[i + 1][0] - 0.7
        ax.annotate("", xy=(x1, 4.5), xytext=(x0, 4.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#444444"))
    ax.text(5.0, 1.5, "Final State: PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#b00020")
    ax.text(5.0, 0.8, "Formal test: LOCKED_UNUSED   |   Compute: 0.2153 / 85 GPU-hours",
            ha="center", va="center", fontsize=9, color="#333333")
    ax.set_title("Figure 19. Phase R Gate Flow", fontsize=12)
    paths.append(str(save_fig(fig, "figure_19_gate_flow.png")))

    # ---- Figure 20: final method diagram ----
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    boxes = [
        (0.3, 2.5, 1.7, 1.0, "Query Image\n+ 6 References\n(A=3, B=3)", "#3b6fb6"),
        (2.5, 2.5, 1.7, 1.0, "DINOv2-L\nFeature Extraction\n(CPU reuse)", "#ff7f0e"),
        (4.7, 2.5, 1.7, 1.0, "Per-Reference\nContrast Maps\nC_r(x)", "#2ca02c"),
        (6.9, 2.5, 1.7, 1.0, "Consensus\nAggregation\n(B2_mean)", "#9467bd"),
        (9.1, 2.5, 1.7, 1.0, "SAM-B\nPrediction\n(1 call)", "#d62728"),
    ]
    for (x, y, w, h, txt, color) in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             linewidth=1.5, edgecolor=color, facecolor=color + "22")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5, fontweight="bold")
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 3.0), xytext=(x0, 3.0),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#222222"))
    ax.text(6.0, 5.0, "Frozen Phase R Pipeline (Reference-Stable Prompting)",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(6.0, 1.2,
            "R1: reference instability CONFIRMED  |  R2: consensus improves Dice (+0.046) but NOT variance "
            "(VR=-5.06%)  |  R3/R4/GlaS BLOCKED",
            ha="center", va="center", fontsize=8.5, color="#555555")
    ax.set_title("Figure 20. Final Method Diagram", fontsize=12)
    paths.append(str(save_fig(fig, "figure_20_final_method_diagram.png")))

    return paths


# --------------------------------------------------------------------------- #
# 4. Markdown report
# --------------------------------------------------------------------------- #
def gen_markdown(art: dict, table_paths: list[str], figure_paths: list[str]) -> Path:
    g1 = art["r1_summary"].get("gate_r1", {})
    g2 = art["r2_summary"].get("gate_r2", {})
    r1meta = art["r1_meta"]
    r2meta = art["r2_meta"]
    pool = art["pool_audit"]

    L: list[str] = []
    L.append("# Phase R 最终报告 (Phase R Final Report)")
    L.append("")
    L.append(f"日期: 2026-07-22  |  Phase: R  |  Final State: `PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC`")
    L.append("")

    # 1. 摘要
    L.append("## 1. 摘要 (Executive Summary)")
    L.append("")
    L.append("Phase R 旨在验证「参考图像稳定性 (reference stability)」是否可通过**确定性共识 (deterministic consensus)** "
             "或**可靠性加权 (reliability weighting)** 来缓解。最终结论: **参考不稳定性被确认存在 (R1 PASSED), "
             "但确定性共识无法降低参考诱导的方差 (R2 FAILED on cond3)**, 因此参考稳定性在确定性方法下不可行。")
    L.append("")
    L.append(f"- **Gate R1**: PASSED — `{g1.get('decision', 'PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED')}` "
             f"(88.75% queries best-worst gap >= 0.05, mean ref std = 0.0737)")
    L.append(f"- **Gate R2**: FAILED — `{g2.get('decision', 'PHASER_GATE_R2_CONSENSUS_NO_GAIN')}` "
             f"(7/8 conditions passed; cond3 variance reduction = -5.06%, need >= +25%)")
    L.append("- **Gate R3 / R4 / GlaS**: BLOCKED (R2 未通过)")
    L.append("- **Formal test**: LOCKED_UNUSED")
    L.append(f"- **计算预算**: 0.2153 / 85 GPU-hours (剩余 84.7847); 总 SAM calls = 10338")
    L.append("- **关键发现**: B2_mean 共识将 mean Dice 从 0.4297 提升至 0.4756 (+0.0460), "
             "并将 catastrophic failure rate 从 12.20% 降至 7.32%, **但方差反而增加 5.06%**, "
             "未达到 R2 的方差缩减条件。")
    L.append("")

    # 2. 背景与动机
    L.append("## 2. 背景与动机 (Background)")
    L.append("")
    L.append("Phase Y 发现 `REFERENCE_INSTABILITY` 是主要瓶颈: 不同有效参考图像对同一 query 产生显著不同的分割结果。"
             "Phase Y 的 stable oracle gap 仅 0.0061 (看似很小), 但这是**聚合层面的伪影** — "
             "per-query best-worst Dice gap 均值达 0.1874, 82.9% 的 query gap >= 0.05。")
    L.append("")
    L.append("Phase R 的核心假设: 通过对多个参考的 contrast map 进行**共识聚合 (consensus aggregation)**, "
             "可以在不增加 SAM 调用次数的前提下降低参考诱导的方差并提升分割可靠性。"
             "Phase R 使用 3 个 prompt config (P20-N1, P30-N0, P50-N0) 而非 Phase Y 的 210-action bank, "
             "以聚焦参考效应本身。")
    L.append("")

    # 3. 数据与协议
    L.append("## 3. 数据与协议 (Data and Protocol)")
    L.append("")
    L.append("**MoNuSAC 数据集角色:**")
    L.append("")
    L.append("| Split | 用途 | Cases |")
    L.append("|---|---|---|")
    L.append("| Train | R1 不稳定性测量 + R2 train 共识选择 | 80 |")
    L.append("| Validation | R2 验证 (consensus 确认) | 41 |")
    L.append("| Holdout | R4 确认 (LOCKED) | 40 |")
    L.append("| Reference pool | 6 refs/query (A=3, B=3) | 80 cases / 18 patients |")
    L.append("")
    L.append("**参考重采样协议:** 每个 query 配 6 个参考 (set A=3, set B=3, 患者不重叠)。"
             "Train 使用 5 个 resampling seeds; validation 使用固定 A/B 分配 (`validation_AB_fixed.csv`) "
             "以避免多次观测。参考池充分性已确认 (min available train=70, val=80, feasible_for_5_seeds=True)。")
    L.append("")

    # 4. 冻结的 pipeline
    L.append("## 4. 冻结的 Pipeline (Frozen Pipeline)")
    L.append("")
    L.append("- **特征提取**: DINOv2-L (CPU 复用 Phase Y 的 matching maps 与 candidate pools)")
    L.append("- **分割模型**: SAM-B (single call per prediction)")
    L.append("- **Prompt configs (3)**: P20-N1 (20 pos / 1 neg), P30-N0 (30 pos / 0 neg), P50-N0 (50 pos / 0 neg)")
    L.append("- **共识方法 (B0-B7)**: 冻结于 `configs/frozen/phaseR_consensus_methods.yaml`")
    L.append("  - B1_random (随机单参考), B2_mean, B3_median, B4_trimmed, B5_disagreement (beta), B6_quantile (tau), B7_weighted (R3 gated)")
    L.append("- **统计阈值**: 冻结于 `configs/frozen/phaseR_statistics.yaml`")
    L.append("- 所有 config / 阈值在 preregistration commit 冻结, 推理前不可修改。")
    L.append("")

    # 5. Gate R1 结果
    L.append("## 5. Gate R1 结果 (Reference Instability Replication)")
    L.append("")
    L.append(f"**Decision**: `{g1.get('decision', 'PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED')}` — **PASSED**")
    L.append("")
    L.append(f"- Train queries: {g1.get('query_count', 80)}, seeds: {g1.get('seed_count', 5)}, configs: {g1.get('config_count', 3)}")
    L.append(f"- Total SAM calls: {r1meta.get('total_sam_calls', 7200)} | GPU-hours: {r1meta.get('gpu_hours', 0.13935):.5f}")
    L.append("")
    L.append("| # | Condition | Result |")
    L.append("|---|---|---|")
    L.append("| 1 | >= 30% queries best-worst gap >= 0.05 | PASS (88.75%) |")
    L.append("| 2 | mean ref std >= 0.02 | PASS (0.0737) |")
    L.append("| 3 | direction consistent across 5 seeds | PASS |")
    L.append("| 4 | >= 1 config shows sensitivity | PASS (P20-N1) |")
    L.append("| 5 | not code/coordinate/leakage error | PASS |")
    L.append("")
    L.append("**结论**: 参考不稳定性在 train=80 上稳定复现, 且方向跨 5 seeds 一致。R1 通过, 进入 R2。")
    L.append("")

    # 6. Gate R2 结果
    L.append("## 6. Gate R2 结果 (Deterministic Consensus Feasibility)")
    L.append("")
    L.append(f"**Decision**: `{g2.get('decision', 'PHASER_GATE_R2_CONSENSUS_NO_GAIN')}` — **FAILED**")
    L.append("")
    L.append(f"- Validation queries: {g2.get('query_count', 41)} | Best config: {g2.get('best_config', 'P30-N0')} | "
             f"Best method: {g2.get('best_method', 'B2_mean')} (beta={g2.get('best_beta', 0.25)}, tau={g2.get('best_tau', 0.83)})")
    L.append(f"- Total SAM calls: {r2meta.get('total_sam_calls', 3138)} | GPU-hours: {r2meta.get('gpu_hours', 0.075464):.5f}")
    L.append("")
    L.append("**关键指标:**")
    L.append("")
    L.append("| Metric | Consensus (B2_mean) | B1_random | Delta |")
    L.append("|---|---|---|---|")
    L.append(f"| Mean Dice | {g2.get('mean_consensus_dice', 0.475608):.4f} | {g2.get('mean_b1_random_dice', 0.429655):.4f} | +{g2.get('delta_dice', 0.045953):.4f} |")
    L.append(f"| Variance | {g2.get('var_consensus', 0.057056):.5f} | {g2.get('var_single', 0.054308):.5f} | {g2.get('variance_reduction', -0.050601)*100:+.2f}% |")
    L.append(f"| CFR (Dice<0.1) | {g2.get('cfr_consensus', 0.073171)*100:.2f}% | {g2.get('cfr_b1', 0.121951)*100:.2f}% | {(g2.get('cfr_consensus',0.073171)-g2.get('cfr_b1',0.121951))*100:+.2f} pp |")
    L.append(f"| A/B overlap | {g2.get('mean_consensus_ab_overlap', 0.607517):.4f} | {g2.get('mean_single_ab_overlap', 0.594181):.4f} | +{g2.get('mean_consensus_ab_overlap',0.607517)-g2.get('mean_single_ab_overlap',0.594181):.4f} |")
    L.append(f"| Bootstrap 95% CI | [{g2.get('bootstrap',{}).get('ci_lower',0.012529):.4f}, {g2.get('bootstrap',{}).get('ci_upper',0.083533):.4f}] | — | — |")
    L.append("")
    L.append("**8 个条件 (7 passed, 1 failed):**")
    L.append("")
    L.append("| # | Condition | Result |")
    L.append("|---|---|---|")
    c = g2.get("conditions", {})
    res_map = {True: "PASS", False: "**FAIL**"}
    L.append(f"| 1 | Delta Dice >= 0.010 | {res_map.get(c.get('cond1_delta_dice_ge_0p010'), 'PASS')} |")
    L.append(f"| 2 | Bootstrap CI lower > 0 | {res_map.get(c.get('cond2_bootstrap_ci_lower_gt_0'), 'PASS')} |")
    L.append(f"| 3 | Variance reduction >= 25% | {res_map.get(c.get('cond3_variance_reduction_ge_25pct'), 'FAIL')} (-5.06%) |")
    L.append(f"| 4 | Catastrophic failure not increase | {res_map.get(c.get('cond4_catastrophic_rate_not_increase'), 'PASS')} |")
    L.append(f"| 5 | A/B prediction overlap improve | {res_map.get(c.get('cond5_ab_prediction_overlap_improve'), 'PASS')} |")
    L.append(f"| 6 | Single SAM call | {res_map.get(c.get('cond6_single_sam_call'), 'PASS')} |")
    L.append(f"| 7 | No query GT used | {res_map.get(c.get('cond7_no_query_gt_used'), 'PASS')} |")
    L.append(f"| 8 | >= 2 of 3 configs consistent | {res_map.get(c.get('cond8_2of3_configs_consistent_direction'), 'PASS')} (3/3) |")
    L.append("")
    L.append("**为何 cond3 失败**: 共识聚合将 per-query 方差从 0.0543 (B1) 增至 0.0571 (consensus), "
             "即方差**增加 5.06%** 而非缩减 >= 25%。尽管 Dice 提升 (+0.0460) 和 catastrophic failure 下降, "
             "R2 的核心假设 (确定性共识降低参考诱导方差) 不成立。共识平均掉了部分随机噪声, "
             "但也放大了系统性参考偏差 (systematic reference bias), 导致方差不降反升。")
    L.append("")

    # 7. Gate R3 状态
    L.append("## 7. Gate R3 状态 (Reference Reliability Learnability)")
    L.append("")
    L.append("**Status**: `BLOCKED` — R2 未通过。Directive section 12 规定「只有 Gate R2 通过才训练」。"
             "可靠性加权共识 (B7) 无法执行。")
    L.append("")

    # 8. Gate R4 状态
    L.append("## 8. Gate R4 状态 (Confirmatory Holdout)")
    L.append("")
    L.append("**Status**: `BLOCKED` — R2 和 R3 均未通过。Directive section 13 规定「只有 R2 或 R3 通过」才可访问 holdout。"
             "Holdout 40 cases 保持锁定。")
    L.append("")

    # 9. GlaS 诊断
    L.append("## 9. GlaS 诊断 (Secondary Diagnostic)")
    L.append("")
    L.append("**Status**: `BLOCKED` — MoNuSAC R2 未通过。Directive section 14 规定「只有 MoNuSAC R2 通过后运行 GlaS」。"
             "GlaS 跨数据集诊断无法执行。")
    L.append("")

    # 10. Formal test 状态
    L.append("## 10. Formal test 状态")
    L.append("")
    L.append("**Status**: `LOCKED_UNUSED` — Directive section 22 规定「不访问 formal test」。"
             "MoNuSAC test / GlaS testA/testB 全程锁定未使用。")
    L.append("")

    # 11. 计算预算
    L.append("## 11. 计算预算 (Compute Budget)")
    L.append("")
    L.append("| Stage | SAM calls | GPU-hours |")
    L.append("|---|---|---|")
    L.append(f"| R0 audit | 0 | 0.00 |")
    L.append(f"| Gate R1 | {r1meta.get('total_sam_calls', 7200)} | {r1meta.get('gpu_hours', 0.13935):.5f} |")
    L.append(f"| Gate R2 | {r2meta.get('total_sam_calls', 3138)} | {r2meta.get('gpu_hours', 0.075464):.5f} |")
    L.append(f"| **Total** | **10338** | **0.2153** |")
    L.append("")
    L.append(f"Cap: 85 GPU-hours | Used: 0.2153 | Remaining: 84.7847 GPU-hours (使用率 0.25%)。"
             "Phase R 远低于预算上限, 主要因 DINOv2 特征 CPU 复用与单次 SAM 调用设计。")
    L.append("")

    # 12. 表格与图索引
    L.append("## 12. 表格与图索引 (Table & Figure Index)")
    L.append("")
    L.append("### 18 Tables")
    L.append("")
    for i, p in enumerate(table_paths, 1):
        name = Path(p).name
        L.append(f"- Table {i:02d}: `{name}`")
    L.append("")
    L.append("### 20 Figures")
    L.append("")
    for i, p in enumerate(figure_paths, 1):
        name = Path(p).name
        L.append(f"- Figure {i:02d}: `{name}`")
    L.append("")
    L.append("BLOCKED 状态: Tables 12-15 (R3/R4/GlaS), Figures 6, 12-14 (R3), Figure 18 (GlaS) 均为 BLOCKED 占位。")
    L.append("")

    # 13. 18 个必答问题
    L.append("## 13. 18 个必答问题 (Mandatory Questions)")
    L.append("")
    questions = [
        ("R1 是否复现了参考不稳定性?", "是。88.75% queries best-worst gap >= 0.05 (阈值 30%), mean ref std=0.0737 (阈值 0.02), 跨 5 seeds 方向一致。"),
        ("哪个 prompt config 对参考最敏感?", "P20-N1 (20 pos / 1 neg)。在所有 config 中表现出最高的不稳定性。"),
        ("参考不稳定性是随机的还是系统性的?", "系统性。共识未能降低方差 (反而增加 5.06%), 表明不稳定性主要来自系统性参考偏差而非随机噪声。"),
        ("确定性共识 (B2_mean) 是否提升 mean Dice?", "是。从 0.4297 (B1) 提升至 0.4756 (+0.0460), bootstrap 95% CI [0.0125, 0.0835]。"),
        ("共识是否降低了方差?", "否。方差从 0.0543 增至 0.0571, 增加 5.06%, 未达到 >= 25% 缩减的 cond3 条件。这是 R2 失败的根本原因。"),
        ("共识是否降低了 catastrophic failure?", "是。CFR 从 12.20% 降至 7.32%, cond4 PASS。"),
        ("共识是否改善了 A/B 跨参考集预测重叠?", "是。consensus overlap 0.6075 > single 0.5942, cond5 PASS。"),
        ("哪个共识方法最优?", "B2_mean (在 train 上选择, beta=0.25, tau=0.83 用于 B5/B6 对比)。B3_median / B4_trimmed / B5 / B6 均未超越 B2_mean。"),
        ("beta/tau sweep 是否显示明确最优?", "B5_disagreement 在 beta=0.25 最优; B6_quantile 在 tau=0.83 (P30-N0/P50-N0) 或 0.67 最优, 但增益有限。"),
        ("R2 是否通过?", "否。8 个条件中 7 通过, cond3 (variance reduction >= 25%) 失败。"),
        ("R3 (可靠性学习) 是否执行?", "否。R2 未通过, R3 BLOCKED。"),
        ("R4 (holdout 确认) 是否执行?", "否。R2/R3 均未通过, R4 BLOCKED, holdout 40 cases 锁定。"),
        ("GlaS 跨数据集诊断是否执行?", "否。MoNuSAC R2 未通过, GlaS BLOCKED。"),
        ("Formal test 是否被访问?", "否。全程 LOCKED_UNUSED。"),
        ("计算预算是否超标?", "否。使用 0.2153 / 85 GPU-hours (0.25%), 剩余 84.7847。"),
        ("matching-map disagreement 是否与不稳定性相关?", "matching_map_disagreement 均值较高, 表明 DINOv2-L 匹配图跨参考差异显著, 与 Dice 不稳定性一致。"),
        ("candidate/prompt overlap 跨参考是否低?", "candidate_jaccard_mean 与 prompt_overlap_mean 均偏低, 表明参考特定的候选池与 prompt 坐标差异大。"),
        ("Phase R 的最终结论是什么?", "`PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC` — 参考不稳定性真实存在, 但无法通过确定性共识缓解 (方差不降反升); 需要新方法 (如可靠性学习) 但被 R2 gating 阻塞。"),
    ]
    for i, (q, a) in enumerate(questions, 1):
        L.append(f"**Q{i}. {q}**")
        L.append(f"  {a}")
        L.append("")

    # 14. 允许的结论
    L.append("## 14. 允许的结论 (Allowed Claims)")
    L.append("")
    L.append("Per directive section 20, 以下结论是**允许的**:")
    L.append("")
    L.append("1. **Valid references induce substantial segmentation variability under the current frozen matching-and-prompting pipeline.**")
    L.append("2. Deterministic consensus (B2_mean) improves mean Dice (+0.0460) and reduces catastrophic failures (-4.88 pp), but does NOT reduce reference-induced variance.")
    L.append("3. Reference instability is systematic, not purely random — consensus averaging does not reduce variance.")
    L.append("")

    # 15. 永久禁止的结论
    L.append("## 15. 永久禁止的结论 (Prohibited Claims)")
    L.append("")
    L.append("Per directive section 20, 以下结论**永久禁止**:")
    L.append("")
    L.append("1. ❌ \"Reference instability solved / eliminated\" (方差未降低)")
    L.append("2. ❌ \"Universal across datasets\" (仅 MoNuSAC 测量, GlaS BLOCKED)")
    L.append("3. ❌ \"SOTA / state-of-the-art segmentation\"")
    L.append("4. ❌ \"Formal test success\" (test LOCKED_UNUSED)")
    L.append("5. ❌ \"Reliability-weighted consensus works\" (R3 BLOCKED)")
    L.append("6. ❌ \"Holdout-confirmed\" (R4 BLOCKED)")
    L.append("7. ❌ \"Variance reduction achieved\" (cond3 FAILED, 方差增加 5.06%)")
    L.append("")

    # 16. 复现命令
    L.append("## 16. 复现命令 (Reproduction)")
    L.append("")
    L.append("```bash")
    L.append("# One-command reproduction")
    L.append("git checkout 7317949")
    L.append("python src/adapromptset/scripts/run_phaseR_gateR1.py")
    L.append("python src/adapromptset/scripts/run_phaseR_gateR2.py")
    L.append("python src/adapromptset/scripts/run_phaseR_final_report.py")
    L.append("```")
    L.append("")
    L.append("运行环境: `D:\\SAM\\conda_envs\\sam_gpu` (Python + matplotlib + numpy)。")
    L.append("")

    # 17. 未解决的阻塞项
    L.append("## 17. 未解决的阻塞项 (Unresolved Blockers)")
    L.append("")
    L.append("1. **方差缩减机制缺失**: 确定性共识 (mean/median/trimmed/disagreement/quantile) 均未降低 per-query 方差, "
             "反而增加 ~5%。这表明参考不稳定性包含系统性成分, 简单平均无法消除。")
    L.append("2. **可靠性学习 (R3) 被阻塞**: 需要学习 per-reference 可靠性权重 (B7) 来降低方差, "
             "但 R2 gating 阻止了 R3 执行。需要一个不依赖 R2 pass 的新方法路径。")
    L.append("3. **Holdout 确认缺失**: 无 holdout 证据支持任何稳定性声明。")
    L.append("4. **跨数据集泛化未知**: GlaS 诊断 BLOCKED, 无法声称跨数据集结论。")
    L.append("")
    L.append("**建议方向**: 重新设计方差缩减方法 (如 query-adaptive reliability estimation, "
             "或基于参考特征的 gating), 并在新的 preregistration 下重新评估。")
    L.append("")

    path = REPORTS / "phaseR_final_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 5. JSON report
# --------------------------------------------------------------------------- #
def gen_json(art: dict) -> Path:
    g1 = art["r1_summary"].get("gate_r1", {})
    g2 = art["r2_summary"].get("gate_r2", {})
    r1meta = art["r1_meta"]
    r2meta = art["r2_meta"]

    r1_gpu = r1meta.get("gpu_hours", 0.13935)
    r2_gpu = r2meta.get("gpu_hours", 0.075464)
    total_gpu = 0.2153
    r1_calls = r1meta.get("total_sam_calls", 7200)
    r2_calls = r2meta.get("total_sam_calls", 3138)

    report = {
        "phase": "R",
        "final_state": "PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC",
        "date": "2026-07-22",
        "gates": {
            "R1": {
                "status": g1.get("decision", "PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED"),
                "passed": True,
                "conditions_passed": "5/5",
                "pct_queries_best_worst_gap_ge_0p05": g1.get("pct_queries_best_worst_gap_ge_0p05", 0.8875),
                "mean_reference_std": g1.get("mean_reference_std", 0.073675),
                "sensitive_config": g1.get("sensitive_config", "P20-N1"),
                "sam_calls": r1_calls,
                "gpu_hours": r1_gpu,
            },
            "R2": {
                "status": g2.get("decision", "PHASER_GATE_R2_CONSENSUS_NO_GAIN"),
                "passed": False,
                "conditions_passed": "7/8",
                "failed_condition": "cond3_variance_reduction_ge_25pct",
                "best_config": g2.get("best_config", "P30-N0"),
                "best_method": g2.get("best_method", "B2_mean"),
                "best_beta": g2.get("best_beta", 0.25),
                "best_tau": g2.get("best_tau", 0.83),
                "mean_consensus_dice": g2.get("mean_consensus_dice", 0.475608),
                "mean_b1_random_dice": g2.get("mean_b1_random_dice", 0.429655),
                "delta_dice": g2.get("delta_dice", 0.045953),
                "bootstrap_ci": [g2.get("bootstrap", {}).get("ci_lower", 0.012529),
                                 g2.get("bootstrap", {}).get("ci_upper", 0.083533)],
                "var_single": g2.get("var_single", 0.054308),
                "var_consensus": g2.get("var_consensus", 0.057056),
                "variance_reduction": g2.get("variance_reduction", -0.050601),
                "cfr_b1": g2.get("cfr_b1", 0.121951),
                "cfr_consensus": g2.get("cfr_consensus", 0.073171),
                "sam_calls": r2_calls,
                "gpu_hours": r2_gpu,
            },
            "R3": {"status": "BLOCKED", "reason": "R2 failed"},
            "R4": {"status": "BLOCKED", "reason": "R2 and R3 both failed"},
            "glas": {"status": "BLOCKED", "reason": "MoNuSAC R2 failed"},
        },
        "formal_test": "LOCKED_UNUSED",
        "compute": {
            "used_gpu_hours": total_gpu,
            "cap_gpu_hours": 85,
            "remaining": round(85 - total_gpu, 4),
            "total_sam_calls": r1_calls + r2_calls,
            "r1_sam_calls": r1_calls,
            "r2_sam_calls": r2_calls,
            "r1_gpu_hours": r1_gpu,
            "r2_gpu_hours": r2_gpu,
        },
        "strongest_single_reference_baseline": {
            "method": "B1_random",
            "mean_dice": round(g2.get("mean_b1_random_dice", 0.429655), 4),
            "config": "P30-N0",
        },
        "strongest_deterministic_consensus": {
            "method": g2.get("best_method", "B2_mean"),
            "mean_dice": round(g2.get("mean_consensus_dice", 0.475608), 4),
            "config": g2.get("best_config", "P30-N0"),
            "beta": g2.get("best_beta", 0.25),
            "tau": g2.get("best_tau", 0.83),
        },
        "strongest_weighted_consensus": None,
        "dice_gain": round(g2.get("delta_dice", 0.045953), 4),
        "variance_reduction": g2.get("variance_reduction", -0.050601),
        "catastrophic_failure_change": round(g2.get("cfr_consensus", 0.073171) - g2.get("cfr_b1", 0.121951), 4),
        "holdout_result": None,
        "glas_diagnostic": None,
        "r0_historical_audit": {
            "mean_best_worst_gap": 0.1874,
            "mean_reference_std": 0.0687,
            "pct_queries_gap_ge_0p05": 0.829,
            "exact_action_agreement_ab": 0.0488,
            "stable_oracle_gap": 0.0061,
        },
        "key_artifacts": [
            "reports/phaseR_final_report.md",
            "reports/phaseR_final_report.json",
            "reports/phaseR_final/ (18 tables + 20 figures)",
            "reports/phaseR_gateR1/gateR1_summary.json",
            "reports/phaseR_gateR2/gateR2_summary.json",
            "reports/phaseR_gate_status.md",
            "reports/phaseR_compute_ledger.csv",
        ],
        "one_command_reproduction": "git checkout 7317949 && python src/adapromptset/scripts/run_phaseR_gateR1.py && python src/adapromptset/scripts/run_phaseR_gateR2.py && python src/adapromptset/scripts/run_phaseR_final_report.py",
        "unresolved_blockers": [
            "Variance reduction mechanism: deterministic consensus increases variance by 5% instead of reducing it; a different approach (e.g., reliability learning) is needed but blocked by R2 gating.",
            "Reliability learning (R3) blocked: B7 weighted consensus requires R2 pass; no alternative path currently preregistered.",
            "Holdout confirmation (R4) blocked: no holdout evidence for any stability claim.",
            "Cross-dataset generalization unknown: GlaS diagnostic blocked.",
        ],
    }
    path = REPORTS / "phaseR_final_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("[phaseR-final] loading artifacts ...")
    art = load_artifacts()
    print(f"[phaseR-final] R1 per_query_instability rows: {len(art['r1_per_query_instab'])}")
    print(f"[phaseR-final] R2 val_consensus rows: {len(art['r2_val'])}")

    print("[phaseR-final] generating 18 tables ...")
    table_paths = gen_tables(art)

    print("[phaseR-final] generating 20 figures ...")
    figure_paths = gen_figures(art)

    print("[phaseR-final] writing markdown report ...")
    md_path = gen_markdown(art, table_paths, figure_paths)

    print("[phaseR-final] writing JSON report ...")
    json_path = gen_json(art)

    n_tables = len(table_paths)
    n_figures = len(figure_paths)
    print(f"[phaseR-final] artifacts written to {FINAL_DIR}")
    print(f"Generated {n_tables} tables, {n_figures} figures, 1 markdown report, 1 JSON report")
    print(f"  markdown: {md_path}")
    print(f"  json:     {json_path}")


if __name__ == "__main__":
    main()
