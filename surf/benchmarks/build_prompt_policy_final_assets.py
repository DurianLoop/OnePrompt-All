from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/research_direction/paper_artifacts_20260715"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_column(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def effect_row(dataset: str, protocol: str, comparison: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "protocol": protocol,
        "comparison": comparison,
        "n": int(summary["case_count"]),
        "anchor_dice": float(summary["mean_fixed_dice"]),
        "method_dice": float(summary["mean_learned_dice"]),
        "delta_dice": float(summary["mean_delta_dice"]),
        "ci95_low": float(summary["ci95_low"]),
        "ci95_high": float(summary["ci95_high"]),
        "wins_ties_losses": f"{summary['wins']}/{summary['ties']}/{summary['losses']}",
        "sign_test_p": summary.get("sign_test_p_two_sided", ""),
    }


def fixed_budget_mean(rows: list[dict[str, Any]], requested: int) -> float:
    matches = []
    for row in rows:
        budget_text = row.get("requested_negative_count")
        budget = int(budget_text if budget_text not in (None, "") else row["negative_count"])
        if budget == requested:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row for N={requested}, found {len(matches)}")
    return float(matches[0]["mean_dice"])


def build_assets() -> None:
    nested = read_json(ROOT / "runs/research_direction/nested_cv5_negative_budget/aggregate/paired_summary.json")
    fg_bg = read_json(OUT / "nested_cv_fg_vs_fgbg/paired_summary.json")
    vs_n5 = read_json(OUT / "nested_cv_vs_global_n5/paired_summary.json")
    oracle = mean_column(read_csv(OUT / "nested_cv_oracle_gap.csv"), "oracle_dice")
    nested_budgets = read_csv(OUT / "nested_cv_fixed_budget_summary.csv")

    zero_patch = read_json(ROOT / "runs/research_direction/monusac_prompt_policy/negative_gate_ref1_frozen_v2/paired_analysis/paired_summary.json")
    zero_patient = read_json(ROOT / "runs/research_direction/monusac_prompt_policy/negative_gate_ref1_frozen_v2/patient_level_analysis/patient_summary.json")
    domain_patch = read_json(ROOT / "runs/research_direction/monusac_prompt_policy/within_domain_negative_gate/policy/paired_analysis/paired_summary.json")
    domain_patient = read_json(ROOT / "runs/research_direction/monusac_prompt_policy/within_domain_negative_gate/policy/patient_level_analysis/patient_summary.json")
    monusac_budgets = read_csv(ROOT / "runs/research_direction/monusac_prompt_policy/negative_gate_ref1_grid_v2/summary.csv")
    sammed2d = read_json(ROOT / "runs/research_direction/nested_cv5_sammed2d_negative_gate_n0n5/aggregate/paired_summary.json")

    effects = [
        effect_row("MoNuSeg", "5-fold nested CV, 37 queries", "FG-BG N0 vs FG-only", fg_bg),
        effect_row("MoNuSeg", "5-fold nested CV, 37 queries", "Learned gate vs fold anchor", nested),
        effect_row("MoNuSeg", "5-fold nested CV, 37 queries", "Learned gate vs global N5", vs_n5),
        effect_row("MoNuSAC", "zero-shot, 101 patches", "Frozen MoNuSeg gate vs N0", zero_patch),
        effect_row("MoNuSAC", "zero-shot, 25 patients", "Frozen MoNuSeg gate vs N0", zero_patient),
        effect_row("MoNuSAC", "within-domain, 101 patches", "Domain gate vs N0", domain_patch),
        effect_row("MoNuSAC", "within-domain, 25 patients", "Domain gate vs N0", domain_patient),
        effect_row("MoNuSeg", "SAM-Med2D 5-fold nested CV", "SAM-Med2D gate vs fold anchor", sammed2d),
    ]
    write_csv(OUT / "final_paired_effects.csv", effects)

    performance = [
        {"dataset": "MoNuSeg", "protocol": "nested_cv", "method": "FG-only K40", "dice": float(fg_bg["mean_fixed_dice"])},
        {"dataset": "MoNuSeg", "protocol": "nested_cv", "method": "FG-BG N0", "dice": float(fg_bg["mean_learned_dice"])},
        {"dataset": "MoNuSeg", "protocol": "nested_cv", "method": "Global fixed N5", "dice": fixed_budget_mean(nested_budgets, 5)},
        {"dataset": "MoNuSeg", "protocol": "nested_cv", "method": "Learned gate", "dice": float(nested["mean_learned_dice"])},
        {"dataset": "MoNuSeg", "protocol": "nested_cv", "method": "Per-query oracle", "dice": oracle},
        {"dataset": "MoNuSAC", "protocol": "patient_disjoint_test", "method": "FG-BG N0", "dice": fixed_budget_mean(monusac_budgets, 0)},
        {"dataset": "MoNuSAC", "protocol": "patient_disjoint_test", "method": "Frozen MoNuSeg gate", "dice": float(zero_patch["mean_learned_dice"])},
        {"dataset": "MoNuSAC", "protocol": "patient_disjoint_test", "method": "Within-domain gate", "dice": float(domain_patch["mean_learned_dice"])},
        {"dataset": "MoNuSAC", "protocol": "patient_disjoint_test", "method": "Global fixed N5", "dice": fixed_budget_mean(monusac_budgets, 5)},
    ]
    write_csv(OUT / "final_performance_table.csv", performance)
    write_csv(OUT / "cross_segmenter_gate_table.csv", [
        {
            "segmenter": "SAM ViT-B",
            "query_count": nested["case_count"],
            "anchor_dice": nested["mean_fixed_dice"],
            "learned_dice": nested["mean_learned_dice"],
            "delta_dice": nested["mean_delta_dice"],
            "ci95_low": nested["ci95_low"],
            "ci95_high": nested["ci95_high"],
        },
        {
            "segmenter": "SAM-Med2D",
            "query_count": sammed2d["case_count"],
            "anchor_dice": sammed2d["mean_fixed_dice"],
            "learned_dice": sammed2d["mean_learned_dice"],
            "delta_dice": sammed2d["mean_delta_dice"],
            "ci95_low": sammed2d["ci95_low"],
            "ci95_high": sammed2d["ci95_high"],
        },
    ])
    runtime_rows = []
    for runtime_file in ("runtime_sam_vit_b.json", "runtime_sammed2d.json"):
        runtime = read_json(OUT / runtime_file)
        runtime_rows.append({
            "backend": runtime["backend"],
            "parameters_million": runtime["model_parameters_million"],
            "model_load_seconds": runtime["model_load_seconds"],
            "image_embedding_seconds": runtime["image_embedding_seconds"],
            "single_mask_decode_median_seconds": runtime["single_mask_decode"]["median_seconds"],
            "two_candidate_decode_seconds": runtime["two_candidate_decode_seconds"],
            "peak_cuda_memory_mib": runtime["peak_cuda_memory_mib"],
            "gpu": runtime["gpu"],
            "scope": "segmenter only; excludes DINO matching and disk I/O",
        })
    write_csv(OUT / "runtime_table.csv", runtime_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    palette = ["#4f6f82", "#3b8f78", "#d0a12e", "#c9553d", "#6d5a8c"]
    for axis, dataset in zip(axes[:2], ("MoNuSeg", "MoNuSAC")):
        rows = [
            row for row in performance
            if row["dataset"] == dataset
            and (dataset != "MoNuSeg" or row["protocol"] == "nested_cv")
        ]
        x = np.arange(len(rows))
        values = [float(row["dice"]) for row in rows]
        axis.bar(x, values, color=palette[: len(rows)])
        axis.set_xticks(x, [str(row["method"]).replace(" ", "\n") for row in rows], fontsize=9)
        axis.set_ylabel("Mean Dice")
        axis.set_title(dataset + (" 5-fold nested CV" if dataset == "MoNuSeg" else " patient-disjoint test"))
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(max(0.0, min(values) - 0.04), min(1.0, max(values) + 0.04))
        for index, value in enumerate(values):
            axis.text(index, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    y = np.arange(len(effects))
    values = np.asarray([float(row["delta_dice"]) for row in effects])
    lows = np.asarray([float(row["ci95_low"]) for row in effects])
    highs = np.asarray([float(row["ci95_high"]) for row in effects])
    labels = [
        "MoNuSeg FG-BG",
        "MoNuSeg gate/fold anchor",
        "MoNuSeg gate/global N5",
        "MoNuSAC zero-shot patch",
        "MoNuSAC zero-shot patient",
        "MoNuSAC domain patch",
        "MoNuSAC domain patient",
        "SAM-Med2D gate/fold anchor",
    ]
    for index, value in enumerate(values):
        color = "#c9553d" if lows[index] > 0 else "#4f6f82"
        axes[2].errorbar(
            value,
            y[index],
            xerr=[[value - lows[index]], [highs[index] - value]],
            fmt="o",
            capsize=4,
            color=color,
            ecolor=color,
        )
    axes[2].axvline(0, color="#222222", linestyle="--", linewidth=1)
    axes[2].set_yticks(y, labels, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Paired Delta Dice with bootstrap 95% CI")
    axes[2].set_title("Effect estimates and uncertainty")
    axes[2].grid(axis="x", alpha=0.22)
    fig.savefig(OUT / "final_main_results.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "monuseg_oracle_dice": oracle,
        "monuseg_nested_gate_closes_oracle_gap_pct": 100.0 * (
            float(nested["mean_learned_dice"]) - fixed_budget_mean(nested_budgets, 5)
        ) / (oracle - fixed_budget_mean(nested_budgets, 5)),
        "monusac_best_fixed_budget": 5,
        "monusac_best_fixed_dice": fixed_budget_mean(monusac_budgets, 5),
        "sammed2d_gate_delta_dice": sammed2d["mean_delta_dice"],
        "sammed2d_gate_ci95": [sammed2d["ci95_low"], sammed2d["ci95_high"]],
        "note": "MoNuSAC learned gates improve N0 anchors but do not exceed the post-hoc global fixed N5.",
    }
    (OUT / "final_results_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    build_assets()
