"""Verify the evidence snapshot used by the ICASSP draft.

This script is intentionally dependency-free.  It verifies that the values printed
in the manuscript's result table are present and mutually consistent in the
copied, frozen CSV/JSON evidence files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
PAPER = ROOT / "paper"


def read_csv(name: str) -> list[dict[str, str]]:
    with (EVIDENCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def assertion(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": passed, "detail": detail}


def main() -> int:
    results = read_csv("final_performance_table.csv")
    effects = read_csv("final_paired_effects.csv")
    metadata = json.loads((EVIDENCE / "final_results_metadata.json").read_text(encoding="utf-8-sig"))
    device = json.loads((EVIDENCE / "device_consistency_audit.json").read_text(encoding="utf-8-sig"))
    fob = json.loads((EVIDENCE / "fob_fold0_adaptation_summary.json").read_text(encoding="utf-8-sig"))
    pannuke_smoke = json.loads((EVIDENCE / "pannuke_interface_smoke.json").read_text(encoding="utf-8-sig"))
    pannuke = json.loads((EVIDENCE / "pannuke_zero_shot_summary.json").read_text(encoding="utf-8-sig"))
    latency = json.loads((EVIDENCE / "end_to_end_latency_summary.json").read_text(encoding="utf-8-sig"))
    pannuke_supports = json.loads((EVIDENCE / "pannuke_support_sensitivity_summary.json").read_text(encoding="utf-8-sig"))
    pannuke_tissues = json.loads((EVIDENCE / "pannuke_tissue_sensitivity_summary.json").read_text(encoding="utf-8-sig"))
    instance_metrics = json.loads((EVIDENCE / "nested_instance_metrics_summary.json").read_text(encoding="utf-8-sig"))
    index = {(r["dataset"], r["method"]): float(r["dice"]) for r in results}
    effect_index = {r["comparison"]: r for r in effects}
    patient_effect = next(
        row for row in effects
        if row["comparison"] == "Frozen MoNuSeg gate vs N0" and row["protocol"] == "zero-shot, 25 patients"
    )

    checks: list[dict[str, object]] = []
    checks.append(assertion(
        "fg_bg_mean_matches_performance_table",
        close(index[("MoNuSeg", "FG-BG N0")], float(effect_index["FG-BG N0 vs FG-only"]["method_dice"])),
        "MoNuSeg FG-BG N0 mean is identical across the performance and paired-effect tables.",
    ))
    checks.append(assertion(
        "learned_gate_mean_matches_two_comparisons",
        close(index[("MoNuSeg", "Learned gate")], float(effect_index["Learned gate vs fold anchor"]["method_dice"]))
        and close(index[("MoNuSeg", "Learned gate")], float(effect_index["Learned gate vs global N5"]["method_dice"])),
        "The main learned-gate Dice is 0.5475548066554488 in all three records.",
    ))
    checks.append(assertion(
        "primary_ci_excludes_zero",
        float(effect_index["Learned gate vs fold anchor"]["ci95_low"]) > 0
        and float(effect_index["Learned gate vs global N5"]["ci95_low"]) > 0,
        "Both MoNuSeg gate comparisons have positive reported 95% CI lower bounds.",
    ))
    checks.append(assertion(
        "oracle_gap_metadata",
        close(float(metadata["monuseg_oracle_dice"]), index[("MoNuSeg", "Per-query oracle")])
        and 66.0 < float(metadata["monuseg_nested_gate_closes_oracle_gap_pct"]) < 67.0,
        "The diagnostic oracle mean and 66.1% gap closure are mutually consistent.",
    ))
    checks.append(assertion(
        "external_boundary_preserved",
        index[("MoNuSAC", "Frozen MoNuSeg gate")] > index[("MoNuSAC", "FG-BG N0")]
        and index[("MoNuSAC", "Frozen MoNuSeg gate")] < index[("MoNuSAC", "Global fixed N5")],
        "The transfer gate improves N0 but does not beat post-hoc global N5; this boundary must remain in the paper.",
    ))
    checks.append(assertion(
        "monusac_patient_aggregate_is_positive_but_sign_test_conservative",
        patient_effect["protocol"] == "zero-shot, 25 patients"
        and int(patient_effect["n"]) == 25
        and float(patient_effect["ci95_low"]) > 0
        and float(patient_effect["sign_test_p"]) > 0.05,
        "The patient-aggregated MoNuSAC bootstrap CI is positive, while its sign test is not below 0.05; both facts must be preserved.",
    ))
    checks.append(assertion(
        "segmenter_negative_transfer_preserved",
        float(effect_index["SAM-Med2D gate vs fold anchor"]["delta_dice"]) < 0,
        "SAM-Med2D is a non-positive transfer result and may not be presented as support.",
    ))
    checks.append(assertion(
        "device_audit_passed",
        bool(device["passed"]) and float(device["max_abs_dice_delta"]) <= float(device["thresholds"]["dice"]),
        "The copied 128-pair CPU/GPU numerical-consistency audit passes its Dice tolerance.",
    ))
    checks.append(assertion(
        "fob_adaptation_is_heldout_negative_diagnostic",
        fob["status"] == "ADAPTED_FOB_FOLD0_DIAGNOSTIC_COMPLETE"
        and bool(fob["outer_test"]["evaluated_once_after_inner_selection"])
        and fob["inner_validation_selection"]["selected_checkpoint"] == "step_0250.pth"
        and float(fob["outer_test"]["mean_dice"]) < 0.01,
        "The one-fold FoB adaptation selected on 30 inner-validation episodes and tested once on 40 held-out episodes; it is a negative diagnostic, not a head-to-head claim.",
    ))
    checks.append(assertion(
        "pannuke_smoke_remains_non_result_interface_check",
        pannuke_smoke["status"] == "PANNUKE_FROZEN_PIPELINE_INTERFACE_PASS"
        and "not a cohort result" in pannuke_smoke["scope"]
        and pannuke_smoke["interface"]["dino_feature_and_coordinate_path"] == "PASS"
        and pannuke_smoke["interface"]["sam_b_decode_path"] == "PASS",
        "The PanNuke artifact validates the frozen pipeline interface on one pair and explicitly forbids treating it as cohort evidence.",
    ))
    checks.append(assertion(
        "pannuke_full_gate_is_version_isolated_and_positive_over_n0",
        pannuke["status"] == "GATE_COMPLETE"
        and int(pannuke["query_count"]) == 2540
        and pannuke["sklearn_versions_by_fold"] == {"0": "1.7.2", "1": "1.7.2", "2": "1.9.0", "3": "1.9.0", "4": "1.9.0"}
        and float(pannuke["gate_vs_n0"]["ci95_low"]) > 0
        and float(pannuke["n5_vs_n0"]["ci95_high"]) < 0,
        "The full field-level PanNuke protocol has 2,540 query tiles; the version-isolated five-fold gate improves N0 while fixed N5 degrades it.",
    ))
    checks.append(assertion(
        "end_to_end_latency_is_declared_and_gt_free",
        latency["status"] == "COMPLETE"
        and int(latency["measurement"]["measured_episodes"]) == 30
        and not bool(latency["measurement"]["query_gt_opened"])
        and latency["pipeline"]["segmenter"] == "SAM ViT-B"
        and latency["pipeline"]["matcher"] == "DINOv2-L foreground-background"
        and 0.0 < float(latency["seconds"]["full_support_query_episode_s"]["median_s"]) < 1.0,
        "The synchronized 30-episode timing uses declared DINOv2-L + SAM ViT-B hardware/input settings and does not open query labels.",
    ))
    checks.append(assertion(
        "pannuke_support_strata_preserve_direction_descriptively",
        pannuke_supports["status"] == "DESCRIPTIVE_COMPLETE"
        and int(pannuke_supports["support_count"]) == 5
        and bool(pannuke_supports["all_gate_vs_n0_ci_lower_positive"])
        and bool(pannuke_supports["all_n5_vs_n0_ci_upper_negative"]),
        "All five fixed-support descriptive strata retain a positive gate-vs-N0 CI lower bound and a negative fixed-N5-vs-N0 CI upper bound; this is not a support-selection result.",
    ))
    checks.append(assertion(
        "pannuke_tissue_strata_report_both_positive_and_indeterminate_results",
        pannuke_tissues["status"] == "DESCRIPTIVE_COMPLETE"
        and int(pannuke_tissues["tissue_code_count"]) == 19
        and int(pannuke_tissues["gate_positive_ci_strata"]) == 15
        and int(pannuke_tissues["gate_negative_ci_strata"]) == 0
        and int(pannuke_tissues["gate_indeterminate_ci_strata"]) == 4,
        "The post-hoc public-tissue analysis reports 15 positive, 4 indeterminate, and no negative gate-vs-N0 strata rather than upgrading subgroup evidence to a primary claim.",
    ))
    checks.append(assertion(
        "nested_instance_diagnostic_preserves_dice_only_claim_boundary",
        instance_metrics["status"] == "DIAGNOSTIC_COMPLETE"
        and int(instance_metrics["query_count"]) == 37
        and float(instance_metrics["metrics"]["object_recall"]["ci95_low"]) < 0 < float(instance_metrics["metrics"]["object_recall"]["ci95_high"])
        and float(instance_metrics["metrics"]["merge_error"]["ci95_low"]) < 0 < float(instance_metrics["metrics"]["merge_error"]["ci95_high"]),
        "The nested instance-oriented diagnostic does not establish a positive object-recall or merge-error effect, so the manuscript must limit its gain claim to union-mask Dice.",
    ))
    required_figures = ["main_results.png", "monuseg_qualitative.png", "monusac_qualitative.png"]
    checks.append(assertion(
        "all_manuscript_figures_present",
        all((PAPER / "figures" / name).is_file() for name in required_figures),
        "All copied raster figures referenced by the manuscript are present.",
    ))

    report = {
        "audit": "icassp_evidence_snapshot",
        "status": "PASS" if all(bool(c["passed"]) for c in checks) else "FAIL",
        "checks": checks,
        "evidence_files": sorted(p.name for p in EVIDENCE.iterdir() if p.is_file()),
        "notes": [
            "This validates cross-file integrity of the frozen evidence snapshot.",
            "It does not re-run GPU feature extraction, SAM inference, or policy training.",
        ],
    }
    output = EVIDENCE / "verification_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(checks)} checks; report written to {output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
