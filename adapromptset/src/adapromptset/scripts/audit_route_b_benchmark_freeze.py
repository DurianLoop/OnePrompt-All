from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/experiments/route_b_promptability_benchmark_v1.json")
DDR_REPORT_PATH = Path("reports/route_b_ddr_manifest/ddr_route_b_manifest_report.json")

REQUIRED_METRICS = {
    "promptability_gap",
    "localization_advantage",
    "interface_gain",
    "oracle_adaptation_value",
    "promptability_reliability",
}
REQUIRED_REPRESENTATIONS = {
    "point",
    "multi_point",
    "grouped_point",
    "box",
    "multi_box",
    "hard_mask",
    "soft_mask",
    "direct_matching_mask",
    "privileged_gt_point",
    "privileged_gt_box",
    "privileged_gt_mask",
}
REQUIRED_CONTROLS = {
    "spatial_random_point",
    "size_matched_random_box",
    "full_image_box",
    "alternative_reference_control",
    "strongest_fixed_representation",
    "raw_matching_baseline",
    "privileged_gt_upper_bound",
}
REQUIRED_ROW_FAMILIES = {
    "automatic",
    "matched_control",
    "privileged_upper_bound",
    "direct_matching",
    "diagnostic_baseline",
}
REQUIRED_OUTPUT_COLUMNS = {
    "dataset",
    "split",
    "image_id",
    "intent",
    "representation",
    "row_family",
    "control_family",
    "matcher",
    "segmenter",
    "dice",
    "iou",
    "sam_calls",
    "runtime_seconds",
    "valid_automatic_row",
    "uses_query_gt_for_generation",
    "formal_test_row",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _missing(required: set[str], observed: list[str] | set[str]) -> list[str]:
    return sorted(required - set(observed))


def audit_route_b_benchmark_freeze(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config_abs = project_root / CONFIG_PATH
    ddr_report_abs = project_root / DDR_REPORT_PATH
    config = _read_json(config_abs)
    ddr_report = _read_json(ddr_report_abs)

    observed_metrics = [item["name"] for item in config.get("metrics", [])]
    output_columns = config.get("output_schema", {}).get("query_level_required_columns", [])
    findings = {
        "missing_metrics": _missing(REQUIRED_METRICS, observed_metrics),
        "missing_representations": _missing(REQUIRED_REPRESENTATIONS, config.get("prompt_representations", [])),
        "missing_controls": _missing(REQUIRED_CONTROLS, config.get("controls", [])),
        "missing_row_families": _missing(REQUIRED_ROW_FAMILIES, config.get("row_families", [])),
        "missing_output_columns": _missing(REQUIRED_OUTPUT_COLUMNS, output_columns),
        "config_disallows_formal_test": config.get("formal_test_access_allowed") is False,
        "ddr_manifest_pass": ddr_report.get("status") == "PASS_DDR_ROUTE_B_DEV_MANIFEST",
        "ddr_test_files_read": ddr_report.get("test_files_read") is True,
        "ddr_formal_test_access": ddr_report.get("formal_test_access") is True,
    }
    issues: list[str] = []
    for key in [
        "missing_metrics",
        "missing_representations",
        "missing_controls",
        "missing_row_families",
        "missing_output_columns",
    ]:
        if findings[key]:
            issues.append(key)
    if not findings["config_disallows_formal_test"]:
        issues.append("config_allows_formal_test")
    if not findings["ddr_manifest_pass"]:
        issues.append("ddr_manifest_not_passed")
    if findings["ddr_test_files_read"]:
        issues.append("ddr_test_files_read")
    if findings["ddr_formal_test_access"]:
        issues.append("ddr_formal_test_access")

    status = "PASS_ROUTE_B_B1_PREREQ_FREEZE" if not issues else "FAIL_ROUTE_B_B1_PREREQ_FREEZE"
    summary = {
        "schema_version": "adapromptset.route_b.benchmark_freeze_audit.v1",
        "date": "2026-07-19",
        "status": status,
        "route": "B_reference_promptability_benchmark",
        "config": str(CONFIG_PATH),
        "config_sha256": _sha256(config_abs),
        "ddr_manifest_report": str(DDR_REPORT_PATH),
        "ddr_manifest_report_sha256": _sha256(ddr_report_abs),
        "findings": findings,
        "issues": issues,
        "authorizes": "development_only_evidence_generation" if status == "PASS_ROUTE_B_B1_PREREQ_FREEZE" else "nothing",
        "does_not_authorize": [
            "formal_test_access",
            "submission_ready_claim",
            "method_positive_claim",
            "test_tuned_thresholds",
        ],
    }

    out_dir = project_root / "reports" / "route_b_benchmark_freeze"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "route_b_benchmark_freeze_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "route_b_benchmark_freeze_audit.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Route B Benchmark Freeze Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Config: `{summary['config']}`",
        f"DDR manifest report: `{summary['ddr_manifest_report']}`",
        "",
        "## Findings",
        "",
    ]
    for key, value in summary["findings"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- Authorizes: `{summary['authorizes']}`",
            "- Does not authorize formal test access, submission-ready claims, method-positive claims, or test-tuned thresholds.",
        ]
    )
    if summary["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in summary["issues"]:
            lines.append(f"- `{issue}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Route B metric/control freeze before development inference.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    audit_route_b_benchmark_freeze(project_root=args.project_root)


if __name__ == "__main__":
    main()
