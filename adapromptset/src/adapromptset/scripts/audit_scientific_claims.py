from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROHIBITED_CLAIM_TOKENS = [
    "state of the art",
    "state-of-the-art",
    "policy improves",
    "adaptive policy improves",
    "model-agnostic",
    "universal",
    "generalizable",
    "generalization",
    "plug-and-play",
    "test-time action search",
    "test time action search",
    "transfers across datasets",
    "transfer across datasets",
    "efficiency gain",
]

CHECKED_TEXT_FILES = [
    "README.md",
    "reports/final_project_report.md",
    "reports/oracle_analysis_paper_assets.md",
    "paper/main.tex",
]

EVIDENCE_LABELS = {
    "gate_f_pivot_documented": "Gate F pivot documented",
    "claim_boundary_oracle_only": "Claim boundary oracle only",
    "policy_training_prohibited": "Policy training prohibited",
    "posthoc_action_redesign_prohibited": "Posthoc action redesign prohibited",
    "test_time_search_prohibited": "Test-time search prohibited",
    "representation_pivot_documented": "Representation pivot documented",
    "glas_not_promoted": "GlaS not promoted",
    "statistics_not_promoted": "Statistics not promoted",
    "pdf_hash_recorded": "PDF hash recorded",
    "oracle_not_deployable": "Oracle not deployable",
    "background_reference_not_explicit_negative": "Background reference not explicit negative",
    "multi_reference_not_independent": "Multi reference not independent",
    "no_in_domain_transfer_claim": "No in-domain transfer claim",
    "failures_and_limitations_disclosed": "Failures and limitations disclosed",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _checked_files(project_root: Path) -> list[Path]:
    files = [project_root / item for item in CHECKED_TEXT_FILES]
    sections_dir = project_root / "paper" / "sections"
    if sections_dir.is_dir():
        files.extend(sorted(sections_dir.glob("*.tex")))
    return [path for path in files if path.is_file()]


def _scan_prohibited_tokens(project_root: Path, files: list[Path]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for token in PROHIBITED_CLAIM_TOKENS:
            if token in text:
                violations.append(
                    {
                        "file": _relative(project_root, path),
                        "token": token,
                        "reason": "prohibited_claim_token",
                    }
                )
    return violations


def _required_evidence_checks(project_root: Path) -> tuple[dict[str, bool], list[str]]:
    final_report = _read_json(project_root / "reports" / "final_project_report.json")
    gate_f = _read_json(project_root / "reports" / "gates" / "gate_f_oracle_gap.json")
    fixed = _read_json(project_root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json")
    representation = _read_json(project_root / "reports" / "representation_analysis" / "representation_summary.json")

    prohibited = set(final_report.get("prohibited_next_actions", []))
    prohibited.update(gate_f.get("prohibited_next_actions", []))
    prohibited.update(representation.get("prohibited_next_actions", []))

    checked_text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in _checked_files(project_root)
    )

    checks = {
        "gate_f_pivot_documented": (
            final_report.get("status") == "PIVOT_ORACLE_ANALYSIS"
            and gate_f.get("status") == "FAIL_ORACLE_GAP_SMALL"
            and gate_f.get("pass_fail") == "FAIL"
        ),
        "claim_boundary_oracle_only": (
            final_report.get("claim_boundary") == "oracle_analysis_only"
            and fixed.get("recommended_claim") == "oracle_analysis_only"
        ),
        "policy_training_prohibited": (
            "monusac_policy_training" in prohibited
            or "policy_training_on_monusac" in prohibited
        ),
        "posthoc_action_redesign_prohibited": (
            "posthoc_monusac_action_redesign" in prohibited
            or "post_hoc_monusac_action_bank_redesign" in prohibited
        ),
        "test_time_search_prohibited": "test_time_action_search" in prohibited,
        "representation_pivot_documented": (
            representation.get("status") == "PIVOT_REPRESENTATION_ANALYSIS"
            and final_report.get("representation", {}).get("claim")
            == "oracle_analysis_only_insufficient_separation"
        ),
        "glas_not_promoted": (
            final_report.get("external_pilots", {})
            .get("glas", {})
            .get("status")
            == "PENDING_INSUFFICIENT_EPISODES"
        ),
        "statistics_not_promoted": (
            final_report.get("statistics", {}).get("status")
            == "PENDING_INSUFFICIENT_CLUSTERS"
        ),
        "pdf_hash_recorded": bool(final_report.get("pdf", {}).get("sha256")),
        "oracle_not_deployable": (
            "oracle rows are analysis upper bounds" in checked_text
            and "not deployable method outputs" in checked_text
        ),
        "background_reference_not_explicit_negative": (
            "background reference evidence is not explicit sam negative point prompting"
            in checked_text
        ),
        "multi_reference_not_independent": (
            "multi-reference episodes are repeated measurements" in checked_text
            and "not independent samples" in checked_text
        ),
        "no_in_domain_transfer_claim": (
            "no in-domain retraining or transfer claim is made" in checked_text
        ),
        "failures_and_limitations_disclosed": (
            "failure results and limitations are disclosed" in checked_text
        ),
    }
    artifacts = [
        "reports/final_project_report.json",
        "reports/gates/gate_f_oracle_gap.json",
        "reports/fixed_oracle_analysis/fixed_oracle_summary.json",
        "reports/representation_analysis/representation_summary.json",
    ]
    return checks, artifacts


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AdaPromptSet Scientific Claim Audit",
        "",
        f"Status: `{summary['status']}`",
        f"Claim boundary: `{summary['claim_boundary']}`",
        f"Gate F status: `{summary['gate_f_status']}`",
        "",
        "## Required Evidence",
        "",
    ]
    for name, passed in summary["required_evidence_checks"].items():
        display = "PASS" if passed else "FAIL"
        label = EVIDENCE_LABELS.get(name, name.replace("_", " "))
        lines.append(f"- {label}: `{display}`")
    lines.extend(["", "## Checked Files", ""])
    lines.extend(f"- `{item}`" for item in summary["checked_files"])
    lines.extend(["", "## Evidence Artifacts", ""])
    lines.extend(f"- `{item}`" for item in summary["evidence_artifacts"])
    lines.extend(["", "## Prohibited Claim Checks", ""])
    lines.extend(f"- `{token}`" for token in summary["prohibited_claim_tokens"])
    lines.extend(["", "## Violations", ""])
    if summary["violations"]:
        lines.extend(
            f"- `{item['file']}` contains `{item['token']}` ({item['reason']})"
            for item in summary["violations"]
        )
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_scientific_claim_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    evidence_checks, evidence_artifacts = _required_evidence_checks(project_root)
    files = _checked_files(project_root)
    violations = _scan_prohibited_tokens(project_root, files)
    violations.extend(
        {
            "file": artifact,
            "token": check_name,
            "reason": "missing_or_inconsistent_required_evidence",
        }
        for check_name, passed in evidence_checks.items()
        if not passed
        for artifact in ["reports/final_project_report.json"]
    )
    final_report = _read_json(project_root / "reports" / "final_project_report.json")
    return {
        "schema_version": "adapromptset.scientific_claim_audit.v1",
        "status": "PASS" if not violations else "FAIL",
        "claim_boundary": final_report.get("claim_boundary"),
        "gate_f_status": final_report.get("gate_f", {}).get("status"),
        "checked_files": [_relative(project_root, path) for path in files],
        "evidence_artifacts": evidence_artifacts,
        "required_evidence_checks": evidence_checks,
        "prohibited_claim_tokens": PROHIBITED_CLAIM_TOKENS,
        "violations": violations,
    }


def export_scientific_claim_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    summary = build_scientific_claim_audit(project_root)
    outputs = {
        "json": project_root / "reports" / "scientific_claim_audit.json",
        "markdown": project_root / "reports" / "scientific_claim_audit.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AdaPromptSet scientific claim boundaries.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    export_scientific_claim_audit(args.project_root)


if __name__ == "__main__":
    main()
