from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _format_float(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _compute_ledger_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "row_count": 0, "total_gpu_hours": 0.0}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    total = sum(float(row.get("gpu_hours") or 0.0) for row in rows)
    return {
        "path": str(path),
        "row_count": len(rows),
        "total_gpu_hours": total,
    }


def _reproduction_commands(project_root: Path) -> list[str]:
    python = r"D:\SAM\conda_envs\sam_gpu\python.exe"
    root = str(project_root)
    return [
        rf"cd {root}",
        f'$env:PYTHONPATH = "{root}\\src"',
        rf"{python} -m unittest discover -s tests",
        rf"{python} scripts\audit_phase0_scope_structure.py --project-root {root}",
        rf"{python} scripts\audit_data.py",
        rf"{python} scripts\audit_data_profile.py --project-root {root}",
        rf"{python} scripts\build_episodes.py",
        rf"{python} scripts\audit_reference_exposure.py --project-root {root}",
        rf"{python} scripts\audit_ddr_protocol.py --project-root {root}",
        rf"{python} scripts\audit_new_domain_pilot_readiness.py --project-root {root}",
        rf"{python} scripts\run_candidate_pilot.py --max-episodes 5",
        rf"{python} scripts\run_oracle_gap_pilot.py --max-episodes 10",
        rf"{python} scripts\build_action_bank.py --bank-version v2",
        rf"{python} scripts\run_oracle_gap_pilot.py --max-episodes 10 --action-bank {root}\configs\actions\prompt_set_bank_v2.json --out-dir {root}\reports\oracle_gap_pilot_v2",
        rf"{python} scripts\write_decision_checkpoint.py",
        rf"{python} scripts\analyze_fixed_oracle_pivot.py",
        rf"{python} scripts\analyze_representation_pivot.py --project-root {root}",
        rf"{python} scripts\audit_gate_f_failure_diagnostics.py --project-root {root}",
        rf"{python} scripts\run_bootstrap.py --episode-regret-csv {root}\reports\fixed_oracle_analysis\episode_regret.csv --episode-manifest-csv {root}\manifests\episodes\monusac\episodes_val.csv --out-dir {root}\reports\bootstrap",
        rf"{python} scripts\export_oracle_analysis_tables.py --project-root {root}",
        rf"{python} scripts\export_oracle_analysis_figures.py --project-root {root}",
        rf"{python} scripts\export_method_overview_figure.py --project-root {root}",
        rf"{python} scripts\export_action_bank_figure.py --project-root {root}",
        rf"{python} scripts\export_gate_reports.py --project-root {root}",
        rf"{python} scripts\audit_paper_sections.py --project-root {root}",
        rf"{python} scripts\build_paper.py --project-root {root} --mode submission --compile",
        rf"{python} scripts\audit_latexmk_build.py --project-root {root}",
        rf"{python} scripts\audit_pdf_metadata.py --project-root {root}",
        rf"{python} scripts\export_final_report.py --project-root {root}",
        rf"{python} scripts\audit_scientific_claims.py --project-root {root}",
    ]


def build_final_summary(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    decision = _read_json(project_root / "reports" / "decision_checkpoint" / "phase5_decision.json")
    gate_f = _read_json(project_root / "reports" / "gates" / "gate_f_oracle_gap.json")
    fixed = _read_json(project_root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json")
    representation = _read_json(project_root / "reports" / "representation_analysis" / "representation_summary.json")
    gate_f_failure = _read_json(project_root / "reports" / "gate_f_failure_diagnostics" / "gate_f_failure_diagnostics.json")
    glas = _read_json(project_root / "reports" / "oracle_gap_pilot_glas" / "oracle_gap_pilot_summary.json")
    bootstrap = _read_json(project_root / "reports" / "bootstrap" / "bootstrap_summary.json")
    reference_exposure = _read_json(project_root / "reports" / "reference_exposure_audit.json")
    data_profile = _read_json(project_root / "reports" / "data_profile_audit.json")
    ddr_protocol = _read_json(project_root / "reports" / "ddr_protocol_audit" / "ddr_protocol_audit.json")
    new_domain_readiness = _read_json(project_root / "reports" / "new_domain_pilot_readiness.json")
    oracle_figures = _read_json(project_root / "reports" / "oracle_analysis_figures.json")
    method_overview = _read_json(project_root / "reports" / "method_overview_figure.json")
    action_bank_figure = _read_json(project_root / "reports" / "action_bank_figure.json")
    phase0_scope = _read_json(project_root / "reports" / "phase0_scope_structure" / "phase0_scope_structure.json")
    paper_section_audit = _read_json(project_root / "reports" / "paper_section_audit.json")
    latexmk_audit = _read_json(project_root / "reports" / "latexmk_audit.json")
    pdf_metadata_audit = _read_json(project_root / "reports" / "pdf_metadata_audit.json")
    metadata = _read_json(
        project_root / "reports" / "run_metadata" / "phase0_reproducibility" / "run_metadata.json"
    )
    pdf = project_root / "paper" / "main.pdf"
    prohibited = sorted(
        set(decision.get("prohibited_next_actions", []))
        | set(gate_f.get("prohibited_next_actions", []))
        | set(representation.get("prohibited_next_actions", []))
    )
    return {
        "schema_version": "adapromptset.final_report.v1",
        "status": decision["phase5_status"],
        "claim_boundary": fixed.get("recommended_claim", "oracle_analysis_only"),
        "gate_f": {
            "status": gate_f["status"],
            "pass_fail": gate_f["pass_fail"],
            "dice_oracle_gap": gate_f["diagnostics"]["dice_oracle_gap"],
            "aji_available": gate_f["diagnostics"].get("aji_available"),
            "aji_oracle_gap": gate_f["diagnostics"].get("aji_oracle_gap"),
            "min_required_gap": gate_f["diagnostics"]["gate_f_min_dice_gap"],
            "episode_count": gate_f["diagnostics"]["episode_count"],
            "record_count": gate_f["diagnostics"]["record_count"],
        },
        "fixed_oracle": {
            "best_fixed_action_id": fixed["best_fixed_action_id"],
            "mean_best_fixed_dice": fixed.get("mean_best_fixed_dice"),
            "mean_oracle_dice": fixed.get("mean_oracle_dice"),
            "mean_regret_to_oracle": fixed["mean_regret_to_oracle"],
            "aji_available": fixed.get("aji_available"),
            "mean_best_fixed_aji": fixed.get("mean_best_fixed_aji"),
            "mean_oracle_aji": fixed.get("mean_oracle_aji"),
            "mean_aji_regret_to_oracle": fixed.get("mean_aji_regret_to_oracle"),
            "deterministic_promotion_status": fixed["deterministic_promotion_status"],
        },
        "representation": {
            "status": representation["status"],
            "claim": representation["representation_claim"],
            "mean_top2_dice_margin": representation["mean_top2_dice_margin"],
            "near_tie_episode_fraction": representation["near_tie_episode_fraction"],
        },
        "gate_f_failure_diagnostics": {
            "status": gate_f_failure["status"],
            "policy_training_allowed": gate_f_failure["policy_training_allowed"],
            "coverage_status": gate_f_failure["coverage_diagnostic"]["status"],
            "mean_component_coverage_at_max_pool": gate_f_failure["coverage_diagnostic"]["mean_component_coverage_at_max_pool"],
            "proposal_status": gate_f_failure["proposal_diagnostic"]["status"],
            "mean_top2_dice_margin": gate_f_failure["proposal_diagnostic"]["mean_top2_dice_margin"],
            "near_tie_episode_fraction": gate_f_failure["proposal_diagnostic"]["near_tie_episode_fraction"],
            "sam_sensitivity_status": gate_f_failure["sam_sensitivity_diagnostic"]["status"],
            "best_high_call_minus_best_one_call": gate_f_failure["sam_sensitivity_diagnostic"]["best_high_call_minus_best_one_call"],
            "interpretation_flags": gate_f_failure["interpretation_flags"],
        },
        "external_pilots": {
            "glas": {
                "status": glas["status"],
                "episode_count": glas["episode_count"],
                "gate_min_episodes": glas["gate_min_episodes"],
                "dice_oracle_gap": glas["dice_oracle_gap"],
            }
        },
        "reference_exposure": {
            "status": reference_exposure["status"],
            "checks": reference_exposure["checks"],
            "dataset_count": reference_exposure["dataset_count"],
            "total_issues": reference_exposure["total_issues"],
        },
        "data_profile": {
            "status": data_profile["status"],
            "dataset_count": data_profile["dataset_count"],
            "row_count": data_profile["row_count"],
            "datasets": data_profile["datasets"],
        },
        "ddr_protocol": {
            "status": ddr_protocol["status"],
            "total_image_count": ddr_protocol["total_image_count"],
            "total_complete_case_count": ddr_protocol["total_complete_case_count"],
            "total_missing_label_count": ddr_protocol["total_missing_label_count"],
            "stage1_mask_rule": ddr_protocol["stage1_union"]["mask_rule"],
            "stage1_allowed_use": ddr_protocol["stage1_union"]["allowed_use"],
            "stage2_intents": ddr_protocol["stage2_four_intents"]["intents"],
            "formal_policy_training_allowed": ddr_protocol["formal_policy_training_allowed"],
        },
        "new_domain_pilot_readiness": {
            "status": new_domain_readiness["status"],
            "blocked_datasets": new_domain_readiness["blocked_datasets"],
            "infrastructure_ready_datasets": new_domain_readiness["infrastructure_ready_datasets"],
            "glas_status": new_domain_readiness["datasets"]["glas"]["status"],
            "glas_episode_count": new_domain_readiness["datasets"]["glas"]["gate"]["episode_count"],
            "glas_gate_min_episodes": new_domain_readiness["datasets"]["glas"]["gate"]["gate_min_episodes"],
            "ddr_status": new_domain_readiness["datasets"]["ddr"]["status"],
            "idrid_status": new_domain_readiness["datasets"]["idrid"]["status"],
            "cpm17_status": new_domain_readiness["datasets"]["cpm17"]["status"],
            "global_allowed_next_actions": new_domain_readiness["global_allowed_next_actions"],
            "global_prohibited_next_actions": new_domain_readiness["global_prohibited_next_actions"],
        },
        "oracle_analysis_figures": {
            "status": oracle_figures["status"],
            "claim_boundary": oracle_figures["claim_boundary"],
            "source_rows": oracle_figures["source_rows"],
            "figures": oracle_figures["figures"],
        },
        "method_overview_figure": {
            "status": method_overview["status"],
            "claim_boundary": method_overview["claim_boundary"],
            "figure": method_overview["figure"],
            "checks": method_overview["checks"],
        },
        "action_bank_figure": {
            "status": action_bank_figure["status"],
            "claim_boundary": action_bank_figure["claim_boundary"],
            "source_bank": action_bank_figure["source_bank"],
            "family_counts": action_bank_figure["family_counts"],
            "figure": action_bank_figure["figure"],
            "checks": action_bank_figure["checks"],
            "deferred_families": action_bank_figure["deferred_families"],
        },
        "phase0_scope_structure": {
            "status": phase0_scope["status"],
            "structure_status": phase0_scope["structure"]["status"],
            "toolchain_status": phase0_scope["toolchain"]["status"],
            "policy_training_allowed": phase0_scope["policy_training_allowed"],
            "research_question": phase0_scope["research_question"]["text"],
            "research_question_does_not_claim_first": phase0_scope["research_question"]["does_not_claim_first_to_learn_prompts"],
            "similar_work_boundaries": phase0_scope["similar_work_boundaries"],
            "rq_count": phase0_scope["rq_hypotheses"]["rq_count"],
            "hypothesis_count": phase0_scope["rq_hypotheses"]["hypothesis_count"],
            "non_mainline_exclusions": phase0_scope["scope"]["non_mainline_exclusions"],
        },
        "paper_section_audit": {
            "status": paper_section_audit["status"],
            "checks": paper_section_audit["checks"],
            "violation_count": len(paper_section_audit.get("violations", [])),
        },
        "latexmk_audit": {
            "status": latexmk_audit["status"],
            "checks": latexmk_audit.get("checks", {}),
            "pdf": latexmk_audit.get("pdf", {}),
            "clean_copy_dir": latexmk_audit.get("clean_copy_dir"),
            "violation_count": len(latexmk_audit.get("violations", [])),
        },
        "pdf_metadata_audit": {
            "status": pdf_metadata_audit["status"],
            "checks": pdf_metadata_audit["checks"],
            "expected": pdf_metadata_audit["expected"],
            "violation_count": len(pdf_metadata_audit.get("violations", [])),
        },
        "statistics": {
            "status": bootstrap["status"],
            "query_count": bootstrap["bootstrap"]["query_count"],
            "cluster_count": bootstrap["bootstrap"]["cluster_count"],
            "observed_mean_delta": bootstrap["bootstrap"]["observed_mean_delta"],
            "paired_sign_flip_p": bootstrap["paired_significance"]["p_value_two_sided"],
        },
        "compute": _compute_ledger_summary(project_root / "reports" / "compute_ledger.csv"),
        "run_metadata": {
            "run_id": metadata["run_id"],
            "config_hash": metadata["config_hash"],
            "python_executable": metadata["environment"].get("python_executable"),
            "gpu_name": metadata["environment"].get("gpu_name"),
            "torch": metadata["environment"].get("torch"),
        },
        "pdf": {
            "path": str(pdf),
            "exists": pdf.exists(),
            "bytes": pdf.stat().st_size if pdf.exists() else 0,
            "sha256": _sha256(pdf) if pdf.exists() else None,
        },
        "allowed_next_actions": decision.get("allowed_next_actions", []),
        "prohibited_next_actions": prohibited,
        "reproduction_commands": _reproduction_commands(project_root),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AdaPromptSet Final Project Report",
        "",
        f"Status: `{summary['status']}`",
        f"Claim boundary: `{summary['claim_boundary']}`",
        "",
        "## Decision",
        "",
        (
            f"Gate F is `{summary['gate_f']['status']}`: oracle gap "
            f"{_format_float(summary['gate_f']['dice_oracle_gap'])} is below the "
            f"predeclared {_format_float(summary['gate_f']['min_required_gap'])} threshold."
        ),
        "Policy training on MoNuSAC remains prohibited.",
        "",
        "## Evidence",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Best fixed action | `{summary['fixed_oracle']['best_fixed_action_id']}` |",
        f"| Mean best fixed Dice | {_format_float(summary['fixed_oracle']['mean_best_fixed_dice'])} |",
        f"| Mean oracle Dice | {_format_float(summary['fixed_oracle']['mean_oracle_dice'])} |",
        f"| Mean best fixed AJI | {_format_float(summary['fixed_oracle']['mean_best_fixed_aji'])} |",
        f"| Mean oracle AJI | {_format_float(summary['fixed_oracle']['mean_oracle_aji'])} |",
        f"| AJI oracle gap | {_format_float(summary['fixed_oracle']['mean_aji_regret_to_oracle'])} |",
        f"| Deterministic promotion | `{summary['fixed_oracle']['deterministic_promotion_status']}` |",
        f"| Representation status | `{summary['representation']['status']}` |",
        f"| Representation claim | `{summary['representation']['claim']}` |",
        f"| Gate F failure diagnostics | `{summary['gate_f_failure_diagnostics']['coverage_status']}`, `{summary['gate_f_failure_diagnostics']['sam_sensitivity_status']}` |",
        f"| GlaS pilot | `{summary['external_pilots']['glas']['status']}` |",
        f"| New-domain readiness | `{summary['new_domain_pilot_readiness']['status']}`; GlaS `{summary['new_domain_pilot_readiness']['glas_status']}`, DDR `{summary['new_domain_pilot_readiness']['ddr_status']}` |",
        f"| Oracle-analysis figures | `{summary['oracle_analysis_figures']['status']}`; {len(summary['oracle_analysis_figures']['figures'])} files |",
        f"| Method overview figure | `{summary['method_overview_figure']['status']}` |",
        f"| Action-bank figure | `{summary['action_bank_figure']['status']}`; `{summary['action_bank_figure']['source_bank']['name']}` |",
        f"| Data profile | `{summary['data_profile']['status']}` ({summary['data_profile']['row_count']} rows) |",
        f"| DDR protocol | `{summary['ddr_protocol']['status']}`; `{summary['ddr_protocol']['stage1_mask_rule']}` |",
        f"| Phase 0 scope | `{summary['phase0_scope_structure']['status']}`; `{summary['phase0_scope_structure']['toolchain_status']}` |",
        f"| Paper section audit | `{summary['paper_section_audit']['status']}`; {summary['paper_section_audit']['violation_count']} violations |",
        f"| Oracle/privileged captions | `{'PASS' if summary['paper_section_audit']['checks'].get('oracle_privileged_caption_boundaries') else 'FAIL'}` |",
        (
            f"| latexmk clean build | `{summary['latexmk_audit']['status']}`; "
            f"{summary['latexmk_audit']['pdf'].get('pages', 'N/A')} pages from clean copy |"
        ),
        (
            f"| PDF metadata | `{summary['pdf_metadata_audit']['status']}`; "
            f"title `{summary['pdf_metadata_audit']['expected']['title']}`; "
            f"author `{summary['pdf_metadata_audit']['expected']['author']}` |"
        ),
        f"| Reference exposure | `{summary['reference_exposure']['status']}` |",
        f"| Bootstrap status | `{summary['statistics']['status']}` |",
        f"| PDF SHA256 | `{summary['pdf']['sha256']}` |",
        "",
        "## Prohibited Next Actions",
        "",
    ]
    lines.extend(f"- `{action}`" for action in summary["prohibited_next_actions"])
    lines.extend(
        [
            "",
            "## Allowed Next Actions",
            "",
        ]
    )
    lines.extend(f"- `{action}`" for action in summary["allowed_next_actions"])
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```powershell",
            *summary["reproduction_commands"],
            "```",
            "",
            "## Notes",
            "",
            "- This report is generated from real local artifacts.",
            "- It does not claim adaptive-policy success, transfer, efficiency, or leaderboard-level performance.",
            "- Oracle rows are analysis upper bounds, not deployable method outputs.",
            "- Phase 0 scope keeps sequential RL, continuous coordinate generation, and online SAM-response refinement outside the current mainline.",
            "- Paper section audit confirms Results cost and accuracy wording, Limitations reference quality coverage, and oracle/privileged caption boundaries.",
            "- Current table captions state oracle columns or labels are analysis upper bounds, not deployable outputs, and no privileged-input row is included.",
            "- latexmk clean-build audit compiles a fresh copy of `paper/` and scans its log for blocking reference, rerun, missing-file, and overfull patterns.",
            "- PDF metadata audit confirms the compiled PDF title, author, subject, and keywords match the internal MVP placeholder metadata.",
            "- Background reference evidence is not explicit SAM negative point prompting.",
            "- Multi-reference episodes are repeated measurements and not independent samples.",
            "- No in-domain retraining or transfer claim is made.",
            "- Failure results and limitations are disclosed.",
            "- GlaS remains a smoke result until a predeclared protocol supplies enough episodes.",
            "- Bootstrap/significance remain diagnostic because the current pilot has insufficient patient clusters.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_final_report(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    summary = build_final_summary(project_root)
    outputs = {
        "json": project_root / "reports" / "final_project_report.json",
        "markdown": project_root / "reports" / "final_project_report.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the AdaPromptSet final project report.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    export_final_report(args.project_root)


if __name__ == "__main__":
    main()
