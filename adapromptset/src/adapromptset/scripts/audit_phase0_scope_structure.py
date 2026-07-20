from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


REQUIRED_PATHS = [
    "pyproject.toml",
    "environment.yml",
    "README.md",
    "execution.md",
    "checklist.md",
    "configs/data",
    "configs/matcher",
    "configs/segmenter",
    "configs/actions",
    "configs/experiments",
    "src/adapromptset/actions",
    "src/adapromptset/candidates",
    "src/adapromptset/data",
    "src/adapromptset/matching",
    "src/adapromptset/policy",
    "src/adapromptset/reporting",
    "src/adapromptset/scripts",
    "src/adapromptset/segmenters",
    "src/adapromptset/statistics",
    "src/adapromptset/training",
    "scripts",
    "tests",
    "manifests",
    "reports",
    "paper",
    "docs",
]


RESEARCH_QUESTION = (
    "Under frozen DINOv2-L and SAM-B, does budget-constrained structured "
    "prompt-set selection show enough query-specific oracle heterogeneity to "
    "justify learned selection?"
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _present(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _structure_summary(project_root: Path) -> dict[str, Any]:
    records = [
        {
            "path": relative,
            "exists": (project_root / relative).exists(),
            "kind": "directory" if (project_root / relative).is_dir() else "file",
        }
        for relative in REQUIRED_PATHS
    ]
    missing = [record["path"] for record in records if not record["exists"]]
    return {
        "status": "PASS" if not missing else "FAIL_MISSING_PATHS",
        "required_path_count": len(REQUIRED_PATHS),
        "missing_paths": missing,
        "paths": records,
    }


def _toolchain_summary(project_root: Path) -> dict[str, Any]:
    pyproject = _read_text(project_root / "pyproject.toml")
    checks = {
        "pyproject_present": (project_root / "pyproject.toml").exists(),
        "environment_yml_present": (project_root / "environment.yml").exists(),
        "pytest_declared_or_tests_present": "pytest" in pyproject.lower() or (project_root / "tests").exists(),
        "package_discovery_configured": "[tool.setuptools.packages.find]" in pyproject,
        "script_entry_points_present": "[project.scripts]" in pyproject,
        "lint_config_present": any(token in pyproject.lower() for token in ["[tool.ruff", "[tool.black", "[tool.flake8"]),
    }
    required_pass = all(
        checks[key]
        for key in [
            "pyproject_present",
            "environment_yml_present",
            "pytest_declared_or_tests_present",
            "package_discovery_configured",
            "script_entry_points_present",
        ]
    )
    if required_pass and checks["lint_config_present"]:
        status = "PASS"
    elif required_pass:
        status = "PASS_WITH_LINT_DEFERRED"
    else:
        status = "FAIL"
    return {"status": status, "checks": checks}


def _scope_summary(project_root: Path) -> dict[str, Any]:
    execution = _read_text(project_root / "execution.md")
    response = _read_text(project_root / "AdaPromptSet_Codex_Response.md")
    combined = "\n".join([execution, response])
    exclusions = {
        "sequential_rl": {
            "present": _present(combined, ["sequential RL"]),
            "mainline": False,
        },
        "continuous_coordinate_generation": {
            "present": _present(
                combined,
                [
                    "unrestricted coordinate generation",
                    "learned coordinate offsets",
                    "continuous clustering-parameter prediction",
                    "continuous coordinate",
                ],
            ),
            "mainline": False,
        },
        "online_sam_response_refinement": {
            "present": _present(
                combined,
                [
                    "iterative SAM-response-conditioned prompting",
                    "online SAM response refinement",
                    "one-step response-conditioned refinement",
                ],
            ),
            "mainline": False,
        },
    }
    return {
        "status": "PASS" if all(item["present"] and item["mainline"] is False for item in exclusions.values()) else "FAIL",
        "non_mainline_exclusions": exclusions,
        "mainline": "offline one-shot set-level utility ranking under a frozen matcher/segmenter",
    }


def _similar_work_boundaries(project_root: Path) -> dict[str, Any]:
    execution = _read_text(project_root / "execution.md")
    return {
        "gf_sam": {"label": "GF-SAM", "present": _present(execution, ["GF-SAM"])},
        "synpo": {"label": "SynPo", "present": _present(execution, ["SynPo"])},
        "alignsam": {"label": "AlignSAM", "present": _present(execution, ["AlignSAM"])},
        "ppd": {
            "label": "Point Prompt Defender",
            "present": _present(execution, ["Point Prompt Defender", "PPD"]),
        },
    }


def _rq_hypothesis_summary(project_root: Path) -> dict[str, Any]:
    execution = _read_text(project_root / "execution.md")
    rq_count = len(set(re.findall(r"\bRQ[1-9]\b", execution)))
    hypothesis_count = len(set(re.findall(r"\bH[1-9]\b", execution)))
    return {
        "status": "PASS" if rq_count == 3 and hypothesis_count == 5 else "FAIL",
        "rq_count": rq_count,
        "hypothesis_count": hypothesis_count,
        "expected_rq_count": 3,
        "expected_hypothesis_count": 5,
    }


def analyze_phase0_scope_structure(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    structure = _structure_summary(project_root)
    toolchain = _toolchain_summary(project_root)
    scope = _scope_summary(project_root)
    similar = _similar_work_boundaries(project_root)
    rq_hypotheses = _rq_hypothesis_summary(project_root)
    research_question = {
        "text": RESEARCH_QUESTION,
        "does_not_claim_first_to_learn_prompts": "first to learn prompts" not in RESEARCH_QUESTION.lower(),
    }
    similar_pass = all(item["present"] for item in similar.values())
    pass_scope = (
        structure["status"] == "PASS"
        and scope["status"] == "PASS"
        and similar_pass
        and rq_hypotheses["status"] == "PASS"
        and research_question["does_not_claim_first_to_learn_prompts"]
    )
    return {
        "schema_version": "adapromptset.phase0_scope_structure.v1",
        "status": "PASS_SCOPE_STRUCTURE" if pass_scope else "FAIL_SCOPE_STRUCTURE",
        "policy_training_allowed": False,
        "post_hoc_monusac_action_redesign_allowed": False,
        "structure": structure,
        "toolchain": toolchain,
        "research_question": research_question,
        "similar_work_boundaries": similar,
        "rq_hypotheses": rq_hypotheses,
        "scope": scope,
        "reader_facing_notes": [
            "The executable mainline remains one-shot, offline, and set-level.",
            "This audit is not deployable policy evidence and does not reopen MoNuSAC policy training.",
            "Sequential RL, continuous coordinate generation, and online SAM-response refinement are non-mainline extensions.",
        ],
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 0 Scope and Structure Audit",
        "",
        f"Status: `{summary['status']}`",
        "Policy training remains prohibited for the current MoNuSAC evidence.",
        "",
        "## Structure",
        "",
        f"- Required paths: {summary['structure']['required_path_count']}",
        f"- Missing paths: {len(summary['structure']['missing_paths'])}",
        f"- Toolchain: `{summary['toolchain']['status']}`",
        "",
        "## Research Boundary",
        "",
        f"- Research question: {summary['research_question']['text']}",
        "- Does not claim first-to-learn prompt policy.",
        f"- RQs: {summary['rq_hypotheses']['rq_count']} / {summary['rq_hypotheses']['expected_rq_count']}",
        f"- Hypotheses: {summary['rq_hypotheses']['hypothesis_count']} / {summary['rq_hypotheses']['expected_hypothesis_count']}",
        "",
        "## Similar Work Boundaries",
        "",
    ]
    for item in summary["similar_work_boundaries"].values():
        lines.append(f"- {item['label']}: `{item['present']}`")
    lines.extend(
        [
            "",
            "## Non-mainline Exclusions",
            "",
        ]
    )
    for key, item in summary["scope"]["non_mainline_exclusions"].items():
        lines.append(f"- `{key}`: present=`{item['present']}`, mainline=`{item['mainline']}`")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary["reader_facing_notes"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase0_scope_structure_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    summary = analyze_phase0_scope_structure(project_root)
    out_dir = project_root / "reports" / "phase0_scope_structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase0_scope_structure.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "phase0_scope_structure.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 0 structure and research-scope boundaries.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_phase0_scope_structure_audit(args.project_root)


if __name__ == "__main__":
    main()
