from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SECTION_FILES = {
    "abstract": "00_abstract.tex",
    "intro": "01_intro.tex",
    "related": "02_related.tex",
    "method": "03_method.tex",
    "experiments": "04_experiments.tex",
    "results": "05_results.tex",
    "limitations": "06_limitations.tex",
    "conclusion": "07_conclusion.tex",
}

TABLE_FILES = {
    "oracle_pivot": "table_oracle_pivot.tex",
    "action_summary": "table_action_summary.tex",
    "representation_pivot": "table_representation_pivot.tex",
}

PROHIBITED_CLAIM_TOKENS = [
    "state of the art",
    "state-of-the-art",
    "policy improves",
    "adaptive policy improves",
    "model-agnostic",
    "generalizable",
    "generalization",
    "plug-and-play",
    "efficiency gain",
]

CHECK_LABELS = {
    "required_sections_present": "Required sections present",
    "abstract_claim_free": "Abstract claim free",
    "related_work_boundaries_present": "Related work boundaries present",
    "method_no_end_to_end_differentiable": "Method no end-to-end differentiable claim",
    "method_figure_frozen_learned_train_only": "Method figure frozen learned train-only paths",
    "method_action_bank_figure_current_subset": "Method action-bank figure current subset boundary",
    "experiments_statistical_units_present": "Experiments statistical units present",
    "results_cost_and_accuracy_present": "Results cost and accuracy present",
    "oracle_privileged_caption_boundaries": "Oracle and privileged caption boundaries",
    "limitations_action_bank_upper_bound": "Limitations action bank upper bound",
    "limitations_reference_quality": "Limitations reference quality",
    "limitations_frozen_matcher_irrecoverable": "Limitations frozen matcher irrecoverable",
    "limitations_utility_generation_cost": "Limitations utility generation cost",
    "conclusion_oracle_analysis_only": "Conclusion oracle analysis only",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _contains_all(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return all(token.lower() in lowered for token in tokens)


def _contains_any(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def build_paper_section_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    section_dir = project_root / "paper" / "sections"
    table_dir = project_root / "paper" / "tables"
    section_paths = {
        name: section_dir / file_name
        for name, file_name in SECTION_FILES.items()
    }
    table_paths = {
        name: table_dir / file_name
        for name, file_name in TABLE_FILES.items()
    }
    sections = {name: _read_text(path) for name, path in section_paths.items()}
    tables = {name: _read_text(path) for name, path in table_paths.items()}
    intro_related = f"{sections['intro']}\n{sections['related']}"
    limitations = sections["limitations"]
    table_text = "\n".join(tables.values())

    checks = {
        "required_sections_present": all(path.exists() for path in section_paths.values()),
        "abstract_claim_free": (
            _contains_any(sections["abstract"], ["oracle-analysis-only", "gate-f pivot", "not an adaptive-policy result"])
            and not _contains_any(sections["abstract"], PROHIBITED_CLAIM_TOKENS)
        ),
        "related_work_boundaries_present": _contains_all(
            intro_related,
            ["GF-SAM", "SynPo", "AlignSAM", "Point Prompt Defender"],
        )
        and "first to learn prompts" in intro_related.lower(),
        "method_no_end_to_end_differentiable": (
            _contains_any(sections["method"], ["frozen", "freezes"])
            and not _contains_any(sections["method"], ["end-to-end differentiable"])
        ),
        "method_figure_frozen_learned_train_only": _contains_all(
            sections["method"],
            [
                "frozen matcher",
                "frozen segmenter",
                "learned utility predictor",
                "train/validation only",
                "test-time safe path",
            ],
        ),
        "method_action_bank_figure_current_subset": _contains_all(
            sections["method"],
            [
                "current v2 mvp action-bank subset",
                "complete prompt-set proposals",
                "sam calls",
                "prompt points",
                "deferred",
                "gf-sam-style graph",
            ],
        ),
        "experiments_statistical_units_present": (
            _contains_any(sections["experiments"], ["query/case", "patient"])
            and "multi-reference episodes are repeated measurements" in sections["experiments"].lower()
            and "not independent samples" in sections["experiments"].lower()
        ),
        "results_cost_and_accuracy_present": (
            _contains_any(sections["results"], ["dice", "iou", "aji"])
            and _contains_any(sections["results"], ["sam calls", "runtime", "gpu-hour", "cost"])
        ),
        "oracle_privileged_caption_boundaries": _contains_all(
            table_text,
            [
                "oracle columns are analysis upper bounds",
                "not deployable method outputs",
                "oracle wins are post-hoc analysis labels",
                "not deployable method selections",
                "no privileged-input row is included",
            ],
        ),
        "limitations_action_bank_upper_bound": _contains_any(
            limitations,
            ["action bank upper bound", "mvp action bank upper bound"],
        ),
        "limitations_reference_quality": "reference quality" in limitations.lower(),
        "limitations_frozen_matcher_irrecoverable": (
            "frozen matcher" in limitations.lower()
            and _contains_any(limitations, ["cannot be recovered", "irrecoverable"])
        ),
        "limitations_utility_generation_cost": (
            "utility generation cost" in limitations.lower()
            or ("action bank" in limitations.lower() and "cost" in limitations.lower())
        ),
        "conclusion_oracle_analysis_only": (
            "oracle-analysis-only" in sections["conclusion"].lower()
            and not _contains_any(sections["conclusion"], ["policy improves"])
        ),
    }

    violations: list[dict[str, Any]] = []
    for name, passed in checks.items():
        if not passed:
            violations.append({"check": name, "reason": "missing_or_inconsistent_section_boundary"})

    for path in section_paths.values():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in PROHIBITED_CLAIM_TOKENS:
            if token in text:
                violations.append(
                    {
                        "check": "prohibited_claim_token",
                        "file": _relative(project_root, path),
                        "token": token,
                        "reason": "unsafe_paper_claim",
                    }
                )

    return {
        "schema_version": "adapromptset.paper_section_audit.v1",
        "status": "PASS" if not violations else "FAIL",
        "checked_sections": {
            name: _relative(project_root, path)
            for name, path in section_paths.items()
            if path.exists()
        },
        "checked_tables": {
            name: _relative(project_root, path)
            for name, path in table_paths.items()
            if path.exists()
        },
        "checks": checks,
        "prohibited_claim_tokens": PROHIBITED_CLAIM_TOKENS,
        "violations": violations,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AdaPromptSet Paper Section Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in summary["checks"].items():
        label = CHECK_LABELS.get(name, name.replace("_", " ").title())
        lines.append(f"- {label}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Checked Sections", ""])
    lines.extend(f"- `{path}`" for path in summary["checked_sections"].values())
    lines.extend(["", "## Checked Tables", ""])
    lines.extend(f"- `{path}`" for path in summary["checked_tables"].values())
    lines.extend(["", "## Violations", ""])
    if summary["violations"]:
        for item in summary["violations"]:
            file_part = f" in `{item['file']}`" if item.get("file") else ""
            token_part = f" token `{item['token']}`" if item.get("token") else ""
            lines.append(f"- `{item['check']}`{file_part}{token_part}: {item['reason']}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_paper_section_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    summary = build_paper_section_audit(project_root)
    outputs = {
        "json": project_root / "reports" / "paper_section_audit.json",
        "markdown": project_root / "reports" / "paper_section_audit.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AdaPromptSet paper section claim boundaries.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    export_paper_section_audit(args.project_root)


if __name__ == "__main__":
    main()
