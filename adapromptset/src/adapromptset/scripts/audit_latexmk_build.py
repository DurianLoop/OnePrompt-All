from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_PROBLEM_PATTERNS = [
    "Overfull",
    "undefined",
    "Citation .* undefined",
    r"LaTeX Warning: Label\(s\) may have changed",
    "Rerun",
    "File .* not found",
    "Missing",
]


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _scan_log(text: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in LOG_PROBLEM_PATTERNS:
            if re.search(pattern, line):
                findings.append({"line": str(line_number), "pattern": pattern, "text": line.strip()})
                break
    return findings


def _pdfinfo_fields(pdf_path: Path) -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def build_latexmk_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paper_dir = project_root / "paper"
    if not (paper_dir / "main.tex").exists():
        return {
            "schema_version": "adapromptset.latexmk_audit.v1",
            "status": "FAIL",
            "violations": [{"check": "main_tex_exists", "reason": "missing_paper_main_tex"}],
        }

    tmp_parent = project_root / "tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    clean_root = tmp_parent / f"latexmk_clean_{uuid.uuid4().hex}"
    clean_paper = clean_root / "paper"
    shutil.copytree(paper_dir, clean_paper)
    completed = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=clean_paper,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = completed.stdout
    stderr = completed.stderr
    log_path = clean_paper / "main.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    pdf_path = clean_paper / "main.pdf"
    log_findings = _scan_log(log_text)
    pdf_exists = pdf_path.exists()
    pdf_bytes = pdf_path.stat().st_size if pdf_exists else 0
    pdfinfo = _pdfinfo_fields(pdf_path) if pdf_exists else {}
    pdf_pages = pdfinfo.get("Pages")

    checks = {
        "latexmk_exit_zero": completed.returncode == 0,
        "clean_copy_pdf_exists": pdf_exists,
        "clean_copy_log_exists": bool(log_text),
        "log_problem_scan_clean": not log_findings,
    }
    violations = [
        {"check": name, "reason": "latexmk_clean_build_failed"}
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "schema_version": "adapromptset.latexmk_audit.v1",
        "status": "PASS" if not violations else "FAIL",
        "source_paper_dir": _relative(project_root, paper_dir),
        "clean_copy_dir": _relative(project_root, clean_root),
        "command": "latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex",
        "checks": checks,
        "returncode": completed.returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "log_problem_patterns": LOG_PROBLEM_PATTERNS,
        "log_findings": log_findings,
        "pdf": {
            "exists": pdf_exists,
            "bytes": pdf_bytes,
            "pages": pdf_pages,
        },
        "violations": violations,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AdaPromptSet latexmk Clean-Build Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Command: `{summary.get('command', 'N/A')}`",
        "",
        "## Checks",
        "",
    ]
    for key, passed in summary.get("checks", {}).items():
        lines.append(f"- {key}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## PDF", ""])
    pdf = summary.get("pdf", {})
    lines.append(f"- exists: `{pdf.get('exists')}`")
    lines.append(f"- bytes: `{pdf.get('bytes')}`")
    lines.append(f"- pages: `{pdf.get('pages')}`")
    lines.extend(["", "## Violations", ""])
    if summary.get("violations"):
        for item in summary["violations"]:
            lines.append(f"- `{item['check']}`: {item['reason']}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_latexmk_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    summary = build_latexmk_audit(project_root)
    outputs = {
        "json": project_root / "reports" / "latexmk_audit.json",
        "markdown": project_root / "reports" / "latexmk_audit.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clean-copy latexmk audit for AdaPromptSet paper.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    export_latexmk_audit(args.project_root)


if __name__ == "__main__":
    main()
