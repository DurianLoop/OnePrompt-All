from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from adapromptset.reporting.pdf_metadata import EXPECTED_PAPER_METADATA

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_pdfinfo_output(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _run_pdfinfo(pdf_path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "error": str(exc),
            "fields": {},
        }
    return {
        "available": True,
        "error": None,
        "fields": parse_pdfinfo_output(completed.stdout),
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_pdf_metadata_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paper_dir = project_root / "paper"
    main_tex = paper_dir / "main.tex"
    pdf = paper_dir / "main.pdf"
    tex = _read_text(main_tex)
    pdfinfo = _run_pdfinfo(pdf)
    fields = pdfinfo["fields"]
    expected = {
        "title": EXPECTED_PAPER_METADATA.title,
        "author": EXPECTED_PAPER_METADATA.author,
        "date": EXPECTED_PAPER_METADATA.date,
        "subject": EXPECTED_PAPER_METADATA.subject,
        "keywords": EXPECTED_PAPER_METADATA.keywords,
    }
    checks = {
        "main_tex_exists": main_tex.exists(),
        "pdf_exists": pdf.exists(),
        "visible_title_present": r"\title{AdaPromptSet:" in tex
        and "Reference-Guided Dense Medical Segmentation}" in tex,
        "visible_author_present": rf"\author{{{EXPECTED_PAPER_METADATA.author}}}" in tex,
        "visible_date_present": rf"\date{{{EXPECTED_PAPER_METADATA.date}}}" in tex,
        "pdfinfo_available": pdfinfo["available"],
        "pdf_title_matches": fields.get("Title") == EXPECTED_PAPER_METADATA.title,
        "pdf_author_matches": fields.get("Author") == EXPECTED_PAPER_METADATA.author,
        "pdf_subject_matches": fields.get("Subject") == EXPECTED_PAPER_METADATA.subject,
        "pdf_keywords_match": fields.get("Keywords") == EXPECTED_PAPER_METADATA.keywords,
    }
    violations: list[dict[str, str]] = []
    for name, passed in checks.items():
        if not passed:
            violations.append({"check": name, "reason": "missing_or_mismatched_pdf_metadata"})
    return {
        "schema_version": "adapromptset.pdf_metadata_audit.v1",
        "status": "PASS" if not violations else "FAIL",
        "expected": expected,
        "observed": {
            "main_tex": str(main_tex),
            "pdf": str(pdf),
            "pdfinfo_available": pdfinfo["available"],
            "pdfinfo_error": pdfinfo["error"],
            "pdfinfo_fields": fields,
        },
        "checks": checks,
        "violations": violations,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AdaPromptSet PDF Metadata Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Expected",
        "",
    ]
    for key, value in summary["expected"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, passed in summary["checks"].items():
        lines.append(f"- {key}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Violations", ""])
    if summary["violations"]:
        for item in summary["violations"]:
            lines.append(f"- `{item['check']}`: {item['reason']}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_pdf_metadata_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    summary = build_pdf_metadata_audit(project_root)
    outputs = {
        "json": project_root / "reports" / "pdf_metadata_audit.json",
        "markdown": project_root / "reports" / "pdf_metadata_audit.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AdaPromptSet PDF metadata and visible paper placeholders.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    export_pdf_metadata_audit(args.project_root)


if __name__ == "__main__":
    main()
