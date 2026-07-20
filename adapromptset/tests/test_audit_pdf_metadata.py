from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.reporting.pdf_metadata import EXPECTED_PAPER_METADATA, pdfinfo_block  # noqa: E402
from adapromptset.scripts.audit_pdf_metadata import (  # noqa: E402
    export_pdf_metadata_audit,
    parse_pdfinfo_output,
)


class PdfMetadataAuditTests(unittest.TestCase):
    def _fixture_root(self, name: str) -> Path:
        root = PROJECT_ROOT / "tmp" / name
        if root.exists():
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _seed_paper(self, root: Path) -> None:
        paper = root / "paper"
        paper.mkdir(parents=True, exist_ok=True)
        (paper / "main.tex").write_text(
            "\n".join(
                [
                    pdfinfo_block(),
                    r"\title{AdaPromptSet: Cost-Aware Prompt Set Selection for\\Reference-Guided Dense Medical Segmentation}",
                    rf"\author{{{EXPECTED_PAPER_METADATA.author}}}",
                    rf"\date{{{EXPECTED_PAPER_METADATA.date}}}",
                ]
            ),
            encoding="utf-8",
        )
        (paper / "main.pdf").write_bytes(b"%PDF-test")

    def test_parse_pdfinfo_output_keeps_expected_fields(self) -> None:
        parsed = parse_pdfinfo_output(
            "\n".join(
                [
                    f"Title:          {EXPECTED_PAPER_METADATA.title}",
                    f"Author:         {EXPECTED_PAPER_METADATA.author}",
                    f"Subject:        {EXPECTED_PAPER_METADATA.subject}",
                    f"Keywords:       {EXPECTED_PAPER_METADATA.keywords}",
                ]
            )
        )

        self.assertEqual(parsed["Title"], EXPECTED_PAPER_METADATA.title)
        self.assertEqual(parsed["Keywords"], EXPECTED_PAPER_METADATA.keywords)

    def test_pdf_metadata_audit_passes_when_visible_and_pdfinfo_match(self) -> None:
        root = self._fixture_root("test_pdf_metadata_audit_pass")
        self._seed_paper(root)
        fake_pdfinfo = {
            "available": True,
            "error": None,
            "fields": {
                "Title": EXPECTED_PAPER_METADATA.title,
                "Author": EXPECTED_PAPER_METADATA.author,
                "Subject": EXPECTED_PAPER_METADATA.subject,
                "Keywords": EXPECTED_PAPER_METADATA.keywords,
            },
        }

        with patch("adapromptset.scripts.audit_pdf_metadata._run_pdfinfo", return_value=fake_pdfinfo):
            outputs = export_pdf_metadata_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertTrue(payload["checks"]["visible_title_present"])
        self.assertTrue(payload["checks"]["pdf_title_matches"])
        self.assertFalse(payload["violations"])

    def test_pdf_metadata_audit_fails_on_missing_pdf_author(self) -> None:
        root = self._fixture_root("test_pdf_metadata_audit_fail")
        self._seed_paper(root)
        fake_pdfinfo = {
            "available": True,
            "error": None,
            "fields": {
                "Title": EXPECTED_PAPER_METADATA.title,
                "Author": "Wrong Author",
                "Subject": EXPECTED_PAPER_METADATA.subject,
                "Keywords": EXPECTED_PAPER_METADATA.keywords,
            },
        }

        with patch("adapromptset.scripts.audit_pdf_metadata._run_pdfinfo", return_value=fake_pdfinfo):
            outputs = export_pdf_metadata_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "FAIL")
        checks = {item["check"] for item in payload["violations"]}
        self.assertIn("pdf_author_matches", checks)


if __name__ == "__main__":
    unittest.main()
