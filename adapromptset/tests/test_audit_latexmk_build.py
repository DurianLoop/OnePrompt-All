from __future__ import annotations

import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_latexmk_build import export_latexmk_audit  # noqa: E402


class LatexmkAuditTests(unittest.TestCase):
    def _fixture_root(self, name: str) -> Path:
        root = PROJECT_ROOT / "tmp" / f"{name}_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "paper").mkdir(parents=True, exist_ok=True)
        (root / "paper" / "main.tex").write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
        return root

    def test_latexmk_audit_passes_clean_copy_build(self) -> None:
        root = self._fixture_root("test_latexmk_audit_pass")

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if args[0] == "latexmk":
                cwd = Path(kwargs["cwd"])  # type: ignore[arg-type]
                (cwd / "main.pdf").write_bytes(b"%PDF-clean")
                (cwd / "main.log").write_text("Clean latexmk log\n", encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, stdout="latexmk ok", stderr="")
            if args[0] == "pdfinfo":
                return subprocess.CompletedProcess(args, 0, stdout="Pages:           1\n", stderr="")
            raise AssertionError(args)

        with patch("adapromptset.scripts.audit_latexmk_build.subprocess.run", side_effect=fake_run):
            outputs = export_latexmk_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertTrue(payload["checks"]["latexmk_exit_zero"])
        self.assertTrue(payload["checks"]["log_problem_scan_clean"])
        self.assertEqual(payload["pdf"]["pages"], "1")
        self.assertFalse(payload["violations"])

    def test_latexmk_audit_fails_on_log_problem(self) -> None:
        root = self._fixture_root("test_latexmk_audit_fail")

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if args[0] == "latexmk":
                cwd = Path(kwargs["cwd"])  # type: ignore[arg-type]
                (cwd / "main.pdf").write_bytes(b"%PDF-clean")
                (cwd / "main.log").write_text("Overfull \\hbox in paragraph\n", encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, stdout="latexmk ok", stderr="")
            if args[0] == "pdfinfo":
                return subprocess.CompletedProcess(args, 0, stdout="Pages:           1\n", stderr="")
            raise AssertionError(args)

        with patch("adapromptset.scripts.audit_latexmk_build.subprocess.run", side_effect=fake_run):
            outputs = export_latexmk_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "FAIL")
        checks = {item["check"] for item in payload["violations"]}
        self.assertIn("log_problem_scan_clean", checks)


if __name__ == "__main__":
    unittest.main()
