from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from adapromptset.matching.runtime import (
    checkpoint_record,
    configure_external_paths,
    dependency_status,
    runtime_versions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAM_ROOT = PROJECT_ROOT.parent


def _append_ledger(project_root: Path) -> None:
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    command = "scripts/audit_matcher_runtime.py"
    if exists:
        with ledger_path.open(newline="", encoding="utf-8") as handle:
            if any(row.get("command") == command for row in csv.DictReader(handle)):
                return
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "3-runtime-audit",
                "command": command,
                "gpu_hours": "0.0",
                "notes": "CPU-only checkpoint/dependency audit; no DINOv2 forward pass.",
            }
        )


def _write_gate_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Matcher Runtime Audit",
        "",
        "| Item | Status | Notes |",
        "|---|---|---|",
    ]
    matcher = payload["mvp_matcher"]
    segmenter = payload["mvp_segmenter"]
    lines.append(f"| matcher checkpoint | {'PASS' if matcher['exists'] else 'FAIL'} | {matcher['path']} |")
    lines.append(f"| segmenter checkpoint | {'PASS' if segmenter['exists'] else 'FAIL'} | {segmenter['path']} |")
    lines.append(f"| formal Gate C | {payload['formal_gate_c']} | requires DINOv2-L forward pass on frozen real episodes |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(out_path: Path | None = None, project_root: Path = PROJECT_ROOT) -> int:
    external_paths = configure_external_paths(SAM_ROOT)
    matcher_path = SAM_ROOT / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    segmenter_path = SAM_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth"
    payload: dict[str, object] = {
        "mvp_matcher": checkpoint_record(matcher_path, role="matcher"),
        "mvp_segmenter": checkpoint_record(segmenter_path, role="segmenter"),
        "external_import_paths": external_paths,
        "dependencies": dependency_status(),
        "runtime": runtime_versions(),
        "formal_gate_c": "PENDING_REAL_MATCHER_EVIDENCE",
        "notes": [
            "This audit records frozen MVP checkpoint hashes and import availability.",
            "It does not run DINOv2 feature extraction and must not be reported as formal Gate C.",
        ],
    }
    output = out_path or (project_root / "reports" / "matcher_runtime.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if out_path is None:
        _write_gate_report(project_root / "reports" / "matcher_runtime_status.md", payload)
        _append_ledger(project_root)
    matcher_ok = bool(payload["mvp_matcher"]["exists"])
    segmenter_ok = bool(payload["mvp_segmenter"]["exists"])
    return 0 if matcher_ok and segmenter_ok else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen AdaPromptSet matcher/segmenter runtime inputs.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    raise SystemExit(run_audit(args.out, args.project_root))


if __name__ == "__main__":
    main()
