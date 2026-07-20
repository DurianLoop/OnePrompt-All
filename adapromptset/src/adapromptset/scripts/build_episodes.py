from __future__ import annotations

import argparse
import csv
from pathlib import Path

from adapromptset.data.audit import GateResult, write_gate_status
from adapromptset.data.protocol import build_episode_protocol, read_csv, validate_gate_b, write_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULTS = {
    "monusac": {"val_patient_count": 9, "references_per_query": 5, "seed": 20260717},
    "monuseg": {"val_patient_count": 8, "references_per_query": 1, "seed": 20260717},
    "glas": {"val_patient_count": 1, "references_per_query": 1, "seed": 20260717},
}


def run(project_root: Path = PROJECT_ROOT) -> int:
    gates: list[GateResult] = []
    for dataset, defaults in DEFAULTS.items():
        rows = read_csv(project_root / "manifests" / f"{dataset}_manifest.csv")
        protocol = build_episode_protocol(rows, **defaults)
        out_dir = project_root / "manifests" / "episodes" / dataset
        write_protocol(out_dir, protocol)
        raw_gate = validate_gate_b(protocol)
        gates.append(
            GateResult(
                name=f"GATE-B/{dataset}",
                passed=raw_gate.passed,
                issues=raw_gate.issues,
                metadata=raw_gate.metadata,
            )
        )
    write_gate_status(project_root / "reports" / "gate_b_status.md", gates)
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "2",
                "command": "scripts/build_episodes.py",
                "gpu_hours": "0.0",
                "notes": "CPU-only deterministic reference-query episode manifests and Gate-B audit.",
            }
        )
    return 0 if all(gate.passed for gate in gates) else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic AdaPromptSet reference-query episode manifests.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    raise SystemExit(run(args.project_root))


if __name__ == "__main__":
    main()
