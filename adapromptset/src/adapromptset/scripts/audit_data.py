from __future__ import annotations

import argparse
from pathlib import Path

from adapromptset.data.audit import (
    DatasetSpec,
    GateResult,
    build_glass_manifest,
    build_monusac_manifest,
    build_monuseg_manifest,
    validate_gate_a,
    write_compute_ledger,
    write_dataset_audit,
    write_exposure_log,
    write_gate_status,
    write_manifest_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAM_ROOT = PROJECT_ROOT.parent


def default_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="MoNuSAC",
            role="main development",
            root=SAM_ROOT / "code" / "datasets" / "MoNuSAC" / "processed",
            tuning_allowed=True,
            notes="Primary MVP development dataset; formal test remains locked by gate protocol.",
        ),
        DatasetSpec(
            name="MoNuSeg",
            role="sanity/historical",
            root=SAM_ROOT / "code" / "datasets" / "MonuSeg" / "official",
            tuning_allowed=False,
            notes="Used for sanity and historical alignment, not for MVP action-bank redesign.",
        ),
        DatasetSpec(
            name="GlaS",
            role="first morphology-shift extension",
            root=SAM_ROOT
            / "external"
            / "research_data"
            / "extracted"
            / "GlaS"
            / "Warwick QU Dataset (Released 2016_07_08)",
            tuning_allowed=False,
            notes="Extension/confirmation only until MoNuSAC gates pass.",
        ),
    ]


def run_audit(project_root: Path = PROJECT_ROOT) -> int:
    specs = default_specs()
    builders = {
        "MoNuSAC": build_monusac_manifest,
        "MoNuSeg": build_monuseg_manifest,
        "GlaS": build_glass_manifest,
    }
    gates = []
    for spec in specs:
        rows = builders[spec.name](spec.root)
        manifest_path = project_root / "manifests" / f"{spec.name.lower()}_manifest.csv"
        write_manifest_csv(manifest_path, rows)
        raw_gate = validate_gate_a(rows)
        gate = GateResult(
            name=f"GATE-A/{spec.name}",
            passed=raw_gate.passed,
            issues=raw_gate.issues,
            metadata=raw_gate.metadata,
        )
        gates.append(gate)
        write_dataset_audit(project_root / "reports" / "dataset_audit" / f"{spec.name.lower()}.json", rows, gate)
    write_exposure_log(project_root / "docs" / "data_exposure_log.md", specs)
    write_gate_status(project_root / "reports" / "gate_status.md", gates)
    write_compute_ledger(project_root / "reports" / "compute_ledger.csv")
    return 0 if all(gate.passed for gate in gates) else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AdaPromptSet MVP dataset manifests and Gate-A audit files.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    raise SystemExit(run_audit(args.project_root))


if __name__ == "__main__":
    main()
