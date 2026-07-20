from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adapromptset.statistics.units import read_csv_rows, summarize_episode_manifest_units


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Statistical Unit Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "Multi-reference episodes are repeated measurements for a query and are not independent samples.",
        "Metrics must be aggregated to query-level before patient/query bootstrap; use patient-cluster bootstrap when multiple queries share a patient.",
        "",
        "| Manifest | Episodes | Queries | Patients | Max refs/query | Independent unit |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for summary in payload["manifests"]:
        lines.append(
            "| "
            f"{summary['relative_path']} | "
            f"{summary['episode_count']} | "
            f"{summary['query_count']} | "
            f"{summary['query_patient_count']} | "
            f"{summary['max_references_per_query']} | "
            f"{summary['recommended_independent_unit']} |"
        )
    lines.extend(
        [
            "",
            "Decision: report query-level metrics; do not treat reference episodes as independent samples.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def run_audit(manifest_paths: list[Path], out_dir: Path, project_root: Path = PROJECT_ROOT) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    issues: list[str] = []
    for manifest_path in manifest_paths:
        rows = read_csv_rows(manifest_path)
        summary = summarize_episode_manifest_units(rows)
        summary["path"] = str(manifest_path)
        summary["relative_path"] = _relative(manifest_path, project_root)
        if summary["episode_count"] and summary["query_count"] and summary["episode_count"] > summary["query_count"]:
            if summary["multi_reference_episodes_are_independent"]:
                issues.append(f"{manifest_path} incorrectly treats references as independent")
        summaries.append(summary)
    payload = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "manifests": summaries,
        "rule": "Aggregate reference episodes to query-level; patient-cluster when multiple queries share a patient.",
    }
    (out_dir / "statistical_units.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out_dir / "statistical_units.md", payload)
    return 0 if not issues else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AdaPromptSet statistical units for episode manifests.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "statistical_units")
    args = parser.parse_args()
    manifest_paths = args.manifest or [
        args.project_root / "manifests" / "episodes" / "monusac" / "episodes_train.csv",
        args.project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv",
        args.project_root / "manifests" / "episodes" / "monusac" / "episodes_test.csv",
        args.project_root / "manifests" / "episodes" / "monuseg" / "episodes_val.csv",
        args.project_root / "manifests" / "episodes" / "glas" / "episodes_val.csv",
    ]
    raise SystemExit(run_audit(manifest_paths=manifest_paths, out_dir=args.out_dir, project_root=args.project_root))


if __name__ == "__main__":
    main()
