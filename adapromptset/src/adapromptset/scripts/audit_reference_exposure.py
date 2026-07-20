from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _is_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return "/test/" in normalized or normalized.startswith("test/") or "/testing/" in normalized


def _dataset_dirs(project_root: Path) -> list[Path]:
    root = project_root / "manifests" / "episodes"
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _audit_dataset(project_root: Path, episode_dir: Path) -> dict[str, Any]:
    case_path = episode_dir / "case_assignments.csv"
    assignments = _read_csv(case_path)
    dataset = assignments[0].get("dataset", episode_dir.name) if assignments else episode_dir.name
    test_cases = {
        row.get("case_id", "")
        for row in assignments
        if row.get("role", row.get("split", "")) == "test"
    }
    test_patients = {
        row.get("patient_id", "")
        for row in assignments
        if row.get("role", row.get("split", "")) == "test"
    }
    issues: list[dict[str, str]] = []
    episode_count = 0
    support_cases: set[str] = set()
    for role in ("train", "val", "test"):
        path = episode_dir / f"episodes_{role}.csv"
        if not path.is_file():
            issues.append(
                {
                    "dataset": dataset,
                    "manifest": _relative(project_root, path),
                    "episode_id": "",
                    "reason": "missing_episode_manifest",
                    "value": role,
                }
            )
            continue
        for row in _read_csv(path):
            episode_count += 1
            episode_id = row.get("episode_id", "")
            support_cases.add(row.get("support_case_id", ""))
            if row.get("support_role") != "train":
                issues.append(
                    {
                        "dataset": dataset,
                        "manifest": _relative(project_root, path),
                        "episode_id": episode_id,
                        "reason": "support_role_not_train",
                        "value": row.get("support_role", ""),
                    }
                )
            if row.get("support_case_id") in test_cases:
                issues.append(
                    {
                        "dataset": dataset,
                        "manifest": _relative(project_root, path),
                        "episode_id": episode_id,
                        "reason": "support_case_from_test_role",
                        "value": row.get("support_case_id", ""),
                    }
                )
            if row.get("support_patient") in test_patients:
                issues.append(
                    {
                        "dataset": dataset,
                        "manifest": _relative(project_root, path),
                        "episode_id": episode_id,
                        "reason": "support_patient_from_test_role",
                        "value": row.get("support_patient", ""),
                    }
                )
            for field in ("support_image_path", "support_mask_path"):
                if _is_test_path(row.get(field, "")):
                    issues.append(
                        {
                            "dataset": dataset,
                            "manifest": _relative(project_root, path),
                            "episode_id": episode_id,
                            "reason": "support_path_points_to_test_split",
                            "value": row.get(field, ""),
                        }
                    )
    return {
        "dataset": dataset,
        "episode_dir": _relative(project_root, episode_dir),
        "episode_count": episode_count,
        "support_case_count": len(support_cases),
        "test_case_count": len(test_cases),
        "test_patient_count": len(test_patients),
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }


def build_reference_exposure_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    datasets = [_audit_dataset(project_root, path) for path in _dataset_dirs(project_root)]
    issues = [issue for item in datasets for issue in item["issues"]]
    reference_pool_train_only = not any(
        issue["reason"] in {"support_role_not_train", "support_path_points_to_test_split"}
        for issue in issues
    )
    reference_test_leakage_zero = not any(
        issue["reason"]
        in {
            "support_role_not_train",
            "support_case_from_test_role",
            "support_patient_from_test_role",
            "support_path_points_to_test_split",
        }
        for issue in issues
    )
    return {
        "schema_version": "adapromptset.reference_exposure_audit.v1",
        "status": "PASS" if not issues and datasets else "FAIL",
        "checks": {
            "reference_pool_train_only": reference_pool_train_only,
            "reference_test_leakage_zero": reference_test_leakage_zero,
        },
        "dataset_count": len(datasets),
        "datasets": datasets,
        "total_issues": len(issues),
        "issues": issues,
        "rule": "Every support/reference row must come from role=train and must not point to test cases, patients, images, or masks.",
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Reference Exposure Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This audit verifies that the reference pool is train-only and that reference-test leakage is zero.",
        "",
        "| Dataset | Episodes | Support cases | Test cases | Test patients | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset in payload["datasets"]:
        lines.append(
            "| "
            f"{dataset['dataset']} | "
            f"{dataset['episode_count']} | "
            f"{dataset['support_case_count']} | "
            f"{dataset['test_case_count']} | "
            f"{dataset['test_patient_count']} | "
            f"{dataset['status']} |"
        )
    lines.extend(["", "## Checks", ""])
    for key, passed in payload["checks"].items():
        lines.append(f"- {key.replace('_', ' ')}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Issues", ""])
    if payload["issues"]:
        lines.extend(
            f"- `{issue['dataset']}` `{issue['episode_id']}`: {issue['reason']} = `{issue['value']}`"
            for issue in payload["issues"]
        )
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_reference_exposure_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    payload = build_reference_exposure_audit(project_root)
    outputs = {
        "json": project_root / "reports" / "reference_exposure_audit.json",
        "markdown": project_root / "reports" / "reference_exposure_audit.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], payload)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AdaPromptSet reference exposure and test leakage.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    export_reference_exposure_audit(args.project_root)


if __name__ == "__main__":
    main()
