from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MASK_SUFFIXES = IMAGE_SUFFIXES


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    role: str
    root: Path
    tuning_allowed: bool
    notes: str


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    issues: tuple[str, ...]
    metadata: Mapping[str, object]


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _paired_rows(
    *,
    dataset: str,
    root: Path,
    split: str,
    image_dir: Path,
    mask_dir: Path,
    patient_by_case: Mapping[str, str] | None = None,
    extra_by_case: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    if not image_dir.is_dir():
        return []
    if not mask_dir.is_dir():
        raise ValueError(f"missing mask directory for {dataset} {split}: {mask_dir}")
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES
    }
    rows: list[dict[str, str]] = []
    for image_path in sorted(image_dir.iterdir(), key=lambda item: item.stem):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            raise ValueError(f"missing mask for {dataset} image {image_path.name}")
        case_id = image_path.stem
        row = {
            "dataset": dataset,
            "case_id": case_id,
            "split": split,
            "role": split,
            "patient_id": patient_by_case.get(case_id, case_id) if patient_by_case else case_id,
            "image_path": _relative(image_path, root.parent),
            "mask_path": _relative(mask_path, root.parent),
            "image_sha256": compute_file_sha256(image_path),
            "mask_sha256": compute_file_sha256(mask_path),
        }
        if extra_by_case and case_id in extra_by_case:
            row.update(extra_by_case[case_id])
        rows.append(row)
    return rows


def build_monuseg_manifest(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        rows.extend(
            _paired_rows(
                dataset="MoNuSeg",
                root=root,
                split=split,
                image_dir=root / split / "images",
                mask_dir=root / split / "masks",
            )
        )
    if not rows:
        raise ValueError(f"no MoNuSeg rows discovered below {root}")
    return rows


def build_monusac_manifest(root: Path) -> list[dict[str, str]]:
    manifest = root / "monusac_processed_manifest.csv"
    if not manifest.is_file():
        raise ValueError(f"missing MoNuSAC processed manifest: {manifest}")
    source_rows = _read_csv(manifest)
    patient_by_case = {row["case_id"]: row.get("patient", row["case_id"]) for row in source_rows}
    extra_by_case = {
        row["case_id"]: {
            "tissue": row.get("tissue", ""),
            "instance_count": row.get("instance_count", ""),
            "classes_present": row.get("classes_present", ""),
        }
        for row in source_rows
    }
    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        rows.extend(
            _paired_rows(
                dataset="MoNuSAC",
                root=root,
                split=split,
                image_dir=root / split / "images",
                mask_dir=root / split / "masks",
                patient_by_case=patient_by_case,
                extra_by_case=extra_by_case,
            )
        )
    expected = {row["case_id"] for row in source_rows}
    observed = {row["case_id"] for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"MoNuSAC manifest cases missing paired files: {missing[:5]}")
    test_patients = {row["patient_id"] for row in rows if row["split"] == "test"}
    for row in rows:
        if row["split"] == "test":
            row["role"] = "test"
        elif row["patient_id"] in test_patients:
            row["role"] = "excluded_overlap"
        else:
            row["role"] = "train"
    return rows


def build_glass_manifest(root: Path) -> list[dict[str, str]]:
    grade_path = root / "Grade.csv"
    if not grade_path.is_file():
        raise ValueError(f"missing GlaS Grade.csv: {grade_path}")
    rows: list[dict[str, str]] = []
    for grade in _read_csv(grade_path):
        case_id = grade["name"].strip()
        image_path = root / f"{case_id}.bmp"
        mask_path = root / f"{case_id}_anno.bmp"
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"missing GlaS image/mask pair for {case_id}")
        split = case_id.rsplit("_", 1)[0]
        rows.append(
            {
                "dataset": "GlaS",
                "case_id": case_id,
                "split": split,
                "role": "test" if split in {"testA", "testB"} else "train",
                "patient_id": grade["patient ID"].strip(),
                "grade": grade[" grade (GlaS)"].strip(),
                "image_path": _relative(image_path, root.parent),
                "mask_path": _relative(mask_path, root.parent),
                "image_sha256": compute_file_sha256(image_path),
                "mask_sha256": compute_file_sha256(mask_path),
            }
        )
    if not rows:
        raise ValueError(f"no GlaS rows discovered below {root}")
    test_patients = {row["patient_id"] for row in rows if row["role"] == "test"}
    for row in rows:
        if row["role"] == "train" and row["patient_id"] in test_patients:
            row["role"] = "excluded_overlap"
    return sorted(rows, key=lambda row: row["case_id"])


def summarize_manifest(rows: Iterable[Mapping[str, str]]) -> dict[str, object]:
    items = list(rows)
    split_counts = Counter(row["split"] for row in items)
    role_counts = Counter(row.get("role", row["split"]) for row in items)
    patient_counts = Counter(row.get("patient_id", "") for row in items)
    return {
        "case_count": len(items),
        "split_counts": dict(sorted(split_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "patient_count": len([key for key in patient_counts if key]),
        "duplicate_case_ids": sorted(
            case for case, count in Counter(row["case_id"] for row in items).items() if count > 1
        ),
    }


def validate_gate_a(rows: Iterable[Mapping[str, str]]) -> GateResult:
    items = list(rows)
    issues: list[str] = []
    summary = summarize_manifest(items)
    if not items:
        issues.append("empty manifest")
    if summary["duplicate_case_ids"]:
        issues.append(f"duplicate case ids: {summary['duplicate_case_ids']}")
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in items:
        patient = row.get("patient_id", "")
        split = row.get("role", row.get("split", ""))
        if split == "excluded_overlap":
            continue
        if not row.get("image_sha256") or not row.get("mask_sha256"):
            issues.append(f"missing file hash for {row.get('case_id', '<unknown>')}")
        if patient:
            by_split[split].add(patient)
    train_like = by_split.get("train", set())
    test_like = set().union(
        by_split.get("test", set()),
        by_split.get("testA", set()),
        by_split.get("testB", set()),
    )
    overlap = sorted(train_like & test_like)
    if overlap:
        issues.append(f"patient overlap between train and test splits: {overlap}")
    return GateResult(
        name="GATE-A",
        passed=not issues,
        issues=tuple(issues),
        metadata=summary,
    )


def write_manifest_csv(path: Path, rows: list[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{compute_file_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )
    summary_path = path.with_name(path.stem + ".summary.json")
    summary_path.write_text(json.dumps(summarize_manifest(rows), indent=2), encoding="utf-8")


def write_dataset_audit(path: Path, rows: list[Mapping[str, str]], gate: GateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": rows[0]["dataset"] if rows else path.stem,
        "summary": summarize_manifest(rows),
        "gate": {
            "name": gate.name,
            "passed": gate.passed,
            "issues": list(gate.issues),
            "metadata": dict(gate.metadata),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_exposure_log(path: Path, specs: Iterable[DatasetSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Exposure Log",
        "",
        "This log records whether each local dataset may influence AdaPromptSet MVP tuning.",
        "",
        "| Dataset | Role | Root | Tuning | Notes |",
        "|---|---|---|---|---|",
    ]
    for spec in specs:
        lines.append(
            f"| {spec.name} | {spec.role} | `{spec.root}` | "
            f"tuning_allowed={spec.tuning_allowed} | {spec.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gate_status(path: Path, gate_results: Iterable[GateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gate Status",
        "",
        "| Gate | Status | Issues |",
        "|---|---|---|",
    ]
    for result in gate_results:
        issues = "; ".join(result.issues) if result.issues else "none"
        lines.append(f"| {result.name} | {'PASS' if result.passed else 'FAIL'} | {issues} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compute_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "phase": "0-1",
                "command": "scripts/audit_data.py",
                "gpu_hours": "0.0",
                "notes": "CPU-only manifest, hash, exposure, and Gate-A audit.",
            }
        )
