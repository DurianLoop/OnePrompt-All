from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_GLAS_ROOT = (
    WORKSPACE_ROOT
    / "external"
    / "research_data"
    / "extracted"
    / "GlaS"
    / "Warwick QU Dataset (Released 2016_07_08)"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "manifests" / "phase2" / "glas"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase2_glas" / "split"
DEFAULT_SEED = 20260719
DEFAULT_TRAIN_COUNT = 55
DEFAULT_VAL_COUNT = 30
DEFAULT_REFERENCES_PER_QUERY = 3


@dataclass(frozen=True)
class GlasCase:
    case_id: str
    source_split: str
    patient_id: str
    grade: str
    sirin_grade: str
    image_path: Path
    mask_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(path: Path, *, base: Path = WORKSPACE_ROOT) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _source_split(case_id: str) -> str:
    if case_id.startswith("testA_"):
        return "testA"
    if case_id.startswith("testB_"):
        return "testB"
    if case_id.startswith("train_"):
        return "train"
    raise ValueError(f"Unsupported GlaS case id: {case_id}")


def read_grade_rows(glas_root: Path) -> list[GlasCase]:
    grade_path = glas_root / "Grade.csv"
    if not grade_path.is_file():
        raise FileNotFoundError(f"Missing GlaS Grade.csv: {grade_path}")
    cases: list[GlasCase] = []
    with grade_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case_id = row["name"].strip()
            image_path = glas_root / f"{case_id}.bmp"
            mask_path = glas_root / f"{case_id}_anno.bmp"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing image/mask pair for {case_id}")
            cases.append(
                GlasCase(
                    case_id=case_id,
                    source_split=_source_split(case_id),
                    patient_id=row["patient ID"].strip(),
                    grade=row[" grade (GlaS)"].strip(),
                    sirin_grade=row[" grade (Sirinukunwattana et al. 2015)"].strip(),
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    return cases


def _stratified_validation_ids(cases: list[GlasCase], val_count: int, seed: int) -> set[str]:
    by_grade: dict[str, list[GlasCase]] = defaultdict(list)
    for case in cases:
        by_grade[case.grade].append(case)
    grade_counts = {grade: len(items) for grade, items in by_grade.items()}
    raw_targets = {
        grade: len(items) * val_count / len(cases)
        for grade, items in by_grade.items()
    }
    targets = {grade: int(raw_targets[grade]) for grade in by_grade}
    while sum(targets.values()) < val_count:
        grade = max(
            by_grade,
            key=lambda item: (raw_targets[item] - targets[item], grade_counts[item], item),
        )
        targets[grade] += 1
    while sum(targets.values()) > val_count:
        grade = max(targets, key=lambda item: (targets[item] - raw_targets[item], targets[item], item))
        targets[grade] -= 1

    selected: set[str] = set()
    for grade, items in sorted(by_grade.items()):
        ordered = sorted(items, key=lambda item: item.case_id)
        random.Random(f"{seed}:val:{grade}").shuffle(ordered)
        selected.update(case.case_id for case in ordered[: targets[grade]])
    return selected


def build_phase2_assignments(
    cases: list[GlasCase],
    *,
    train_count: int = DEFAULT_TRAIN_COUNT,
    val_count: int = DEFAULT_VAL_COUNT,
    seed: int = DEFAULT_SEED,
    path_base: Path = WORKSPACE_ROOT,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    split_counts = Counter(case.source_split for case in cases)
    expected = {"train": 85, "testA": 60, "testB": 20}
    if dict(split_counts) != expected:
        raise ValueError(f"Expected GlaS source split counts {expected}, got {dict(split_counts)}")
    if train_count + val_count != split_counts["train"]:
        raise ValueError("Phase 2 train_count + val_count must equal the 85 official train cases.")

    official_train = [case for case in cases if case.source_split == "train"]
    val_ids = _stratified_validation_ids(official_train, val_count, seed)
    rows: list[dict[str, str]] = []
    for case in sorted(cases, key=lambda item: (item.source_split, item.case_id)):
        if case.source_split == "train":
            role = "utility_train" if case.case_id not in val_ids else "strict_val"
        elif case.source_split == "testA":
            role = "locked_testA"
        else:
            role = "locked_testB"
        rows.append(
            {
                "dataset": "GlaS",
                "case_id": case.case_id,
                "source_split": case.source_split,
                "phase2_role": role,
                "patient_id": case.patient_id,
                "grade": case.grade,
                "sirin_grade": case.sirin_grade,
                "image_path": _relative(case.image_path, base=path_base),
                "mask_path": _relative(case.mask_path, base=path_base),
                "image_sha256": _sha256(case.image_path),
                "mask_sha256": _sha256(case.mask_path),
            }
        )

    role_counts = Counter(row["phase2_role"] for row in rows)
    if role_counts["utility_train"] != train_count or role_counts["strict_val"] != val_count:
        raise AssertionError(f"Unexpected Phase 2 role counts: {dict(role_counts)}")

    train_patients = {row["patient_id"] for row in rows if row["phase2_role"] == "utility_train"}
    val_patients = {row["patient_id"] for row in rows if row["phase2_role"] == "strict_val"}
    test_patients = {
        row["patient_id"]
        for row in rows
        if row["phase2_role"] in {"locked_testA", "locked_testB"}
    }
    metadata = {
        "protocol_version": "adapromptset.phase2.glas_strict.v2",
        "seed": seed,
        "source_split_counts": dict(sorted(split_counts.items())),
        "phase2_role_counts": dict(sorted(role_counts.items())),
        "grade_counts_by_role": {
            role: dict(sorted(Counter(row["grade"] for row in rows if row["phase2_role"] == role).items()))
            for role in sorted(role_counts)
        },
        "patient_counts_by_role": {
            role: len({row["patient_id"] for row in rows if row["phase2_role"] == role})
            for role in sorted(role_counts)
        },
        "train_val_patient_overlap": sorted(train_patients & val_patients),
        "train_locked_test_patient_overlap": sorted(train_patients & test_patients),
        "val_locked_test_patient_overlap": sorted(val_patients & test_patients),
        "formal_test_boundary": "locked_not_fully_blind_due_to_prior_manifest_profile_exposure",
    }
    return rows, metadata


def build_reference_episodes(
    assignments: list[dict[str, str]],
    *,
    references_per_query: int = DEFAULT_REFERENCES_PER_QUERY,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, str]]]:
    train_pool = [row for row in assignments if row["phase2_role"] == "utility_train"]
    query_pools = {
        "train": train_pool,
        "val": [row for row in assignments if row["phase2_role"] == "strict_val"],
    }
    episodes: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    for role, queries in query_pools.items():
        for query in sorted(queries, key=lambda item: item["case_id"]):
            eligible = [
                support
                for support in train_pool
                if support["case_id"] != query["case_id"]
            ]
            same_grade = [support for support in eligible if support["grade"] == query["grade"]]
            pool = same_grade if len(same_grade) >= references_per_query else eligible
            if len(pool) < references_per_query:
                raise ValueError(f"Not enough references for {query['case_id']}")
            ordered = sorted(pool, key=lambda item: item["case_id"])
            rng = random.Random(f"{seed}:{role}:{query['case_id']}:refs")
            selected = rng.sample(ordered, references_per_query)
            fallback = "false" if pool is same_grade else "true"
            for index, support in enumerate(selected):
                episodes[role].append(
                    {
                        "episode_id": f"glas_phase2_{role}_{query['case_id']}_ref{index + 1}_{support['case_id']}",
                        "dataset": "GlaS",
                        "role": role,
                        "support_role": "utility_train",
                        "support_case_id": support["case_id"],
                        "support_patient_id": support["patient_id"],
                        "support_grade": support["grade"],
                        "support_image_path": support["image_path"],
                        "support_mask_path": support["mask_path"],
                        "query_case_id": query["case_id"],
                        "query_patient_id": query["patient_id"],
                        "query_grade": query["grade"],
                        "query_source_split": query["source_split"],
                        "query_image_path": query["image_path"],
                        "query_mask_path": query["mask_path"],
                        "reference_index": str(index + 1),
                        "grade_matched_reference_fallback": fallback,
                    }
                )
    return episodes


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GlaS Phase 2 Split Report",
        "",
        f"Status: `{summary['status']}`",
        f"Protocol: `{summary['metadata']['protocol_version']}`",
        f"Seed: `{summary['metadata']['seed']}`",
        "",
        "## Role Counts",
        "",
    ]
    for role, count in summary["metadata"]["phase2_role_counts"].items():
        lines.append(f"- `{role}`: {count}")
    lines.extend(["", "## Episode Counts", ""])
    for role, count in summary["episode_counts"].items():
        lines.append(f"- `{role}`: {count}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- Formal test boundary: `{summary['metadata']['formal_test_boundary']}`",
            "- testA/testB locked cases are recorded but no Phase 2 test episodes are generated.",
            "- Query-level aggregation is the Gate F unit; per-reference rows are retained for sensitivity.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in summary["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(
    glas_root: Path = DEFAULT_GLAS_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    *,
    seed: int = DEFAULT_SEED,
    train_count: int = DEFAULT_TRAIN_COUNT,
    val_count: int = DEFAULT_VAL_COUNT,
    references_per_query: int = DEFAULT_REFERENCES_PER_QUERY,
) -> dict[str, Any]:
    cases = read_grade_rows(glas_root)
    assignments, metadata = build_phase2_assignments(
        cases,
        train_count=train_count,
        val_count=val_count,
        seed=seed,
        path_base=glas_root.parent,
    )
    episodes = build_reference_episodes(assignments, references_per_query=references_per_query, seed=seed)
    locked_tests = [
        row
        for row in assignments
        if row["phase2_role"] in {"locked_testA", "locked_testB"}
    ]
    metadata["references_per_query"] = references_per_query
    metadata["episode_counts"] = {role: len(rows) for role, rows in episodes.items()}
    metadata["query_counts"] = {
        "train": len({row["query_case_id"] for row in episodes["train"]}),
        "val": len({row["query_case_id"] for row in episodes["val"]}),
    }
    metadata["locked_test_case_counts"] = dict(Counter(row["phase2_role"] for row in locked_tests))

    artifacts = {
        "case_assignments": (out_dir / "case_assignments.csv").relative_to(PROJECT_ROOT).as_posix(),
        "episodes_train": (out_dir / "episodes_train.csv").relative_to(PROJECT_ROOT).as_posix(),
        "episodes_val": (out_dir / "episodes_val.csv").relative_to(PROJECT_ROOT).as_posix(),
        "locked_test_cases": (out_dir / "locked_test_cases.csv").relative_to(PROJECT_ROOT).as_posix(),
        "protocol_metadata": (out_dir / "protocol_metadata.json").relative_to(PROJECT_ROOT).as_posix(),
        "report_json": (report_dir / "split_report.json").relative_to(PROJECT_ROOT).as_posix(),
        "report_md": (report_dir / "split_report.md").relative_to(PROJECT_ROOT).as_posix(),
    }
    summary = {
        "schema_version": "adapromptset.phase2.glas_split.v1",
        "status": "PASS_PHASE2_GLAS_SPLIT",
        "metadata": metadata,
        "artifacts": artifacts,
        "episode_counts": metadata["episode_counts"],
    }
    _write_csv(out_dir / "case_assignments.csv", assignments)
    _write_csv(out_dir / "episodes_train.csv", episodes["train"])
    _write_csv(out_dir / "episodes_val.csv", episodes["val"])
    _write_csv(out_dir / "locked_test_cases.csv", locked_tests)
    _write_json(out_dir / "protocol_metadata.json", metadata)
    _write_json(report_dir / "split_report.json", summary)
    _write_markdown(report_dir / "split_report.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 GlaS 55/30 strict diagnostic manifests.")
    parser.add_argument("--glas-root", type=Path, default=DEFAULT_GLAS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument("--references-per-query", type=int, default=DEFAULT_REFERENCES_PER_QUERY)
    args = parser.parse_args()
    summary = build_outputs(
        args.glas_root,
        args.out_dir,
        args.report_dir,
        seed=args.seed,
        train_count=args.train_count,
        val_count=args.val_count,
        references_per_query=args.references_per_query,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
