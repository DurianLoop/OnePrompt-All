from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from adapromptset.data.audit import GateResult


def _eligible_rows(rows: list[Mapping[str, str]], role: str) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("role", row.get("split", "")) == role]


def _select_validation_patients(
    train_patients: set[str],
    val_patient_count: int,
    seed: int,
) -> set[str]:
    if val_patient_count < 0:
        raise ValueError("val_patient_count must be non-negative")
    if val_patient_count == 0:
        return set()
    if val_patient_count >= len(train_patients):
        raise ValueError("val_patient_count must leave at least one train patient")
    ordered = sorted(train_patients)
    random.Random(seed).shuffle(ordered)
    return set(ordered[:val_patient_count])


def build_episode_protocol(
    rows: list[Mapping[str, str]],
    *,
    val_patient_count: int,
    references_per_query: int,
    seed: int,
) -> dict[str, Any]:
    if references_per_query < 1:
        raise ValueError("references_per_query must be positive")
    source_train = _eligible_rows(rows, "train")
    source_test = _eligible_rows(rows, "test")
    if not source_train:
        raise ValueError("protocol requires at least one train row")
    if not source_test:
        raise ValueError("protocol requires at least one test row")

    val_patients = _select_validation_patients(
        {row["patient_id"] for row in source_train},
        val_patient_count,
        seed,
    )
    assignments: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: (item.get("dataset", ""), item.get("case_id", ""))):
        assigned = dict(row)
        role = row.get("role", row.get("split", ""))
        if role == "train" and row["patient_id"] in val_patients:
            assigned["role"] = "val"
        assignments.append(assigned)

    train_supports = [row for row in assignments if row["role"] == "train"]
    protocol: dict[str, Any] = {"case_assignments": assignments}
    for role in ("train", "val", "test"):
        episodes: list[dict[str, str]] = []
        queries = [row for row in assignments if row["role"] == role]
        for query in queries:
            eligible = [
                support
                for support in train_supports
                if support["case_id"] != query["case_id"]
                and support["patient_id"] != query["patient_id"]
            ]
            if len(eligible) < references_per_query:
                raise ValueError(
                    f"Query {query['case_id']} has only {len(eligible)} patient-disjoint train supports."
                )
            rng = random.Random(f"{seed}:{role}:{query['case_id']}")
            selected = rng.sample(sorted(eligible, key=lambda item: item["case_id"]), references_per_query)
            for support in selected:
                episodes.append(
                    {
                        "episode_id": (
                            f"{query['dataset'].lower()}_{role}__"
                            f"{support['case_id']}__to__{query['case_id']}"
                        ),
                        "dataset": query["dataset"],
                        "role": role,
                        "support_role": "train",
                        "support_case_id": support["case_id"],
                        "support_patient": support["patient_id"],
                        "support_image_path": support["image_path"],
                        "support_mask_path": support["mask_path"],
                        "query_case_id": query["case_id"],
                        "query_patient": query["patient_id"],
                        "query_source_split": query["split"],
                        "query_image_path": query["image_path"],
                        "query_mask_path": query["mask_path"],
                    }
                )
        protocol[f"episodes_{role}"] = episodes

    protocol["metadata"] = {
        "seed": seed,
        "references_per_query": references_per_query,
        "val_patient_count": val_patient_count,
        "role_counts": dict(sorted(Counter(row["role"] for row in assignments).items())),
        "episode_counts": {
            role: len(protocol[f"episodes_{role}"]) for role in ("train", "val", "test")
        },
        "leakage_rules": [
            "Support rows always come from role=train.",
            "Support and query patient_id must differ.",
            "Rows marked excluded_overlap are never sampled as support or query.",
        ],
    }
    return protocol


def validate_gate_b(protocol: Mapping[str, Any]) -> GateResult:
    issues: list[str] = []
    seen_episode_ids: set[str] = set()
    for role in ("train", "val", "test"):
        for episode in protocol.get(f"episodes_{role}", []):
            episode_id = episode.get("episode_id", "")
            if episode_id in seen_episode_ids:
                issues.append(f"duplicate episode_id: {episode_id}")
            seen_episode_ids.add(episode_id)
            if episode.get("support_role") != "train":
                issues.append(f"support_role leak in {episode_id}: {episode.get('support_role')}")
            if episode.get("support_patient") == episode.get("query_patient"):
                issues.append(f"support/query patient leak in {episode_id}")
            if episode.get("support_case_id") == episode.get("query_case_id"):
                issues.append(f"support/query case leak in {episode_id}")
    return GateResult(
        name="GATE-B",
        passed=not issues,
        issues=tuple(issues),
        metadata=protocol.get("metadata", {}),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_protocol(out_dir: Path, protocol: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "case_assignments.csv", protocol["case_assignments"])
    for role in ("train", "val", "test"):
        write_csv(out_dir / f"episodes_{role}.csv", protocol[f"episodes_{role}"])
    (out_dir / "protocol_metadata.json").write_text(
        json.dumps(protocol["metadata"], indent=2),
        encoding="utf-8",
    )

