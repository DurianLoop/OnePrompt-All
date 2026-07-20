from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_GROUP_KEYS = ("dataset", "role", "query_case_id", "query_patient", "action_id")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_values(rows: Sequence[Mapping[str, str]], metric_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(metric_name, "")
        if raw == "":
            continue
        values.append(float(raw))
    return values


def aggregate_reference_metrics(
    rows: Iterable[Mapping[str, str]],
    *,
    metric_names: Sequence[str],
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        record: dict[str, Any] = dict(zip(group_keys, key))
        support_ids = sorted({row.get("support_case_id", "") for row in group_rows if row.get("support_case_id")})
        record["independent_unit"] = "query"
        record["reference_episode_count"] = len(group_rows)
        record["support_case_ids"] = ";".join(support_ids)
        for metric_name in metric_names:
            values = _float_values(group_rows, metric_name)
            if values:
                record[f"{metric_name}_mean"] = sum(values) / len(values)
                record[f"{metric_name}_count"] = len(values)
            else:
                record[f"{metric_name}_mean"] = None
                record[f"{metric_name}_count"] = 0
        aggregated.append(record)
    return aggregated


def summarize_episode_manifest_units(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    query_to_references: dict[str, set[str]] = defaultdict(set)
    query_to_patient: dict[str, str] = {}
    patient_to_queries: dict[str, set[str]] = defaultdict(set)
    role_counts = Counter(row.get("role", "") for row in rows)
    dataset_counts = Counter(row.get("dataset", "") for row in rows)
    for row in rows:
        query_id = row.get("query_case_id", "")
        if not query_id:
            continue
        patient_id = row.get("query_patient", "")
        support_id = row.get("support_case_id", "")
        query_to_patient[query_id] = patient_id
        patient_to_queries[patient_id].add(query_id)
        if support_id:
            query_to_references[query_id].add(support_id)
    ref_counts = {query_id: len(refs) for query_id, refs in sorted(query_to_references.items())}
    max_patient_queries = max((len(queries) for queries in patient_to_queries.values()), default=0)
    return {
        "episode_count": len(rows),
        "query_count": len(query_to_patient),
        "query_patient_count": len(patient_to_queries),
        "role_counts": dict(sorted(role_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "reference_episode_count_by_query": ref_counts,
        "min_references_per_query": min(ref_counts.values(), default=0),
        "max_references_per_query": max(ref_counts.values(), default=0),
        "queries_per_patient": {
            patient: len(queries) for patient, queries in sorted(patient_to_queries.items())
        },
        "max_queries_per_patient": max_patient_queries,
        "recommended_independent_unit": "patient_cluster" if max_patient_queries > 1 else "query",
        "multi_reference_episodes_are_independent": False,
    }
