from __future__ import annotations

import random
from collections import defaultdict
from itertools import product
from typing import Any, Mapping, Sequence


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile for an empty sequence")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def build_query_level_comparison(
    episode_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_metric: str,
    comparator_metric: str,
) -> list[dict[str, Any]]:
    manifest_by_episode = {str(row["episode_id"]): row for row in manifest_rows}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    support_by_query: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in episode_rows:
        manifest = manifest_by_episode.get(str(row["episode_id"]))
        if manifest is None:
            continue
        query_case_id = str(manifest.get("query_case_id", ""))
        query_patient = str(manifest.get("query_patient", query_case_id))
        key = (query_case_id, query_patient)
        grouped[key].append(row)
        support_id = str(manifest.get("support_case_id", ""))
        if support_id:
            support_by_query[key].add(support_id)

    rows: list[dict[str, Any]] = []
    for query_case_id, query_patient in sorted(grouped):
        items = grouped[(query_case_id, query_patient)]
        baseline_mean = sum(_float(row, baseline_metric) for row in items) / len(items)
        comparator_mean = sum(_float(row, comparator_metric) for row in items) / len(items)
        rows.append(
            {
                "query_case_id": query_case_id,
                "query_patient": query_patient,
                "reference_episode_count": len(items),
                "support_case_ids": ";".join(sorted(support_by_query[(query_case_id, query_patient)])),
                f"{baseline_metric}_mean": baseline_mean,
                f"{comparator_metric}_mean": comparator_mean,
                "paired_delta": comparator_mean - baseline_mean,
            }
        )
    return rows


def paired_cluster_bootstrap(
    query_rows: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str = "query_patient",
    delta_key: str = "paired_delta",
    iterations: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
    min_clusters: int = 2,
) -> dict[str, Any]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in query_rows:
        clusters[str(row.get(cluster_key, row.get("query_case_id", "")))].append(row)

    cluster_ids = sorted(clusters)
    deltas = [_float(row, delta_key) for row in query_rows]
    observed = sum(deltas) / len(deltas) if deltas else 0.0
    base = {
        "unit": "patient_cluster" if cluster_key == "query_patient" else cluster_key,
        "cluster_key": cluster_key,
        "query_count": len(query_rows),
        "cluster_count": len(cluster_ids),
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
        "observed_mean_delta": observed,
    }
    if len(cluster_ids) < min_clusters or not query_rows:
        return {
            **base,
            "status": "PENDING_INSUFFICIENT_CLUSTERS",
            "ci_low": None,
            "ci_high": None,
            "bootstrap_mean_delta": None,
        }

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_rows: list[Mapping[str, Any]] = []
        for _ in cluster_ids:
            sampled_rows.extend(clusters[rng.choice(cluster_ids)])
        samples.append(sum(_float(row, delta_key) for row in sampled_rows) / len(sampled_rows))

    alpha = (1.0 - confidence) / 2.0
    return {
        **base,
        "status": "PASS",
        "ci_low": _quantile(samples, alpha),
        "ci_high": _quantile(samples, 1.0 - alpha),
        "bootstrap_mean_delta": sum(samples) / len(samples),
    }


def win_tie_loss(
    query_rows: Sequence[Mapping[str, Any]],
    *,
    delta_key: str = "paired_delta",
    tie_threshold: float = 0.005,
) -> dict[str, int]:
    counts = {"win": 0, "tie": 0, "loss": 0}
    for row in query_rows:
        delta = _float(row, delta_key)
        if abs(delta) < tie_threshold:
            counts["tie"] += 1
        elif delta > 0:
            counts["win"] += 1
        else:
            counts["loss"] += 1
    return counts


def paired_permutation_test(
    query_rows: Sequence[Mapping[str, Any]],
    *,
    delta_key: str = "paired_delta",
    max_exact_queries: int = 20,
    iterations: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    deltas = [_float(row, delta_key) for row in query_rows if abs(_float(row, delta_key)) > 0.0]
    observed = sum(deltas) / len(deltas) if deltas else 0.0
    base = {
        "query_count": len(query_rows),
        "nonzero_delta_count": len(deltas),
        "observed_mean_delta": observed,
        "alternative": "two_sided",
    }
    if not deltas:
        return {
            **base,
            "status": "PENDING_NO_NONZERO_DELTAS",
            "method": "exact_sign_flip",
            "p_value_two_sided": None,
        }

    observed_abs = abs(observed)
    if len(deltas) <= max_exact_queries:
        null_means = [
            sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas)
            for signs in product((-1.0, 1.0), repeat=len(deltas))
        ]
        extreme = sum(1 for mean in null_means if abs(mean) >= observed_abs - 1e-15)
        return {
            **base,
            "status": "PASS",
            "method": "exact_sign_flip",
            "permutation_count": len(null_means),
            "p_value_two_sided": extreme / len(null_means),
        }

    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        mean = sum((1.0 if rng.random() < 0.5 else -1.0) * delta for delta in deltas) / len(deltas)
        if abs(mean) >= observed_abs - 1e-15:
            extreme += 1
    return {
        **base,
        "status": "PASS",
        "method": "monte_carlo_sign_flip",
        "permutation_count": iterations,
        "seed": seed,
        "p_value_two_sided": extreme / iterations,
    }
