from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def summarize_candidate_pilot(
    rows: list[Mapping[str, Any]],
    *,
    precision_margin: float,
    min_episodes: int,
) -> dict[str, Any]:
    if not rows:
        return {"status": "FAIL", "issues": ["empty candidate pilot"], "episode_count": 0}
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    issues: list[str] = []
    if len(episode_ids) < min_episodes:
        issues.append(f"episode_count {len(episode_ids)} < min_episodes {min_episodes}")
    if any(int(row.get("validation_issue_count", 0)) > 0 for row in rows):
        issues.append("candidate validation issues present")
    if any(float(row.get("negative_blank_fraction", 0.0)) > 0.5 for row in rows):
        issues.append("negative blank/padding fraction exceeds 0.5")

    max_limit = max(int(row["positive_limit"]) for row in rows)
    max_rows = [row for row in rows if int(row["positive_limit"]) == max_limit]
    mean_precision = mean(_float(row, "point_precision") for row in max_rows)
    mean_random = mean(_float(row, "random_precision") for row in max_rows)
    precision_delta = mean_precision - mean_random
    if precision_delta < precision_margin:
        issues.append(
            f"precision margin {precision_delta:.4f} < required {precision_margin:.4f}"
        )

    coverage_by_episode: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        coverage_by_episode[str(row["episode_id"])].append(
            (int(row["positive_limit"]), _float(row, "component_coverage"))
        )
    bad_coverage = []
    for episode_id, values in coverage_by_episode.items():
        ordered = sorted(values)
        for (_, earlier), (_, later) in zip(ordered, ordered[1:]):
            if later + 1e-12 < earlier:
                bad_coverage.append(episode_id)
                break
    if bad_coverage:
        issues.append(f"coverage decreased with pool size for episodes: {bad_coverage}")

    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "episode_count": len(episode_ids),
        "max_positive_limit": max_limit,
        "mean_point_precision": mean_precision,
        "mean_random_precision": mean_random,
        "precision_delta": precision_delta,
        "mean_component_coverage": mean(_float(row, "component_coverage") for row in max_rows),
        "mean_negative_blank_fraction": mean(float(row.get("negative_blank_fraction", 0.0)) for row in max_rows),
    }
