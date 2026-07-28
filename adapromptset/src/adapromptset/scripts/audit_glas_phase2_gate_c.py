from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_METRICS = PROJECT_ROOT / "reports" / "phase2_glas" / "candidates" / "candidate_pilot_metrics.csv"
DEFAULT_CANDIDATES = PROJECT_ROOT / "reports" / "phase2_glas" / "candidates" / "candidate_pilot_candidates.csv"
DEFAULT_SPLIT = PROJECT_ROOT / "reports" / "phase2_glas" / "split" / "split_report.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "phase2_glas" / "candidates"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "paper_phase2" / "figures" / "candidate_diagnostics"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "glas_candidates"

MIN_PRECISION_DELTA = 0.15
MIN_PRECISION_RATIO = 1.5
MIN_COMPONENT_COVERAGE = 0.60
MAX_NEGATIVE_BLANK_FRACTION = 0.50


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value in {None, ""} else float(value)


def _limits(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(row["positive_limit"]) for row in rows})


def _aggregate_queries(rows: list[dict[str, str]], *, positive_limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["positive_limit"]) == positive_limit:
            grouped[row["query_case_id"]].append(row)
    query_rows: list[dict[str, Any]] = []
    for query_case_id, items in sorted(grouped.items()):
        query_rows.append(
            {
                "query_case_id": query_case_id,
                "reference_episode_count": len(items),
                "positive_limit": positive_limit,
                "point_precision": mean(_float(row, "point_precision") for row in items),
                "random_precision": mean(_float(row, "random_precision") for row in items),
                "precision_delta": mean(_float(row, "point_precision") - _float(row, "random_precision") for row in items),
                "component_coverage": mean(_float(row, "component_coverage") for row in items),
                "negative_blank_fraction": mean(_float(row, "negative_blank_fraction") for row in items),
                "validation_issue_count": sum(int(float(row.get("validation_issue_count") or 0)) for row in items),
            }
        )
    return query_rows


def _coverage_curve(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for limit in _limits(rows):
        query_rows = _aggregate_queries(rows, positive_limit=limit)
        curve.append(
            {
                "positive_limit": limit,
                "query_count": len(query_rows),
                "mean_point_precision": mean(row["point_precision"] for row in query_rows),
                "mean_random_precision": mean(row["random_precision"] for row in query_rows),
                "mean_precision_delta": mean(row["precision_delta"] for row in query_rows),
                "mean_component_coverage": mean(row["component_coverage"] for row in query_rows),
                "mean_negative_blank_fraction": mean(row["negative_blank_fraction"] for row in query_rows),
            }
        )
    return curve


def _is_nondecreasing(values: list[float]) -> bool:
    return all(later + 1e-12 >= earlier for earlier, later in zip(values, values[1:], strict=False))


def _plot_curve(path: Path, curve: list[dict[str, Any]], y_key: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=180)
    ax.plot(
        [row["positive_limit"] for row in curve],
        [row[y_key] for row in curve],
        marker="o",
        linewidth=1.8,
    )
    ax.set_xlabel("Positive candidate pool size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GlaS Phase 2 Gate C Candidate Diagnostics",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        "",
        "## Core Metrics",
        "",
        f"- Validation queries: `{summary['query_count']}`",
        f"- Reference episodes: `{summary['episode_count']}`",
        f"- Max positive limit: `{summary['max_positive_limit']}`",
        f"- Mean point precision: `{summary['max_pool_metrics']['mean_point_precision']:.6f}`",
        f"- Mean random precision: `{summary['max_pool_metrics']['mean_random_precision']:.6f}`",
        f"- Precision delta: `{summary['max_pool_metrics']['mean_precision_delta']:.6f}`",
        f"- Precision ratio: `{summary['max_pool_metrics']['precision_ratio']:.6f}`",
        f"- Mean component coverage: `{summary['max_pool_metrics']['mean_component_coverage']:.6f}`",
        f"- Mean negative blank fraction: `{summary['max_pool_metrics']['mean_negative_blank_fraction']:.6f}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Issues", ""])
    if summary["issues"]:
        lines.extend(f"- {issue}" for issue in summary["issues"])
    else:
        lines.append("- none")
    lines.extend(["", "## Figures", ""])
    for item in summary["figures"].values():
        lines.append(f"- `{item['path']}` SHA256 `{item['sha256']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_gate_c(
    metrics_rows: list[dict[str, str]],
    *,
    expected_query_count: int,
    references_per_query: int,
) -> dict[str, Any]:
    if not metrics_rows:
        return {
            "status": "FAIL",
            "decision": "STOP_GLAS_CANDIDATE_FAILURE",
            "issues": ["empty candidate metrics"],
        }
    max_limit = max(_limits(metrics_rows))
    max_query_rows = _aggregate_queries(metrics_rows, positive_limit=max_limit)
    curve = _coverage_curve(metrics_rows)
    query_count = len(max_query_rows)
    episode_count = len({row["episode_id"] for row in metrics_rows})
    expected_episode_count = expected_query_count * references_per_query
    max_precision = mean(row["point_precision"] for row in max_query_rows)
    max_random = mean(row["random_precision"] for row in max_query_rows)
    precision_delta = max_precision - max_random
    precision_ratio = max_precision / max_random if max_random > 0 else float("inf")
    max_coverage = mean(row["component_coverage"] for row in max_query_rows)
    max_blank = mean(row["negative_blank_fraction"] for row in max_query_rows)
    validation_issue_count = sum(row["validation_issue_count"] for row in max_query_rows)
    checks = {
        "expected_query_count": query_count == expected_query_count,
        "expected_reference_episode_count": episode_count == expected_episode_count,
        "precision_enrichment": precision_delta >= MIN_PRECISION_DELTA or precision_ratio >= MIN_PRECISION_RATIO,
        "component_coverage": max_coverage >= MIN_COMPONENT_COVERAGE,
        "coverage_curve_nondecreasing": _is_nondecreasing([row["mean_component_coverage"] for row in curve]),
        "negative_pool_not_easy_blank": max_blank <= MAX_NEGATIVE_BLANK_FRACTION,
        "coordinate_validation": validation_issue_count == 0,
    }
    issues: list[str] = []
    if not checks["expected_query_count"]:
        issues.append(f"query_count {query_count} != expected {expected_query_count}")
    if not checks["expected_reference_episode_count"]:
        issues.append(f"episode_count {episode_count} != expected {expected_episode_count}")
    if not checks["precision_enrichment"]:
        issues.append(
            f"precision enrichment delta {precision_delta:.4f} / ratio {precision_ratio:.4f} below thresholds"
        )
    if not checks["component_coverage"]:
        issues.append(f"component coverage {max_coverage:.4f} < required {MIN_COMPONENT_COVERAGE:.4f}")
    if not checks["coverage_curve_nondecreasing"]:
        issues.append("coverage curve decreases with pool size")
    if not checks["negative_pool_not_easy_blank"]:
        issues.append(f"negative blank fraction {max_blank:.4f} > allowed {MAX_NEGATIVE_BLANK_FRACTION:.4f}")
    if not checks["coordinate_validation"]:
        issues.append(f"validation_issue_count {validation_issue_count} != 0")

    status = "PASS" if all(checks.values()) else "FAIL"
    decision = "CONTINUE_TO_GATE_D" if status == "PASS" else "REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED"
    return {
        "schema_version": "adapromptset.phase2.glas_gate_c.v1",
        "status": status,
        "decision": decision,
        "issues": issues,
        "query_count": query_count,
        "episode_count": episode_count,
        "expected_query_count": expected_query_count,
        "expected_episode_count": expected_episode_count,
        "references_per_query": references_per_query,
        "max_positive_limit": max_limit,
        "max_pool_metrics": {
            "mean_point_precision": max_precision,
            "mean_random_precision": max_random,
            "mean_precision_delta": precision_delta,
            "precision_ratio": precision_ratio,
            "mean_component_coverage": max_coverage,
            "mean_negative_blank_fraction": max_blank,
            "validation_issue_count": validation_issue_count,
        },
        "coverage_curve": curve,
        "query_level_rows": max_query_rows,
        "checks": checks,
        "thresholds": {
            "min_precision_delta": MIN_PRECISION_DELTA,
            "min_precision_ratio": MIN_PRECISION_RATIO,
            "min_component_coverage": MIN_COMPONENT_COVERAGE,
            "max_negative_blank_fraction": MAX_NEGATIVE_BLANK_FRACTION,
        },
        "prohibited_next_actions": [
            "glas_policy_training_before_gate_f",
            "formal_testA_or_testB_before_freeze",
            "posthoc_validation_oracle_action_addition",
        ],
    }


def write_gate_c_audit(
    project_root: Path = PROJECT_ROOT,
    metrics_path: Path = DEFAULT_METRICS,
    candidates_path: Path = DEFAULT_CANDIDATES,
    split_path: Path = DEFAULT_SPLIT,
    out_dir: Path = DEFAULT_OUT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_query_count = int(split["metadata"]["query_counts"]["val"])
    references_per_query = int(split["metadata"]["references_per_query"])
    metrics_rows = _read_csv(metrics_path)
    summary = analyze_gate_c(
        metrics_rows,
        expected_query_count=expected_query_count,
        references_per_query=references_per_query,
    )
    query_rows_path = out_dir / "query_level_candidate_metrics.csv"
    curve_path = out_dir / "candidate_pool_curve.csv"
    json_path = out_dir / "gate_c_summary.json"
    md_path = out_dir / "gate_c_summary.md"
    cache_manifest_path = cache_dir / "candidate_cache_manifest.json"
    precision_fig = figure_dir / "precision_vs_pool.png"
    coverage_fig = figure_dir / "coverage_vs_pool.png"
    negative_fig = figure_dir / "negative_blank_vs_pool.png"

    _write_csv(query_rows_path, summary.get("query_level_rows", []))
    _write_csv(curve_path, summary.get("coverage_curve", []))
    _plot_curve(precision_fig, summary["coverage_curve"], "mean_point_precision", "Mean point precision", "GlaS Gate C precision")
    _plot_curve(coverage_fig, summary["coverage_curve"], "mean_component_coverage", "Mean component coverage", "GlaS Gate C coverage")
    _plot_curve(
        negative_fig,
        summary["coverage_curve"],
        "mean_negative_blank_fraction",
        "Mean negative blank fraction",
        "GlaS Gate C negative candidates",
    )
    cache_manifest = {
        "schema_version": "adapromptset.phase2.glas_candidate_cache.v1",
        "status": "RECORDED_DIAGNOSTIC_CACHE",
        "source_candidates_csv": _relative(project_root, candidates_path),
        "source_candidates_sha256": _sha256(candidates_path),
        "source_metrics_csv": _relative(project_root, metrics_path),
        "source_metrics_sha256": _sha256(metrics_path),
        "notes": [
            "This cache records candidate coordinates and metrics generated under the Phase 2 GlaS protocol.",
            "It is not a policy-training artifact by itself.",
        ],
    }
    _write_json(cache_manifest_path, cache_manifest)
    summary["artifacts"] = {
        "metrics_csv": {"path": _relative(project_root, metrics_path), "sha256": _sha256(metrics_path)},
        "candidates_csv": {"path": _relative(project_root, candidates_path), "sha256": _sha256(candidates_path)},
        "query_level_metrics_csv": {"path": _relative(project_root, query_rows_path), "sha256": _sha256(query_rows_path)},
        "candidate_pool_curve_csv": {"path": _relative(project_root, curve_path), "sha256": _sha256(curve_path)},
        "cache_manifest": {"path": _relative(project_root, cache_manifest_path), "sha256": _sha256(cache_manifest_path)},
    }
    summary["figures"] = {
        "precision_vs_pool": {
            "path": _relative(project_root, precision_fig),
            "sha256": _sha256(precision_fig),
            "bytes": precision_fig.stat().st_size,
        },
        "coverage_vs_pool": {
            "path": _relative(project_root, coverage_fig),
            "sha256": _sha256(coverage_fig),
            "bytes": coverage_fig.stat().st_size,
        },
        "negative_blank_vs_pool": {
            "path": _relative(project_root, negative_fig),
            "sha256": _sha256(negative_fig),
            "bytes": negative_fig.stat().st_size,
        },
    }
    summary.pop("query_level_rows", None)
    _write_json(json_path, summary)
    _write_markdown(md_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 2 GlaS Gate C candidate diagnostics.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--candidates-path", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = write_gate_c_audit(
        project_root=args.project_root,
        metrics_path=args.metrics_path or args.project_root / "reports" / "phase2_glas" / "candidates" / "candidate_pilot_metrics.csv",
        candidates_path=args.candidates_path or args.project_root / "reports" / "phase2_glas" / "candidates" / "candidate_pilot_candidates.csv",
        split_path=args.split_path or args.project_root / "reports" / "phase2_glas" / "split" / "split_report.json",
        out_dir=args.out_dir or args.project_root / "reports" / "phase2_glas" / "candidates",
        figure_dir=args.figure_dir or args.project_root / "paper_phase2" / "figures" / "candidate_diagnostics",
        cache_dir=args.cache_dir or args.project_root / "cache" / "glas_candidates",
    )
    raise SystemExit(0 if summary["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
