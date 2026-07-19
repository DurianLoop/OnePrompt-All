from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = Path("manifests/phase4plus")
REPORT_DIR = Path("reports/route_b_ddr_episodes")
DEFAULT_REFERENCES_PER_QUERY = 3
DEFAULT_SEED = "route-b-ddr-episode-v1"
ALLOWED_SPLITS = ("train", "valid")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _positive_manifest_path(project_root: Path, split: str) -> Path:
    return project_root / MANIFEST_DIR / f"ddr_route_b_{split}_positive_queries.csv"


def _select_references(
    *,
    query: dict[str, str],
    candidates: list[dict[str, str]],
    references_per_query: int,
    seed: str,
) -> list[dict[str, str]]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["intent"] == query["intent"] and candidate["image_id"] != query["image_id"]
    ]
    ranked = sorted(
        eligible,
        key=lambda candidate: _sha256_text(f"{seed}:{query['query_unit_id']}:{candidate['query_unit_id']}"),
    )
    return ranked[:references_per_query]


def build_route_b_ddr_episodes(
    *,
    project_root: Path = PROJECT_ROOT,
    references_per_query: int = DEFAULT_REFERENCES_PER_QUERY,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    if references_per_query < 1:
        raise ValueError("references_per_query must be >= 1")

    train_rows = _read_csv(_positive_manifest_path(project_root, "train"))
    valid_rows = _read_csv(_positive_manifest_path(project_root, "valid"))
    split_to_queries = {"train": train_rows, "valid": valid_rows}
    reference_pools = {
        "train": train_rows,
        "valid": train_rows,
    }
    output_dir = project_root / MANIFEST_DIR
    report_dir = project_root / REPORT_DIR
    fieldnames = [
        "dataset",
        "split",
        "intent",
        "query_unit_id",
        "query_image_id",
        "query_image_path",
        "query_mask_path",
        "query_mask_sha256",
        "reference_rank",
        "reference_source_split",
        "reference_unit_id",
        "reference_image_id",
        "reference_image_path",
        "reference_mask_path",
        "reference_mask_sha256",
        "selection_seed",
        "support_pool_rule",
        "formal_test_locked",
    ]

    outputs: dict[str, str] = {}
    split_summaries: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    all_episode_rows: list[dict[str, Any]] = []

    for split in ALLOWED_SPLITS:
        episode_rows: list[dict[str, Any]] = []
        for query in split_to_queries[split]:
            references = _select_references(
                query=query,
                candidates=reference_pools[split],
                references_per_query=references_per_query,
                seed=seed,
            )
            if len(references) < references_per_query:
                issues.append(
                    {
                        "split": split,
                        "query_unit_id": query["query_unit_id"],
                        "intent": query["intent"],
                        "issue": "insufficient_same_intent_references",
                    }
                )
            for rank, reference in enumerate(references, start=1):
                if split == "train" and reference["image_id"] == query["image_id"]:
                    issues.append(
                        {
                            "split": split,
                            "query_unit_id": query["query_unit_id"],
                            "intent": query["intent"],
                            "issue": "train_reference_matches_query_image",
                        }
                    )
                if split == "valid" and reference["split"] != "train":
                    issues.append(
                        {
                            "split": split,
                            "query_unit_id": query["query_unit_id"],
                            "intent": query["intent"],
                            "issue": "valid_reference_not_from_train",
                        }
                    )
                episode_rows.append(
                    {
                        "dataset": "DDR",
                        "split": split,
                        "intent": query["intent"],
                        "query_unit_id": query["query_unit_id"],
                        "query_image_id": query["image_id"],
                        "query_image_path": query["image_path"],
                        "query_mask_path": query["mask_path"],
                        "query_mask_sha256": query["mask_sha256"],
                        "reference_rank": rank,
                        "reference_source_split": reference["split"],
                        "reference_unit_id": reference["query_unit_id"],
                        "reference_image_id": reference["image_id"],
                        "reference_image_path": reference["image_path"],
                        "reference_mask_path": reference["mask_path"],
                        "reference_mask_sha256": reference["mask_sha256"],
                        "selection_seed": seed,
                        "support_pool_rule": "train_excluding_query" if split == "train" else "train_only",
                        "formal_test_locked": True,
                    }
                )
        out_path = output_dir / f"ddr_route_b_{split}_reference_query_episodes.csv"
        _write_csv(out_path, episode_rows, fieldnames)
        outputs[f"{split}_episodes"] = str(out_path)
        all_episode_rows.extend(episode_rows)
        queries = {row["query_unit_id"] for row in episode_rows}
        intent_counts = Counter(row["intent"] for row in episode_rows)
        split_summaries[split] = {
            "positive_query_count": len(split_to_queries[split]),
            "episode_rows": len(episode_rows),
            "covered_query_count": len(queries),
            "references_per_query": references_per_query,
            "episode_rows_by_intent": dict(sorted(intent_counts.items())),
        }

    status = "PASS_DDR_ROUTE_B_EPISODES" if not issues else "FAIL_DDR_ROUTE_B_EPISODES"
    output_hashes = {name: _sha256_file(Path(path)) for name, path in outputs.items()}
    summary = {
        "schema_version": "adapromptset.route_b.ddr_episodes.v1",
        "date": "2026-07-19",
        "status": status,
        "route": "B_reference_promptability_benchmark",
        "dataset": "DDR",
        "references_per_query": references_per_query,
        "selection_seed": seed,
        "generated_splits": list(ALLOWED_SPLITS),
        "formal_test_access": False,
        "test_files_read": False,
        "test_as_reference_source": False,
        "valid_reference_source_rule": "train_only",
        "train_reference_source_rule": "train_excluding_query",
        "split_summaries": split_summaries,
        "total_episode_rows": len(all_episode_rows),
        "issues": issues,
        "outputs": outputs,
        "output_sha256": output_hashes,
        "statistical_unit_note": "Query image/case is the independent unit; reference rows and image-intent rows are repeated task views.",
        "authorizes": "development_input_manifests_only",
        "does_not_authorize": [
            "formal_test_access",
            "submission_ready_claim",
            "method_positive_claim",
        ],
    }
    _write_json(report_dir / "ddr_route_b_episode_report.json", summary)
    _write_markdown(report_dir / "ddr_route_b_episode_report.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Route B DDR Episode Report",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"References per query: `{summary['references_per_query']}`",
        f"Formal test access: `{str(summary['formal_test_access']).lower()}`",
        f"DDR test files read: `{str(summary['test_files_read']).lower()}`",
        "",
        "## Split Summary",
        "",
        "| Split | Positive queries | Covered queries | Episode rows | References/query |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in summary["generated_splits"]:
        item = summary["split_summaries"][split]
        lines.append(
            f"| {split} | {item['positive_query_count']} | {item['covered_query_count']} | "
            f"{item['episode_rows']} | {item['references_per_query']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Valid references come only from DDR train positive-query rows.",
            "- Train references exclude the query image.",
            "- DDR test remains locked and unused.",
            "- This report creates development input manifests only.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, output in summary["outputs"].items():
        lines.append(f"- `{name}`: `{output}`")
    if summary["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in summary["issues"]:
            lines.append(f"- `{issue}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic DDR reference/query episodes for Route B.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--references-per-query", type=int, default=DEFAULT_REFERENCES_PER_QUERY)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    build_route_b_ddr_episodes(
        project_root=args.project_root,
        references_per_query=args.references_per_query,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
