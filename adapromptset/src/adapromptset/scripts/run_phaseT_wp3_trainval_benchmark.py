"""Phase T WP3 Train/Val Benchmark (reuses the WP1 holdout pipeline).

Runs the WP1 reference-instability pipeline (phases 1-3 only) on the
MoNuSAC *train* (80 queries) and *validation* (41 queries) splits to
generate episode data for WP3 (M1 dynamic budget + M2 selective
prediction) training. This is **not** a confirmatory run -- phase 4
(the WP1 gate evaluation) is intentionally skipped.

The script imports :mod:`run_phaseT_wp1_holdout` and reuses its
``run_phase1`` / ``run_phase2`` / ``run_phase3`` functions by overriding
the WP1 module-level constants (``HOLDOUT_QUERY_MANIFEST``,
``PHASET_MANIFEST_DIR``, ``SEED_COUNT``) before each invocation. The WP1
constants are looked up at call time, so overriding them on the module
object is sufficient to redirect inputs/outputs to split-specific paths.

Pipeline (per split)::

    Phase 1 (CPU)  -- reference resampling.
                      Manifests -> manifests/phaseT/wp3_{split}/
                      holdout_resampling_seed_{i}.csv
    Phase 2 (GPU)  -- DINOv2-L matching + SAM-B segmentation.
                      Checkpoints -> reports/phaseT_wp3_{split}/
                      phase2_raw_per_call.csv,
                      phase2_episode_instability.csv
    Phase 3 (CPU)  -- metric computation (8 WP1 metric CSVs).

CLI::

    python src/adapromptset/scripts/run_phaseT_wp3_trainval_benchmark.py \\
        --split {train,validation,both} --phase {1,2,3,all} \\
        --smoke --device {cpu,cuda,auto} --seeds 5

``--smoke`` restricts to 2 queries x 2 seeds for fast verification.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# OpenMP runtime conflict workaround (conda + torch on Windows).
# MUST be set before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# adapromptset root (D:\\SAM\\adapromptset).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Make the scripts directory importable so we can ``import run_phaseT_wp1_holdout``.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402

import run_phaseT_wp1_holdout as wp1  # noqa: E402


# ---------------------------------------------------------------------------
# Split -> query manifest mapping
# ---------------------------------------------------------------------------

SPLIT_MANIFESTS: dict[str, Path] = {
    "train": PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv",
    "validation": PROJECT_ROOT / "manifests" / "phaseY" / "policy_validation_queries.csv",
}

REPORTS_ROOT = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _count_query_rows(manifest_path: Path) -> int:
    """Count data rows (excluding header) in a query manifest CSV."""
    if not manifest_path.is_file():
        return 0
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def _list_output_files(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    return sorted(p for p in out_dir.iterdir() if p.is_file())


# ---------------------------------------------------------------------------
# Per-split pipeline
# ---------------------------------------------------------------------------

def run_split(
    split: str,
    *,
    phase: str,
    smoke: bool,
    device: torch.device,
    seeds_n: int,
) -> dict:
    """Run phases 1-3 of the WP1 pipeline for a single split.

    Overrides the WP1 module-level constants so the existing functions
    read from the split-specific query manifest and write split-specific
    resampling manifests.
    """
    manifest_path = SPLIT_MANIFESTS[split]
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Query manifest not found for split '{split}': {manifest_path}"
        )

    manifest_rows = _count_query_rows(manifest_path)

    if smoke:
        max_queries = 2
        seeds = range(2)
    else:
        max_queries = None
        seeds = range(seeds_n)

    # ---- Override WP1 module constants (looked up at call time) ------------
    # (a) query manifest -> split's manifest
    wp1.HOLDOUT_QUERY_MANIFEST = manifest_path
    # (b) resampling manifest dir -> split-specific to avoid collisions
    wp1.PHASET_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "phaseT" / f"wp3_{split}"
    # (c) seed count -> requested seed budget (2 in smoke mode)
    wp1.SEED_COUNT = len(seeds)
    # NOTE: REFERENCE_POOL_MANIFEST is left as the WP1 default
    # (manifests/phaseY/reference_pool.csv), which is the correct shared pool.

    out_dir = REPORTS_ROOT / f"phaseT_wp3_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wp1.PHASET_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[phaseT-WP3] SPLIT = {split}")
    print(f"[phaseT-WP3]   query manifest     : {manifest_path}")
    print(f"[phaseT-WP3]   manifest rows      : {manifest_rows}")
    print(f"[phaseT-WP3]   reference pool     : {wp1.REFERENCE_POOL_MANIFEST}")
    print(f"[phaseT-WP3]   resampling dir     : {wp1.PHASET_MANIFEST_DIR}")
    print(f"[phaseT-WP3]   output dir         : {out_dir}")
    print(f"[phaseT-WP3]   device             : {device}")
    print(f"[phaseT-WP3]   seeds              : {list(seeds)} "
          f"(SEED_COUNT={wp1.SEED_COUNT})")
    print(f"[phaseT-WP3]   max_queries        : {max_queries}")
    print(f"[phaseT-WP3]   smoke              : {smoke}")
    print(f"[phaseT-WP3]   phase              : {phase}")
    print("=" * 78)

    summary: dict = {
        "split": split,
        "manifest": str(manifest_path),
        "manifest_rows": manifest_rows,
        "out_dir": str(out_dir),
        "resampling_dir": str(wp1.PHASET_MANIFEST_DIR),
        "seeds": list(seeds),
        "max_queries": max_queries,
        "query_count": 0,
        "phase1": None,
        "phase2": None,
        "phase3": None,
    }

    # ---- Phase 1: reference resampling -------------------------------------
    if phase in ("1", "all"):
        print()
        print(f"[phaseT-WP3/{split}] Phase 1: reference resampling "
              f"({len(seeds)} seeds)")
        p1 = wp1.run_phase1(seeds=seeds, max_queries=max_queries)
        summary["phase1"] = p1
        summary["query_count"] = p1["query_count"]

    # ---- Phase 2: DINOv2-L + SAM-B segmentation ----------------------------
    if phase in ("2", "all"):
        print()
        print(f"[phaseT-WP3/{split}] Phase 2: DINOv2-L matching + "
              f"SAM-B segmentation")
        feature_cache = wp1.FeatureCache(device=device)
        predictor = wp1.load_sam_predictor(device)
        p2 = wp1.run_phase2(
            feature_cache=feature_cache,
            predictor=predictor,
            out_dir=out_dir,
            seeds=seeds,
            max_queries=max_queries,
        )
        summary["phase2"] = p2
        if summary["query_count"] == 0:
            summary["query_count"] = p2.get("query_count", 0)
        print(f"[phaseT-WP3/{split}] phase2 done: "
              f"{p2.get('total_episodes', 0)} episodes, "
              f"{p2.get('total_sam_calls', 0)} SAM calls, "
              f"{p2.get('elapsed_seconds', 0.0)}s")

    # ---- Phase 3: metric computation ---------------------------------------
    if phase in ("3", "all"):
        print()
        print(f"[phaseT-WP3/{split}] Phase 3: metric computation")
        p3 = wp1.run_phase3(out_dir)
        summary["phase3"] = {
            "raw_rows": len(p3.get("raw_rows", [])),
            "episode_rows": len(p3.get("episode_rows", [])),
        }
        if summary["query_count"] == 0:
            qs = {r["query_case_id"] for r in p3.get("raw_rows", [])}
            summary["query_count"] = len(qs)

    print()
    print(f"[phaseT-WP3/{split}] artifacts written to {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase T WP3 train/val benchmark: run the WP1 reference-instability "
            "pipeline (phases 1-3) on MoNuSAC train (80) and validation (41) "
            "queries to generate WP3 training episode data."
        )
    )
    parser.add_argument(
        "--split", default="both",
        choices=["train", "validation", "both"],
        help="Which split(s) to run.",
    )
    parser.add_argument(
        "--phase", default="all",
        choices=["1", "2", "3", "all"],
        help="Which phases to run. Phase 4 (gate) is NOT run by this script.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick verification: 2 queries x 2 seeds.",
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of resampling seeds (default 5).",
    )
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    device = _resolve_device(args.device)

    splits = ["train", "validation"] if args.split == "both" else [args.split]

    print("[phaseT-WP3] Phase T WP3 train/val benchmark (WP1 pipeline reuse)")
    print(f"[phaseT-WP3] splits={splits} phase={args.phase} device={device} "
          f"seeds={args.seeds} smoke={args.smoke}")
    print(f"[phaseT-WP3] project root: {PROJECT_ROOT}")

    all_summaries: list[dict] = []
    for split in splits:
        summary = run_split(
            split,
            phase=args.phase,
            smoke=args.smoke,
            device=device,
            seeds_n=args.seeds,
        )
        all_summaries.append(summary)

    # ---- Final summary ------------------------------------------------------
    print()
    print("#" * 78)
    print("[phaseT-WP3] FINAL SUMMARY")
    print("#" * 78)
    for s in all_summaries:
        print()
        print(f"  split            : {s['split']}")
        print(f"  manifest rows    : {s['manifest_rows']}")
        print(f"  query count run  : {s['query_count']}")
        print(f"  seeds            : {s['seeds']}")
        print(f"  max_queries      : {s['max_queries']}")
        print(f"  output dir       : {s['out_dir']}")
        print(f"  resampling dir   : {s['resampling_dir']}")
        out_dir = Path(s["out_dir"])
        files = _list_output_files(out_dir)
        print(f"  output files ({len(files)}):")
        for f in files:
            size = f.stat().st_size if f.is_file() else 0
            print(f"    - {f.name}  ({size} bytes)")
    print()
    print("[phaseT-WP3] done.")


if __name__ == "__main__":
    main()
