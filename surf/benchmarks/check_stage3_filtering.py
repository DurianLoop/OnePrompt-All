from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "filter_stage3_proposals.py"


REQUIRED_TOKENS = [
    "--proposals-dir",
    "--min-score",
    "--min-area-pct",
    "--max-area-pct",
    "--max-bbox-area-pct",
    "--min-cluster-coverage",
    "--nms-iou",
    "filtered_scores.csv",
    "filtered_scores.json",
    "retained_overlay.png",
    "retained_grid.png",
    "load_proposals",
    "load_cluster_points",
    "compute_proposal_metrics",
    "filter_candidates",
    "mask_iou",
    "run_nms",
    "save_filter_outputs",
]


def main() -> None:
    if not SCRIPT.exists():
        raise SystemExit(f"Missing expected Stage 3 filtering script: {SCRIPT}")
    text = SCRIPT.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    if missing:
        raise SystemExit("Missing required Stage 3 filtering tokens: " + ", ".join(missing))
    print("Stage 3 filtering static checks passed.")


if __name__ == "__main__":
    main()
