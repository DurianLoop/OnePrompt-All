from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "feature_match_stage3_proposals.py"


REQUIRED_TOKENS = [
    "--proposals-dir",
    "--stage1-dir",
    "--dinov2-weights",
    "--min-instance-similarity",
    "--nms-iou",
    "--max-retained-per-cluster",
    "feature_scores.csv",
    "feature_scores.json",
    "feature_retained_overlay.png",
    "feature_retained_grid.png",
    "load_stage1_metadata",
    "compute_reference_prototype",
    "compute_proposal_prototype",
    "compute_instance_similarity",
    "filter_feature_candidates",
    "run_feature_nms",
    "save_feature_outputs",
]


def main() -> None:
    if not SCRIPT.exists():
        raise SystemExit(f"Missing expected Stage 3 feature matching script: {SCRIPT}")
    text = SCRIPT.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    if missing:
        raise SystemExit("Missing required Stage 3 feature matching tokens: " + ", ".join(missing))
    print("Stage 3 feature matching static checks passed.")


if __name__ == "__main__":
    main()
