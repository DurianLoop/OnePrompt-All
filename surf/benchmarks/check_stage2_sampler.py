from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "sample_stage2_prompts.py"


REQUIRED_TOKENS = [
    "--feedback-dir",
    "--top-ratio",
    "--cluster-method",
    "--eps",
    "--min-samples",
    "--sample-sizes",
    "--samples-per-cluster",
    "--max-anchor-distance",
    "prompt_groups.csv",
    "prompt_groups.json",
    "cluster_overlay.png",
    "sample_overlay.png",
    "summary.md",
    "load_feedback_points",
    "select_candidate_points",
    "cluster_points",
    "filter_cluster_points_by_anchor",
    "sample_prompt_groups",
    "save_stage2_outputs",
]


def main() -> None:
    if not SCRIPT.exists():
        raise SystemExit(f"Missing expected sampler script: {SCRIPT}")
    text = SCRIPT.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    if missing:
        raise SystemExit("Missing required Stage 2 sampler tokens: " + ", ".join(missing))
    print("Stage 2 sampler static checks passed.")


if __name__ == "__main__":
    main()
