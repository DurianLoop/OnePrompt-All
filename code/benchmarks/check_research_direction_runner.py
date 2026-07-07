from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "run_research_direction_case.py"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = [
        "stage1",
        "feature_feedback",
        "stage2_sampling",
        "stage2_proposals",
        "stage3_feature",
        "learning_policy",
        "--labels-json",
        "run_matcher_feedback_learning_policy.py",
        "subprocess.run",
        "pipeline_summary.json",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Missing required tokens in {SCRIPT}: {missing}")
    print("Research direction runner static checks passed.")


if __name__ == "__main__":
    main()
