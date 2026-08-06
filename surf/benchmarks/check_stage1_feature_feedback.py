from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "apply_stage1_feature_feedback.py"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = [
        "--positive-points",
        "--negative-points",
        "compute_feedback_prototype",
        "updated_proto",
        "feedback_points.json",
        "similarity_after.npy",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Missing required tokens in {SCRIPT}: {missing}")
    print("Stage 1 feature feedback static checks passed.")


if __name__ == "__main__":
    main()
