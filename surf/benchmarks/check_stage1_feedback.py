from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "apply_stage1_feedback.py"


def main() -> None:
    if not SCRIPT.exists():
        raise AssertionError(f"Missing feedback script: {SCRIPT}")

    source = SCRIPT.read_text(encoding="utf-8")
    required_tokens = [
        "--stage1-dir",
        "--positive-points",
        "--negative-points",
        "--sigma",
        "--positive-weight",
        "--negative-weight",
        "--top-k",
        "--threshold",
        "feedback_points.csv",
        "feedback_points.json",
        "feedback_overlay_before_after.png",
        "feedback_heatmap_before_after.png",
        "summary.md",
        "load_stage1",
        "apply_feedback",
        "save_feedback_outputs",
    ]
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise AssertionError(f"Feedback script is missing required tokens: {missing}")

    compile(source, str(SCRIPT), "exec")
    print("Stage 1 feedback static checks passed.")


if __name__ == "__main__":
    main()
