from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "code" / "benchmarks" / "export_matcher_stage1_points.py"


def main() -> None:
    if not EXPORTER.exists():
        raise AssertionError(f"Missing exporter script: {EXPORTER}")

    source = EXPORTER.read_text(encoding="utf-8")
    required_tokens = [
        "--reference-image",
        "--target-image",
        "--reference-box",
        "--reference-mask",
        "--dinov2-weights",
        "--top-k",
        "--threshold",
        "points.csv",
        "points.json",
        "similarity.npy",
        "topk_overlay.png",
        "similarity_heatmap.png",
        "summary.md",
        "load_dinov2",
        "compute_stage1_matches",
        "save_outputs",
    ]
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise AssertionError(f"Exporter is missing required tokens: {missing}")

    compile(source, str(EXPORTER), "exec")
    print("Stage 1 exporter static checks passed.")


if __name__ == "__main__":
    main()
