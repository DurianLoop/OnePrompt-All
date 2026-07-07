from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / ".cache" / "teacher_image_batch_check"


def main() -> None:
    script = ROOT / "code" / "benchmarks" / "run_teacher_image_batch.py"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--image-dir",
            str(ROOT / "图片"),
            "--out-dir",
            str(OUT_DIR),
            "--max-images",
            "30",
            "--skip-hog",
        ],
        cwd=ROOT,
        check=True,
    )
    for name in [
        "image_batch_scores.csv",
        "summary.md",
        "top_point_feedback_candidates.jpg",
        "top_text_failure_candidates.jpg",
        "top_matcher_pairs.jpg",
    ]:
        assert (OUT_DIR / name).exists(), name
    text = (OUT_DIR / "summary.md").read_text(encoding="utf-8")
    assert "老师要求" in text
    assert "point" in text
    assert "text" in text
    print(f"[ok] teacher image batch smoke check: {OUT_DIR}")


if __name__ == "__main__":
    main()
