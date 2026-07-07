from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "export_sliding_window_stage1_points.py"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = [
        "iter_tiles",
        "restore_tile_coordinates",
        "tile_x0",
        "tile_y0",
        "merged_points.csv",
        "sliding_window_overlay.png",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Missing required tokens in {SCRIPT}: {missing}")
    print("Sliding-window Stage 1 static checks passed.")


if __name__ == "__main__":
    main()
