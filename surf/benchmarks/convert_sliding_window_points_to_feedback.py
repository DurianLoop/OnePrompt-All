from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sliding-window Stage 1 merged points to the feedback_points format consumed by Stage 2 sampler."
    )
    parser.add_argument("--merged-points-json", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_path = resolve_path(args.merged_points_json)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(merged_path.read_text(encoding="utf-8"))
    converted_points: list[dict[str, Any]] = []
    for row in payload["points"]:
        similarity = float(row["similarity"])
        converted_points.append(
            {
                "rank": int(row["rank"]),
                "x_original": float(row["x_original"]),
                "y_original": float(row["y_original"]),
                "similarity_before": similarity,
                "similarity_after": similarity,
                "delta": 0.0,
                "selected_by_threshold": bool(row["selected_by_threshold"]),
                "tile_id": row.get("tile_id"),
            }
        )

    feedback_payload = {
        "source": str(merged_path.relative_to(WORKSPACE_ROOT)),
        "reference_image": payload["reference_image"],
        "target_image": payload["target_image"],
        "reference_box": payload["reference_box"],
        "threshold": payload["threshold"],
        "points": converted_points,
        "note": "Converted from sliding-window Stage 1 points; similarity_before == similarity_after because no feedback reranking is applied here.",
    }
    (out_dir / "feedback_points.json").write_text(json.dumps(feedback_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "feedback_points.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(converted_points[0].keys()) if converted_points else [
            "rank",
            "x_original",
            "y_original",
            "similarity_before",
            "similarity_after",
            "delta",
            "selected_by_threshold",
            "tile_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted_points)
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Sliding-Window Points Feedback Adapter",
                "",
                f"- Source: `{merged_path.relative_to(WORKSPACE_ROOT)}`",
                f"- Target image: `{payload['target_image']}`",
                f"- Points: `{len(converted_points)}`",
                f"- Selected points: `{sum(1 for row in converted_points if row['selected_by_threshold'])}`",
                "",
                "## Artifacts",
                "",
                "- `feedback_points.json`",
                "- `feedback_points.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Converted {len(converted_points)} points")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
