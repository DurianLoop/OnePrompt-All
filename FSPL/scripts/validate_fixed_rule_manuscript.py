from __future__ import annotations

import argparse
import json
from pathlib import Path


ROW_TO_REPLICATE = {
    "DINOv2-L + SAM-B / 20260716": "dinov2_l_sam_b_seed20260716",
    "DINOv2-L + SAM-B / 20260717": "dinov2_l_sam_b_seed20260717",
    "DINOv2-B + SAM-B / 20260716": "dinov2_b_sam_b_seed20260716",
    "DINOv2-L + SAM-L / 20260716": "dinov2_l_sam_l_seed20260716",
    "DINOv2-L + SAM-H / 20260716": "dinov2_l_sam_h_seed20260716",
}


def _table_rows(manuscript_text: str) -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = {}
    for line in manuscript_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
        if len(cells) != 4 or cells[0] not in ROW_TO_REPLICATE:
            continue
        rows.setdefault(cells[0], []).append(cells)
    return rows


def validate_replication_table(
    manuscript_text: str,
    replicates: dict[str, dict[str, object]],
) -> list[str]:
    """Return precise errors when the manuscript replication table drifts from JSON."""
    rows = _table_rows(manuscript_text)
    errors: list[str] = []
    for label, replicate_key in ROW_TO_REPLICATE.items():
        matching_rows = rows.get(label, [])
        if not matching_rows:
            errors.append(f"{label}: missing replication row")
            continue
        if len(matching_rows) != 1:
            errors.append(f"{label}: found {len(matching_rows)} replication rows")
            continue
        if replicate_key not in replicates:
            errors.append(f"{label}: missing artifact replicate {replicate_key}")
            continue

        _, top5, spread, gain = matching_rows[0]
        artifact = replicates[replicate_key]
        expected = {
            "top5": f"{float(artifact['top5_n3_mean_dice']):.4f}",
            "spread": f"{float(artifact['spread5_n3_mean_dice']):.4f}",
            "gain": f"{float(artifact['mean_delta']):+.4f}",
        }
        actual = {"top5": top5, "spread": spread, "gain": gain}
        for field in ("top5", "spread", "gain"):
            if actual[field] != expected[field]:
                errors.append(
                    f"{label}: {field} expected {expected[field]}, got {actual[field]}"
                )
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fixed-layout manuscript rows against artifact JSON."
    )
    parser.add_argument("--manuscript-path", type=Path, required=True)
    parser.add_argument(
        "--effects-path",
        type=Path,
        default=Path("artifacts/fixed_layout_effects.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manuscript_text = args.manuscript_path.read_text(encoding="utf-8")
    effects = json.loads(args.effects_path.read_text(encoding="utf-8"))
    errors = validate_replication_table(manuscript_text, effects["replicates"])
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(ROW_TO_REPLICATE)} fixed-layout replication rows.")


if __name__ == "__main__":
    main()
