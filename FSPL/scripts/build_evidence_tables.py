from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready tables from completed FSPL experiment summaries only.")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args(argv)


def _metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: float(values["mean_dice"])
        for name, values in payload["official_test"]["fixed_rules"].items()
    }


def _row(name: str, value: float | None) -> str:
    return f"| {name} | {value:.4f} |" if value is not None else f"| {name} | pending |"


def _fixed_layout_row(name: str, metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return f"| {name} | pending | pending | pending |"
    gain = metrics["spread5_n3"] - metrics["top5_n3"]
    return f"| {name} | {metrics['top5_n3']:.4f} | {metrics['spread5_n3']:.4f} | {gain:+.4f} |"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stage9 = _metrics(Path("artifacts/stage9_monusac_full_fixed_ablation/summary.json"))
    stage6 = _metrics(Path("artifacts/stage6_monusac_heldout/summary.json"))
    stage8 = _metrics(Path("artifacts/stage8_monusac_seed20260717/summary.json"))
    stage13 = _metrics(Path("artifacts/stage13_dinob_samb_monusac/summary.json"))
    stage16_path = Path("artifacts/stage16_dinol_saml_monusac/summary.json")
    stage16 = _metrics(stage16_path) if stage16_path.is_file() else None
    stage18_path = Path("artifacts/stage18_dinol_samh_monusac/summary.json")
    stage18 = _metrics(stage18_path) if stage18_path.is_file() else None

    lines = [
        "# Evidence Tables (Auto-generated)",
        "",
        "Only completed experiment summaries are rendered. `pending` means no result artifact exists.",
        "",
        "## MoNuSAC fixed-rule ablation (seed 20260716; 303 episodes/rule)",
        "",
        "| Rule | Mean Dice |",
        "|---|---:|",
        _row("Top-1 / 0 negative", stage9["top1_n0"]),
        _row("Top-3 / 0 negative", stage9["top3_n0"]),
        _row("Top-5 / 0 negative", stage9["top5_n0"]),
        _row("Top-5 / 3 negative", stage6["top5_n3"]),
        _row("Spread radius 3 / 3 negative", stage6["spread5_n3"]),
        "",
        "## Fixed spatial-layout replication", "", "| Combination / reference seed | Top-5 + 3 neg. | Spread r=3 + 3 neg. | Gain |", "|---|---:|---:|---:|",
    ]
    for label, metrics in (("DINOv2-L + SAM-B / 20260716", stage6), ("DINOv2-L + SAM-B / 20260717", stage8), ("DINOv2-B + SAM-B / 20260716", stage13)):
        lines.append(_fixed_layout_row(label, metrics))
    lines.append(_fixed_layout_row("DINOv2-L + SAM-L / 20260716", stage16))
    lines.append(_fixed_layout_row("DINOv2-L + SAM-H / 20260716", stage18))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_path)


if __name__ == "__main__":
    main()
