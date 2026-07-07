from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "medical_small_data_gate.py"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = [
        "discover_medical_files",
        "medical_gate_report.json",
        "mask_candidates",
        "can_run_dice_iou",
        "modality_summary",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Missing required tokens in {SCRIPT}: {missing}")
    print("Medical small-data gate static checks passed.")


if __name__ == "__main__":
    main()
