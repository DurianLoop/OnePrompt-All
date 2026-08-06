from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "adapter_lora_gate.py"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = [
        "adapter_gate_report.json",
        "ready_for_training",
        "medical data with masks",
        "training_not_recommended",
        "gate_criteria",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Missing required tokens in {SCRIPT}: {missing}")
    print("Adapter/LoRA gate static checks passed.")


if __name__ == "__main__":
    main()
