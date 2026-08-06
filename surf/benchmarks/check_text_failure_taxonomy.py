from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "build_sam3_text_failure_taxonomy.py"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = [
        "taxonomy.csv",
        "taxonomy.json",
        "fallback_required",
        "prompt_type",
        "instances == 0",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Missing required tokens in {SCRIPT}: {missing}")
    print("Text failure taxonomy static checks passed.")


if __name__ == "__main__":
    main()
