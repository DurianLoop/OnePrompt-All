from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "generate_stage2_sam_proposals.py"


REQUIRED_TOKENS = [
    "--stage2-dir",
    "--checkpoint",
    "--model-type",
    "--device",
    "--max-groups",
    "proposal_scores.csv",
    "proposal_scores.json",
    "proposal_grid.png",
    "proposal_masks",
    "proposal_overlays",
    "load_prompt_groups",
    "load_sam_predictor",
    "predict_prompt_group",
    "save_proposal_outputs",
]


def main() -> None:
    if not SCRIPT.exists():
        raise SystemExit(f"Missing expected proposal script: {SCRIPT}")
    text = SCRIPT.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    if missing:
        raise SystemExit("Missing required Stage 2 proposal tokens: " + ", ".join(missing))
    print("Stage 2 proposal static checks passed.")


if __name__ == "__main__":
    main()
