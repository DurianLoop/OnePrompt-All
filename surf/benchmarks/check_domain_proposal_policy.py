from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "research_direction" / "feedback_learned_matcher_mainline" / "domain_policy_transfer"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary_path = OUT_DIR / "domain_policy_transfer_summary.csv"
    predictions_path = OUT_DIR / "domain_policy_predictions.csv"
    results_path = OUT_DIR / "domain_policy_transfer_results.json"
    assert_true(summary_path.exists(), f"Missing {summary_path}")
    assert_true(predictions_path.exists(), f"Missing {predictions_path}")
    assert_true(results_path.exists(), f"Missing {results_path}")

    summary = read_csv(summary_path)
    predictions = read_csv(predictions_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    assert_true(len(predictions) == 600, f"Expected 600 target predictions, got {len(predictions)}")
    domains = {row["domain"] for row in summary}
    assert_true({"IDRID", "MonuSeg"}.issubset(domains), f"Missing domains: {domains}")
    for domain in ("IDRID", "MonuSeg"):
        domain_rows = [row for row in summary if row["domain"] == domain]
        splits = {row["split"] for row in domain_rows}
        assert_true("target_50_heldout" in splits, f"{domain} missing heldout split")
        heldout = next(row for row in domain_rows if row["split"] == "target_50_heldout")
        assert_true(int(heldout["apply_cases"]) > 0, f"{domain} heldout split has no cases")
        assert_true(float(heldout["scalar_f1"]) >= 0.0, f"{domain} invalid scalar F1")
        assert_true(heldout["scalar_feature"] != "", f"{domain} missing learned feature")
        assert_true(domain in payload["domains"], f"{domain} missing model payload")

    print("Domain proposal policy checks passed.")


if __name__ == "__main__":
    main()
