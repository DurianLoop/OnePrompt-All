from __future__ import annotations

import csv
import json
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = WORKSPACE_ROOT / "runs" / "research_direction" / "feedback_learned_matcher_mainline"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    labels_path = OUT_DIR / "manual_labels_20.json"
    assert_true(labels_path.exists(), f"Missing {labels_path}")
    labels = load_json(labels_path)
    assert_true(labels.get("proposal_count") == 240, f"Expected 240 labels, got {labels.get('proposal_count')}")
    assert_true(labels.get("case_count") == 20, f"Expected 20 cases, got {labels.get('case_count')}")
    assert_true(labels.get("positive_count", 0) > 0, "Need positive labels")
    assert_true(labels.get("negative_count", 0) > 0, "Need negative labels")

    for name, expected_rows in [("overall_20", 240), ("idrid_20", 144), ("monuseg_20", 96)]:
        policy_path = OUT_DIR / name / "medical_casebook_policy.json"
        rows_path = OUT_DIR / name / "medical_casebook_rows.csv"
        assert_true(policy_path.exists(), f"Missing {policy_path}")
        assert_true(rows_path.exists(), f"Missing {rows_path}")
        policy = load_json(policy_path)
        rows = read_csv(rows_path)
        assert_true(len(rows) == expected_rows, f"{name} expected {expected_rows} rows, got {len(rows)}")
        assert_true("best_policy" in policy, f"{name} missing best policy")
        assert_true("leave_one_case_out" in policy, f"{name} missing LOCO metrics")

    table_path = OUT_DIR / "paper_main_table.csv"
    assert_true(table_path.exists(), f"Missing {table_path}")
    table = read_csv(table_path)
    table_names = {row["setting"] for row in table}
    for setting in {"overall_20", "idrid_20", "monuseg_20", "overall_50"}:
        assert_true(setting in table_names, f"Missing table setting {setting}")

    manifest_50 = WORKSPACE_ROOT / "runs" / "research_direction" / "matcher_medical_expanded_50" / "medical_matcher_pilot_manifest.csv"
    rows_50 = WORKSPACE_ROOT / "runs" / "research_direction" / "matcher_medical_expanded_50" / "casebook" / "medical_casebook_rows.csv"
    assert_true(manifest_50.exists(), f"Missing {manifest_50}")
    assert_true(rows_50.exists(), f"Missing {rows_50}")
    assert_true(len(read_csv(manifest_50)) == 50, "Expected 50 medical cases")
    assert_true(len(read_csv(rows_50)) == 600, "Expected 600 proposals for 50 cases")

    print("Feedback-learned Matcher mainline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
