from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "research_direction" / "feedback_learned_matcher_mainline" / "case_split_50"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary_path = OUT_DIR / "case_split_50_summary.csv"
    folds_path = OUT_DIR / "case_split_50_folds.csv"
    predictions_path = OUT_DIR / "case_split_50_predictions.csv"
    for path in (summary_path, folds_path, predictions_path, OUT_DIR / "case_split_50_models.json", OUT_DIR / "summary.md"):
        assert_true(path.exists(), f"Missing {path}")

    summary = read_csv(summary_path)
    folds = read_csv(folds_path)
    predictions = read_csv(predictions_path)
    assert_true({row["setting"] for row in summary} == {"overall_5fold", "domain_5fold"}, "Unexpected summary settings")

    by_setting: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        by_setting[row["setting"]].append(row)
    for setting, rows in by_setting.items():
        assert_true(len(rows) == 600, f"{setting} expected 600 prediction rows, got {len(rows)}")
        counts = Counter(row["case_name"] for row in rows)
        assert_true(len(counts) == 50, f"{setting} expected 50 cases, got {len(counts)}")
        assert_true(all(count == 12 for count in counts.values()), f"{setting} each case should contribute 12 proposals")

    for row in folds:
        assert_true(int(row["train_cases"]) == 40, f"{row['setting']} fold {row['fold']} train cases mismatch")
        assert_true(int(row["test_cases"]) == 10, f"{row['setting']} fold {row['fold']} test cases mismatch")
        assert_true(int(row["test_proposals"]) == 120, f"{row['setting']} fold {row['fold']} test proposals mismatch")

    print("Case-level 50-case split checks passed.")


if __name__ == "__main__":
    main()
