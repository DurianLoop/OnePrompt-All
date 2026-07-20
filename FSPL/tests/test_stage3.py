import csv
from pathlib import Path

from fspl.stage3 import _write_predictions_csv


def test_prediction_csv_accepts_predicted_utility_column() -> None:
    path = Path(__file__).resolve().parent.parent / "artifacts" / "test-data" / "predictions.csv"
    _write_predictions_csv(
        path,
        [
            {
                "case_id": "case_a",
                "split": "test",
                "rule_name": "top3_n0",
                "utility": 0.5,
                "predicted_utility": 0.6,
                "prompt_count": 3,
            }
        ],
    )

    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["predicted_utility"] == "0.6"
