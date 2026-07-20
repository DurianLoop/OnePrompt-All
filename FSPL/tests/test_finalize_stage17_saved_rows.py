from __future__ import annotations

import csv
import json
from pathlib import Path


def _rows(split: str, case_id: str) -> list[dict[str, float | int | str]]:
    from fspl.stage3 import POLICY_FEATURE_NAMES

    rows: list[dict[str, float | int | str]] = []
    for index, rule_name in enumerate(("top5_n3", "spread5_n3_d2", "spread5_n3_d3", "spread5_n3_d4")):
        rows.append(
            {
                "case_id": case_id,
                "reference_case_id": "reference_001",
                "query_case_id": case_id.split("::")[-1],
                "split": split,
                "rule_name": rule_name,
                "utility": 0.1 * (index + 1),
                "sam_score": 0.8,
                "positive_count": 5,
                "negative_count": 3,
                "prompt_count": 8,
                **{name: float(index) for name in POLICY_FEATURE_NAMES},
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_finalize_saved_rows_emits_auditable_summary_and_policy_selections() -> None:
    from scripts.finalize_stage17_saved_rows import finalize_saved_rows

    tmp_path = Path(__file__).resolve().parent.parent / "artifacts" / "test-data" / "finalize-stage17"
    tmp_path.mkdir(parents=True, exist_ok=True)
    development_rows = [*_rows("train", "reference_001::train_query"), *_rows("validation", "reference_001::validation_query")]
    idrid_rows = _rows("test", "reference_001::IDRiD_68")
    _write(tmp_path / "development_case_action_utilities.csv", development_rows)
    _write(tmp_path / "idrid_test_case_action_utilities.csv", idrid_rows)

    summary = finalize_saved_rows(tmp_path, seed=20260718)

    assert summary["development"]["train_action_row_count"] == 4
    assert summary["development"]["validation_action_row_count"] == 4
    assert summary["idrid_external_test"]["test_case_count"] == 1
    assert summary["idrid_external_test"]["episodes_per_rule"] == 1
    assert (tmp_path / "summary.json").is_file()
    assert "predicted_utility" in (tmp_path / "validation_policy_selections.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "predicted_utility" in (tmp_path / "idrid_policy_selections.csv").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
