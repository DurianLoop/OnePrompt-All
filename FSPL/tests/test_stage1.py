import json
from pathlib import Path

from fspl.stage1 import write_stage1_summary


def test_write_stage1_summary_preserves_required_serializable_fields() -> None:
    output = Path(__file__).resolve().parent.parent / "artifacts" / "test-data" / "stage1-summary.json"
    summary = {
        "reference_case_id": "case_a",
        "query_case_id": "case_b",
        "positive_count": 5,
        "negative_count": 3,
        "dice": 0.5,
        "elapsed_seconds": 1.25,
    }

    write_stage1_summary(output, summary)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == summary
