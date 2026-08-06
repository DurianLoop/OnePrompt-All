from audit_monusac_dataset import audit_rows


def test_audit_rows_counts_instances_and_categories() -> None:
    rows = [
        {"patient": "p1", "instances": [{"mask": 1}, {"mask": 2}], "categories": [1, 2], "tissue": 3},
        {"patient": "p2", "instances": [{"mask": 3}], "categories": [2], "tissue": 3},
    ]

    result = audit_rows(rows, "train")

    assert result["split"] == "train"
    assert result["patch_count"] == 2
    assert result["patient_count"] == 2
    assert result["instance_count"] == 3
    assert result["category_counts"] == {"1": 1, "2": 2}
