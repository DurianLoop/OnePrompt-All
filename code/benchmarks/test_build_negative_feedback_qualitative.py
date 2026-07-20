from build_negative_feedback_qualitative import paired_extremes


def test_paired_extremes_selects_largest_gain_and_largest_loss() -> None:
    rows = []
    for query, base, treatment in (("a", 0.4, 0.7), ("b", 0.6, 0.5), ("c", 0.5, 0.55)):
        rows.extend([
            {"query_case_id": query, "negative_count": "0", "dice": str(base)},
            {"query_case_id": query, "negative_count": "5", "dice": str(treatment)},
        ])

    gain, loss = paired_extremes(rows)

    assert gain["query_case_id"] == "a"
    assert loss["query_case_id"] == "b"
