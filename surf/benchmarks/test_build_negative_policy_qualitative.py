from build_negative_policy_qualitative import select_policy_examples


def test_select_policy_examples_separates_helpful_switches_and_avoided_failures() -> None:
    negative = []
    for query, n0, n5 in (("a", 0.4, 0.7), ("b", 0.6, 0.2), ("c", 0.5, 0.55)):
        negative.extend([
            {"query_case_id": query, "negative_count": "0", "dice": str(n0)},
            {"query_case_id": query, "negative_count": "5", "dice": str(n5)},
        ])
    learned = [
        {"query_case_id": "a", "selected_action": "neg5"},
        {"query_case_id": "b", "selected_action": "neg0"},
        {"query_case_id": "c", "selected_action": "neg0"},
    ]

    switched, avoided = select_policy_examples(negative, learned, count=1)

    assert switched[0]["query_case_id"] == "a"
    assert avoided[0]["query_case_id"] == "b"
