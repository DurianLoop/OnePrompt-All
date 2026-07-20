from build_negative_gate_domain_qualitative import assemble_examples


def test_assemble_examples_joins_reference_query_grid_and_policy() -> None:
    manifests = [{"pilot_id": "episode_1", "support_case_id": "ref", "query_case_id": "query"}]
    negative_rows = [
        {"episode_id": "episode_1", "requested_negative_count": "0", "dice": "0.4"},
        {"episode_id": "episode_1", "requested_negative_count": "5", "dice": "0.6"},
    ]
    learned_rows = [{"episode_id": "episode_1", "selected_action": "neg5"}]

    examples = assemble_examples(manifests, negative_rows, learned_rows)

    assert len(examples) == 1
    assert examples[0]["manifest"]["support_case_id"] == "ref"
    assert examples[0]["n0"]["dice"] == "0.4"
    assert examples[0]["n5"]["dice"] == "0.6"
    assert examples[0]["selected_action"] == "neg5"


def test_assemble_examples_uses_actual_count_for_legacy_rows() -> None:
    manifests = [{"pilot_id": "episode_1", "support_case_id": "ref", "query_case_id": "query"}]
    negative_rows = [
        {"episode_id": "episode_1", "negative_count": "0", "dice": "0.4"},
        {"episode_id": "episode_1", "negative_count": "5", "dice": "0.6"},
    ]
    learned_rows = [{"episode_id": "episode_1", "action": "neg0"}]

    examples = assemble_examples(manifests, negative_rows, learned_rows)

    assert examples[0]["selected_action"] == "neg0"
