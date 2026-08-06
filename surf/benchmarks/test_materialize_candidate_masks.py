from materialize_candidate_masks import build_episode_image_map, group_candidates_by_query


def test_build_episode_image_map_keys_rows_by_episode_id() -> None:
    mapping = build_episode_image_map([
        {"episode_id": "support_a_to_query_1", "query_image_path": "query_1.png"},
        {"episode_id": "support_b_to_query_1", "query_image_path": "query_1.png"},
    ])

    assert mapping == {
        "support_a_to_query_1": "query_1.png",
        "support_b_to_query_1": "query_1.png",
    }


def test_build_episode_image_map_accepts_real_protocol_pilot_id() -> None:
    mapping = build_episode_image_map([
        {"pilot_id": "support_a_to_query_1", "query_image_path": "query_1.png"},
    ])

    assert mapping == {"support_a_to_query_1": "query_1.png"}


def test_group_candidates_by_query_preserves_all_reference_actions() -> None:
    groups = group_candidates_by_query([
        {"query_case_id": "q1", "case_name": "r1_to_q1", "method": "a"},
        {"query_case_id": "q2", "case_name": "r1_to_q2", "method": "a"},
        {"query_case_id": "q1", "case_name": "r2_to_q1", "method": "b"},
    ])

    assert sorted(groups) == ["q1", "q2"]
    assert [row["case_name"] for row in groups["q1"]] == ["r1_to_q1", "r2_to_q1"]
