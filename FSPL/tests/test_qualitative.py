from fspl.qualitative import select_qualitative_case_ids


def test_qualitative_cases_are_seeded_sorted_and_without_replacement() -> None:
    selected = select_qualitative_case_ids(["case_c", "case_a", "case_b", "case_d"], count=3, seed=7)

    assert selected == select_qualitative_case_ids(["case_d", "case_b", "case_c", "case_a"], count=3, seed=7)
    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_panel_manifest_uses_all_locked_references_without_utility_fields() -> None:
    from fspl.qualitative import build_panel_manifest

    manifest = build_panel_manifest(
        ["IDRiD_68"],
        ["IDRiD_20", "IDRiD_47"],
        [
            {"case_id": "IDRiD_20::IDRiD_68", "rule_name": "spread5_n3_d2"},
            {"case_id": "IDRiD_47::IDRiD_68", "rule_name": "top5_n3"},
        ],
    )

    assert manifest["case_ids"] == ["IDRiD_68"]
    assert manifest["panels"][0]["reference_panels"] == [
        {"reference_case_id": "IDRiD_20", "baseline_rule": "top5_n3", "selected_policy_rule": "spread5_n3_d2"},
        {"reference_case_id": "IDRiD_47", "baseline_rule": "top5_n3", "selected_policy_rule": "top5_n3"},
    ]
    assert all(
        set(reference_panel) == {"reference_case_id", "baseline_rule", "selected_policy_rule"}
        for reference_panel in manifest["panels"][0]["reference_panels"]
    )
