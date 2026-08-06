import pytest

from build_cpm17_strict_policy_splits import build_cpm17_protocol, validate_complete_pairs


def row(case_id: str, source_split: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "source_split": source_split,
        "image_path": f"images/{case_id}.png",
        "mat_path": f"labels/{case_id}.mat",
    }


def test_validator_rejects_missing_image_or_label_pair() -> None:
    with pytest.raises(ValueError, match="missing label"):
        validate_complete_pairs([row("image_00", "train")], {"image_00"}, set())


def test_protocol_preserves_official_test_and_uses_train_only_supports() -> None:
    rows = [row(f"train_{index:02d}", "train") for index in range(8)] + [
        row(f"test_{index:02d}", "test") for index in range(3)
    ]

    assignments, episodes, metadata = build_cpm17_protocol(
        rows, val_count=2, references_per_query=2, seed=17
    )

    roles = {item["case_id"]: item["role"] for item in assignments}
    assert {case_id for case_id, role in roles.items() if role == "test"} == {
        "test_00", "test_01", "test_02"
    }
    assert metadata["official_test_case_count"] == 3
    assert all(item["support_role"] == "train" for role in episodes.values() for item in role)
    assert all(item["support_case_id"] != item["query_case_id"] for role in episodes.values() for item in role)
    assert all(item["support_source_split"] == "train" for role in episodes.values() for item in role)
