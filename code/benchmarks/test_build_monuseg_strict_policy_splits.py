from build_monuseg_strict_policy_splits import build_protocol


def make_row(split: str, sample_id: str, ratio: float, regions: int) -> dict[str, str]:
    return {
        "dataset": "MoNuSeg",
        "split": split,
        "sample_id": sample_id,
        "image_path": f"images/{sample_id}.tif",
        "mask_path": f"masks/{sample_id}.png",
        "mask_area_ratio": str(ratio),
        "nuclei_region_count": str(regions),
    }


def test_build_protocol_keeps_test_held_out_and_reference_in_training() -> None:
    rows = [
        make_row("train", f"train_{index}", 0.10 + 0.05 * index, 100 + index)
        for index in range(6)
    ] + [
        make_row("test", f"test_{index}", 0.20 + 0.05 * index, 200 + index)
        for index in range(2)
    ]

    assignments, episodes, metadata = build_protocol(rows, val_count=2)

    roles = {row["sample_id"]: row["role"] for row in assignments}
    assert list(roles.values()).count("train") == 4
    assert list(roles.values()).count("val") == 2
    assert list(roles.values()).count("test") == 2
    assert len(episodes["train"]) == 3
    assert len(episodes["val"]) == 2
    assert len(episodes["test"]) == 2
    assert metadata["case_counts"] == {"train": 4, "val": 2, "test": 2}
    assert metadata["protocol"] == "observed_monuseg_37_train_14_test_strict_holdout"

    support_ids = {episode["support_case_id"] for role in episodes.values() for episode in role}
    assert len(support_ids) == 1
    support_id = support_ids.pop()
    assert roles[support_id] == "train"
    assert all(episode["query_case_id"] != support_id for episode in episodes["train"])
    assert {episode["query_case_id"] for episode in episodes["test"]} == {"test_0", "test_1"}
    assert all(episode["support_split"] == "train" for role in episodes.values() for episode in role)


def test_build_protocol_can_create_multiple_train_references_per_query_without_self_support() -> None:
    rows = [
        make_row("train", f"train_{index}", 0.10 + 0.05 * index, 100 + index)
        for index in range(6)
    ] + [make_row("test", "test_0", 0.2, 200)]

    assignments, episodes, metadata = build_protocol(rows, val_count=2, train_reference_count=2)

    train_ids = {row["sample_id"] for row in assignments if row["role"] == "train"}
    assert len(episodes["train"]) == 8
    assert {episode["query_case_id"] for episode in episodes["train"]} == train_ids
    assert all(episode["support_case_id"] != episode["query_case_id"] for episode in episodes["train"])
    assert all(episode["support_case_id"] in train_ids for episode in episodes["train"])
    assert metadata["train_reference_count_per_query"] == 2


def test_build_protocol_can_repeat_validation_and_test_over_training_references() -> None:
    rows = [
        make_row("train", f"train_{index}", 0.10 + 0.05 * index, 100 + index)
        for index in range(8)
    ] + [
        make_row("test", f"test_{index}", 0.20 + 0.05 * index, 200 + index)
        for index in range(2)
    ]

    assignments, episodes, metadata = build_protocol(
        rows,
        val_count=2,
        train_reference_count=2,
        evaluation_reference_count=3,
    )

    train_ids = {row["sample_id"] for row in assignments if row["role"] == "train"}
    val_ids = {row["sample_id"] for row in assignments if row["role"] == "val"}
    test_ids = {row["sample_id"] for row in assignments if row["role"] == "test"}
    assert len(episodes["val"]) == len(val_ids) * 3
    assert len(episodes["test"]) == len(test_ids) * 3
    assert all(episode["support_case_id"] in train_ids for episode in episodes["val"] + episodes["test"])
    assert all(episode["support_case_id"] != episode["query_case_id"] for episode in episodes["val"])
    assert all(
        len({episode["support_case_id"] for episode in episodes[role] if episode["query_case_id"] == query_id}) == 3
        for role, query_ids in (("val", val_ids), ("test", test_ids))
        for query_id in query_ids
    )
    assert metadata["evaluation_reference_count_per_query"] == 3
