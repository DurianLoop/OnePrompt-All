from build_monuseg_nested_cv_splits import build_nested_cv_protocol


def _row(split: str, sample_id: str, ratio: float) -> dict[str, str]:
    return {
        "dataset": "MoNuSeg",
        "split": split,
        "sample_id": sample_id,
        "image_path": f"images/{sample_id}.tif",
        "mask_path": f"masks/{sample_id}.png",
        "mask_area_ratio": str(ratio),
        "nuclei_region_count": str(int(ratio * 1000)),
    }


def test_nested_cv_tests_each_source_train_case_once_and_never_uses_official_test() -> None:
    rows = [_row("train", f"train_{index}", 0.1 + index * 0.01) for index in range(10)]
    rows += [_row("test", f"official_test_{index}", 0.3 + index * 0.01) for index in range(2)]

    folds, metadata = build_nested_cv_protocol(
        rows,
        fold_count=3,
        inner_val_count=2,
        train_reference_count=2,
        evaluation_reference_count=2,
    )

    outer_test_ids = [
        row["sample_id"]
        for fold in folds
        for row in fold["assignments"]
        if row["role"] == "outer_test"
    ]
    assert sorted(outer_test_ids) == [f"train_{index}" for index in range(10)]
    assert len(outer_test_ids) == len(set(outer_test_ids))
    assert all("official_test" not in row["sample_id"] for fold in folds for row in fold["assignments"])
    assert metadata["official_test_cases_reserved"] == 2

    for fold in folds:
        roles = {row["sample_id"]: row["role"] for row in fold["assignments"]}
        inner_train = {sample_id for sample_id, role in roles.items() if role == "train"}
        assert all(
            episode["support_case_id"] in inner_train
            for split in ("train", "val", "test")
            for episode in fold["episodes"][split]
        )
        assert all(
            episode["support_case_id"] != episode["query_case_id"]
            for split in ("train", "val", "test")
            for episode in fold["episodes"][split]
        )
