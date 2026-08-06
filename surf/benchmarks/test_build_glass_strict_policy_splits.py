from build_glass_strict_policy_splits import build_glass_protocol


def row(case_id: str, source_split: str, patient: int) -> dict[str, str]:
    return {
        "case_id": case_id,
        "source_split": source_split,
        "patient": str(patient),
        "image_path": f"images/{case_id}.bmp",
        "mask_path": f"masks/{case_id}_anno.bmp",
        "grade": "benign",
    }


def test_protocol_excludes_train_patients_seen_in_official_test() -> None:
    rows = [
        row("train_1", "train", 1),
        row("train_2", "train", 1),
        row("train_3", "train", 2),
        row("train_4", "train", 2),
        row("train_5", "train", 3),
        row("train_6", "train", 3),
        row("train_7", "train", 4),
        row("train_8", "train", 4),
        row("testA_1", "testA", 1),
        row("testB_1", "testB", 9),
    ]

    assignments, episodes, metadata = build_glass_protocol(
        rows,
        val_patient_count=1,
        references_per_query=2,
        seed=7,
    )

    train_patients = {item["patient"] for item in assignments if item["role"] == "train"}
    val_patients = {item["patient"] for item in assignments if item["role"] == "val"}
    test_patients = {item["patient"] for item in assignments if item["role"] == "test"}
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
    assert {item["case_id"] for item in assignments if item["role"] == "excluded_overlap"} == {
        "train_1",
        "train_2",
    }
    assert metadata["official_test_case_count"] == 2
    assert metadata["excluded_overlap_case_count"] == 2
    assert all(item["support_role"] == "train" for role in episodes.values() for item in role)
    assert all(item["support_patient"] != item["query_patient"] for role in episodes.values() for item in role)


def test_protocol_is_deterministic_and_never_uses_query_as_support() -> None:
    rows = [
        row("train_1", "train", 1),
        row("train_2", "train", 1),
        row("train_3", "train", 2),
        row("train_4", "train", 2),
        row("train_5", "train", 3),
        row("train_6", "train", 3),
        row("train_7", "train", 4),
        row("train_8", "train", 4),
        row("testA_1", "testA", 8),
    ]

    first = build_glass_protocol(rows, val_patient_count=1, references_per_query=2, seed=11)
    second = build_glass_protocol(rows, val_patient_count=1, references_per_query=2, seed=11)

    assert first == second
    assert all(
        item["support_case_id"] != item["query_case_id"]
        for role in first[1].values()
        for item in role
    )


def test_protocol_records_source_provenance() -> None:
    rows = [
        row("train_1", "train", 1),
        row("train_2", "train", 1),
        row("train_3", "train", 2),
        row("train_4", "train", 2),
        row("train_5", "train", 3),
        row("train_6", "train", 3),
        row("train_7", "train", 4),
        row("train_8", "train", 4),
        row("testA_1", "testA", 8),
    ]

    _, _, metadata = build_glass_protocol(
        rows,
        val_patient_count=1,
        references_per_query=1,
        seed=5,
        source_url="https://example.test/glas.zip",
        source_sha256="abc123",
    )

    assert metadata["source_url"] == "https://example.test/glas.zip"
    assert metadata["source_sha256"] == "abc123"
