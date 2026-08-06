from build_monusac_strict_policy_splits import build_strict_protocol


def sample(case_id: str, split: str, patient: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "split": split,
        "patient": patient,
        "instance_count": "10",
        "classes_present": "1;2",
    }


def test_strict_protocol_removes_train_patient_overlap_and_keeps_full_test() -> None:
    rows = [
        sample("tr_a0", "train", "pa"),
        sample("tr_a1", "train", "pa"),
        sample("tr_b0", "train", "pb"),
        sample("tr_b1", "train", "pb"),
        sample("tr_c0", "train", "pc"),
        sample("tr_c1", "train", "pc"),
        sample("tr_d0", "train", "pd"),
        sample("tr_d1", "train", "pd"),
        sample("te_a0", "test", "pa"),
        sample("te_z0", "test", "pz"),
    ]

    protocol, metadata = build_strict_protocol(
        rows,
        val_patient_count=1,
        references_per_query=2,
        seed=7,
    )

    assignments = protocol["case_assignments"]
    assert {row["case_id"] for row in assignments if row["role"] == "excluded_overlap"} == {
        "tr_a0",
        "tr_a1",
    }
    assert {row["case_id"] for row in assignments if row["role"] == "test"} == {
        "te_a0",
        "te_z0",
    }
    assert metadata["official_test_patch_count"] == 2
    assert metadata["overlap_patient_count"] == 1

    train_patients = {row["patient"] for row in assignments if row["role"] == "train"}
    val_patients = {row["patient"] for row in assignments if row["role"] == "val"}
    test_patients = {row["patient"] for row in assignments if row["role"] == "test"}
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)

    for split in ("train", "val", "test"):
        episodes = protocol[f"episodes_{split}"]
        query_counts: dict[str, int] = {}
        for row in episodes:
            query_counts[row["query_case_id"]] = query_counts.get(row["query_case_id"], 0) + 1
            assert row["support_role"] == "train"
            assert row["support_patient"] != row["query_patient"]
        assert set(query_counts.values()) == {2}
