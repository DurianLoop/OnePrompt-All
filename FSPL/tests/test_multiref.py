from pathlib import Path

from fspl.dataset import CaseRecord
from fspl.multiref import build_multireference_protocol, episode_id


def _case(case_id: str, split: str = "train") -> CaseRecord:
    return CaseRecord(case_id, Path(f"{case_id}.tif"), Path(f"{case_id}.png"), split)


def test_multireference_protocol_has_disjoint_train_only_roles() -> None:
    train = [_case(f"train_{index:03d}") for index in range(20)]
    test = [_case(f"test_{index:03d}", "test") for index in range(4)]

    protocol = build_multireference_protocol(
        train,
        test,
        seed=11,
        reference_count=3,
        policy_train_query_count=8,
        validation_query_count=4,
    )

    role_ids = [
        {case.case_id for case in protocol.references},
        {case.case_id for case in protocol.policy_train_queries},
        {case.case_id for case in protocol.validation_queries},
        {case.case_id for case in protocol.test_cases},
    ]
    assert all(first.isdisjoint(second) for index, first in enumerate(role_ids) for second in role_ids[index + 1 :])
    assert len(protocol.references) == 3
    assert len(protocol.policy_train_queries) == 8
    assert len(protocol.validation_queries) == 4
    assert protocol == build_multireference_protocol(
        train,
        test,
        seed=11,
        reference_count=3,
        policy_train_query_count=8,
        validation_query_count=4,
    )


def test_episode_id_preserves_reference_query_boundary() -> None:
    assert episode_id("ref_a", "query_b") == "ref_a::query_b"
