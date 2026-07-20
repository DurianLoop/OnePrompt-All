from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from fspl.dataset import CaseRecord


@dataclass(frozen=True)
class MultiReferenceProtocol:
    seed: int
    references: tuple[CaseRecord, ...]
    policy_train_queries: tuple[CaseRecord, ...]
    validation_queries: tuple[CaseRecord, ...]
    test_cases: tuple[CaseRecord, ...]


def episode_id(reference_case_id: str, query_case_id: str) -> str:
    return f"{reference_case_id}::{query_case_id}"


def build_multireference_protocol(
    train_cases: Sequence[CaseRecord],
    test_cases: Sequence[CaseRecord],
    *,
    seed: int,
    reference_count: int,
    policy_train_query_count: int,
    validation_query_count: int,
) -> MultiReferenceProtocol:
    if reference_count < 1 or policy_train_query_count < 1 or validation_query_count < 1:
        raise ValueError("all multi-reference partition counts must be positive")
    ordered_train = sorted(train_cases, key=lambda case: case.case_id)
    ordered_test = tuple(sorted(test_cases, key=lambda case: case.case_id))
    required = reference_count + policy_train_query_count + validation_query_count
    if len(ordered_train) < required:
        raise ValueError(f"multi-reference protocol requires at least {required} train cases")
    if any(case.split != "train" for case in ordered_train):
        raise ValueError("references and policy queries must come from the train split")
    if any(case.split != "test" for case in ordered_test):
        raise ValueError("test cases must come from the test split")

    shuffled = list(ordered_train)
    random.Random(seed).shuffle(shuffled)
    references = tuple(sorted(shuffled[:reference_count], key=lambda case: case.case_id))
    validation_start = reference_count
    validation_end = validation_start + validation_query_count
    validation = tuple(sorted(shuffled[validation_start:validation_end], key=lambda case: case.case_id))
    train_end = validation_end + policy_train_query_count
    policy_train = tuple(sorted(shuffled[validation_end:train_end], key=lambda case: case.case_id))
    return MultiReferenceProtocol(
        seed=seed,
        references=references,
        policy_train_queries=policy_train,
        validation_queries=validation,
        test_cases=ordered_test,
    )
