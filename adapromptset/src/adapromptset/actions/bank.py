from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from adapromptset.candidates.points import CandidatePools


@dataclass(frozen=True)
class PromptGroup:
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.positive_indices:
            raise ValueError("prompt group requires at least one positive index")


@dataclass(frozen=True)
class PromptAction:
    action_id: str
    family: str
    groups: tuple[PromptGroup, ...]
    config: dict[str, Any]
    valid: bool = True
    invalid_reason: str = ""

    def content_hash(self) -> str:
        payload = {
            "action_id": self.action_id,
            "family": self.family,
            "groups": [
                {
                    "positive_indices": list(group.positive_indices),
                    "negative_indices": list(group.negative_indices),
                }
                for group in self.groups
            ],
            "config": self.config,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def build_prompt_arrays(
    group: PromptGroup,
    coordinates: Sequence[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[int]]:
    points: list[tuple[int, int]] = []
    labels: list[int] = []
    for index in group.positive_indices:
        points.append(coordinates[index])
        labels.append(1)
    for index in group.negative_indices:
        points.append(coordinates[index])
        labels.append(0)
    return points, labels


def candidate_coordinates(pools: CandidatePools) -> list[tuple[int, int]]:
    return [candidate.coordinate for candidate in pools.positives] + [candidate.coordinate for candidate in pools.negatives]


def sam_call_count(action: PromptAction) -> int:
    if not action.valid:
        return 0
    return len(action.groups)


def action_point_count(action: PromptAction) -> int:
    return sum(len(group.positive_indices) + len(group.negative_indices) for group in action.groups)


def group_cache_key(group: PromptGroup, coordinates: Sequence[tuple[int, int]]) -> str:
    points, labels = build_prompt_arrays(group, coordinates)
    payload = {"points": points, "labels": labels}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def action_cache_keys(action: PromptAction, coordinates: Sequence[tuple[int, int]]) -> tuple[str, ...]:
    return tuple(group_cache_key(group, coordinates) for group in action.groups)


def _invalid_action(action_id: str, family: str, required: int, kind: str, config: dict[str, Any]) -> PromptAction:
    return PromptAction(
        action_id=action_id,
        family=family,
        groups=(),
        config=config,
        valid=False,
        invalid_reason=f"requires_{required}_{kind}_candidates",
    )


def _one_group_positive(action_id: str, family: str, k: int, positive_count: int, config: dict[str, Any]) -> PromptAction:
    if positive_count < k:
        return _invalid_action(action_id, family, k, "positive", config)
    return PromptAction(
        action_id=action_id,
        family=family,
        groups=(PromptGroup(positive_indices=tuple(range(k))),),
        config=config,
    )


def _one_group_pos_neg(
    action_id: str,
    family: str,
    positive_k: int,
    negative_k: int,
    positive_count: int,
    negative_count: int,
    config: dict[str, Any],
) -> PromptAction:
    if positive_count < positive_k:
        return _invalid_action(action_id, family, positive_k, "positive", config)
    if negative_count < negative_k:
        return _invalid_action(action_id, family, negative_k, "negative", config)
    negative_offset = positive_count
    return PromptAction(
        action_id=action_id,
        family=family,
        groups=(
            PromptGroup(
                positive_indices=tuple(range(positive_k)),
                negative_indices=tuple(negative_offset + index for index in range(negative_k)),
            ),
        ),
        config=config,
    )


def _independent_positive(action_id: str, family: str, k: int, positive_count: int, config: dict[str, Any]) -> PromptAction:
    if positive_count < k:
        return _invalid_action(action_id, family, k, "positive", config)
    return PromptAction(
        action_id=action_id,
        family=family,
        groups=tuple(PromptGroup(positive_indices=(index,)) for index in range(k)),
        config=config,
    )


def build_mvp_action_bank(
    pools: CandidatePools,
    *,
    topk_limits: Sequence[int] = (1, 3, 5, 10),
) -> tuple[PromptAction, ...]:
    positive_count = len(pools.positives)
    negative_count = len(pools.negatives)
    actions: list[PromptAction] = []
    for k in topk_limits:
        actions.append(
            _one_group_positive(
                f"A_top{k}_one_group",
                "single_group_positive",
                k,
                positive_count,
                {"positive_limit": k},
            )
        )
    for positive_k, negative_k in ((3, 1), (3, 3), (5, 1), (5, 3)):
        actions.append(
            _one_group_pos_neg(
                f"B_top{positive_k}_pos{negative_k}neg",
                "single_group_positive_negative",
                positive_k,
                negative_k,
                positive_count,
                negative_count,
                {"positive_limit": positive_k, "negative_limit": negative_k},
            )
        )
    for k in (3, 5):
        actions.append(
            _independent_positive(
                f"C_top{k}_independent",
                "independent_positive_groups",
                k,
                positive_count,
                {"positive_limit": k},
            )
        )
    return tuple(actions)


def _nearest_negative_indices(pools: CandidatePools, positive_indices: Sequence[int]) -> tuple[int, ...]:
    offset = len(pools.positives)
    if not pools.negatives:
        return ()
    selected: list[int] = []
    for pos_index in positive_indices:
        px, py = pools.positives[pos_index].coordinate
        nearest = min(
            range(len(pools.negatives)),
            key=lambda neg_index: math.hypot(
                pools.negatives[neg_index].coordinate[0] - px,
                pools.negatives[neg_index].coordinate[1] - py,
            ),
        )
        selected.append(offset + nearest)
    return tuple(selected)


def _group_nearest_negative_index(pools: CandidatePools, positive_indices: Sequence[int]) -> tuple[int, ...]:
    nearest = _nearest_negative_indices(pools, positive_indices)
    if not nearest:
        return ()
    counts: dict[int, int] = {}
    for index in nearest:
        counts[index] = counts.get(index, 0) + 1
    return (max(counts, key=counts.get),)


def build_rescue_action_bank(pools: CandidatePools) -> tuple[PromptAction, ...]:
    positive_count = len(pools.positives)
    negative_count = len(pools.negatives)
    actions: list[PromptAction] = []
    if positive_count < 5:
        return (
            _invalid_action(
                "R_split5_2groups",
                "rescue_split_groups",
                5,
                "positive",
                {"positive_limit": 5, "groups": 2},
            ),
        )
    split_groups = ((0, 1, 2), (3, 4))
    actions.append(
        PromptAction(
            action_id="R_split5_2groups",
            family="rescue_split_groups",
            groups=tuple(PromptGroup(positive_indices=group) for group in split_groups),
            config={"positive_limit": 5, "groups": 2, "local_negative": False},
        )
    )
    if negative_count:
        actions.append(
            PromptAction(
                action_id="R_split5_2groups_localneg",
                family="rescue_group_specific_negatives",
                groups=tuple(
                    PromptGroup(
                        positive_indices=group,
                        negative_indices=_group_nearest_negative_index(pools, group),
                    )
                    for group in split_groups
                ),
                config={"positive_limit": 5, "groups": 2, "local_negative": True},
            )
        )
        actions.append(
            PromptAction(
                action_id="R_top5_independent_localneg",
                family="rescue_group_specific_negatives",
                groups=tuple(
                    PromptGroup(
                        positive_indices=(index,),
                        negative_indices=_nearest_negative_indices(pools, (index,)),
                    )
                    for index in range(5)
                ),
                config={"positive_limit": 5, "groups": 5, "local_negative": True},
            )
        )
    else:
        actions.append(
            _invalid_action(
                "R_split5_2groups_localneg",
                "rescue_group_specific_negatives",
                1,
                "negative",
                {"positive_limit": 5, "groups": 2, "local_negative": True},
            )
        )
        actions.append(
            _invalid_action(
                "R_top5_independent_localneg",
                "rescue_group_specific_negatives",
                1,
                "negative",
                {"positive_limit": 5, "groups": 5, "local_negative": True},
            )
        )
    return tuple(actions)


def action_manifest_record(action: PromptAction) -> dict[str, Any]:
    num_positive = sum(len(group.positive_indices) for group in action.groups)
    num_negative = sum(len(group.negative_indices) for group in action.groups)
    return {
        "action_id": action.action_id,
        "family": action.family,
        "config": action.config,
        "groups": [
            {
                "positive_indices": list(group.positive_indices),
                "negative_indices": list(group.negative_indices),
            }
            for group in action.groups
        ],
        "valid": action.valid,
        "invalid_reason": action.invalid_reason,
        "num_groups": len(action.groups),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "sam_calls": sam_call_count(action),
        "prompt_points": num_positive + num_negative,
        "content_hash": action.content_hash(),
    }


def action_bank_manifest(name: str, actions: Sequence[PromptAction]) -> dict[str, Any]:
    records = [action_manifest_record(action) for action in actions]
    payload: dict[str, Any] = {
        "name": name,
        "action_count": len(records),
        "valid_action_count": sum(1 for record in records if record["valid"]),
        "actions": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["bank_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()
    return payload
