from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class FrozenParameterViolation(RuntimeError):
    """Raised when an optimizer contains frozen matcher or segmenter parameters."""


def freeze_module_parameters(module: Any) -> None:
    """Mark every parameter in a torch-like module as non-trainable."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def _optimizer_parameter_ids(optimizer: Any) -> set[int]:
    parameter_ids: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            parameter_ids.add(id(parameter))
    return parameter_ids


def _module_parameter_ids(module: Any) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def assert_optimizer_excludes_frozen_modules(
    optimizer: Any,
    frozen_modules: Mapping[str, Any],
) -> dict[str, Any]:
    """Assert that optimizer param groups do not contain frozen module parameters.

    This is intentionally independent of ``requires_grad``. A frozen matcher or
    segmenter parameter in an optimizer is a protocol violation even if gradients
    are disabled, because it weakens the audit trail for future training runs.
    """
    optimizer_ids = _optimizer_parameter_ids(optimizer)
    violations: dict[str, int] = {}
    frozen_count = 0
    for role, module in frozen_modules.items():
        module_ids = _module_parameter_ids(module)
        frozen_count += len(module_ids)
        overlap_count = len(optimizer_ids.intersection(module_ids))
        if overlap_count:
            violations[role] = overlap_count
    if violations:
        details = ", ".join(f"{role}={count}" for role, count in sorted(violations.items()))
        raise FrozenParameterViolation(
            f"optimizer contains frozen matcher/segmenter parameters: {details}"
        )
    return {
        "status": "PASS",
        "optimizer_parameter_count": len(optimizer_ids),
        "frozen_parameter_count": frozen_count,
        "checked_roles": sorted(frozen_modules),
    }
