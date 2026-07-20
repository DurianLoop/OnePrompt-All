"""Training entry points and safety guards for lightweight policies."""

from adapromptset.training.optimizer_guard import (
    FrozenParameterViolation,
    assert_optimizer_excludes_frozen_modules,
    freeze_module_parameters,
)

__all__ = [
    "FrozenParameterViolation",
    "assert_optimizer_excludes_frozen_modules",
    "freeze_module_parameters",
]
