from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from adapromptset.training.optimizer_guard import (
    FrozenParameterViolation,
    assert_optimizer_excludes_frozen_modules,
    freeze_module_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _toy_modules() -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    matcher = torch.nn.Linear(2, 2)
    segmenter = torch.nn.Linear(2, 2)
    policy = torch.nn.Linear(2, 1)
    freeze_module_parameters(matcher)
    freeze_module_parameters(segmenter)
    return matcher, segmenter, policy


def run_audit(out_dir: Path, include_frozen_violation: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    matcher, segmenter, policy = _toy_modules()
    optimizer_params: list[Any] = [{"params": policy.parameters()}]
    if include_frozen_violation:
        optimizer_params.append({"params": matcher.parameters()})
    optimizer = torch.optim.AdamW(optimizer_params, lr=1e-3)
    payload: dict[str, Any] = {
        "status": "PENDING",
        "policy_training": "not_started",
        "note": "Toy audit only; no AdaPromptSet policy training is executed.",
    }
    try:
        payload["guard_report"] = assert_optimizer_excludes_frozen_modules(
            optimizer,
            frozen_modules={"matcher": matcher, "segmenter": segmenter},
        )
        payload["status"] = "PASS"
        exit_code = 0
    except FrozenParameterViolation as exc:
        payload["status"] = "FAIL"
        payload["error"] = str(exc)
        exit_code = 2
    (out_dir / "optimizer_guard.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit that optimizer param groups exclude frozen modules.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "optimizer_guard")
    parser.add_argument("--include-frozen-violation", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_audit(out_dir=args.out_dir, include_frozen_violation=args.include_frozen_violation))


if __name__ == "__main__":
    main()
