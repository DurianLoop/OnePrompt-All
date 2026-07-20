from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from adapromptset.candidates.points import (
    build_candidate_pools,
    component_coverage,
    point_precision,
    validate_candidate_pools,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _write_gate_c_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gate C Status",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
        (
            f"| synthetic candidate kernel smoke | {payload['synthetic_smoke']} | "
            "score-map mechanics only; no query GT used for candidate generation |"
        ),
        (
            f"| formal matcher candidate gate | {payload['formal_gate_c']} | "
            "requires frozen DINOv2-L matching evidence on real episodes |"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_ledger(project_root: Path) -> None:
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "3-smoke",
                "command": "scripts/run_candidate_smoke.py",
                "gpu_hours": "0.0",
                "notes": "CPU-only synthetic candidate kernel smoke; formal Gate C still pending real matcher evidence.",
            }
        )


def run_smoke(out_path: Path | None = None, project_root: Path = PROJECT_ROOT) -> int:
    foreground = [
        [10.0, 9.0, 1.0, 1.0, 1.0],
        [9.0, 8.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 8.0, 9.0],
        [1.0, 1.0, 1.0, 9.0, 10.0],
    ]
    background = [
        [0.0, 0.0, 8.0, 8.0, 8.0],
        [0.0, 0.0, 8.0, 8.0, 8.0],
        [8.0, 8.0, 8.0, 0.0, 0.0],
        [8.0, 8.0, 8.0, 0.0, 0.0],
    ]
    valid_mask = [
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
    ]
    target_mask = [
        [True, True, False, False, False],
        [True, True, False, False, False],
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]
    pools = build_candidate_pools(
        foreground,
        background,
        image_size=(5, 4),
        positive_limit=2,
        negative_limit=2,
        valid_mask=valid_mask,
        positive_layout="spread",
        min_positive_distance=2.0,
    )
    candidates = list(pools.positives + pools.negatives)
    issues = validate_candidate_pools(candidates, image_size=(5, 4))
    metrics = {
        "point_precision": point_precision(candidates, target_mask),
        "component_coverage": component_coverage(candidates, target_mask),
        "positive_count": len(pools.positives),
        "negative_count": len(pools.negatives),
        "validation_issues": issues,
    }
    payload: dict[str, object] = {
        "synthetic_smoke": "PASS" if not issues and metrics["point_precision"] >= 0.5 else "FAIL",
        "formal_gate_c": "PENDING_REAL_MATCHER",
        "metrics": metrics,
        "candidate_generation_uses_query_gt": False,
        "notes": [
            "This smoke uses synthetic foreground/background score maps.",
            "It validates candidate mechanics only and must not be reported as formal Gate C.",
        ],
    }
    output = out_path or (project_root / "reports" / "candidate_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if out_path is None:
        _write_gate_c_status(project_root / "reports" / "gate_c_status.md", payload)
        _append_ledger(project_root)
    return 0 if payload["synthetic_smoke"] == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only synthetic candidate kernel smoke.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    raise SystemExit(run_smoke(args.out, args.project_root))


if __name__ == "__main__":
    main()

