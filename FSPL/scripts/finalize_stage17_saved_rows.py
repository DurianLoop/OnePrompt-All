from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from fspl.policy import fit_frozen_utility_policy, fit_utility_policy
from fspl.stage3 import POLICY_FEATURE_NAMES, _policy_outcome, _write_predictions_csv
from fspl.stage4 import _metrics


DEVELOPMENT_ROWS = "development_case_action_utilities.csv"
IDRID_ROWS = "idrid_test_case_action_utilities.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the clean adaptive-layout result from saved utility rows without rerunning image inference."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args(argv)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing saved action utility CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"saved action utility CSV is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no {split!r} rows in saved utility table")
    return selected


def _unique(rows: list[dict[str, str]], field: str) -> list[str]:
    return sorted({row[field] for row in rows})


def finalize_saved_rows(artifact_dir: Path, *, seed: int = 20260718) -> dict:
    """Write deterministic policy summaries from existing utility tables only.

    This intentionally does not discover cases, load image/mask files, or invoke
    a matcher or segmenter.  It is for a completed external evaluation whose
    raw per-action utility rows were saved before a later reporting interruption.
    """

    development_path = artifact_dir / DEVELOPMENT_ROWS
    idrid_path = artifact_dir / IDRID_ROWS
    development_rows = _read_rows(development_path)
    idrid_rows = _read_rows(idrid_path)
    train_rows = _split_rows(development_rows, "train")
    validation_rows = _split_rows(development_rows, "validation")
    test_rows = _split_rows(idrid_rows, "test")

    fit = fit_utility_policy(train_rows, validation_rows, feature_names=POLICY_FEATURE_NAMES)
    validation_selected, validation_policy_dice = _policy_outcome(validation_rows, fit.estimator)
    frozen_estimator = fit_frozen_utility_policy(
        [*train_rows, *validation_rows], feature_names=POLICY_FEATURE_NAMES, min_samples_leaf=fit.min_samples_leaf
    )
    idrid_selected, idrid_policy_dice = _policy_outcome(test_rows, frozen_estimator)

    _write_predictions_csv(artifact_dir / "validation_policy_selections.csv", validation_selected)
    _write_predictions_csv(artifact_dir / "idrid_policy_selections.csv", idrid_selected)
    summary = {
        "method_status": "postprocessed clean-development adaptive-layout policy; no image or mask inference was rerun",
        "result_provenance": {
            "seed": seed,
            "development_utility_csv_sha256": _sha256(development_path),
            "idrid_test_utility_csv_sha256": _sha256(idrid_path),
        },
        "development_protocol": {
            "reference_case_ids": _unique(development_rows, "reference_case_id"),
            "policy_train_query_ids": _unique(train_rows, "query_case_id"),
            "validation_query_ids": _unique(validation_rows, "query_case_id"),
        },
        "rules": _unique(development_rows, "rule_name"),
        "development": {
            "train_action_row_count": len(train_rows),
            "validation_action_row_count": len(validation_rows),
            "validation_fixed_rules": _metrics(validation_rows, "validation"),
            "selected_min_samples_leaf": fit.min_samples_leaf,
            "validation_policy_selected_mean_dice": validation_policy_dice,
        },
        "idrid_external_test": {
            "reference_case_ids": _unique(test_rows, "reference_case_id"),
            "test_case_count": len(_unique(test_rows, "query_case_id")),
            "episodes_per_rule": len(_unique(test_rows, "case_id")),
            "fixed_rules": _metrics(test_rows, "test"),
            "policy_selected_mean_dice": idrid_policy_dice,
        },
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = finalize_saved_rows(args.artifact_dir, seed=args.seed)
    print(json.dumps(summary["idrid_external_test"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
