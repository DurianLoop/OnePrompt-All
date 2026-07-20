from pathlib import Path

from scripts.run_stage17_clean_adaptive_layout import _prior_used_monusac_ids


def test_clean_adaptive_runner_has_locked_protocol_defaults() -> None:
    from scripts.run_stage17_clean_adaptive_layout import parse_args

    args = parse_args(["--output-dir", "artifacts/example"])

    assert args.output_dir == Path("artifacts/example")
    assert args.seed == 20260718
    assert args.reference_count == 3
    assert args.policy_train_query_count == 48
    assert args.validation_query_count == 12
    assert args.radii == (2.0, 3.0, 4.0)


def test_clean_policy_exclusion_manifest_contains_all_previously_used_train_cases() -> None:
    used = _prior_used_monusac_ids()

    assert len(used) == 66
    assert {"train_028", "train_115", "train_183", "train_035", "train_036", "train_154"}.issubset(used)


def test_policy_selection_writer_preserves_predicted_utility_column() -> None:
    from scripts.run_stage17_clean_adaptive_layout import write_policy_selections

    path = Path(__file__).resolve().parent.parent / "artifacts" / "test-data" / "policy-selections.csv"
    write_policy_selections(
        path,
        [
            {
                "case_id": "reference::query",
                "split": "test",
                "rule_name": "spread5_n3_d3",
                "utility": 0.7,
                "predicted_utility": 0.6,
                "prompt_count": 8,
            }
        ],
    )

    assert "predicted_utility" in path.read_text(encoding="utf-8").splitlines()[0]
