from pathlib import Path


def test_effect_report_parser_defaults_to_completed_replicates_only() -> None:
    from scripts.analyze_fixed_layout_effects import parse_args

    args = parse_args(["--output-path", "artifacts/fixed-layout-effects.json"])

    assert args.output_path == Path("artifacts/fixed-layout-effects.json")
    assert args.include_stage18 is False
    assert args.resamples == 10_000
