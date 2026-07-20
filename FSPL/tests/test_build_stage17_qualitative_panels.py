from pathlib import Path


def test_qualitative_panel_builder_parses_required_artifact_arguments() -> None:
    from scripts.build_stage17_qualitative_panels import parse_args

    args = parse_args(
        [
            "--case-manifest",
            "artifacts/cases.json",
            "--policy-selections",
            "artifacts/policy.csv",
            "--output-path",
            "artifacts/panels.json",
        ]
    )

    assert args.case_manifest == Path("artifacts/cases.json")
    assert args.policy_selections == Path("artifacts/policy.csv")
    assert args.output_path == Path("artifacts/panels.json")
