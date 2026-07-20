from pathlib import Path


def test_validator_parses_artifact_and_expected_episode_arguments() -> None:
    from scripts.validate_layout_artifact import parse_args

    args = parse_args(["--artifact-dir", "artifacts/stage16", "--expected-episodes", "303"])

    assert args.artifact_dir == Path("artifacts/stage16")
    assert args.expected_episodes == 303
