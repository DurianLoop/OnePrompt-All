from pathlib import Path


def test_manifest_runner_parses_dataset_root_and_output_path() -> None:
    from scripts.make_qualitative_manifest import parse_args

    args = parse_args(["--dataset-root", "data", "--output-path", "artifacts/qual.json"])

    assert args.dataset_root == Path("data")
    assert args.output_path == Path("artifacts/qual.json")
    assert args.count == 3
    assert args.seed == 20260718
