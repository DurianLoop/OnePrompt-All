from pathlib import Path


def test_runner_parses_explicit_backbone_and_output_arguments() -> None:
    from scripts.run_frozen_layout_combo import parse_args

    args = parse_args(
        [
            "--dino-checkpoint",
            "dinov2_vitl14_pretrain.pth",
            "--sam-checkpoint",
            "sam_vit_l_0b3195.pth",
            "--output-dir",
            "artifacts/example",
        ]
    )

    assert args.dino_checkpoint == Path("dinov2_vitl14_pretrain.pth")
    assert args.sam_checkpoint == Path("sam_vit_l_0b3195.pth")
    assert args.output_dir == Path("artifacts/example")
    assert args.seed == 20260716


def test_runner_limits_test_cases_only_when_requested() -> None:
    from scripts.run_frozen_layout_combo import parse_args

    args = parse_args(
        [
            "--dino-checkpoint",
            "dino.pth",
            "--sam-checkpoint",
            "sam.pth",
            "--output-dir",
            "artifacts/example",
            "--max-test-cases",
            "2",
        ]
    )

    assert args.max_test_cases == 2


def test_runner_accepts_a_test_case_offset_for_resumable_batches() -> None:
    from scripts.run_frozen_layout_combo import parse_args

    args = parse_args(
        [
            "--dino-checkpoint",
            "dino.pth",
            "--sam-checkpoint",
            "sam.pth",
            "--output-dir",
            "artifacts/example",
            "--test-start",
            "6",
            "--max-test-cases",
            "2",
        ]
    )

    assert args.test_start == 6
