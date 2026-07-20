from pathlib import Path


def test_evidence_table_builder_parses_output_path() -> None:
    from scripts.build_evidence_tables import parse_args

    args = parse_args(["--output-path", "artifacts/report.md"])

    assert args.output_path == Path("artifacts/report.md")


def test_fixed_layout_row_marks_missing_stage_as_pending() -> None:
    from scripts.build_evidence_tables import _fixed_layout_row

    assert _fixed_layout_row("DINOv2-L + SAM-H / 20260716", None) == (
        "| DINOv2-L + SAM-H / 20260716 | pending | pending | pending |"
    )


def test_fixed_layout_row_renders_completed_stage_metrics() -> None:
    from scripts.build_evidence_tables import _fixed_layout_row

    rendered = _fixed_layout_row(
        "DINOv2-L + SAM-H / 20260716",
        {"top5_n3": 0.4, "spread5_n3": 0.43},
    )

    assert rendered == "| DINOv2-L + SAM-H / 20260716 | 0.4000 | 0.4300 | +0.0300 |"
