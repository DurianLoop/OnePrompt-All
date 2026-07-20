from pathlib import Path


def test_renderer_parses_locked_manifest_and_cpu_defaults() -> None:
    from scripts.render_stage17_qualitative import parse_args

    args = parse_args(
        [
            "--panel-manifest",
            "artifacts/panels.json",
            "--output-dir",
            "artifacts/rendered",
        ]
    )

    assert args.panel_manifest == Path("artifacts/panels.json")
    assert args.output_dir == Path("artifacts/rendered")
    assert args.device == "cpu"
    assert args.dino_checkpoint.name == "dinov2_vitl14_pretrain.pth"
    assert args.sam_checkpoint.name == "sam_vit_b.pth"
