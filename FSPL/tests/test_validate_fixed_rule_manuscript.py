from __future__ import annotations


def _replicates() -> dict[str, dict[str, float]]:
    return {
        "dinov2_l_sam_b_seed20260716": {
            "top5_n3_mean_dice": 0.4196113079050684,
            "spread5_n3_mean_dice": 0.4526176561123518,
            "mean_delta": 0.033006348207283376,
        },
        "dinov2_l_sam_b_seed20260717": {
            "top5_n3_mean_dice": 0.400792706067821,
            "spread5_n3_mean_dice": 0.4362002476210994,
            "mean_delta": 0.035407541553278264,
        },
        "dinov2_b_sam_b_seed20260716": {
            "top5_n3_mean_dice": 0.3835448092126562,
            "spread5_n3_mean_dice": 0.4343868594983669,
            "mean_delta": 0.050842050285710705,
        },
        "dinov2_l_sam_l_seed20260716": {
            "top5_n3_mean_dice": 0.4507474131934137,
            "spread5_n3_mean_dice": 0.4659352047998432,
            "mean_delta": 0.015187791606429532,
        },
        "dinov2_l_sam_h_seed20260716": {
            "top5_n3_mean_dice": 0.4289334891379113,
            "spread5_n3_mean_dice": 0.4438820252302129,
            "mean_delta": 0.014948536092301628,
        },
    }


def _manuscript(samh_gain: str = "+0.0149") -> str:
    return "\n".join(
        [
            "| Combination / reference seed | Top-5 + 3 neg. | Spread r=3 + 3 neg. | Gain |",
            "|---|---:|---:|---:|",
            "| DINOv2-L + SAM-B / 20260716 | 0.4196 | 0.4526 | +0.0330 |",
            "| DINOv2-L + SAM-B / 20260717 | 0.4008 | 0.4362 | +0.0354 |",
            "| DINOv2-B + SAM-B / 20260716 | 0.3835 | 0.4344 | +0.0508 |",
            "| DINOv2-L + SAM-L / 20260716 | 0.4507 | 0.4659 | +0.0152 |",
            f"| DINOv2-L + SAM-H / 20260716 | 0.4289 | 0.4439 | {samh_gain} |",
        ]
    )


def test_validate_replication_table_accepts_artifact_rounded_rows() -> None:
    from scripts.validate_fixed_rule_manuscript import validate_replication_table

    assert validate_replication_table(_manuscript(), _replicates()) == []


def test_validate_replication_table_rejects_stale_samh_gain() -> None:
    from scripts.validate_fixed_rule_manuscript import validate_replication_table

    errors = validate_replication_table(_manuscript("+0.0200"), _replicates())

    assert any("DINOv2-L + SAM-H / 20260716" in error and "gain" in error for error in errors)
