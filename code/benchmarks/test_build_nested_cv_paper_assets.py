import pytest

from build_nested_cv_paper_assets import prepare_fold_rows


def test_prepare_fold_rows_sorts_folds_and_preserves_effect_interval() -> None:
    rows = [
        {
            "fold": "1",
            "case_count": "7",
            "mean_fixed_dice": "0.50",
            "mean_learned_dice": "0.52",
            "mean_delta_dice": "0.02",
            "ci95_low": "0.01",
            "ci95_high": "0.03",
            "wins": "4",
            "ties": "2",
            "losses": "1",
        },
        {
            "fold": "0",
            "case_count": "8",
            "mean_fixed_dice": "0.49",
            "mean_learned_dice": "0.51",
            "mean_delta_dice": "0.02",
            "ci95_low": "-0.01",
            "ci95_high": "0.05",
            "wins": "3",
            "ties": "5",
            "losses": "0",
        },
    ]

    prepared = prepare_fold_rows(rows)

    assert [row["fold"] for row in prepared] == [0, 1]
    assert prepared[0]["case_count"] == 8
    assert prepared[0]["ci95_low"] == pytest.approx(-0.01)
    assert prepared[1]["mean_learned_dice"] == pytest.approx(0.52)


def test_prepare_fold_rows_rejects_duplicate_folds() -> None:
    row = {
        "fold": "0",
        "case_count": "8",
        "mean_fixed_dice": "0.49",
        "mean_learned_dice": "0.51",
        "mean_delta_dice": "0.02",
        "ci95_low": "-0.01",
        "ci95_high": "0.05",
        "wins": "3",
        "ties": "5",
        "losses": "0",
    }

    with pytest.raises(ValueError, match="Duplicate fold"):
        prepare_fold_rows([row, dict(row)])
