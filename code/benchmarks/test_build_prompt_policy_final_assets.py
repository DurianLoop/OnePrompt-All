from build_prompt_policy_final_assets import effect_row, fixed_budget_mean, mean_column


def test_effect_row_preserves_paired_statistics() -> None:
    summary = {
        "case_count": 37,
        "mean_fixed_dice": 0.53,
        "mean_learned_dice": 0.55,
        "mean_delta_dice": 0.02,
        "ci95_low": 0.01,
        "ci95_high": 0.03,
        "wins": 20,
        "ties": 10,
        "losses": 7,
    }

    row = effect_row("MoNuSeg", "nested_cv", "learned_vs_fixed", summary)

    assert row["dataset"] == "MoNuSeg"
    assert row["n"] == 37
    assert row["delta_dice"] == 0.02
    assert row["wins_ties_losses"] == "20/10/7"


def test_mean_column_averages_numeric_csv_values() -> None:
    rows = [{"oracle_dice": "0.5"}, {"oracle_dice": "0.7"}]

    assert mean_column(rows, "oracle_dice") == 0.6


def test_fixed_budget_mean_accepts_requested_and_legacy_budget_columns() -> None:
    assert fixed_budget_mean([{"requested_negative_count": "5", "mean_dice": "0.6"}], 5) == 0.6
    assert fixed_budget_mean([{"negative_count": "5", "mean_dice": "0.7"}], 5) == 0.7
