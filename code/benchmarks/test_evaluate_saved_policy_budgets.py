from evaluate_saved_policy_budgets import choose_validation_budget


def test_choose_validation_budget_prefers_higher_dice_then_smaller_budget() -> None:
    rows = [
        {"candidate_budget": 3, "mean_dice": 0.52},
        {"candidate_budget": 5, "mean_dice": 0.53},
        {"candidate_budget": 7, "mean_dice": 0.53},
    ]

    assert choose_validation_budget(rows)["candidate_budget"] == 5
