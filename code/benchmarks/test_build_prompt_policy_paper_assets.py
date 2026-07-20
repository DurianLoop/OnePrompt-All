from build_prompt_policy_paper_assets import (
    find_row,
    five_reference_plot_payload,
    optional_float,
    paired_effect_tuple,
)


def test_find_row_requires_all_requested_fields() -> None:
    rows = [
        {"split": "test", "method": "fixed", "mean_dice": "0.5"},
        {"split": "test", "method": "learned", "mean_dice": "0.6"},
    ]

    assert find_row(rows, split="test", method="learned")["mean_dice"] == "0.6"


def test_optional_float_returns_blank_for_unavailable_metric() -> None:
    assert optional_float({"mean_dice": "0.5"}, "mean_merge_error") == ""


def test_paired_effect_tuple_uses_query_level_summary_fields() -> None:
    summary = {
        "mean_delta_dice": 0.00875,
        "ci95_low": 0.003765,
        "ci95_high": 0.015351,
    }

    assert paired_effect_tuple("V7 5-ref", summary) == (
        "V7 5-ref",
        0.00875,
        0.003765,
        0.015351,
    )


def test_five_reference_plot_payload_includes_negative_budget_gate() -> None:
    rows = [
        {"method": "Fixed action (5-ref mean)", "mean_dice": 0.491946},
        {"method": "Anchored V7 (5-ref frozen)", "mean_dice": 0.500696},
        {"method": "Learned N-budget (5-ref frozen)", "mean_dice": 0.518099},
    ]

    labels, values = five_reference_plot_payload(rows)

    assert labels == ["Fixed", "Frozen V7", "N-gate"]
    assert values == [0.491946, 0.500696, 0.518099]
