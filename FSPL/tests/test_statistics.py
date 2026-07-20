from __future__ import annotations

import pytest


def test_paired_rule_effect_reports_mean_ci_and_wins_without_reordering_cases() -> None:
    from fspl.statistics import paired_rule_effect

    rows = [
        {"case_id": "episode_b", "rule_name": "top5_n3", "utility": 0.4},
        {"case_id": "episode_a", "rule_name": "spread5_n3", "utility": 0.6},
        {"case_id": "episode_c", "rule_name": "top5_n3", "utility": 0.2},
        {"case_id": "episode_b", "rule_name": "spread5_n3", "utility": 0.3},
        {"case_id": "episode_a", "rule_name": "top5_n3", "utility": 0.5},
        {"case_id": "episode_c", "rule_name": "spread5_n3", "utility": 0.2},
    ]

    result = paired_rule_effect(rows, baseline_rule="top5_n3", variant_rule="spread5_n3", seed=11, resamples=1000)

    assert result["episode_count"] == 3
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["standard_error"] == pytest.approx(0.0577350269)
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert result["bootstrap_95_ci"] == pytest.approx([-0.1, 0.1])


def test_paired_rule_effect_rejects_missing_or_duplicate_actions() -> None:
    from fspl.statistics import paired_rule_effect

    with pytest.raises(ValueError, match="exactly one"):
        paired_rule_effect(
            [
                {"case_id": "episode", "rule_name": "top5_n3", "utility": 0.1},
                {"case_id": "episode", "rule_name": "top5_n3", "utility": 0.2},
            ],
            baseline_rule="top5_n3",
            variant_rule="spread5_n3",
        )
