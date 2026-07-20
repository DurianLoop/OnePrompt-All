import pytest

from analyze_all_nuclei_utility import compare_oracles
from all_nuclei_utility import AllNucleiWeights


def _row(episode: str, action: str, dice: float, recall: float, merge: float) -> dict[str, str]:
    return {
        "episode_id": episode,
        "query_case_id": episode,
        "method": action,
        "dice": str(dice),
        "object_recall": str(recall),
        "merge_error": str(merge),
        "fg_precision": "0.8",
        "fg_recall": "0.8",
    }


def test_compare_oracles_reports_action_disagreement_and_metric_tradeoff() -> None:
    rows = [
        _row("a", "dice_best", 0.80, 0.40, 0.40),
        _row("a", "utility_best", 0.75, 0.90, 0.00),
        _row("b", "same", 0.70, 0.80, 0.10),
        _row("b", "worse", 0.50, 0.20, 0.50),
    ]
    weights = AllNucleiWeights(object_recall=0.5, area_error=0.0, merge_error=0.5)

    per_episode, summary = compare_oracles(rows, weights)

    assert summary["episode_count"] == 2
    assert summary["oracle_action_agreement"] == pytest.approx(0.5)
    assert summary["mean_utility_oracle_object_recall"] > summary["mean_dice_oracle_object_recall"]
    assert {row["episode_id"]: row["same_action"] for row in per_episode} == {"a": 0, "b": 1}
