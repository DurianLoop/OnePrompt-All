import numpy as np

from train_pairwise_prompt_ranker import (
    build_pairwise_examples,
    score_candidates_pairwise,
    train_pairwise_ranker,
)


def _row(action: str, value: float, dice: float) -> dict[str, float | str]:
    return {"episode_id": "episode_a", "action": action, "feature": value, "dice": dice}


def test_build_pairwise_examples_balances_forward_and_reverse_comparisons() -> None:
    rows = [_row("low", 0.0, 0.2), _row("mid", 1.0, 0.5), _row("high", 2.0, 0.8)]

    x, y = build_pairwise_examples(rows, ["feature"], target_key="dice")

    assert x.shape == (6, 1)
    assert y.shape == (6,)
    assert y.sum() == 3
    assert sorted(x[:, 0].tolist()) == [-2.0, -1.0, -1.0, 1.0, 1.0, 2.0]


def test_pairwise_ranker_scores_higher_quality_candidate_above_lower_one() -> None:
    train_rows = []
    for episode in ("a", "b", "c"):
        train_rows.extend([
            {"episode_id": episode, "action": "low", "feature": 0.0, "dice": 0.2},
            {"episode_id": episode, "action": "high", "feature": 2.0, "dice": 0.8},
        ])
    model = train_pairwise_ranker(train_rows, ["feature"], target_key="dice", c=1.0)

    scored = score_candidates_pairwise([
        {"episode_id": "test", "action": "low", "feature": 0.1},
        {"episode_id": "test", "action": "high", "feature": 1.9},
    ], model)

    scores = {row["action"]: row["predicted_pairwise_score"] for row in scored}
    assert scores["high"] > scores["low"]
