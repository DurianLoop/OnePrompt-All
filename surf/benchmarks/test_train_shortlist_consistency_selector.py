from train_shortlist_consistency_selector import (
    CONSISTENCY_FEATURE_NAMES,
    base_selector_features,
    consistency_selector_features,
    normalize_shortlist_rows,
)


def test_normalize_shortlist_rows_restores_candidate_method() -> None:
    rows = [{"episode_id": "e1", "action": "contrast_topk_K40_lam0.75", "method": ""}]

    normalized = normalize_shortlist_rows(rows)

    assert normalized[0]["method"] == "contrast_topk_K40_lam0.75"
    assert normalized[0]["action"] == "contrast_topk_K40_lam0.75"


def test_consistency_ablation_changes_only_mask_matcher_features() -> None:
    base = base_selector_features()
    full = consistency_selector_features()

    assert set(base).isdisjoint(CONSISTENCY_FEATURE_NAMES)
    assert set(full) == set(base) | set(CONSISTENCY_FEATURE_NAMES)
    assert len(full) > len(base)
