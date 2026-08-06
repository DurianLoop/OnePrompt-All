from run_monuseg_dense_prompt_experiment import (
    cached_value,
    completed_case_names,
    filter_manifest_rows,
    merge_unique_candidate_rows,
    requested_rounds,
    runtime_metrics,
    select_round3_config,
    should_refresh_query_embedding,
    visual_artifacts_enabled,
)


def test_filter_manifest_rows_accepts_requested_dataset_case_insensitively() -> None:
    rows = [{"dataset": "MonuSeg"}, {"dataset": "MoNuSAC"}]

    selected = filter_manifest_rows(rows, "monusac")

    assert selected == [{"dataset": "MoNuSAC"}]


def test_filter_manifest_rows_can_select_one_policy_role() -> None:
    rows = [
        {"dataset": "MoNuSeg", "role": "train", "query_case_id": "train_case"},
        {"dataset": "MoNuSeg", "role": "val", "query_case_id": "val_case"},
        {"dataset": "MoNuSeg", "role": "test", "query_case_id": "test_case"},
    ]

    selected = filter_manifest_rows(rows, "monuseg", "train")

    assert selected == [{"dataset": "MoNuSeg", "role": "train", "query_case_id": "train_case"}]


def test_select_round3_config_uses_validation_frozen_values_when_supplied() -> None:
    round1 = [{"k": 20, "mean_dice": 0.5}, {"k": 40, "mean_dice": 0.6}]
    round2 = [{"k": 40, "lambda_bg": 0.25, "mean_dice": 0.6}]

    config = select_round3_config(round1, round2, fixed_k=75, fixed_lambda=0.75)

    assert config == (75, 0.75, "external_fixed")


def test_query_embedding_refreshes_only_when_query_case_changes() -> None:
    assert should_refresh_query_embedding(None, "query_a") is True
    assert should_refresh_query_embedding("query_a", "query_a") is False
    assert should_refresh_query_embedding("query_a", "query_b") is True


def test_cached_value_computes_each_image_feature_once() -> None:
    cache = {}
    calls = []

    first = cached_value(cache, "image_a", lambda: calls.append("computed") or 42)
    second = cached_value(cache, "image_a", lambda: calls.append("computed_again") or 99)

    assert first == 42
    assert second == 42
    assert calls == ["computed"]


def test_minimal_artifacts_and_skip_round3_define_fast_training_mode() -> None:
    assert visual_artifacts_enabled(minimal_artifacts=True) is False
    assert requested_rounds(skip_round3=True) == ("round1", "round2")
    assert requested_rounds(skip_round3=False) == ("round1", "round2", "round3")


def test_completed_case_names_requires_the_full_round_action_count() -> None:
    rows = [
        {"case_name": "a", "round_name": "round2", "method": "m1"},
        {"case_name": "a", "round_name": "round2", "method": "m2"},
        {"case_name": "b", "round_name": "round2", "method": "m1"},
    ]

    assert completed_case_names(rows, "round2", expected_candidates=2) == {"a"}


def test_merge_unique_candidate_rows_replaces_duplicate_checkpoint_key() -> None:
    old = [{"case_name": "a", "round_name": "round2", "method": "m1", "dice": 0.1}]
    new = [{"case_name": "a", "round_name": "round2", "method": "m1", "dice": 0.2}]

    merged = merge_unique_candidate_rows(old, new)

    assert len(merged) == 1
    assert merged[0]["dice"] == 0.2


def test_runtime_metrics_reports_reproducible_throughput_units() -> None:
    metrics = runtime_metrics(elapsed_seconds=20.0, candidate_count=40, case_count=5, peak_gpu_mb=123.0)

    assert metrics["elapsed_seconds"] == 20.0
    assert metrics["seconds_per_candidate"] == 0.5
    assert metrics["seconds_per_episode"] == 4.0
    assert metrics["peak_gpu_memory_mb"] == 123.0
