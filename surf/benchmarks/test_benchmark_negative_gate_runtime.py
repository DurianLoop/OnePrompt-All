from benchmark_negative_gate_runtime import summarize_durations


def test_summarize_durations_reports_mean_median_and_total() -> None:
    summary = summarize_durations([0.1, 0.3, 0.2])

    assert summary["repeat_count"] == 3
    assert summary["mean_seconds"] == 0.2
    assert summary["median_seconds"] == 0.2
    assert summary["total_seconds"] == 0.6
