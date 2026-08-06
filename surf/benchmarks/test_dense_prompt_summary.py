from run_monuseg_dense_prompt_experiment import summarize


def _row(k: object, lambda_bg: object, dice: float) -> dict[str, object]:
    metrics = {
        "dice": dice,
        "iou": dice,
        "fg_precision": dice,
        "fg_recall": dice,
        "object_recall": dice,
        "object_precision": dice,
        "component_count_error": dice,
        "merge_error": dice,
        "sam_score": dice,
        "pred_area_pct": dice,
    }
    return {"k": k, "lambda_bg": lambda_bg, **metrics}


def test_summarize_merges_resumed_csv_numeric_strings_with_new_numeric_rows() -> None:
    rows = [
        _row("40", "0.75", 0.4),
        _row(40, 0.75, 0.6),
    ]

    summary = summarize(rows, ["k", "lambda_bg"])

    assert len(summary) == 1
    assert summary[0]["k"] == 40
    assert summary[0]["lambda_bg"] == 0.75
    assert summary[0]["mean_dice"] == 0.5
    assert summary[0]["case_count"] == 2
