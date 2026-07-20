from fspl.protocol import FixedRuleConfig
from fspl.stage2 import BaselineRow, aggregate_rule_metrics, select_rule_from_rows


def test_validation_selection_uses_only_validation_rows() -> None:
    rules = (
        FixedRuleConfig("top1_n0", 1, 0),
        FixedRuleConfig("top5_n3", 5, 3),
    )
    rows = [
        BaselineRow("val_a", "validation", "top1_n0", 0.5, 0.1, 1, 0),
        BaselineRow("val_a", "validation", "top5_n3", 0.7, 0.1, 5, 3),
        BaselineRow("test_a", "test", "top1_n0", 0.99, 0.1, 1, 0),
        BaselineRow("test_a", "test", "top5_n3", 0.01, 0.1, 5, 3),
    ]

    selection = select_rule_from_rows(rows, rules)

    assert selection.selected.name == "top5_n3"
    assert aggregate_rule_metrics(rows, split="test") == {
        "top1_n0": {"case_count": 1, "mean_dice": 0.99},
        "top5_n3": {"case_count": 1, "mean_dice": 0.01},
    }
