from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def select_policy_examples(
    negative_rows: list[dict[str, Any]],
    learned_rows: list[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_query: dict[str, dict[int, dict[str, Any]]] = {}
    for row in negative_rows:
        by_query.setdefault(str(row["query_case_id"]), {})[int(row["negative_count"])] = row
    selected_by_query = {
        str(row["query_case_id"]): str(row.get("selected_action") or row.get("action"))
        for row in learned_rows
    }
    comparisons: list[dict[str, Any]] = []
    for query_id, settings in by_query.items():
        if 0 not in settings or 5 not in settings or query_id not in selected_by_query:
            continue
        gain = float(settings[5]["dice"]) - float(settings[0]["dice"])
        comparisons.append({
            "query_case_id": query_id,
            "selected_action": selected_by_query[query_id],
            "n0": settings[0],
            "n5": settings[5],
            "n5_delta": gain,
        })
    switched = sorted(
        [row for row in comparisons if row["selected_action"] == "neg5"],
        key=lambda row: (-float(row["n5_delta"]), str(row["query_case_id"])),
    )[:count]
    avoided = sorted(
        [row for row in comparisons if row["selected_action"] == "neg0"],
        key=lambda row: (float(row["n5_delta"]), str(row["query_case_id"])),
    )[:count]
    return switched, avoided


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize learned negative-budget switches and avoided N5 failures.")
    parser.add_argument(
        "--negative-rows",
        default="runs/research_direction/mechanisms/negative_prompt_ablation_test14/per_case.csv",
    )
    parser.add_argument(
        "--learned",
        default="runs/research_direction/prompt_policy/adaptive_negative_budget_v1/test_learned_predictions.csv",
    )
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument(
        "--out",
        default="runs/research_direction/paper_artifacts_20260715/negative_policy_qualitative.png",
    )
    args = parser.parse_args()

    switched, avoided = select_policy_examples(
        read_csv(resolve(args.negative_rows)),
        read_csv(resolve(args.learned)),
        count=args.count,
    )
    examples = [("Gate accepts N=5", row) for row in switched] + [
        ("Gate keeps N=0", row) for row in avoided
    ]
    if not examples:
        raise ValueError("No policy examples are available.")
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 4.7 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    for row_axes, (decision, example) in zip(axes, examples):
        for axis, setting, label in zip(row_axes, (example["n0"], example["n5"]), ("N=0", "N=5")):
            axis.imshow(Image.open(resolve(str(setting["overlay_path"]))).convert("RGB"))
            axis.set_title(f"{label}: Dice {float(setting['dice']):.3f}")
            axis.axis("off")
        row_axes[0].text(
            0.02,
            0.98,
            f"{decision}\n{example['query_case_id']}\nN5-N0 {float(example['n5_delta']):+.3f}",
            transform=row_axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#333333", "alpha": 0.9, "pad": 5},
        )
    fig.suptitle(
        "Learned negative-point gate: accepted corrections and avoided failures\n"
        "Green: TP   Red: FP   Blue: FN   Yellow dots: positive prompts   Cyan crosses: negative prompts",
        fontsize=15,
        linespacing=1.45,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white", pil_kwargs={"compress_level": 9})
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
