import numpy as np
import pytest
import torch

from evaluate_prompt_execution_modes import keep_prompt_component, select_best_multimasks, union_prompt_components


def test_keep_prompt_component_removes_disconnected_regions() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:3, 1:3] = True
    mask[4:6, 4:6] = True

    kept = keep_prompt_component(mask, x=1, y=1)

    assert kept[1:3, 1:3].all()
    assert not kept[4:6, 4:6].any()


def test_union_prompt_components_keeps_one_local_region_per_prompt() -> None:
    first = np.zeros((7, 7), dtype=bool)
    first[1:3, 1:3] = True
    first[4:6, 4:6] = True
    second = first.copy()

    union = union_prompt_components([first, second], [(1, 1), (5, 5)])

    assert union[1:3, 1:3].all()
    assert union[4:6, 4:6].all()


def test_select_best_multimasks_preserves_batch_height_width() -> None:
    masks = torch.zeros((2, 3, 4, 5), dtype=torch.bool)
    masks[0, 2, 1, 1] = True
    masks[1, 1, 2, 2] = True
    scores = torch.tensor([[0.1, 0.2, 0.9], [0.1, 0.8, 0.2]])

    selected, selected_scores = select_best_multimasks(masks, scores)

    assert selected.shape == (2, 4, 5)
    assert selected[0, 1, 1]
    assert selected[1, 2, 2]
    assert selected_scores.tolist() == pytest.approx([0.9, 0.8])
