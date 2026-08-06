import io

import numpy as np
from PIL import Image

from export_monusac_dataset import union_masks


def test_union_masks_combines_instances_and_class_masks() -> None:
    left = np.zeros((3, 3), dtype=np.uint8)
    left[0, 0] = 255
    right = np.zeros((3, 3), dtype=np.uint8)
    right[1, 1] = 255
    encoded = []
    for mask in (left, right):
        buffer = io.BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        encoded.append(buffer.getvalue())

    all_mask, class_masks = union_masks(encoded, [1, 2])

    assert int(all_mask.sum()) == 2
    assert int(class_masks[1].sum()) == 1
    assert int(class_masks[2].sum()) == 1
