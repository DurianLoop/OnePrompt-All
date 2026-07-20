from pathlib import Path

from fspl.models import dino_architecture_from_checkpoint, sam_architecture_from_checkpoint


def test_dino_architecture_is_inferred_from_local_checkpoint_name() -> None:
    assert dino_architecture_from_checkpoint(Path("dinov2_vitb14_pretrain.pth")) == "base"
    assert dino_architecture_from_checkpoint(Path("dinov2_vitl14_pretrain.pth")) == "large"


def test_sam_architecture_is_inferred_from_local_checkpoint_name() -> None:
    assert sam_architecture_from_checkpoint(Path("sam_vit_b.pth")) == "vit_b"
    assert sam_architecture_from_checkpoint(Path("sam_vit_l_0b3195.pth")) == "vit_l"
    assert sam_architecture_from_checkpoint(Path("sam_vit_h_4b8939.pth")) == "vit_h"
