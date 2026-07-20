from pathlib import Path

import pytest

from fspl.dataset import CaseRecord, build_smoke_episode, discover_cases, discover_idrid_cases


TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "test-data"


def make_case_tree(name: str) -> Path:
    root = TEST_DATA_ROOT / name
    (root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (root / "train" / "masks").mkdir(parents=True, exist_ok=True)
    return root


def test_discover_cases_returns_stem_matched_records_in_sorted_order() -> None:
    root = make_case_tree("paired-cases")
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    (image_dir / "case_b.tif").touch()
    (image_dir / "case_a.tif").touch()
    (mask_dir / "case_b.png").touch()
    (mask_dir / "case_a.png").touch()

    cases = discover_cases(root, "train")

    assert [case.case_id for case in cases] == ["case_a", "case_b"]
    assert all(case.split == "train" for case in cases)
    assert cases[0].image_path.name == "case_a.tif"
    assert cases[0].mask_path.name == "case_a.png"


def test_discover_cases_rejects_an_image_without_a_matching_mask() -> None:
    root = make_case_tree("missing-mask")
    image_dir = root / "train" / "images"
    (image_dir / "missing_mask.tif").touch()

    with pytest.raises(ValueError, match="missing mask"):
        discover_cases(root, "train")


def test_build_smoke_episode_uses_distinct_first_two_sorted_cases() -> None:
    cases = [
        CaseRecord("case_a", Path("a.tif"), Path("a.png"), "train"),
        CaseRecord("case_b", Path("b.tif"), Path("b.png"), "train"),
    ]

    episode = build_smoke_episode(cases)

    assert episode.reference.case_id == "case_a"
    assert episode.query.case_id == "case_b"
    assert episode.reference.case_id != episode.query.case_id


def test_discover_idrid_cases_pairs_native_image_and_mask_directories() -> None:
    root = TEST_DATA_ROOT / "IDRID"
    image_dir = root / "DR_Training_Set" / "Fundus Images"
    mask_dir = root / "DR_Training_Set" / "Combined Masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "IDRiD_02.jpg").touch()
    (image_dir / "IDRiD_01.jpg").touch()
    (image_dir / "IDRiD_14_sam_result.jpg").touch()
    (mask_dir / "IDRiD_01.png").touch()
    (mask_dir / "IDRiD_02.png").touch()

    cases = discover_idrid_cases(root, "train")

    assert [case.case_id for case in cases] == ["IDRiD_01", "IDRiD_02"]
    assert all(case.split == "train" for case in cases)
