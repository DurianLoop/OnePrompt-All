from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "runs" / "nnunet" / "nnUNet_raw" / "Dataset902_OCSAMPhase3Stratified"
TRAIN = ROOT / "scripts" / "nnunet_train_phase3_stratified_fold0_250epochs.ps1"
PREDICT = ROOT / "scripts" / "nnunet_predict_phase3_stratified_fold0_250epochs.ps1"


def main() -> None:
    dataset_json = json.loads((RAW / "dataset.json").read_text(encoding="utf-8"))
    split = json.loads((RAW / "ocsam_split_manifest.json").read_text(encoding="utf-8"))
    counts = Counter((row["dataset"], row["split"]) for row in split)

    assert dataset_json["name"] == "OCSAMPhase3Stratified"
    assert dataset_json["numTraining"] == 80
    assert counts[("IDRID", "train")] == 40
    assert counts[("IDRID", "test")] == 10
    assert counts[("MonuSeg", "train")] == 40
    assert counts[("MonuSeg", "test")] == 10
    assert len(list((RAW / "imagesTr").glob("*_0000.png"))) == 80
    assert len(list((RAW / "labelsTr").glob("*.png"))) == 80
    assert len(list((RAW / "imagesTs").glob("*_0000.png"))) == 20
    assert len(list((RAW / "labelsTs").glob("*.png"))) == 20

    train_text = TRAIN.read_text(encoding="utf-8")
    predict_text = PREDICT.read_text(encoding="utf-8")
    assert "901" not in train_text and "Dataset901_OCSAMPhase3" not in predict_text
    assert "902" in train_text and "Dataset902_OCSAMPhase3Stratified" in predict_text
    assert "-tr nnUNetTrainer_250epochs" in train_text
    assert "-tr nnUNetTrainer_250epochs" in predict_text
    assert "-chk checkpoint_best.pth" in predict_text
    print("nnU-Net stratified Phase3 setup ok")


if __name__ == "__main__":
    main()
