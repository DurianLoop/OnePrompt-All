from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "nnunet_train_phase3_fold0.ps1"
PREDICT = ROOT / "scripts" / "nnunet_predict_phase3_fold0.ps1"
TRAIN_250 = ROOT / "scripts" / "nnunet_train_phase3_fold0_250epochs.ps1"
PREDICT_250 = ROOT / "scripts" / "nnunet_predict_phase3_fold0_250epochs.ps1"


def main() -> None:
    train_text = TRAIN.read_text(encoding="utf-8")
    predict_text = PREDICT.read_text(encoding="utf-8")
    train_250_text = TRAIN_250.read_text(encoding="utf-8")
    predict_250_text = PREDICT_250.read_text(encoding="utf-8")

    assert "-tr nnUNetTrainer_100epochs" in train_text, "training script must default to 100-epoch sanity trainer"
    assert "-chk checkpoint_best.pth" in predict_text, "prediction script must evaluate checkpoint_best.pth"
    assert "-npp 1" in predict_text and "-nps 1" in predict_text, "prediction script should keep Windows-safe single-process inference"
    assert "-tr nnUNetTrainer_250epochs" in train_250_text, "250-epoch training script must use matching trainer"
    assert "-tr nnUNetTrainer_250epochs" in predict_250_text, "250-epoch prediction script must use matching trainer"
    assert "-chk checkpoint_best.pth" in predict_250_text, "250-epoch prediction script must evaluate checkpoint_best.pth"
    print("nnU-Net Phase3 scripts ok")


if __name__ == "__main__":
    main()
