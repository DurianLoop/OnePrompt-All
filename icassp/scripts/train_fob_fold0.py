"""Train the adapted FoB policy on a frozen MoNuSeg nested fold.

The ``--fold`` argument selects one frozen split and this script intentionally trains only `episodes_train.csv`.  Checkpoints are
selection candidates; validation and outer-test decoding are separate scripts
so no held-out label can influence optimization.
"""
from __future__ import annotations

import csv
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2  # keep OpenCV cached before importing the FoB snapshot
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FOB_ROOT = ROOT / "baselines" / "fob_source"
DATA = ROOT / "data" / "monuseg_hf" / "materialized_train"
SPLIT = WORKSPACE / "runs" / "research_direction" / "prompt_policy_splits" / "monuseg_nested_cv5" / "fold_0" / "episodes_train.csv"
RESNET = ROOT / "models" / "deeplabv3_resnet101_coco-586e9e4e.pth"
RUN = ROOT / "runs" / "fob_monuseg_fold0"
STEPS = 1000
CHECKPOINT_STEPS = {250, 500, 750, 1000}
SEED = 20260809


def tensor(path: Path, *, mask: bool) -> torch.Tensor:
    with Image.open(path) as image:
        if mask:
            return torch.from_numpy((np.asarray(image.convert("L")) > 0).astype(np.float32)).unsqueeze(0)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    value = torch.from_numpy(array).permute(2, 0, 1)
    return ((value - value.mean(dim=(1, 2), keepdim=True)) /
            value.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)).unsqueeze(0)


def load_episode(row: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support, query = row["support_case_id"], row["query_case_id"]
    if support == query:
        raise RuntimeError("support/query self-pair")
    paths = [
        DATA / "images" / f"{support}.png", DATA / "masks" / f"{support}.png",
        DATA / "images" / f"{query}.png", DATA / "masks" / f"{query}.png",
    ]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(paths)
    return tensor(paths[0], mask=False), tensor(paths[1], mask=True), tensor(paths[2], mask=False), tensor(paths[3], mask=True).long()


def write_json(name: str, payload: dict[str, object]) -> None:
    (RUN / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global SPLIT, RUN
    parser = argparse.ArgumentParser(description="Train the frozen-protocol FoB diagnostic on one nested MoNuSeg fold.")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    args = parser.parse_args()
    SPLIT = WORKSPACE / "runs" / "research_direction" / "prompt_policy_splits" / "monuseg_nested_cv5" / f"fold_{args.fold}" / "episodes_train.csv"
    RUN = ROOT / "runs" / f"fob_monuseg_fold{args.fold}"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["FOB_RESNET101_COCO"] = str(RESNET)
    sys.path.insert(0, str(FOB_ROOT))
    if not torch.cuda.is_available():
        raise RuntimeError("FoB fold training requires CUDA")
    RUN.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS = RUN / "checkpoints"
    CHECKPOINTS.mkdir(exist_ok=True)
    with SPLIT.open(encoding="utf-8", newline="") as handle:
        episodes = list(csv.DictReader(handle))
    expected_train = 115 if args.fold < 2 else 120
    if len(episodes) != expected_train or any(row["role"] != "train" for row in episodes):
        raise RuntimeError(f"fold-{args.fold} train split is not the expected {expected_train} train-only episodes")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    from models.FoB import FewShotSeg
    model = FewShotSeg({}).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng = random.Random(SEED)
    order = list(range(len(episodes)))
    rng.shuffle(order)
    started = time.time()
    metadata = {
        "status": "RUNNING", "fold": args.fold, "steps": STEPS, "seed": SEED,
        "train_episode_count": len(episodes), "split": str(SPLIT),
        "data_root": str(DATA), "normalization": "per-image RGB channel z-score; no augmentation",
        "checkpoint_steps": sorted(CHECKPOINT_STEPS),
        "note": "validation and outer test are not read by this training process",
    }
    write_json("run_metadata.json", metadata)
    history_path = RUN / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "episode_id", "prompt_loss", "contrastive_loss", "foreground_loss", "total_loss", "seconds"])
        writer.writeheader()
        for step in range(1, STEPS + 1):
            if (step - 1) % len(order) == 0 and step > 1:
                rng.shuffle(order)
            row = episodes[order[(step - 1) % len(order)]]
            supp_img, supp_mask, qry_img, qry_mask = [x.cuda(non_blocking=True) for x in load_episode(row)]
            prompt, contrastive, foreground = model([[supp_img]], [[supp_mask]], [qry_img], qry_mask, True)
            total = prompt + contrastive + foreground
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            record = {
                "step": step, "episode_id": row["pilot_id"],
                "prompt_loss": float(prompt.detach().cpu()), "contrastive_loss": float(contrastive.detach().cpu()),
                "foreground_loss": float(foreground.detach().cpu()), "total_loss": float(total.detach().cpu()),
                "seconds": time.time() - started,
            }
            writer.writerow(record); handle.flush()
            if step % 25 == 0:
                write_json("status.json", {**metadata, "status": "RUNNING", "last": record, "elapsed_seconds": time.time() - started})
            if step in CHECKPOINT_STEPS:
                torch.save({"step": step, "seed": SEED, "model_state_dict": model.state_dict()}, CHECKPOINTS / f"step_{step:04d}.pth")
                write_json("status.json", {**metadata, "status": "RUNNING", "last": record, "checkpoint": f"checkpoints/step_{step:04d}.pth", "elapsed_seconds": time.time() - started})
    torch.cuda.synchronize()
    final = {**metadata, "status": "TRAINING_COMPLETE", "elapsed_seconds": time.time() - started, "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024)}
    write_json("status.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
