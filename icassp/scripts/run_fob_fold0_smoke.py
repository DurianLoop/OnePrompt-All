"""Run one real FoB optimization step on a train-only MoNuSeg fold-0 episode.

This is deliberately a feasibility smoke, not a reported baseline result.  It
uses only an episode designated `train` by the pre-existing nested protocol and
records enough provenance to prove no validation/outer-test target was used.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FOB_ROOT = ROOT / "baselines" / "fob_source"
DATA = ROOT / "data" / "monuseg_hf" / "materialized_train"
FOLD_TRAIN = WORKSPACE / "runs" / "research_direction" / "prompt_policy_splits" / "monuseg_nested_cv5" / "fold_0" / "episodes_train.csv"
WEIGHTS = ROOT / "models" / "deeplabv3_resnet101_coco-586e9e4e.pth"
OUTPUT = ROOT / "evidence" / "fob_fold0_train_smoke.json"


def load_tensor(path: Path, *, mask: bool) -> torch.Tensor:
    with Image.open(path) as image:
        if mask:
            values = np.asarray(image.convert("L"), dtype=np.float32) > 0
            return torch.from_numpy(values.astype(np.float32)).unsqueeze(0)
        values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1)
    # Per-image channel normalization is a deterministic smoke preprocessing;
    # the full baseline must pre-register its augmentation/normalization policy.
    return (tensor - tensor.mean(dim=(1, 2), keepdim=True)) / (tensor.std(dim=(1, 2), keepdim=True).clamp_min(1e-6))


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["FOB_RESNET101_COCO"] = str(WEIGHTS)
    sys.path.insert(0, str(FOB_ROOT))
    if not torch.cuda.is_available():
        raise RuntimeError("FoB smoke requires CUDA")
    if not WEIGHTS.is_file() or not FOLD_TRAIN.is_file():
        raise FileNotFoundError("missing FoB weights or frozen fold-0 train manifest")
    random.seed(20260809)
    np.random.seed(20260809)
    torch.manual_seed(20260809)
    torch.cuda.manual_seed_all(20260809)
    with FOLD_TRAIN.open(encoding="utf-8", newline="") as handle:
        episode = next(csv.DictReader(handle))
    support_id = episode["support_case_id"]
    query_id = episode["query_case_id"]
    if support_id == query_id:
        raise RuntimeError("support/query self-pair violates the episode policy")
    support_image = DATA / "images" / f"{support_id}.png"
    support_mask = DATA / "masks" / f"{support_id}.png"
    query_image = DATA / "images" / f"{query_id}.png"
    query_mask = DATA / "masks" / f"{query_id}.png"
    if not all(path.is_file() for path in (support_image, support_mask, query_image, query_mask)):
        raise FileNotFoundError("materialized image/mask pair missing")

    from models.FoB import FewShotSeg  # imported after the vendored path is set

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = FewShotSeg({}).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    supp_img = load_tensor(support_image, mask=False).unsqueeze(0).cuda(non_blocking=True)
    supp_msk = load_tensor(support_mask, mask=True).cuda(non_blocking=True)
    qry_img = load_tensor(query_image, mask=False).unsqueeze(0).cuda(non_blocking=True)
    qry_msk = load_tensor(query_mask, mask=True).long().cuda(non_blocking=True)
    prompt_loss, contrastive_loss, foreground_loss = model(
        [[supp_img]], [[supp_msk]], [qry_img], qry_msk, True
    )
    total_loss = prompt_loss + contrastive_loss + foreground_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    finite_gradients = all(
        param.grad is None or bool(torch.isfinite(param.grad).all())
        for param in model.parameters()
    )
    optimizer.step()
    torch.cuda.synchronize()
    report = {
        "status": "FOB_TRAIN_STEP_PASS" if finite_gradients else "FOB_TRAIN_STEP_NONFINITE",
        "scope": "single train-only fold-0 episode; feasibility only; not a performance result",
        "fold": 0,
        "episode_id": episode["pilot_id"],
        "support_case_id": support_id,
        "query_case_id": query_id,
        "support_role": episode["role"],
        "query_source_split": episode["query_source_split"],
        "seed": 20260809,
        "losses": {
            "prompt": float(prompt_loss.detach().cpu()),
            "contrastive": float(contrastive_loss.detach().cpu()),
            "foreground": float(foreground_loss.detach().cpu()),
            "total": float(total_loss.detach().cpu()),
        },
        "finite_gradients": finite_gradients,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(WEIGHTS.relative_to(ROOT)),
        "checkpoint_bytes": WEIGHTS.stat().st_size,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not finite_gradients:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
