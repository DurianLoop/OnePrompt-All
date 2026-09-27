"""Verify GT-free FoB prompt generation followed by a real SAM-H decode."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2  # import before adding `surf`, which contains an unrelated cv2.py
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FOB_ROOT = ROOT / "baselines" / "fob_source"
SURF_ROOT = WORKSPACE / "surf"
DATA = ROOT / "data" / "monuseg_hf" / "materialized_train"
MANIFEST = WORKSPACE / "runs" / "research_direction" / "prompt_policy_splits" / "monuseg_nested_cv5" / "fold_0" / "episodes_val.csv"
RESNET = ROOT / "models" / "deeplabv3_resnet101_coco-586e9e4e.pth"
SAM_H = WORKSPACE / "assets" / "checkpoints" / "sam_vit_h_4b8939.pth"
OUTPUT = ROOT / "evidence" / "fob_sam_h_gtfree_smoke.json"


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return ((tensor - tensor.mean(dim=(1, 2), keepdim=True)) /
            tensor.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)).unsqueeze(0)


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["FOB_RESNET101_COCO"] = str(RESNET)
    sys.path[:0] = [str(FOB_ROOT), str(SURF_ROOT)]
    if not all(path.is_file() for path in (MANIFEST, RESNET, SAM_H)):
        raise FileNotFoundError("required manifest or checkpoint missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        episode = next(csv.DictReader(handle))
    support_id, query_id = episode["support_case_id"], episode["query_case_id"]
    supp_image = image_tensor(DATA / "images" / f"{support_id}.png").cuda()
    query_image = image_tensor(DATA / "images" / f"{query_id}.png").cuda()
    with Image.open(DATA / "masks" / f"{support_id}.png") as source_mask:
        supp_mask = torch.from_numpy((np.asarray(source_mask.convert("L")) > 0).astype(np.float32)).unsqueeze(0).cuda()
    # This tensor contains no query annotation. The patched FoB inference path
    # must not inspect it to produce prompts.
    query_label_placeholder = torch.zeros((1, query_image.shape[-2], query_image.shape[-1]), dtype=torch.long, device="cuda")
    from models.FoB import FewShotSeg
    from SAM import SAM
    torch.manual_seed(20260809)
    torch.cuda.manual_seed_all(20260809)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    policy = FewShotSeg({}).cuda().eval()
    with torch.inference_mode():
        negative, positive = policy([[supp_image]], [[supp_mask]], [query_image], query_label_placeholder, False)
    segmenter = SAM(sam_pretrained_path=str(SAM_H))
    with torch.inference_mode():
        prediction = segmenter(query_image[0], positive, negative, {"dataset": "MoNuSeg"})
    torch.cuda.synchronize()
    report = {
        "status": "FOB_SAM_H_GTFREE_SMOKE_PASS",
        "scope": "untrained FoB prompt policy + SAM-H decode; wiring/GT-free check only; not a result",
        "fold": 0,
        "episode_id": episode["pilot_id"],
        "split": "inner_validation",
        "support_case_id": support_id,
        "query_case_id": query_id,
        "query_gt_loaded": False,
        "positive_point_count": int(sum(len(points) for points in positive if points is not None)),
        "negative_point_count": int(sum(len(points) for points in negative if points is not None)),
        "prediction_shape": list(prediction.shape),
        "prediction_foreground_fraction": float(np.asarray(prediction, dtype=np.float32).mean()),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "sam_checkpoint_bytes": SAM_H.stat().st_size,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
