"""Nested-CV evaluation for the adapted FoB comparator on one frozen fold.

Run ``--stage select`` only after training completes: it decodes every saved
training checkpoint on the *inner-validation* episodes and deterministically
selects the largest mean Dice (ties: smaller step).  ``--stage test`` then
loads only that recorded checkpoint and decodes the untouched outer-test
episodes.  Query masks are loaded only after FoB prompts and SAM-H predictions
have been generated, solely to calculate metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2  # Cache OpenCV before `surf` on sys.path can shadow it.
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FOB_ROOT = ROOT / "baselines" / "fob_source"
SURF_ROOT = WORKSPACE / "surf"
DATA = ROOT / "data" / "monuseg_hf" / "materialized_train"
FOLD = WORKSPACE / "runs" / "research_direction" / "prompt_policy_splits" / "monuseg_nested_cv5" / "fold_0"
RUN = ROOT / "runs" / "fob_monuseg_fold0"
RESNET = ROOT / "models" / "deeplabv3_resnet101_coco-586e9e4e.pth"
SAM_H = WORKSPACE / "assets" / "checkpoints" / "sam_vit_h_4b8939.pth"
SEED = 20260809


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    value = torch.from_numpy(array).permute(2, 0, 1)
    return ((value - value.mean(dim=(1, 2), keepdim=True)) /
            value.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)).unsqueeze(0)


def support_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return torch.from_numpy((np.asarray(image.convert("L")) > 0).astype(np.float32)).unsqueeze(0)


def dice(prediction: np.ndarray, target_path: Path) -> float:
    # The query annotation is intentionally opened after decoding, never passed
    # into FoB or SAM-H.
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("L")) > 0
    pred = np.asarray(prediction, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    return float(2.0 * np.logical_and(pred, target).sum() / (pred.sum() + target.sum() + 1e-8))


def rows(split: str) -> list[dict[str, str]]:
    manifest = FOLD / f"episodes_{split}.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        output = list(csv.DictReader(handle))
    expected = 30 if split == "val" else (40 if FOLD.name in {"fold_0", "fold_1"} else 35)
    if len(output) != expected or any(row["role"] != split for row in output):
        raise RuntimeError(f"unexpected frozen fold-0 {split} manifest")
    return output


def decode(policy: torch.nn.Module, segmenter: torch.nn.Module, row: dict[str, str]) -> tuple[np.ndarray, int, int]:
    support_id, query_id = row["support_case_id"], row["query_case_id"]
    supp_image = image_tensor(DATA / "images" / f"{support_id}.png").cuda(non_blocking=True)
    qry_image = image_tensor(DATA / "images" / f"{query_id}.png").cuda(non_blocking=True)
    supp_mask = support_mask(DATA / "masks" / f"{support_id}.png").cuda(non_blocking=True)
    # Deliberately all-zero and annotation-free placeholder: FoB inference must
    # not consult query ground truth to create SAM prompts.
    placeholder = torch.zeros((1, qry_image.shape[-2], qry_image.shape[-1]), dtype=torch.long, device="cuda")
    with torch.inference_mode():
        negative, positive = policy([[supp_image]], [[supp_mask]], [qry_image], placeholder, False)
        prediction = segmenter(qry_image[0], positive, negative, {"dataset": "MoNuSeg"})
    return (np.asarray(prediction, dtype=bool),
            int(sum(len(p) for p in positive if p is not None)),
            int(sum(len(p) for p in negative if p is not None)))


def evaluate(policy: torch.nn.Module, segmenter: torch.nn.Module, split: str, checkpoint: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, row in enumerate(rows(split), start=1):
        started = time.perf_counter()
        prediction, positives, negatives = decode(policy, segmenter, row)
        score = dice(prediction, DATA / "masks" / f"{row['query_case_id']}.png")
        output.append({
            "checkpoint": checkpoint.name, "split": split, "episode_index": index,
            "episode_id": row["pilot_id"], "support_case_id": row["support_case_id"],
            "query_case_id": row["query_case_id"], "dice": score,
            "positive_point_count": positives, "negative_point_count": negatives,
            "query_gt_loaded_after_prediction": True,
            "seconds": time.perf_counter() - started,
        })
    return output


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def build_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["FOB_RESNET101_COCO"] = str(RESNET)
    sys.path[:0] = [str(FOB_ROOT), str(SURF_ROOT)]
    if not torch.cuda.is_available():
        raise RuntimeError("FoB evaluation requires CUDA")
    if not all(path.is_file() for path in (RESNET, SAM_H)):
        raise FileNotFoundError("required FoB/SAM-H weight missing")
    from models.FoB import FewShotSeg
    from SAM import SAM
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    policy = FewShotSeg({}).cuda().eval()
    segmenter = SAM(sam_pretrained_path=str(SAM_H))
    return policy, segmenter


def main() -> None:
    global FOLD, RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("select", "test"), required=True)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    args = parser.parse_args()
    FOLD = WORKSPACE / "runs" / "research_direction" / "prompt_policy_splits" / "monuseg_nested_cv5" / f"fold_{args.fold}"
    RUN = ROOT / "runs" / f"fob_monuseg_fold{args.fold}"
    status_path = RUN / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "TRAINING_COMPLETE":
        raise RuntimeError("refusing evaluation until the recorded train-only run completes")
    policy, segmenter = build_models()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.stage == "select":
        checkpoints = sorted((RUN / "checkpoints").glob("step_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
        if [path.stem for path in checkpoints] != ["step_0250", "step_0500", "step_0750", "step_1000"]:
            raise RuntimeError("expected exactly the four pre-registered training checkpoints")
        summary = []
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
            policy.load_state_dict(payload["model_state_dict"], strict=True); policy.eval()
            record_path = RUN / "selection" / f"{checkpoint.stem}_val_per_episode.csv"
            # Selection jobs can be interrupted by the orchestration time cap.
            # A complete checkpoint-level validation table is deterministic and
            # immutable for this run, so safely reuse it rather than decoding
            # the same validation episodes again. Partial tables are discarded.
            if record_path.is_file():
                with record_path.open(encoding="utf-8", newline="") as handle:
                    records = list(csv.DictReader(handle))
                if len(records) != 30 or any(row.get("checkpoint") != checkpoint.name or row.get("split") != "val" for row in records):
                    records = evaluate(policy, segmenter, "val", checkpoint)
                    write_csv(record_path, records)
                else:
                    records = [{**row, "dice": float(row["dice"])} for row in records]
            else:
                records = evaluate(policy, segmenter, "val", checkpoint)
                write_csv(record_path, records)
            summary.append({"checkpoint": checkpoint.name, "step": int(payload["step"]), "episodes": len(records),
                            "mean_dice": float(np.mean([record["dice"] for record in records]))})
        summary.sort(key=lambda item: (-item["mean_dice"], item["step"]))
        selected = {"status": "INNER_VALIDATION_SELECTION_COMPLETE", "fold": args.fold, "seed": SEED,
                    "selection_rule": "largest inner-validation mean Dice; ties use smaller training step",
                    "query_gt_policy": "loaded only after each prediction to score Dice; never supplied to FoB/SAM-H decoding",
                    "candidates_ranked": summary, "selected": summary[0],
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024)}
        target = RUN / "selection" / "selected_checkpoint.json"; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(selected, indent=2))
    else:
        selection = json.loads((RUN / "selection" / "selected_checkpoint.json").read_text(encoding="utf-8"))
        if selection.get("status") != "INNER_VALIDATION_SELECTION_COMPLETE":
            raise RuntimeError("no valid inner-validation selection record")
        checkpoint = RUN / "checkpoints" / str(selection["selected"]["checkpoint"])
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        policy.load_state_dict(payload["model_state_dict"], strict=True); policy.eval()
        records = evaluate(policy, segmenter, "test", checkpoint)
        write_csv(RUN / "test" / "per_episode.csv", records)
        report = {"status": "OUTER_TEST_COMPLETE", "fold": args.fold, "selected_checkpoint": checkpoint.name,
                  "selection_record": str(RUN / "selection" / "selected_checkpoint.json"), "episodes": len(records),
                  "mean_dice": float(np.mean([record["dice"] for record in records])),
                  "query_gt_policy": "loaded only after each prediction to score Dice; never supplied to FoB/SAM-H decoding",
                  "elapsed_seconds": time.perf_counter() - started,
                  "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024)}
        target = RUN / "test" / "summary.json"; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
