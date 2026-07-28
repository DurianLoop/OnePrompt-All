"""Phase U External Comparator: MAUP (MICCAI 2025).

Adapts MAUP's multi-center uncertainty-aware prompting to our
reference-resampling protocol on MoNuSAC. For each (query, reference) pair,
runs MAUP's full pipeline (RPG + similarity computation + MAUP prompting +
SAM prediction) and computes Dice.

MAUP interface (FewShotSeg.forward):
    model([supp_image], [supp_mask], [query_image], train=False)
    Returns: tensor of shape (B, 2, H, W) [BG, FG]

Output: per-episode Dice and comparison with our internal B2 baseline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add MAUP and its bundled segment-anything to path
_MAUP_DIR = Path(__file__).resolve().parents[3] / "external_baselines" / "MAUP"
sys.path.insert(0, str(_MAUP_DIR))
sys.path.insert(0, str(_MAUP_DIR / "segment-anything"))

# Override MAUP model paths before import
_orig_cwd = os.getcwd()
os.chdir(str(_MAUP_DIR))

from models.fewshot import FewShotSeg

os.chdir(_orig_cwd)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_monusac_reference_manifests(manifest_dir: Path) -> list[dict]:
    """Load all reference-resampling manifests from a directory."""
    import pandas as pd

    rows = []
    for csv_path in sorted(manifest_dir.glob("holdout_resampling_seed_*.csv")):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rows.append({
                "query_case_id": row["query_case_id"],
                "seed": int(row["seed"]),
                "ref_case_id": row["reference_case_id"],
            })
    return rows


def load_monusac_image(case_id: str, data_dir: Path) -> np.ndarray:
    """Load a MoNuSAC image as RGB numpy array (H, W, 3)."""
    import cv2

    for ext in [".png", ".tif", ".tiff"]:
        img_path = data_dir / "images" / f"{case_id}{ext}"
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise FileNotFoundError(f"Cannot load image for {case_id}")


def load_monusac_mask(case_id: str, data_dir: Path) -> np.ndarray:
    """Load a MoNuSAC mask as binary numpy array (H, W)."""
    import cv2

    for ext in [".png", ".tif", ".tiff"]:
        mask_path = data_dir / "masks" / f"{case_id}{ext}"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return (mask > 0).astype(np.uint8)

    raise FileNotFoundError(f"Cannot load mask for {case_id}")


def preprocess_for_maup(img: np.ndarray, mask: np.ndarray | None,
                         target_size: int = 256) -> tuple[torch.Tensor, ...]:
    """Preprocess image and mask for MAUP.

    MAUP expects images as float tensors in [0, 1] and masks as binary.
    Returns (img_tensor, mask_tensor_or_None).
    """
    import cv2

    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    if mask is not None:
        mask_t = torch.from_numpy(mask).float()  # (H, W)
        return img_t, mask_t
    return img_t, None


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice score between two binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / denom


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_maup(
    model: FewShotSeg,
    manifests: list[dict],
    data_dir: Path,
    device: torch.device,
    target_size: int = 256,
) -> list[dict]:
    """Run MAUP on all (query, reference) pairs from manifests."""
    results = []
    n_total = len(manifests)
    t_start = time.time()

    # Pre-load all images and masks
    case_ids = set()
    for m in manifests:
        case_ids.add(m["query_case_id"])
        case_ids.add(m["ref_case_id"])

    print(f"Pre-loading {len(case_ids)} unique cases...")
    images = {}
    masks = {}
    for cid in sorted(case_ids):
        images[cid] = load_monusac_image(cid, data_dir)
        masks[cid] = load_monusac_mask(cid, data_dir)

    print(f"Running MAUP on {n_total} episodes...")
    model.eval()

    for i, m in enumerate(manifests):
        qid = m["query_case_id"]
        rid = m["ref_case_id"]
        seed = m["seed"]

        # Prepare inputs
        ref_img, ref_mask = preprocess_for_maup(images[rid], masks[rid], target_size)
        query_img, _ = preprocess_for_maup(images[qid], None, target_size)
        gt_mask = masks[qid]

        # MAUP expects: supp_imgs=[[tensor]], supp_mask=[[tensor]], qry_imgs=[tensor]
        supp_imgs = [[ref_img.unsqueeze(0).to(device)]]   # [[1, 3, 256, 256]]
        supp_mask = [[ref_mask.unsqueeze(0).to(device)]]   # [[1, 256, 256]]
        qry_imgs = [query_img.unsqueeze(0).to(device)]     # [1, 3, 256, 256]

        try:
            with torch.no_grad():
                # Model returns (B, 2, H, W) [BG, FG]
                output = model(supp_imgs, supp_mask, qry_imgs, None, train=False)

            # output shape: (1, 2, H, W), take argmax for binary mask
            pred = output.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)  # (H, W)

            # Resize prediction back to original resolution
            if pred.shape != gt_mask.shape:
                import cv2
                pred = cv2.resize(
                    pred, (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            dice = compute_dice(pred, gt_mask)

        except Exception as e:
            print(f"  ERROR [{qid} ref={rid} seed={seed}]: {e}")
            dice = 0.0

        results.append({
            "query_case_id": qid,
            "ref_case_id": rid,
            "seed": seed,
            "dice": dice,
            "method": "MAUP",
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            dice_vals = [r["dice"] for r in results[-100:]]
            print(f"  [{i+1}/{n_total}] elapsed={elapsed:.0f}s "
                  f"mean_dice(last100)={np.mean(dice_vals):.4f}")

    elapsed = time.time() - t_start
    dice_all = [r["dice"] for r in results]
    print(f"MAUP complete: {n_total} episodes, elapsed={elapsed:.0f}s, "
          f"mean_dice={np.mean(dice_all):.4f}, median_dice={np.median(dice_all):.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAUP External Comparator")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="MoNuSAC data directory (contains images/ and masks/)")
    parser.add_argument("--manifest-dir", type=str, required=True,
                        help="Directory with holdout_resampling_seed_*.csv manifests")
    parser.add_argument("--internal-csv", type=str, default=None,
                        help="Path to internal phase2_episode_instability.csv for comparison")
    parser.add_argument("--sam-weights", type=str,
                        default="D:/SAM/assets/checkpoints/sam_vit_h_4b8939.pth",
                        help="Path to SAM-ViT-H checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="reports/phaseU_external_maup")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes (for smoke testing)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifests
    manifests = load_monusac_reference_manifests(manifest_dir)
    print(f"Loaded {len(manifests)} episodes from manifests")

    if args.max_episodes:
        manifests = manifests[:args.max_episodes]
        print(f"Limited to {len(manifests)} episodes (smoke test)")

    # Build MAUP model
    print("Building MAUP model...")
    # Symlink or copy SAM checkpoint to MAUP's expected location
    maup_checkpoint_dir = _MAUP_DIR / "checkpoints"
    maup_checkpoint_dir.mkdir(exist_ok=True)
    target_sam_path = maup_checkpoint_dir / "sam_vit_h_4b8939.pth"
    if not target_sam_path.exists():
        import shutil
        shutil.copy(args.sam_weights, target_sam_path)
        print(f"Copied SAM checkpoint to {target_sam_path}")

    os.chdir(str(_MAUP_DIR))  # Switch to MAUP dir for relative paths
    model = FewShotSeg()
    os.chdir(_orig_cwd)  # Restore original CWD
    model.to(device)
    model.device = device
    print("MAUP model ready.")

    # Run evaluation
    results = evaluate_maup(model, manifests, data_dir, device)
    del model
    torch.cuda.empty_cache()

    # Save raw results
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "maup_per_episode.csv", index=False)

    # Summary
    summary = {
        "method": "MAUP",
        "n_episodes": len(results),
        "mean_dice": float(np.mean([r["dice"] for r in results])),
        "median_dice": float(np.median([r["dice"] for r in results])),
        "std_dice": float(np.std([r["dice"] for r in results])),
        "min_dice": float(np.min([r["dice"] for r in results])),
        "max_dice": float(np.max([r["dice"] for r in results])),
    }

    # Compare with internal baseline if provided
    if args.internal_csv:
        internal_csv = Path(args.internal_csv)
        if internal_csv.exists():
            import pandas as pd
            from scipy.stats import wilcoxon

            df_internal = pd.read_csv(internal_csv)
            internal_by_query = df_internal.groupby(
                ["query_case_id", "seed"]
            )["mean_dice_k1"].mean().reset_index()

            df_maup = pd.DataFrame(results)
            maup_by_query = df_maup.groupby(
                ["query_case_id", "seed"]
            )["dice"].mean().reset_index()

            merged = pd.merge(
                maup_by_query, internal_by_query,
                on=["query_case_id", "seed"],
                suffixes=("_maup", "_internal"),
            )

            if len(merged) > 0:
                # Columns: 'dice' (from maup) and 'mean_dice_k1' (from internal) — no conflict so no suffix
                delta = merged["dice"] - merged["mean_dice_k1"]
                try:
                    stat, pval = wilcoxon(delta)
                except Exception:
                    stat, pval = float("nan"), float("nan")

                comparison = {
                    "n_queries": len(merged),
                    "maup_mean_dice": float(merged["dice"].mean()),
                    "internal_mean_dice": float(merged["mean_dice_k1"].mean()),
                    "delta_mean": float(delta.mean()),
                    "wilcoxon_p": float(pval) if not np.isnan(pval) else None,
                }
                summary["comparison"] = comparison
                print(f"\nMAUP vs Internal: delta={comparison['delta_mean']:.4f}, "
                      f"p={comparison.get('wilcoxon_p', 'N/A')}")

    with open(out_dir / "maup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()