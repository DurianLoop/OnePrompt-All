"""Phase U External Comparator: GF-SAM (NeurIPS 2024 Spotlight).

Adapts GF-SAM's graph-based few-shot segmentation to our reference-resampling
protocol on MoNuSAC. For each (query, reference) pair, runs GF-SAM's full
pipeline (PNA + PMC + Post-Gating) and computes Dice.

GF-SAM interface:
    GFSAM.set_reference(imgs, masks)   # imgs: (nshot, 3, 1024, 1024)
    GFSAM.set_target(img)              # img: (1, 3, 1024, 1024)
    pred_mask, _ = GFSAM.predict()     # pred_mask: (1, 1024, 1024) binary

Output: per-episode Dice and comparison with our internal B2 (mean consensus)
baseline from the same reference-resampling manifests.
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

# Add GF-SAM to path (it bundles its own segment_anything and dinov2)
_GF_SAM_DIR = Path(__file__).resolve().parents[3] / "external_baselines" / "GF-SAM"
sys.path.insert(0, str(_GF_SAM_DIR))

from matcher.GFSAM import GFSAM, build_model as build_gfsam
from matcher.common.evaluation import Evaluator


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_monusac_reference_manifests(manifest_dir: Path) -> list[dict]:
    """Load all reference-resampling manifests from a directory.

    Each manifest CSV has columns: query_case_id, seed, ref_idx, ref_case_id.
    Returns a list of dicts with keys: query_case_id, seed, ref_case_id.
    """
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

    img_path = data_dir / "images" / f"{case_id}.png"
    if not img_path.exists():
        img_path = data_dir / "images" / f"{case_id}.tif"
    if not img_path.exists():
        # Try MoNuSAC 2020 naming convention
        for ext in [".png", ".tif", ".tiff"]:
            candidate = data_dir / "images" / f"{case_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image for {case_id} from {data_dir}/images/")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_monusac_mask(case_id: str, data_dir: Path) -> np.ndarray:
    """Load a MoNuSAC mask as binary numpy array (H, W)."""
    import cv2

    mask_path = data_dir / "masks" / f"{case_id}.png"
    if not mask_path.exists():
        mask_path = data_dir / "masks" / f"{case_id}.tif"
    if not mask_path.exists():
        for ext in [".png", ".tif", ".tiff"]:
            candidate = data_dir / "masks" / f"{case_id}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask for {case_id} from {data_dir}/masks/")
    mask = (mask > 0).astype(np.uint8)
    return mask


def preprocess_for_gfsam(img: np.ndarray, target_size: int = 1024) -> torch.Tensor:
    """Preprocess image for GF-SAM: resize to target_size, normalize to [0,1].

    Returns tensor of shape (1, 3, target_size, target_size).
    """
    import cv2

    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img_t.unsqueeze(0)  # (1, 3, H, W)


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

def evaluate_gfsam(
    model: GFSAM,
    manifests: list[dict],
    data_dir: Path,
    device: torch.device,
    target_size: int = 1024,
) -> list[dict]:
    """Run GF-SAM on all (query, reference) pairs from manifests.

    Returns list of per-episode result dicts.
    """
    results = []
    n_total = len(manifests)
    t_start = time.time()

    # Pre-load all images and masks to avoid repeated I/O
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

    print(f"Running GF-SAM on {n_total} episodes...")
    for i, m in enumerate(manifests):
        qid = m["query_case_id"]
        rid = m["ref_case_id"]
        seed = m["seed"]

        # Prepare inputs
        # GF-SAM set_reference expects (batch, nshot, 3, H, W) — add nshot=1 dim
        ref_img = preprocess_for_gfsam(images[rid], target_size).to(device)  # (1, 3, H, W)
        ref_img = ref_img.unsqueeze(1)  # (1, 1, 3, H, W)
        # GF-SAM expects binary 0/1 mask (NOT 0/255)
        ref_mask = torch.from_numpy(masks[rid]).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        ref_mask = F.interpolate(ref_mask, size=(target_size, target_size), mode='nearest')
        ref_mask = ref_mask.to(device)

        query_img = preprocess_for_gfsam(images[qid], target_size).to(device)

        # Load GT mask at original resolution for Dice computation
        gt_mask = masks[qid]

        try:
            # GF-SAM inference
            model.set_reference(ref_img, ref_mask)
            model.set_target(query_img)
            with torch.no_grad():
                pred_mask, _ = model.predict()
            model.clear()

            # pred_mask is (1, H, W) binary tensor at target_size
            pred_np = pred_mask.squeeze(0).cpu().numpy().astype(bool)

            # Resize prediction back to original resolution for Dice
            if pred_np.shape != gt_mask.shape:
                import cv2
                pred_np = cv2.resize(
                    pred_np.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            dice = compute_dice(pred_np, gt_mask)

        except Exception as e:
            import traceback
            print(f"  ERROR [{qid} ref={rid} seed={seed}]: {e}")
            traceback.print_exc()
            dice = 0.0

        results.append({
            "query_case_id": qid,
            "ref_case_id": rid,
            "seed": seed,
            "dice": dice,
            "method": "GF-SAM",
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            dice_vals = [r["dice"] for r in results[-100:]]
            print(f"  [{i+1}/{n_total}] elapsed={elapsed:.0f}s "
                  f"mean_dice(last100)={np.mean(dice_vals):.4f}")

    elapsed = time.time() - t_start
    dice_all = [r["dice"] for r in results]
    print(f"GF-SAM complete: {n_total} episodes, elapsed={elapsed:.0f}s, "
          f"mean_dice={np.mean(dice_all):.4f}, median_dice={np.median(dice_all):.4f}")

    return results


# ---------------------------------------------------------------------------
# Compare with internal B2 baseline
# ---------------------------------------------------------------------------

def compare_with_internal(
    gfsam_results: list[dict],
    internal_episode_csv: Path,
) -> dict:
    """Compare GF-SAM results with our internal B2 (mean consensus) baseline.

    Computes per-query Dice comparison and statistical tests.
    """
    import pandas as pd
    from scipy.stats import wilcoxon

    # Load internal results
    df_internal = pd.read_csv(internal_episode_csv)
    # Group by query and seed, compute mean Dice across references
    internal_by_query = df_internal.groupby(["query_case_id", "seed"])["mean_dice_k1"].mean().reset_index()

    # Group GF-SAM results by query and seed
    df_gfsam = pd.DataFrame(gfsam_results)
    gfsam_by_query = df_gfsam.groupby(["query_case_id", "seed"])["dice"].mean().reset_index()

    # Merge
    merged = pd.merge(
        gfsam_by_query, internal_by_query,
        on=["query_case_id", "seed"],
        suffixes=("_gfsam", "_internal"),
    )

    if len(merged) == 0:
        return {"error": "No matching queries between GF-SAM and internal results"}

    # Compute statistics
    # Columns: 'dice' (from gfsam) and 'mean_dice_k1' (from internal) — no conflict so no suffix
    delta = merged["dice"] - merged["mean_dice_k1"]
    try:
        stat, pval = wilcoxon(delta)
    except Exception:
        stat, pval = float("nan"), float("nan")

    return {
        "n_queries": len(merged),
        "gfsam_mean_dice": float(merged["dice"].mean()),
        "gfsam_median_dice": float(merged["dice"].median()),
        "internal_mean_dice": float(merged["mean_dice_k1"].mean()),
        "internal_median_dice": float(merged["mean_dice_k1"].median()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
        "wilcoxon_p": float(pval) if not np.isnan(pval) else None,
        "gfsam_wins": int((delta > 0).sum()),
        "internal_wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GF-SAM External Comparator")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="MoNuSAC data directory (contains images/ and masks/)")
    parser.add_argument("--manifest-dir", type=str, required=True,
                        help="Directory with holdout_resampling_seed_*.csv manifests")
    parser.add_argument("--internal-csv", type=str, default=None,
                        help="Path to internal phase2_episode_instability.csv for comparison")
    parser.add_argument("--dinov2-weights", type=str,
                        default="D:/SAM/adapromptset/.test_tmp/dinov2_vitl14_pretrain.pth",
                        help="Path to DINOv2-L pretrained weights")
    parser.add_argument("--sam-weights", type=str,
                        default="D:/SAM/code/models/sam_vit_h_4b8939.pth",
                        help="Path to SAM-ViT-H checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="reports/phaseU_external_gfsam")
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

    # Build GF-SAM model
    print("Building GF-SAM model...")
    # Monkey-patch load_pretrained_weights to handle Windows paths
    # (urlparse('D:/...').scheme == 'd' which is truthy, causing torch.hub
    # to try downloading the local path as a URL)
    import dinov2.utils.utils as dinov2_utils
    _orig_load = dinov2_utils.load_pretrained_weights
    def _patched_load(model, pretrained_weights, checkpoint_key):
        from urllib.parse import urlparse
        import os
        if os.path.isfile(pretrained_weights):
            # Local file: use torch.load directly
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                state_dict = state_dict[checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # Handle DINOv2 teacher format
            msg = model.load_state_dict(state_dict, strict=False)
            return
        return _orig_load(model, pretrained_weights, checkpoint_key)
    dinov2_utils.load_pretrained_weights = _patched_load

    # Create a minimal args object for build_model
    from types import SimpleNamespace
    gfsam_args = SimpleNamespace(
        dinov2_size="vit_large",
        sam_size="vit_h",
        dinov2_weights=args.dinov2_weights,
        sam_weights=args.sam_weights,
        device=device,
    )
    model = build_gfsam(gfsam_args)
    model.device = device
    Evaluator.initialize()
    print("GF-SAM model ready.")

    # Run evaluation
    results = evaluate_gfsam(model, manifests, data_dir, device)
    model.clear()

    # Save raw results
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "gfsam_per_episode.csv", index=False)

    # Summary
    summary = {
        "method": "GF-SAM",
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
            comparison = compare_with_internal(results, internal_csv)
            summary["comparison"] = comparison
            print(f"\nComparison with internal baseline:")
            print(f"  GF-SAM mean Dice:   {comparison['gfsam_mean_dice']:.4f}")
            print(f"  Internal mean Dice:  {comparison['internal_mean_dice']:.4f}")
            print(f"  Delta (GF-SAM - Int): {comparison['delta_mean']:.4f}")
            if comparison.get("wilcoxon_p") is not None:
                print(f"  Wilcoxon p: {comparison['wilcoxon_p']:.4f}")
        else:
            print(f"Internal CSV not found: {internal_csv}")

    with open(out_dir / "gfsam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()