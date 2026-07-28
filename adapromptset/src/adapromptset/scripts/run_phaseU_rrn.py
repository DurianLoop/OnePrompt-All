"""Phase U RRN: Reference Reliability Network.

Implements the learned Reference Reliability Network (RRN) described in the paper.
RRN is a lightweight MLP that maps five pipeline-level features to a calibrated
reference reliability score P(success | query, reference).

Features:
  f1: matching_map_disagreement (episode-level, proxy for matching entropy)
  f2: candidate_jaccard_mean_k1 (episode-level, proxy for feature similarity)
  f3: prompt_overlap_mean_k1 (episode-level, proxy for prompt dispersion)
  f4: mask_consistency (per-reference, Jaccard vs other refs)
  f5: prediction_uncertainty (per-reference, SAM score)

Architecture: MLP [5 -> 64 -> 32 -> 1] + ReLU + Sigmoid
Training: Binary cross-entropy, 70/15/15 train/val/test split by query
Output: Per-reference reliability score, selective prediction CFR reduction

Usage:
    python src/adapromptset/scripts/run_phaseU_rrn.py \
        --wp-dir reports/phaseU_wp1/sam_b \
        --out-dir reports/phaseU_rrn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import base64
import io
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def decode_mask_b64(b64_str: str, target_h: int = 128, target_w: int = 128) -> np.ndarray:
    """Decode base64-encoded mask string to binary numpy array."""
    try:
        raw = base64.b64decode(b64_str)
        # SAM prediction masks are run-length encoded; use simple approach
        # The mask is stored as a binary string; reconstruct as numpy array
        pixel_count = target_h * target_w
        if len(raw) >= pixel_count:
            mask = np.frombuffer(raw[:pixel_count], dtype=np.uint8).reshape(target_h, target_w)
        else:
            # Pad if undersized
            mask = np.zeros(pixel_count, dtype=np.uint8)
            mask[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
            mask = mask.reshape(target_h, target_w)
        return (mask > 0).astype(np.float32)
    except Exception:
        return np.zeros((target_h, target_w), dtype=np.float32)


def compute_mask_jaccard(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Jaccard index between two binary masks."""
    inter = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()
    union = np.logical_or(mask1 > 0.5, mask2 > 0.5).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def load_raw_per_call(wp_dir: Path) -> pd.DataFrame:
    """Load raw per-call CSV."""
    path = wp_dir / "phase2_raw_per_call.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw per-call data not found: {path}")
    return pd.read_csv(path)


def load_episode_data(wp_dir: Path) -> pd.DataFrame:
    """Load episode instability CSV."""
    path = wp_dir / "phase2_episode_instability.csv"
    if not path.exists():
        raise FileNotFoundError(f"Episode data not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Per-reference feature computation
# ---------------------------------------------------------------------------

def build_rrn_dataset(raw_df: pd.DataFrame, episode_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-reference dataset for RRN training.

    For each (query, seed, config, reference) row in raw_df:
      - Get episode-level features (f1-f3) from episode_df
      - Compute per-reference features (f4, f5) from raw_df
      - Label: success = 1 if Dice >= 0.5, else 0
    """
    rows = []

    # Pre-compute mask consistency within each (query, seed, config) group
    # Group by query_case_id, seed, config_name
    for (qid, seed, cfg), group in raw_df.groupby(["query_case_id", "seed", "config_name"]):
        # Get episode-level features for this config
        ep_row = episode_df[
            (episode_df["query_case_id"] == qid)
            & (episode_df["seed"] == seed)
            & (episode_df["config_name"] == cfg)
        ]
        if ep_row.empty:
            continue
        ep_row = ep_row.iloc[0]

        f1 = float(ep_row["matching_map_disagreement"])
        f2 = float(ep_row["candidate_jaccard_mean_k1"])
        f3 = float(ep_row["prompt_overlap_mean_k1"])

        # Decode all masks for this group
        ref_indices = group["reference_index"].values
        masks = {}
        for _, row in group.iterrows():
            ref_idx = int(row["reference_index"])
            b64 = row.get("pred_mask_b64_128", "")
            if pd.notna(b64) and b64:
                masks[ref_idx] = decode_mask_b64(str(b64))

        # Compute per-reference features
        for _, row in group.iterrows():
            ref_idx = int(row["reference_index"])
            dice = float(row["dice"])

            # f4: mask consistency = mean Jaccard with other refs
            jaccards = []
            for other_idx, other_mask in masks.items():
                if other_idx != ref_idx and ref_idx in masks:
                    jaccards.append(compute_mask_jaccard(masks[ref_idx], other_mask))
            f4 = np.mean(jaccards) if jaccards else 0.0

            # f5: SAM score as prediction uncertainty proxy
            f5 = float(row.get("sam_score", 0.5))

            success = 1.0 if dice >= 0.5 else 0.0

            rows.append({
                "query_case_id": qid,
                "seed": int(seed),
                "config_name": cfg,
                "reference_index": ref_idx,
                "reference_case_id": row["reference_case_id"],
                "dice": dice,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f4": f4,
                "f5": f5,
                "success": success,
            })

    return pd.DataFrame(rows)


class RRNNuclearDataset(Dataset):
    """PyTorch Dataset for RRN training."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# RRN Model
# ---------------------------------------------------------------------------

class ReferenceReliabilityNetwork(nn.Module):
    """Lightweight MLP for reference reliability estimation.

    Architecture: [5 -> 64 -> 32 -> 1] + ReLU + Sigmoid
    """

    def __init__(self, input_dim: int = 5, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rrn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> tuple[nn.Module, dict]:
    """Train RRN with binary cross-entropy loss and early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy().tolist())
                val_labels.extend(y_batch.cpu().numpy().tolist())

        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))
        val_auroc = float(roc_auc_score(val_labels, val_preds))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auroc"].append(val_auroc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best state
    model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Evaluation: Selective prediction
# ---------------------------------------------------------------------------

def evaluate_selective_prediction(
    model: nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    scaler: StandardScaler,
    coverage: float = 0.80,
) -> dict[str, Any]:
    """Evaluate RRN as a selective predictor.

    For each query, compute the RRN reliability score for each reference.
    Select top-K references (K = 3, matching the paper's protocol).
    Compute CFR at the given coverage.
    """
    feature_cols = ["f1", "f2", "f3", "f4", "f5"]

    # Get per-reference predictions
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    X_tensor = torch.FloatTensor(X_test).to(device)
    model.eval()
    with torch.no_grad():
        reliability_scores = model(X_tensor).cpu().numpy()

    test_df = test_df.copy()
    test_df["rrn_score"] = reliability_scores

    # Group by (query_case_id, seed, config_name)
    # For each query, rank references by RRN score, select top-3
    results = []
    for (qid, seed, cfg), group in test_df.groupby(["query_case_id", "seed", "config_name"]):
        group = group.sort_values("rrn_score", ascending=False)
        top_k = group.head(3)

        mean_dice_rrn = float(top_k["dice"].mean())
        mean_dice_all = float(group["dice"].mean())
        n_refs = len(group)

        # Catastrophic failure: mean Dice < 0.1
        is_catastrophic = mean_dice_all < 0.1
        is_catastrophic_rrn = mean_dice_rrn < 0.1

        results.append({
            "query_case_id": qid,
            "seed": int(seed),
            "config_name": cfg,
            "mean_dice_all": mean_dice_all,
            "mean_dice_rrn": mean_dice_rrn,
            "is_catastrophic": is_catastrophic,
            "is_catastrophic_rrn": is_catastrophic_rrn,
            "n_refs": n_refs,
            "rrn_scores": group["rrn_score"].tolist(),
        })

    result_df = pd.DataFrame(results)

    # Compute metrics
    cfr_all = float(result_df["is_catastrophic"].mean())
    cfr_rrn = float(result_df["is_catastrophic_rrn"].mean())
    cfr_reduction = (cfr_all - cfr_rrn) / max(cfr_all, 1e-8) if cfr_all > 0 else 0.0

    mean_dice_all = float(result_df["mean_dice_all"].mean())
    mean_dice_rrn = float(result_df["mean_dice_rrn"].mean())

    # Compute AUROC for per-reference success prediction
    per_ref_labels = test_df["success"].values
    per_ref_preds = test_df["rrn_score"].values
    auroc = float(roc_auc_score(per_ref_labels, per_ref_preds))
    auprc = float(average_precision_score(per_ref_labels, per_ref_preds))

    return {
        "n_queries": len(result_df),
        "cfr_all": cfr_all,
        "cfr_rrn": cfr_rrn,
        "cfr_reduction_pct": cfr_reduction * 100,
        "mean_dice_all": mean_dice_all,
        "mean_dice_rrn": mean_dice_rrn,
        "auroc": auroc,
        "auprc": auprc,
        "prevalence_catastrophic": cfr_all,
    }


# ---------------------------------------------------------------------------
# Reference selection benchmark
# ---------------------------------------------------------------------------

def evaluate_reference_selection(
    test_df: pd.DataFrame,
    rrn_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate multiple reference selection strategies.

    Strategies:
      - random: random reference (mean of 100 trials)
      - f2 (similarity): highest candidate_jaccard_mean_k1
      - entropy (f1): lowest matching_map_disagreement
      - rrn: highest RRN score
      - oracle: highest Dice (upper bound)
    """
    feature_cols = ["f1", "f2", "f3", "f4", "f5"]

    if rrn_scores is not None:
        test_df = test_df.copy()
        test_df["rrn_score"] = rrn_scores

    strategies = {}

    # Oracle: select top-3 references by Dice
    oracle_dice = []
    for (qid, seed, cfg), group in test_df.groupby(["query_case_id", "seed", "config_name"]):
        top = group.nlargest(3, "dice")
        oracle_dice.append(float(top["dice"].mean()))
    strategies["oracle"] = {"mean_dice": float(np.mean(oracle_dice))}

    # Random: mean of 100 trials
    random_dice = []
    for (qid, seed, cfg), group in test_df.groupby(["query_case_id", "seed", "config_name"]):
        trial_means = []
        for _ in range(100):
            sample = group.sample(min(3, len(group)), replace=False)
            trial_means.append(float(sample["dice"].mean()))
        random_dice.append(float(np.mean(trial_means)))
    strategies["random"] = {"mean_dice": float(np.mean(random_dice))}

    # f2 (similarity): highest f2
    f2_dice = []
    for (qid, seed, cfg), group in test_df.groupby(["query_case_id", "seed", "config_name"]):
        top = group.nlargest(3, "f2")
        f2_dice.append(float(top["dice"].mean()))
    strategies["similarity"] = {"mean_dice": float(np.mean(f2_dice))}

    # Entropy (f1): lowest f1
    f1_dice = []
    for (qid, seed, cfg), group in test_df.groupby(["query_case_id", "seed", "config_name"]):
        top = group.nsmallest(3, "f1")
        f1_dice.append(float(top["dice"].mean()))
    strategies["entropy"] = {"mean_dice": float(np.mean(f1_dice))}

    # RRN: highest RRN score
    if "rrn_score" in test_df.columns:
        rrn_dice = []
        for (qid, seed, cfg), group in test_df.groupby(["query_case_id", "seed", "config_name"]):
            top = group.nlargest(3, "rrn_score")
            rrn_dice.append(float(top["dice"].mean()))
        strategies["rrn"] = {"mean_dice": float(np.mean(rrn_dice))}

    # Baseline: all references mean
    all_dice = []
    for (qid, seed, cfg), group in test_df.groupby(["query_case_id", "seed", "config_name"]):
        all_dice.append(float(group["dice"].mean()))
    strategies["all_refs"] = {"mean_dice": float(np.mean(all_dice))}

    return strategies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase U RRN: Reference Reliability Network")
    parser.add_argument(
        "--wp-dir", type=Path,
        default=REPORTS_DIR / "phaseU_wp1" / "sam_b",
        help="Working point directory with raw per-call and episode CSV files",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPORTS_DIR / "phaseU_rrn",
        help="Output directory for RRN results",
    )
    parser.add_argument(
        "--config", type=str, default="P20-N1",
        help="Prompt configuration to use",
    )
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[RRN] Loading data from {args.wp_dir}")
    raw_df = load_raw_per_call(args.wp_dir)
    ep_df = load_episode_data(args.wp_dir)

    # Filter to specific config and individual references (k=1 only)
    raw_df = raw_df[raw_df["config_name"] == args.config]
    raw_df = raw_df[raw_df["k"] == 1]  # Only individual reference calls
    ep_df = ep_df[ep_df["config_name"] == args.config]

    print(f"[RRN] Raw per-call: {len(raw_df)} rows")
    print(f"[RRN] Episode data: {len(ep_df)} rows")

    # Build per-reference dataset
    print("[RRN] Building RRN dataset...")
    rrn_df = build_rrn_dataset(raw_df, ep_df)
    print(f"[RRN] RRN dataset: {len(rrn_df)} rows, "
          f"success rate: {rrn_df['success'].mean():.3f}")

    # Split by query (70/15/15)
    unique_queries = rrn_df["query_case_id"].unique()
    np.random.shuffle(unique_queries)
    n = len(unique_queries)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_queries = set(unique_queries[:n_train])
    val_queries = set(unique_queries[n_train:n_train + n_val])
    test_queries = set(unique_queries[n_train + n_val:])

    train_df = rrn_df[rrn_df["query_case_id"].isin(train_queries)]
    val_df = rrn_df[rrn_df["query_case_id"].isin(val_queries)]
    test_df = rrn_df[rrn_df["query_case_id"].isin(test_queries)].copy()

    print(f"[RRN] Train: {len(train_df)} rows ({len(train_queries)} queries), "
          f"Val: {len(val_df)} rows ({len(val_queries)} queries), "
          f"Test: {len(test_df)} rows ({len(test_queries)} queries)")

    feature_cols = ["f1", "f2", "f3", "f4", "f5"]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    X_val = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))

    y_train = train_df["success"].values.astype(np.float32)
    y_val = val_df["success"].values.astype(np.float32)
    y_test = test_df["success"].values.astype(np.float32)

    # Create data loaders
    train_loader = DataLoader(RRNNuclearDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RRNNuclearDataset(X_val, y_val),
                            batch_size=args.batch_size, shuffle=False)

    # Train model
    print("[RRN] Training RRN...")
    model = ReferenceReliabilityNetwork(input_dim=5).to(device)
    model, history = train_rrn(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )

    # Evaluate on test set
    print("[RRN] Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        test_preds = model(X_test_t).cpu().numpy()

    test_df["rrn_score"] = test_preds

    # Per-reference metrics
    test_auroc = float(roc_auc_score(y_test, test_preds))
    test_auprc = float(average_precision_score(y_test, test_preds))
    print(f"[RRN] Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")

    # Selective prediction evaluation
    sel_result = evaluate_selective_prediction(
        model, test_df, device, scaler, coverage=0.80,
    )
    print(f"[RRN] CFR all: {sel_result['cfr_all']:.4f}, "
          f"CFR RRN: {sel_result['cfr_rrn']:.4f}, "
          f"Reduction: {sel_result['cfr_reduction_pct']:.1f}%")
    print(f"[RRN] Mean Dice all: {sel_result['mean_dice_all']:.4f}, "
          f"Mean Dice RRN: {sel_result['mean_dice_rrn']:.4f}")

    # Reference selection benchmark
    sel_bench = evaluate_reference_selection(test_df)
    print("[RRN] Reference selection benchmark:")
    for strategy, metrics in sel_bench.items():
        print(f"  {strategy}: Dice = {metrics['mean_dice']:.4f}")

    # Feature importance (simple correlation analysis)
    correlations = {}
    for col in feature_cols:
        corr = float(np.corrcoef(test_df[col].values, test_df["dice"].values)[0, 1])
        correlations[col] = corr

    # Save results
    output = {
        "method": "RRN",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "wp_dir": str(args.wp_dir),
        "train_queries": sorted(list(train_queries)),
        "val_queries": sorted(list(val_queries)),
        "test_queries": sorted(list(test_queries)),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "training": {
            "epochs_trained": len(history["train_loss"]),
            "best_val_loss": float(min(history["val_loss"])),
            "best_val_auroc": float(max(history["val_auroc"])),
        },
        "test_metrics": {
            "auroc": test_auroc,
            "auprc": test_auprc,
            "success_prevalence": float(y_test.mean()),
        },
        "selective_prediction": sel_result,
        "reference_selection_benchmark": sel_bench,
        "feature_correlations": correlations,
    }

    out_json = args.out_dir / "rrn_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[RRN] Results saved to {out_json}")

    # Save model
    import joblib
    model_path = args.out_dir / "rrn_model.pkl"
    joblib.dump({
        "model_state": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "feature_cols": feature_cols,
    }, model_path)
    print(f"[RRN] Model saved to {model_path}")


if __name__ == "__main__":
    main()