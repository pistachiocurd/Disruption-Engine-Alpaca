"""
train_tcn.py - Supervised training of TCNSpikePredictor.

Reads a feature_history.csv (real or synthetic) with columns:
    timestamp_ms, ce_ratio, obi, liquidation_rate, vpin, ..., best_bid, best_ask

Process:
    1. Reconstruct ob_snapshots and vpin_series.
    2. Call calibration.identify_shock_events() to label each tick.
    3. Build (60-tick window, label) pairs. Label=1 iff a shock starts within
       the next MAX_DIFFUSION_TICKS (200) ticks. Otherwise 0.
    4. Train TCNSpikePredictor with BCE loss.
    5. Save weights to ./calibration/tcn_weights.pt (engine auto-loads on start).

Usage:
    # Synthetic data path (recommended for plumbing test):
    python synthetic_data.py
    python train_tcn.py --csv ./calibration/synthetic_history.csv

    # Real data path (requires enough shocks in your captured data):
    python train_tcn.py --csv ./calibration/feature_history.csv

    python train_tcn.py --epochs 20 --val-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from calibration import identify_shock_events
from layer2_alpha import TCNSpikePredictor


# ----------------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------------
def load_csv(path: Path) -> dict:
    """Load CSV produced by FeatureDumper / synthetic_data.py into arrays."""
    cols: dict[str, list] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for h in header:
            cols[h] = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            for h, p in zip(header, parts):
                cols[h].append(p)

    required = {
        "timestamp_ms", "ce_ratio", "obi", "liquidation_rate", "vpin",
        "best_bid", "best_ask",
    }
    missing = required - set(cols)
    if missing:
        raise SystemExit(
            f"ERROR: CSV {path} is missing required columns: {missing}\n"
            f"  found: {list(cols)}\n"
            f"  Hint: regenerate via synthetic_data.py, or re-run the engine "
            f"with the latest FeatureDumper to add best_bid/best_ask."
        )

    return {
        "timestamp_ms": np.array([int(x) for x in cols["timestamp_ms"]], dtype=np.int64),
        "ce_ratio": np.array([float(x) for x in cols["ce_ratio"]], dtype=np.float32),
        "obi": np.array([float(x) for x in cols["obi"]], dtype=np.float32),
        "liquidation_rate": np.array([float(x) for x in cols["liquidation_rate"]], dtype=np.float32),
        "vpin": np.array([float(x) for x in cols["vpin"]], dtype=np.float32),
        "best_bid": np.array([float(x) for x in cols["best_bid"]], dtype=np.float32),
        "best_ask": np.array([float(x) for x in cols["best_ask"]], dtype=np.float32),
    }


# ----------------------------------------------------------------------------
# Labeling
# ----------------------------------------------------------------------------
def build_labels(data: dict) -> tuple[np.ndarray, list]:
    """
    For each tick t, label = 1 iff a shock event starts within the next
    MAX_DIFFUSION_TICKS ticks. Returns (labels[N], events).
    """
    n = len(data["timestamp_ms"])
    ob_snapshots = [
        {
            "timestamp": int(data["timestamp_ms"][i]),
            "bids": [[float(data["best_bid"][i]), 1.0]],
            "asks": [[float(data["best_ask"][i]), 1.0]],
        }
        for i in range(n)
    ]
    vpin_series = data["vpin"].tolist()

    events = identify_shock_events(ob_snapshots, vpin_series)
    print(f"identify_shock_events found {len(events)} shock events.")
    if len(events) < 5:
        print(
            "WARNING: very few shocks. Training will be near-trivial. "
            "Consider increasing --shocks or collecting more real data.",
            file=sys.stderr,
        )

    labels = np.zeros(n, dtype=np.float32)
    horizon = config.MAX_DIFFUSION_TICKS
    for ev in events:
        # Mark every tick in [ev.t_index - horizon, ev.t_index - 1] as positive
        # ("a shock will occur within the next horizon ticks").
        lo = max(0, ev.t_index - horizon)
        hi = ev.t_index
        labels[lo:hi] = 1.0
    return labels, events


# ----------------------------------------------------------------------------
# Window construction
# ----------------------------------------------------------------------------
def build_windows(
    data: dict,
    labels: np.ndarray,
    window: int = config.TCN_INPUT_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (N, 3, window) input tensor and (N,) label tensor.

    For each anchor tick t in [window-1, len-1], include the window ending at t
    and label[t]. (We label the END of each window so the model is causal.)
    """
    n = len(labels)
    if n < window:
        raise SystemExit(f"ERROR: not enough rows ({n}) for window {window}.")

    feats = np.stack(
        [data["ce_ratio"], data["obi"], data["liquidation_rate"]], axis=0
    )  # (3, N)
    n_windows = n - window + 1
    X = np.zeros((n_windows, 3, window), dtype=np.float32)
    y = np.zeros((n_windows,), dtype=np.float32)
    for i in range(n_windows):
        X[i] = feats[:, i:i + window]
        y[i] = labels[i + window - 1]
    return X, y


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    max_pos_weight: float = 10.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
) -> tuple[TCNSpikePredictor, np.ndarray, np.ndarray]:
    """Train and return (model, val_predictions, val_labels)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.from_numpy(X[train_idx]).float()
    y_tr = torch.from_numpy(y[train_idx]).float()
    X_va = torch.from_numpy(X[val_idx]).float()
    y_va = torch.from_numpy(y[val_idx]).float()

    raw_pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    pos_weight = min(raw_pos_weight, max_pos_weight)
    capped_note = f"  (capped from {raw_pos_weight:.1f})" if pos_weight < raw_pos_weight else ""
    print(f"train: {len(X_tr)}  val: {len(X_va)}  "
          f"pos rate: {y_tr.mean().item():.4f}  pos_weight: {pos_weight:.2f}{capped_note}")

    model = TCNSpikePredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], device=device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # Get logits by re-doing the forward without sigmoid, so BCE-with-logits
            # gives numerically stable + class-weighted loss.
            x = model.input_proj(xb)
            for blk in model.blocks:
                x = blk(x)
            logits = model.head(x[:, :, -1]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        train_loss = total / max(n_batches, 1)

        # Validation at default 0.5 threshold for per-epoch progress reporting.
        model.eval()
        with torch.no_grad():
            preds_val = model(X_va.to(device)).cpu().numpy()
        thr = 0.5
        tp = float(((preds_val >= thr) & (y_va.numpy() == 1)).sum())
        fp = float(((preds_val >= thr) & (y_va.numpy() == 0)).sum())
        fn = float(((preds_val < thr) & (y_va.numpy() == 1)).sum())
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        print(
            f"epoch {epoch:2d}  train_loss {train_loss:.4f}  "
            f"val_pos_rate {(preds_val >= thr).mean():.3f}  "
            f"prec@0.5 {prec:.3f}  rec@0.5 {rec:.3f}"
        )

    return model, preds_val, y_va.numpy()


# ----------------------------------------------------------------------------
# PR curve + threshold selection
# ----------------------------------------------------------------------------
def sweep_pr_curve(preds: np.ndarray, labels: np.ndarray, n_points: int = 99) -> list[dict]:
    """Sweep thresholds and return precision/recall at each."""
    thresholds = np.linspace(0.01, 0.99, n_points)
    curve = []
    for t in thresholds:
        pred_pos = preds >= t
        actual_pos = labels == 1
        tp = float((pred_pos & actual_pos).sum())
        fp = float((pred_pos & ~actual_pos).sum())
        fn = float((~pred_pos & actual_pos).sum())
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        curve.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "pred_pos_rate": float(pred_pos.mean()),
        })
    return curve


def pick_threshold(curve: list[dict], target_precision: float) -> tuple[dict, bool]:
    """
    Lowest threshold whose precision >= target. Lowest = highest recall among
    qualifying. Returns (chosen_point, target_satisfied).
    """
    qualifying = [p for p in curve if p["precision"] >= target_precision]
    if not qualifying:
        # Best we can do is the highest-precision point.
        best = max(curve, key=lambda p: p["precision"])
        return best, False
    chosen = min(qualifying, key=lambda p: p["threshold"])
    return chosen, True


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=config.FEATURE_DUMP_PATH,
        help=(
            "Input CSV. Defaults to the per-symbol live CSV "
            f"({config.FEATURE_DUMP_PATH}). "
            "Pass ./calibration/synthetic_history.csv for synthetic training."
        ),
    )
    parser.add_argument(
        "--out",
        default=config.TCN_WEIGHTS_PATH,
        help=(
            "Output weights file. Engine auto-loads from this per-symbol path: "
            f"{config.TCN_WEIGHTS_PATH}"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=10.0,
        help="Cap on the positive-class weight in BCE-with-logits. Prevents "
             "the 'fire on everything' collapse when the class split is very "
             "imbalanced (default 10).",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.7,
        help="After training, sweep validation thresholds and pick the lowest "
             "where precision >= this value. The chosen threshold is written "
             "to a sidecar JSON loaded by the engine on startup (default 0.7).",
    )
    parser.add_argument(
        "--threshold-out",
        default=config.TCN_THRESHOLD_PATH,
        help=f"Sidecar JSON for chosen threshold + PR curve. "
             f"Default: {config.TCN_THRESHOLD_PATH}",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    data = load_csv(csv_path)
    n = len(data["timestamp_ms"])
    print(f"loaded {n} rows from {csv_path}")

    labels, events = build_labels(data)
    pos_rate = float(labels.mean())
    print(f"labels: {int(labels.sum())} positive / {n} total  ({pos_rate:.4f})")

    if pos_rate == 0.0:
        print(
            "ERROR: no positive labels - identify_shock_events found no shocks.\n"
            "       The TCN cannot learn from this data. Either:\n"
            "         - generate more shocks via synthetic_data.py --shocks N\n"
            "         - or collect more real data with actual market shocks.",
            file=sys.stderr,
        )
        sys.exit(1)

    X, y = build_windows(data, labels)
    print(f"windows: X={X.shape}  y={y.shape}")

    model, val_preds, val_labels = train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        max_pos_weight=args.max_pos_weight,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved TCN weights to {out_path}")

    # PR curve sweep + recommended threshold ---------------------------------
    curve = sweep_pr_curve(val_preds, val_labels)
    chosen, ok = pick_threshold(curve, args.target_precision)

    sidecar = {
        "threshold": chosen["threshold"],
        "target_precision": float(args.target_precision),
        "achieved_precision": chosen["precision"],
        "achieved_recall": chosen["recall"],
        "achieved_f1": chosen["f1"],
        "satisfied_target": bool(ok),
        "trained_on_rows": int(len(data["vpin"])),
        "trained_on_shocks": int(len(events)),
        "trained_on_csv": str(csv_path.resolve()),
        "pr_curve": curve,
    }
    sidecar_path = Path(args.threshold_out)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print()
    print(f"PR curve sweep + threshold selection -> {sidecar_path}")
    print(f"  target precision:       {args.target_precision:.3f}")
    if ok:
        print(f"  chosen threshold:       {chosen['threshold']:.4f}")
        print(f"  precision @ threshold:  {chosen['precision']:.3f}")
        print(f"  recall @ threshold:     {chosen['recall']:.3f}")
        print(f"  F1 @ threshold:         {chosen['f1']:.3f}")
    else:
        print(f"  *** TARGET NOT REACHED *** model can't hit {args.target_precision:.2f} precision anywhere.")
        print(f"  best precision available: {chosen['precision']:.3f} at t={chosen['threshold']:.4f} (recall={chosen['recall']:.3f})")
        print("  Consider: more training data, more epochs, lower target, or accept that the model isn't ready.")
    print()
    print("Restart the engine to load. Look for 'loaded TCN weights' and 'TCN threshold'.")


if __name__ == "__main__":
    main()
