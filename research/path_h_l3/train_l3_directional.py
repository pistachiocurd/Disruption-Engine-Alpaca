"""
train_l3_directional.py - TCN directional classifier on Bitfinex L3 ticks.

Reads per-symbol event-clock tick CSVs produced by aggregate_mbo_events.py
(11 L3 feature channels + last_trade_price + timestamp), trains a TCN at
horizon H=100 ticks with the directional target (label[t]=1 iff
last_trade_price[t+H] > last_trade_price[t]), and reports an F2-tuned
threshold sweep on held-out val.

Protocol matches LAYER2_TRAINING.md §14.4 so the L3 result is directly
comparable to the L2 baseline (F2=0.106 / max prec 1.5x base rate /
gross edge 0.06 bps per trade -> all need to be beaten for L3 to be
the right answer per L3_RESEARCH_PLAN.md §6).

Inputs: calibration/l3_ticks_{tBTCUSD,tETHUSD,tSOLUSD}.csv (default).
Outputs:
  - calibration/tcn_weights_l3.pt              (trained TCN state dict)
  - calibration/l3_feature_stats.json          (per-channel mean/std from train pool)
  - calibration/l3_directional_metrics.json    (val sweep + best F2 threshold)

Usage:
    python train_l3_directional.py
    python train_l3_directional.py --horizon 100 --seq-len 60 --epochs 1
    python train_l3_directional.py --horizon 300                 # 5-min-ish horizon test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

# research/path_h_l3/ → repo root is two levels up
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from layer2_alpha import TCNSpikePredictor

# Single source of truth for the L3 feature channel list. Imported here so
# updating the sensor stack automatically propagates to training without
# a manual edit (and any column drift between aggregator + trainer +
# backtester is impossible).
from layer1_l3_sensors import FEATURE_COLS  # noqa: F401 (re-exported for legacy callers)


DEFAULT_CSV_DIR = Path(__file__).parent / "calibration"
DEFAULT_SYMBOLS = ["tBTCUSD", "tETHUSD", "tSOLUSD"]


def load_split(csv_path: Path, split: str, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Read one symbol's CSV, return (features, prices) for the requested
    temporal split. Per-symbol split avoids cross-coin leakage in val.
    """
    df = pl.read_csv(csv_path)
    n = df.height
    cut = int(n * train_frac)
    if split == "train":
        df = df.slice(0, cut)
    elif split == "val":
        df = df.slice(cut, n - cut)
    else:
        raise ValueError(split)
    features = np.stack(
        [df[c].to_numpy().astype(np.float32) for c in FEATURE_COLS],
        axis=1,
    )
    prices = df["last_trade_price"].to_numpy().astype(np.float32)
    return features, prices


def compute_norm_stats(feature_blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a list of feature arrays (each is
    (n_i, n_channels)). Used only on the training pool to avoid val
    leakage. Std floored at 1e-6 so degenerate channels don't blow up.
    """
    stacked = np.concatenate(feature_blocks, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def make_directional_labels(prices: np.ndarray, horizon: int) -> np.ndarray:
    """label[t] = 1 iff prices[t+H] > prices[t]. Last H positions get 0
    (no future available — negligible bias when H << n)."""
    n = len(prices)
    y = np.zeros(n, dtype=np.float32)
    if n > horizon:
        y[: n - horizon] = (prices[horizon:] > prices[:-horizon]).astype(np.float32)
    return y


class PerSymbolWindowDataset(IterableDataset):
    """Yields (window, label) for one symbol-split's pre-loaded arrays.
    Sliding window of `seq_len` past ticks; label is mid[t+H] > mid[t].
    Boundary-clean (each symbol is processed independently)."""

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        seq_len: int,
        horizon: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.features = features
        self.prices = prices
        self.seq_len = seq_len
        self.horizon = horizon
        self.mean = mean
        self.std = std

    def __iter__(self):
        # Z-score normalize at iter time so we don't hold double the memory.
        feats_norm = (self.features - self.mean) / self.std
        labels = make_directional_labels(self.prices, self.horizon)
        n = len(feats_norm)
        # Window i uses features[i-seq_len:i] to predict at time i (label[i]).
        # We need both a full backlook (i >= seq_len) AND a valid label
        # (i + horizon < n) since the last `horizon` positions have y=0
        # by construction but they're not "future down" - they're "future unknown."
        last_valid = n - self.horizon
        for i in range(self.seq_len, last_valid):
            X = feats_norm[i - self.seq_len : i].T  # (n_ch, seq_len)
            y = labels[i]
            yield torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


class MultiSymbolDataset(IterableDataset):
    """Concatenates per-symbol streams in fixed order. With ~605K total
    ticks across 3 symbols and the model's stationarity assumption, the
    coin-by-coin order is fine for a single-epoch run; for multi-epoch
    consider wrapping in ShuffledBufferDataset like train_stream.py does."""

    def __init__(self, datasets: list[IterableDataset]) -> None:
        self.datasets = datasets

    def __iter__(self):
        for ds in self.datasets:
            yield from ds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--horizon", type=int, default=100,
                   help="Prediction horizon in ticks. 100 matches L2 §14.4.")
    p.add_argument("--seq-len", type=int, default=60,
                   help="TCN input backlook in ticks. 60 matches L2 baseline.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--weights-out", type=Path,
                   default=DEFAULT_CSV_DIR / "tcn_weights_l3.pt")
    p.add_argument("--stats-out", type=Path,
                   default=DEFAULT_CSV_DIR / "l3_feature_stats.json")
    p.add_argument("--metrics-out", type=Path,
                   default=DEFAULT_CSV_DIR / "l3_directional_metrics.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, symbols={args.symbols}, H={args.horizon}, seq_len={args.seq_len}")

    # ----- load + split + normalize -----
    train_blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    val_blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym in args.symbols:
        path = args.csv_dir / f"l3_ticks_{sym}.csv"
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        train_blocks[sym] = load_split(path, "train", args.train_frac)
        val_blocks[sym] = load_split(path, "val", args.train_frac)
        print(f"  {sym}: train={train_blocks[sym][0].shape[0]:,} ticks  "
              f"val={val_blocks[sym][0].shape[0]:,} ticks")

    mean, std = compute_norm_stats([feat for feat, _ in train_blocks.values()])
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_out, "w") as f:
        json.dump({
            "feature_cols": FEATURE_COLS,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }, f, indent=2)
    print(f"[setup] wrote normalization stats -> {args.stats_out}")

    # ----- val label base rate -----
    val_labels_total = []
    for sym, (feat, prices) in val_blocks.items():
        val_labels_total.append(make_directional_labels(prices, args.horizon)[: len(prices) - args.horizon])
    val_labels_concat = np.concatenate(val_labels_total)
    base_rate = float(val_labels_concat.mean())
    print(f"[setup] val base rate (P(up at H={args.horizon})) = {base_rate:.4f}")

    # ----- model -----
    model = TCNSpikePredictor(in_channels=len(FEATURE_COLS)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] TCN in_channels={len(FEATURE_COLS)} params={n_params:,}")

    criterion = nn.BCEWithLogitsLoss()  # pos_weight=1.0 default; balanced target.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----- train -----
    train_ds = MultiSymbolDataset([
        PerSymbolWindowDataset(feat, prices, args.seq_len, args.horizon, mean, std)
        for sym, (feat, prices) in train_blocks.items()
    ])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        n_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits = model.forward_logits(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            n_samples += X_batch.size(0)
            if n_batches % 50 == 0:
                print(f"  epoch {epoch}  batch {n_batches}  "
                      f"running avg loss {total_loss/n_batches:.4f}  "
                      f"samples {n_samples:,}")
        print(f"[train] epoch {epoch} done: "
              f"avg_loss={total_loss/max(n_batches,1):.4f}  total_samples={n_samples:,}")

    # ----- val: threshold sweep -----
    val_ds = MultiSymbolDataset([
        PerSymbolWindowDataset(feat, prices, args.seq_len, args.horizon, mean, std)
        for sym, (feat, prices) in val_blocks.items()
    ])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    model.eval()
    probs_chunks = []
    labels_chunks = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model.forward_logits(X_batch)
            probs_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels_chunks.append(y_batch.numpy())
    probs = np.concatenate(probs_chunks)
    labels = np.concatenate(labels_chunks)
    y_bool = labels >= 0.5
    n_val = len(probs)
    val_pos_rate = float(y_bool.mean())
    print(f"[val] N={n_val:,}  base rate={val_pos_rate:.4f}")

    thresholds = np.concatenate([
        np.linspace(0.30, 0.45, 4),
        np.linspace(0.46, 0.54, 9),  # dense near 0.5 since balanced target
        np.linspace(0.55, 0.70, 4),
        np.array([0.75, 0.80, 0.85, 0.90]),
    ])
    sweep = []
    for thr in thresholds:
        pred = probs >= thr
        tp = int(np.logical_and(pred, y_bool).sum())
        fp = int(np.logical_and(pred, ~y_bool).sum())
        fn = int(np.logical_and(~pred, y_bool).sum())
        pred_pos = tp + fp
        prec = tp / max(pred_pos, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
        lift = prec / max(val_pos_rate, 1e-9)
        sweep.append({
            "thr": round(float(thr), 4),
            "pred_pos": pred_pos,
            "pred_pos_frac": round(pred_pos / max(n_val, 1), 4),
            "tp": tp, "fp": fp, "fn": fn,
            "prec": round(prec, 4),
            "rec": round(rec, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "lift_vs_base": round(lift, 3),
        })

    best_f2 = max(sweep, key=lambda r: r["f2"])
    best_prec = max(sweep, key=lambda r: r["prec"])

    print(f"\n[val] threshold sweep (n_val={n_val:,}, base_rate={val_pos_rate:.4f}):")
    print(f"  {'thr':>6} {'pred%':>8} {'prec':>7} {'rec':>7} {'f2':>7} {'lift':>7}")
    for r in sweep:
        print(f"  {r['thr']:>6.3f} {r['pred_pos_frac']*100:>7.2f}% "
              f"{r['prec']:>7.4f} {r['rec']:>7.4f} {r['f2']:>7.4f} {r['lift_vs_base']:>6.3f}x")

    print(f"\n[best F2]   thr={best_f2['thr']}  prec={best_f2['prec']}  rec={best_f2['rec']}  "
          f"F2={best_f2['f2']}  lift={best_f2['lift_vs_base']}x")
    print(f"[best prec] thr={best_prec['thr']}  prec={best_prec['prec']}  rec={best_prec['rec']}  "
          f"F2={best_prec['f2']}  lift={best_prec['lift_vs_base']}x  "
          f"(pred_pos={best_prec['pred_pos']})")

    print(f"\nL2 baselines to beat (LAYER2_TRAINING.md §14.4):")
    print(f"  F2 (directional, H=100, val) = 0.106    -> L3 best F2 = {best_f2['f2']}")
    print(f"  max prec / base rate         = 1.5x     -> L3 max lift = {best_prec['lift_vs_base']}x")

    # ----- save -----
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.weights_out)
    with open(args.metrics_out, "w") as f:
        json.dump({
            "args": {
                "horizon": args.horizon, "seq_len": args.seq_len,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "train_frac": args.train_frac,
                "symbols": args.symbols,
            },
            "n_val": n_val,
            "val_base_rate": val_pos_rate,
            "best_f2": best_f2,
            "best_prec": best_prec,
            "sweep": sweep,
            "l2_baselines": {"f2_directional": 0.106, "max_prec_lift": 1.5},
        }, f, indent=2)
    print(f"\n[save] weights  -> {args.weights_out}")
    print(f"[save] metrics  -> {args.metrics_out}")


if __name__ == "__main__":
    main()
