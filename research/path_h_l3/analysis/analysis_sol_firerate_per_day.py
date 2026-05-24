"""
analysis_sol_firerate_per_day.py — Per-day fire-rate + prediction-distribution
diagnostic on the SOL Phase 2 model, run separately on the train and val splits.

Discriminates between two failure modes for the SOL signal:
  A. Regime overfit  — model fires on only 1-2 training days; May 23 was just
     another instance of that rare condition.
  B. Rare-firer by design — model fires on most training days but only 1 of 3
     val days because val happens to contain few "fire-worthy" days.

Output also includes per-day prediction quantiles (min/p5/med/p95/max) so we
can see whether non-firing days are "model is neutral (0.3-0.7)" or "model is
confident but just under threshold (0.85-0.94)".
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent  # path-h-l3/ — where layer1_l3_sensors etc. live
sys.path.insert(0, str(ROOT))

from layer1_l3_sensors import FEATURE_COLS
from train_l3_directional import TCNSpikePredictor
from backtest_l3_directional import batched_tcn_predict

SYMBOL = "tSOLUSD"
HORIZON = 300
SEQ_LEN = 200
THR_LONG = 0.95
THR_SHORT = 0.05
TRAIN_FRAC = 0.8

CSV_PATH = ROOT / "calibration" / f"l3_ticks_{SYMBOL}.csv"
WEIGHTS = ROOT / "calibration" / f"tcn_weights_l3_phase2_persym_{SYMBOL}_H{HORIZON}.pt"
STATS_FILE = ROOT / "calibration" / f"l3_feature_stats_phase2_persym_{SYMBOL}.json"
OUT_JSON = ROOT / "calibration" / "fulltest_sol_firerate_per_day.json"


def per_day(split_name: str, sub_df: pl.DataFrame, mean, std, tcn, device):
    n = sub_df.height
    features = np.stack(
        [sub_df[c].to_numpy().astype(np.float32) for c in FEATURE_COLS],
        axis=1,
    )
    timestamps_ms = sub_df["timestamp_ms"].to_numpy()
    feats_norm = (features - mean) / std
    preds = batched_tcn_predict(tcn, feats_norm, SEQ_LEN, device)

    pred_offset = SEQ_LEN - 1
    pred_indices = np.arange(len(preds)) + pred_offset
    valid = pred_indices < n
    pred_indices = pred_indices[valid]
    preds = preds[valid]

    tick_dates = np.array(
        [datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).date() for ts in timestamps_ms]
    )
    pred_dates = tick_dates[pred_indices]

    long_sig = preds > THR_LONG
    short_sig = preds < THR_SHORT

    rows = []
    print(f"\n  {split_name:<5}  {'date':<12} {'ticks':>8} {'preds':>8} "
          f"{'long':>5} {'short':>5} {'fire%':>7} "
          f"{'p_min':>7} {'p_p5':>7} {'p_med':>7} {'p_p95':>7} {'p_max':>7}")
    for day in sorted(set(tick_dates)):
        tick_n = int((tick_dates == day).sum())
        mask = pred_dates == day
        p_n = int(mask.sum())
        l_n = int(long_sig[mask].sum())
        s_n = int(short_sig[mask].sum())
        fire = (l_n + s_n) / p_n * 100 if p_n else 0.0
        dp = preds[mask]
        q = {
            "min": float(dp.min()) if len(dp) else 0.0,
            "p5":  float(np.percentile(dp, 5)) if len(dp) else 0.0,
            "med": float(np.median(dp)) if len(dp) else 0.0,
            "p95": float(np.percentile(dp, 95)) if len(dp) else 0.0,
            "max": float(dp.max()) if len(dp) else 0.0,
        }
        rows.append({
            "date": str(day),
            "ticks": tick_n,
            "predictions": p_n,
            "long_signals": l_n,
            "short_signals": s_n,
            "fire_rate_pct": fire,
            "pred_quantiles": q,
        })
        print(f"  {split_name:<5}  {str(day):<12} {tick_n:>8,} {p_n:>8,} "
              f"{l_n:>5} {s_n:>5} {fire:>6.2f}% "
              f"{q['min']:>7.3f} {q['p5']:>7.3f} {q['med']:>7.3f} "
              f"{q['p95']:>7.3f} {q['max']:>7.3f}")
    return rows


def main():
    print(f"Loading {CSV_PATH}...")
    df = pl.read_csv(CSV_PATH)
    n_total = df.height
    cut = int(n_total * TRAIN_FRAC)
    print(f"  total: {n_total:,} ticks  (train: {cut:,}  val: {n_total-cut:,})")

    with open(STATS_FILE) as f:
        stats_blob = json.load(f)
    mean = np.array(stats_blob["mean"], dtype=np.float32)
    std = np.array(stats_blob["std"], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcn = TCNSpikePredictor(in_channels=len(FEATURE_COLS)).to(device).eval()
    tcn.load_state_dict(torch.load(WEIGHTS, map_location=device))
    print(f"  device={device}")

    train_rows = per_day("train", df.slice(0, cut), mean, std, tcn, device)
    val_rows = per_day("val", df.slice(cut, n_total - cut), mean, std, tcn, device)

    train_fire_days = sum(1 for d in train_rows if d['long_signals'] + d['short_signals'] > 0)
    val_fire_days = sum(1 for d in val_rows if d['long_signals'] + d['short_signals'] > 0)

    print()
    print("=== SUMMARY ===")
    print(f"  Train: fired on {train_fire_days}/{len(train_rows)} days")
    print(f"  Val:   fired on {val_fire_days}/{len(val_rows)} days")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "config": f"{SYMBOL} H={HORIZON} thr={THR_LONG}",
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_fire_days": train_fire_days,
            "train_total_days": len(train_rows),
            "val_fire_days": val_fire_days,
            "val_total_days": len(val_rows),
        }, f, indent=2)
    print(f"  JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
