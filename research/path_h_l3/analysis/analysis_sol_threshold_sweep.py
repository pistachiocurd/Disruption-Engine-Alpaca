"""
analysis_sol_threshold_sweep.py — Sweep thresholds on val and break down per-day.

Tests whether May 21-22 fire at lower thresholds (where train showed close-but-
no-fire behavior, pred_max = 0.900 and 0.945) and what the per-trade economics
look like. Discriminator between:
  - Strategy is real with tight threshold calibration (May 21-22 fire with
    positive per-trade economics at thr=0.93-0.94)
  - Threshold is doing real noise-filtering (May 21-22 fire but lose money
    at lower thresholds)
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import json

HERE = Path(__file__).parent
ROOT = HERE.parent  # path-h-l3/ — where layer1_l3_sensors etc. live
sys.path.insert(0, str(ROOT))

from layer1_l3_sensors import FEATURE_COLS
from train_l3_directional import TCNSpikePredictor
from backtest_l3_directional import batched_tcn_predict

SYMBOL = "tSOLUSD"
HORIZON = 300
SEQ_LEN = 200
MAKER_FEE = 0.0010
TAKER_FEE = 0.0020
SIZE = 1.0
TRAIN_FRAC = 0.8
SEED = 42

THRESHOLDS = [0.90, 0.92, 0.93, 0.94, 0.95, 0.97, 0.98]

CSV_PATH = ROOT / "calibration" / f"l3_ticks_{SYMBOL}.csv"
WEIGHTS = ROOT / "calibration" / f"tcn_weights_l3_phase2_persym_{SYMBOL}_H{HORIZON}.pt"
STATS_FILE = ROOT / "calibration" / f"l3_feature_stats_phase2_persym_{SYMBOL}.json"
OUT_JSON = ROOT / "calibration" / "fulltest_sol_threshold_sweep.json"


def simulate(preds, prices, ts_ms, thr_long: float, thr_short: float, rng):
    pred_offset = SEQ_LEN - 1
    trades = []
    for i in range(len(preds)):
        t = i + pred_offset
        if t + HORIZON >= len(prices):
            break
        signal = float(preds[i])
        if signal > thr_long:
            side = "long"
        elif signal < thr_short:
            side = "short"
        else:
            continue
        entry_price = float(prices[t])

        upper = min(t + HORIZON + 1, len(prices))
        adverse_fill_t = -1
        if side == "long":
            for tt in range(t + 1, upper):
                if prices[tt] < entry_price:
                    adverse_fill_t = tt
                    break
        else:
            for tt in range(t + 1, upper):
                if prices[tt] > entry_price:
                    adverse_fill_t = tt
                    break
        queue_fill_t = t + 1 + int(rng.integers(upper - (t + 1))) if upper - (t + 1) > 0 else -1
        if adverse_fill_t < 0 and queue_fill_t < 0:
            continue

        exit_t = t + HORIZON
        exit_price = float(prices[exit_t])
        if side == "long":
            gross = (exit_price - entry_price) * SIZE
        else:
            gross = (entry_price - exit_price) * SIZE
        net = gross - MAKER_FEE * SIZE * entry_price - TAKER_FEE * SIZE * exit_price
        trades.append({
            "ts_ms": int(ts_ms[t]),
            "side": side,
            "net_bps": (net / entry_price) * 10000.0,
            "win": int(net > 0),
        })
    return trades


def main():
    print(f"Loading {CSV_PATH}...")
    df = pl.read_csv(CSV_PATH)
    n_total = df.height
    cut = int(n_total * TRAIN_FRAC)
    val_df = df.slice(cut, n_total - cut)
    features = np.stack(
        [val_df[c].to_numpy().astype(np.float32) for c in FEATURE_COLS], axis=1,
    )
    prices = val_df["last_trade_price"].to_numpy().astype(np.float32)
    ts_ms = val_df["timestamp_ms"].to_numpy()

    with open(STATS_FILE) as f:
        stats_blob = json.load(f)
    mean = np.array(stats_blob["mean"], dtype=np.float32)
    std = np.array(stats_blob["std"], dtype=np.float32)
    feats_norm = (features - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcn = TCNSpikePredictor(in_channels=len(FEATURE_COLS)).to(device).eval()
    tcn.load_state_dict(torch.load(WEIGHTS, map_location=device))
    preds = batched_tcn_predict(tcn, feats_norm, SEQ_LEN, device)
    print(f"  val: {len(prices):,} ticks  preds: {len(preds):,}")

    summary = {"config": f"{SYMBOL} H={HORIZON} q=1.0 fees=10/20bps", "by_threshold": []}
    for thr in THRESHOLDS:
        rng = np.random.default_rng(SEED)
        trades = simulate(preds, prices, ts_ms, thr, 1.0 - thr, rng)
        N = len(trades)
        per_day_rows = []
        if N > 0:
            bps = np.array([t["net_bps"] for t in trades])
            wins = np.array([t["win"] for t in trades])
            sides = np.array([t["side"] for t in trades])
            dates = np.array([
                datetime.fromtimestamp(t["ts_ms"]/1000, tz=timezone.utc).date()
                for t in trades
            ])
            mean_bps = float(bps.mean())
            n_long = int((sides == "long").sum())
            n_short = int((sides == "short").sum())
            print()
            print(f"=== thr = {thr:.2f}   (long > {thr}, short < {1-thr:.2f}) ===")
            print(f"  total: N={N}  L={n_long} S={n_short}  win={100*wins.mean():.1f}%  mean={mean_bps:+.2f} bps")
            print(f"  {'date':<12} {'N':>5} {'L/S':>9} {'win%':>6} {'mean_bps':>10}")
            for day in sorted(set(dates)):
                mask = dates == day
                d_n = int(mask.sum())
                d_l = int((sides[mask] == "long").sum())
                d_s = int((sides[mask] == "short").sum())
                d_win = float(wins[mask].mean() * 100)
                d_mean = float(bps[mask].mean())
                per_day_rows.append({
                    "date": str(day), "n": d_n, "long": d_l, "short": d_s,
                    "win_pct": d_win, "mean_bps": d_mean,
                })
                print(f"  {str(day):<12} {d_n:>5} {d_l:>4}/{d_s:<4} {d_win:>5.1f}% {d_mean:>+9.2f}")
            summary["by_threshold"].append({
                "thr": thr, "n_total": N, "long": n_long, "short": n_short,
                "win_pct": float(wins.mean() * 100), "mean_bps": mean_bps,
                "per_day": per_day_rows,
            })
        else:
            print()
            print(f"=== thr = {thr:.2f}   N=0 (no signals fired) ===")
            summary["by_threshold"].append({
                "thr": thr, "n_total": 0, "long": 0, "short": 0,
                "win_pct": 0.0, "mean_bps": 0.0, "per_day": [],
            })

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
