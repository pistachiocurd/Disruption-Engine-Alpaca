"""
analysis_sol_w2_feature_drift.py — Why did SOL fail on w2?

Compares window-1 and window-2 SOL feature distributions per channel.
For each of the 19 channels:
  - KS-D two-sample statistic (distribution similarity)
  - Mean shift expressed in training-time σ (how far the w2 mean is from
    where the training normalization centered it)
  - Std ratio w2/w1
  - Cross-reference with the prior ablation's load-bearing ranking

Also runs the model on w1 val + w2 and compares the prediction-output
distribution: if w2's TCN output histogram is shifted right of w1's,
that explains "fires more often" mechanically.

Outputs:
  calibration/feature_drift_w1_vs_w2_sol.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import stats as sp_stats

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from layer1_l3_sensors import FEATURE_COLS
from train_l3_directional import TCNSpikePredictor
from backtest_l3_directional import batched_tcn_predict

SYMBOL = "tSOLUSD"
HORIZON = 300
SEQ_LEN = 200
TRAIN_FRAC = 0.8

W1_CSV = ROOT / "calibration" / f"l3_ticks_{SYMBOL}.csv"
W2_CSV = ROOT / "calibration" / f"l3_ticks_w2_{SYMBOL}.csv"
WEIGHTS = ROOT / "calibration" / f"tcn_weights_l3_phase2_persym_{SYMBOL}_H{HORIZON}.pt"
STATS_FILE = ROOT / "calibration" / f"l3_feature_stats_phase2_persym_{SYMBOL}.json"
OUT_JSON = ROOT / "calibration" / "feature_drift_w1_vs_w2_sol.json"

# Top load-bearing channels from the prior inference-time ablation.
# drop_from_baseline_bps: positive = zeroing the channel HURT the model
# (channel is load-bearing); negative = zeroing IMPROVED the model
# (channel was net noise at thr=0.95).
ABLATION_DROP_BPS = {
    "lifespan_ask_p50_ms":       49.96,
    "lifespan_bid_p50_ms":       34.49,
    "hidden_trade_rate":         29.19,
    "event_density_per_s":       11.89,
    "queue_depletion_bid_per_s": 11.03,
    "queue_depletion_ask_per_s": 10.41,
    "top_bid_size":               9.82,
    "depth_imbalance_top5":       2.41,
    "spread_bps":                 0.62,
    "cancel_rate_ask_per_s":     -0.86,
    "arrival_rate_bid_per_s":    -1.31,
    "top_ask_size":              -1.55,
    "cancel_rate_bid_per_s":     -5.29,
    "aggressor_imbalance":       -6.58,
    "aggressor_autocorr_lag5":   -7.10,
    "arrival_rate_ask_per_s":   -12.83,
    "aggressor_autocorr_lag1":  -16.34,
    "lifespan_ask_p95_ms":      -28.98,
    "lifespan_bid_p95_ms":      -66.68,
}


def load_features(csv_path):
    df = pl.read_csv(csv_path)
    return np.stack(
        [df[c].to_numpy().astype(np.float32) for c in FEATURE_COLS], axis=1
    )


def main():
    print(f"Loading w1 features from {W1_CSV.name}...")
    w1 = load_features(W1_CSV)
    print(f"  w1: {w1.shape[0]:,} ticks")

    print(f"Loading w2 features from {W2_CSV.name}...")
    w2 = load_features(W2_CSV)
    print(f"  w2: {w2.shape[0]:,} ticks")

    with open(STATS_FILE) as f:
        stats_blob = json.load(f)
    train_mean = np.array(stats_blob["mean"], dtype=np.float32)
    train_std = np.array(stats_blob["std"], dtype=np.float32)

    print()
    print("=== Per-channel drift (w1 vs w2), sorted by KS-D ===")
    print(f"  {'channel':<28} {'KS-D':>6} {'w1_mean':>10} {'w2_mean':>10} "
          f"{'shift_in_train_σ':>18} {'std_ratio':>10} {'ablation_load':>14}")

    rows = []
    for i, ch in enumerate(FEATURE_COLS):
        w1_col = w1[:, i]
        w2_col = w2[:, i]
        # Subsample for KS-D to keep it tractable (uses ranks; still need O(n+m))
        # ~150k ticks per window is fine.
        ks_stat, ks_p = sp_stats.ks_2samp(w1_col, w2_col)
        w1_mean = float(w1_col.mean()); w1_std = float(w1_col.std())
        w2_mean = float(w2_col.mean()); w2_std = float(w2_col.std())
        sigma_used = float(train_std[i])
        mean_shift_in_sigmas = (w2_mean - float(train_mean[i])) / sigma_used if sigma_used > 1e-9 else 0.0
        std_ratio = w2_std / w1_std if w1_std > 1e-9 else float("inf")
        rows.append({
            "channel": ch,
            "ks_d": float(ks_stat),
            "ks_p": float(ks_p),
            "w1_mean": w1_mean, "w1_std": w1_std,
            "w2_mean": w2_mean, "w2_std": w2_std,
            "train_mean": float(train_mean[i]),
            "train_std": sigma_used,
            "mean_shift_in_train_sigmas": mean_shift_in_sigmas,
            "std_ratio": std_ratio,
            "ablation_drop_bps": ABLATION_DROP_BPS.get(ch),
        })

    rows_sorted = sorted(rows, key=lambda r: -r["ks_d"])
    for r in rows_sorted:
        load_str = (f"{r['ablation_drop_bps']:+6.2f}" if r["ablation_drop_bps"] is not None else "    -  ")
        print(f"  {r['channel']:<28} {r['ks_d']:>6.3f} "
              f"{r['w1_mean']:>10.3g} {r['w2_mean']:>10.3g} "
              f"{r['mean_shift_in_train_sigmas']:>17.2f}σ {r['std_ratio']:>10.2f}  {load_str:>14}")

    print()
    print("=== Dangerous overlap: load-bearing (drop > 0) AND high drift (KS-D >= 0.10) ===")
    dangerous = [r for r in rows
                 if r["ablation_drop_bps"] is not None
                 and r["ablation_drop_bps"] > 0
                 and r["ks_d"] >= 0.10]
    dangerous_sorted = sorted(dangerous, key=lambda r: -r["ablation_drop_bps"])
    if not dangerous_sorted:
        print("  (none) — load-bearing channels were stable across windows")
    for r in dangerous_sorted:
        print(f"  {r['channel']:<28}  KS-D={r['ks_d']:.3f}  "
              f"shift={r['mean_shift_in_train_sigmas']:+.2f}σ  "
              f"std_ratio={r['std_ratio']:.2f}  "
              f"ablation_drop={r['ablation_drop_bps']:+.2f} bps")

    # Prediction distribution shift
    print()
    print("=== Prediction-output distribution shift ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcn = TCNSpikePredictor(in_channels=len(FEATURE_COLS)).to(device).eval()
    tcn.load_state_dict(torch.load(WEIGHTS, map_location=device))

    cut = int(w1.shape[0] * TRAIN_FRAC)
    w1_val = w1[cut:]
    w1_val_norm = (w1_val - train_mean) / train_std
    w1_preds = batched_tcn_predict(tcn, w1_val_norm, SEQ_LEN, device)

    w2_norm = (w2 - train_mean) / train_std
    w2_preds = batched_tcn_predict(tcn, w2_norm, SEQ_LEN, device)

    def pquant(arr):
        return {
            "n": int(len(arr)),
            "min": float(arr.min()),
            "p1": float(np.percentile(arr, 1)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "med": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }

    w1q = pquant(w1_preds)
    w2q = pquant(w2_preds)
    print(f"  w1_val: N={w1q['n']:,}  min={w1q['min']:.3f}  p5={w1q['p5']:.3f}  "
          f"med={w1q['med']:.3f}  p95={w1q['p95']:.3f}  max={w1q['max']:.3f}  mean={w1q['mean']:.3f}")
    print(f"  w2    : N={w2q['n']:,}  min={w2q['min']:.3f}  p5={w2q['p5']:.3f}  "
          f"med={w2q['med']:.3f}  p95={w2q['p95']:.3f}  max={w2q['max']:.3f}  mean={w2q['mean']:.3f}")

    print()
    print("=== Threshold-crossing rates ===")
    print(f"  {'thr':>5} {'w1_long_fire%':>14} {'w1_short_fire%':>16} {'w2_long_fire%':>14} {'w2_short_fire%':>16}")
    threshold_rates = []
    for thr in [0.90, 0.93, 0.95, 0.97, 0.98]:
        w1_long = float((w1_preds > thr).mean() * 100)
        w2_long = float((w2_preds > thr).mean() * 100)
        w1_short = float((w1_preds < (1-thr)).mean() * 100)
        w2_short = float((w2_preds < (1-thr)).mean() * 100)
        threshold_rates.append({
            "thr": thr,
            "w1_long_fire_pct": w1_long, "w1_short_fire_pct": w1_short,
            "w2_long_fire_pct": w2_long, "w2_short_fire_pct": w2_short,
        })
        print(f"  {thr:>5.2f} {w1_long:>14.3f} {w1_short:>16.3f} {w2_long:>14.3f} {w2_short:>16.3f}")

    # Save summary
    summary = {
        "config": f"{SYMBOL} H={HORIZON} (window-1-trained weights applied to both windows)",
        "per_channel_drift_sorted_by_ks_d": rows_sorted,
        "dangerous_overlap_load_bearing_AND_drifted": dangerous_sorted,
        "prediction_distribution": {"w1_val": w1q, "w2": w2q},
        "threshold_crossing_rates": threshold_rates,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
