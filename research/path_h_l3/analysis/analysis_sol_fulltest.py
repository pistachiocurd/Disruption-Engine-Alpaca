"""
analysis_sol_fulltest.py — In-sample full-test suite for SOL Phase 2 model.

Runs the trained per-symbol SOL TCN on the val split and produces:
  1. Per-day edge breakdown (UTC date)
  4. Random-entry bootstrap (model vs null hypothesis)
  5. Bootstrap CI on mean bps/trade (statistical significance)
  6. Trade-level distribution (payoff ratio, win/loss asymmetry)
  7. Time-of-day pattern (UTC hour bins)
  8. Max drawdown + longest losing streak (chronological)

Outputs:
  calibration/fulltest_sol_trades.csv  — per-trade ledger
  calibration/fulltest_sol_summary.json — aggregated results
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
MAKER_FEE = 0.0010
TAKER_FEE = 0.0020
SIZE = 1.0
TRAIN_FRAC = 0.8
SEED = 42
N_BOOTSTRAP = 1000

CSV_PATH = ROOT / "calibration" / f"l3_ticks_{SYMBOL}.csv"
WEIGHTS = ROOT / "calibration" / f"tcn_weights_l3_phase2_persym_{SYMBOL}_H{HORIZON}.pt"
STATS_FILE = ROOT / "calibration" / f"l3_feature_stats_phase2_persym_{SYMBOL}.json"

OUT_TRADES_CSV = ROOT / "calibration" / "fulltest_sol_trades.csv"
OUT_SUMMARY_JSON = ROOT / "calibration" / "fulltest_sol_summary.json"


def main() -> None:
    print(f"Loading {CSV_PATH}...")
    df = pl.read_csv(CSV_PATH)
    n_total = df.height
    cut = int(n_total * TRAIN_FRAC)
    val_df = df.slice(cut, n_total - cut)
    n = val_df.height
    val_ts_first = int(val_df["timestamp_ms"][0])
    val_ts_last = int(val_df["timestamp_ms"][-1])
    full_ts_first = int(df["timestamp_ms"][0])
    full_ts_last = int(df["timestamp_ms"][-1])
    print(f"  full: {n_total:,} ticks  val: {n:,} ticks  ({TRAIN_FRAC:.0%} train cut)")
    print(f"  full span: {datetime.fromtimestamp(full_ts_first/1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S} "
          f"-> {datetime.fromtimestamp(full_ts_last/1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S}  "
          f"({(full_ts_last-full_ts_first)/86400000:.2f} days)")
    print(f"  val  span: {datetime.fromtimestamp(val_ts_first/1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S} "
          f"-> {datetime.fromtimestamp(val_ts_last/1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S}  "
          f"({(val_ts_last-val_ts_first)/86400000:.2f} days)")

    features = np.stack(
        [val_df[c].to_numpy().astype(np.float32) for c in FEATURE_COLS],
        axis=1,
    )
    prices = val_df["last_trade_price"].to_numpy().astype(np.float32)
    timestamps_ms = val_df["timestamp_ms"].to_numpy()

    with open(STATS_FILE) as f:
        stats_blob = json.load(f)
    mean = np.array(stats_blob["mean"], dtype=np.float32)
    std = np.array(stats_blob["std"], dtype=np.float32)
    feats_norm = (features - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcn = TCNSpikePredictor(in_channels=len(FEATURE_COLS)).to(device).eval()
    tcn.load_state_dict(torch.load(WEIGHTS, map_location=device))
    print(f"Inferring {len(feats_norm):,} ticks on {device}...")
    preds = batched_tcn_predict(tcn, feats_norm, SEQ_LEN, device)
    print(f"  predictions: {len(preds):,} windows")

    rng = np.random.default_rng(SEED)
    trades: list[dict] = []
    pred_offset = SEQ_LEN - 1

    for i in range(len(preds)):
        t = i + pred_offset
        if t + HORIZON >= len(prices):
            break
        signal = float(preds[i])
        if signal > THR_LONG:
            side = "long"
        elif signal < THR_SHORT:
            side = "short"
        else:
            continue
        entry_price = float(prices[t])
        entry_ts_ms = int(timestamps_ms[t])

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

        queue_fill_t = -1
        window_len = upper - (t + 1)
        if window_len > 0:
            queue_fill_t = t + 1 + int(rng.integers(window_len))

        if adverse_fill_t < 0 and queue_fill_t < 0:
            continue

        exit_t = t + HORIZON
        if exit_t >= len(prices):
            continue
        exit_price = float(prices[exit_t])

        if side == "long":
            gross = (exit_price - entry_price) * SIZE
        else:
            gross = (entry_price - exit_price) * SIZE

        maker_rebate = -MAKER_FEE * SIZE * entry_price
        taker_fee = TAKER_FEE * SIZE * exit_price
        net = gross + maker_rebate - taker_fee
        net_bps = (net / entry_price) * 10000.0

        trades.append({
            "entry_ts_ms": entry_ts_ms,
            "entry_t": t,
            "exit_t": exit_t,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross,
            "net_pnl": net,
            "net_bps": net_bps,
            "win": int(net > 0),
            "signal": signal,
        })

    N = len(trades)
    if N == 0:
        print("No trades — aborting.")
        return
    print(f"  trades: N={N}")

    pl.DataFrame(trades).write_csv(OUT_TRADES_CSV)
    print(f"  per-trade CSV: {OUT_TRADES_CSV}")

    net_bps_arr = np.array([t["net_bps"] for t in trades])
    wins = np.array([t["win"] for t in trades])
    sides = np.array([t["side"] for t in trades])
    entry_ts = np.array([t["entry_ts_ms"] for t in trades])

    baseline_mean_bps = float(net_bps_arr.mean())
    n_long = int((sides == "long").sum())
    n_short = int((sides == "short").sum())
    print()
    print(f"=== Baseline (val window, thr={THR_LONG}, fees={MAKER_FEE*1e4:.0f}/{TAKER_FEE*1e4:.0f}bps) ===")
    print(f"  N={N}  long={n_long}  short={n_short}  win%={100*wins.mean():.1f}  mean bps/trade={baseline_mean_bps:+.2f}")

    # Test 1: Per-day breakdown — also report tick density to distinguish
    # model-signal concentration from uneven tick distribution
    print()
    print("=== Test 1: Per-day breakdown (UTC) ===")
    val_ts_arr = val_df["timestamp_ms"].to_numpy()
    val_tick_days = np.array([datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).date() for ts in val_ts_arr])
    days = np.array([datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).date() for ts in entry_ts])
    unique_days = sorted(set(val_tick_days))
    per_day = []
    print(f"  {'date':<12} {'val_ticks':>10} {'trades':>7} {'fire_rate':>10} {'win%':>6} {'mean_bps':>10}")
    for day in unique_days:
        tick_n = int((val_tick_days == day).sum())
        mask = days == day
        d_n = int(mask.sum())
        d_mean = float(net_bps_arr[mask].mean()) if d_n > 0 else 0.0
        d_win = float(wins[mask].mean() * 100) if d_n > 0 else 0.0
        fire_rate = (d_n / tick_n * 100) if tick_n > 0 else 0.0
        per_day.append({"date": str(day), "val_ticks": tick_n, "trades": d_n,
                        "fire_rate_pct": fire_rate, "mean_bps": d_mean, "win_pct": d_win})
        print(f"  {str(day):<12} {tick_n:>10,} {d_n:>7} {fire_rate:>9.2f}% {d_win:>5.1f}% {d_mean:>+9.2f}")

    # Test 7: Time-of-day pattern
    print()
    print("=== Test 7: Time-of-day (UTC hour) ===")
    hours = np.array([datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).hour for ts in entry_ts])
    per_hour = []
    for h in range(24):
        mask = hours == h
        h_n = int(mask.sum())
        if h_n == 0:
            continue
        h_mean = float(net_bps_arr[mask].mean())
        h_win = float(wins[mask].mean() * 100)
        per_hour.append({"hour_utc": h, "n": h_n, "mean_bps": h_mean, "win_pct": h_win})
        print(f"  {h:02d}:00  N={h_n:4d}  win={h_win:5.1f}%  mean={h_mean:+7.2f} bps")

    # Test 4: Random-entry bootstrap
    print()
    print(f"=== Test 4: Random-entry bootstrap ({N_BOOTSTRAP} samples of N={N}) ===")
    valid_starts = np.arange(pred_offset, len(prices) - HORIZON)
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        random_starts = rng.choice(valid_starts, size=N, replace=True)
        boot_nets_bps = np.empty(N)
        for k, t in enumerate(random_starts):
            side = "long" if k < n_long else "short"
            ep = float(prices[t])
            xp = float(prices[t + HORIZON])
            if side == "long":
                g = (xp - ep) * SIZE
            else:
                g = (ep - xp) * SIZE
            mr = -MAKER_FEE * SIZE * ep
            tf = TAKER_FEE * SIZE * xp
            n_pnl = g + mr - tf
            boot_nets_bps[k] = (n_pnl / ep) * 10000.0
        boot_means[b] = float(boot_nets_bps.mean())

    random_mean = float(boot_means.mean())
    random_p5 = float(np.percentile(boot_means, 5))
    random_p95 = float(np.percentile(boot_means, 95))
    random_p99 = float(np.percentile(boot_means, 99))
    percentile_of_model = float((boot_means < baseline_mean_bps).mean() * 100)
    edge_over_random = baseline_mean_bps - random_mean
    print(f"  Random:  mean={random_mean:+.2f}  p5={random_p5:+.2f}  p95={random_p95:+.2f}  p99={random_p99:+.2f}")
    print(f"  Model:   {baseline_mean_bps:+.2f} bps  ->  {percentile_of_model:.1f}th percentile  (edge over random: {edge_over_random:+.2f})")

    # Test 5: Bootstrap CI on mean bps/trade
    print()
    print(f"=== Test 5: Bootstrap CI on mean bps/trade ({N_BOOTSTRAP} resamples) ===")
    boot_self_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, N, size=N)
        boot_self_means[b] = float(net_bps_arr[idx].mean())
    ci_low = float(np.percentile(boot_self_means, 2.5))
    ci_high = float(np.percentile(boot_self_means, 97.5))
    ci_p5 = float(np.percentile(boot_self_means, 5))
    crosses_zero = bool(ci_low < 0 < ci_high)
    print(f"  Mean: {baseline_mean_bps:+.2f}  95% CI: [{ci_low:+.2f}, {ci_high:+.2f}]  one-sided p5: {ci_p5:+.2f}")
    print(f"  CI crosses zero: {crosses_zero}")

    # Test 6: Trade-level distribution
    print()
    print("=== Test 6: Trade distribution ===")
    print(f"  min={net_bps_arr.min():+.2f}  p5={np.percentile(net_bps_arr, 5):+.2f}  p25={np.percentile(net_bps_arr, 25):+.2f}  "
          f"med={float(np.median(net_bps_arr)):+.2f}  p75={np.percentile(net_bps_arr, 75):+.2f}  p95={np.percentile(net_bps_arr, 95):+.2f}  "
          f"max={net_bps_arr.max():+.2f}")
    win_bps = net_bps_arr[net_bps_arr > 0]
    loss_bps = net_bps_arr[net_bps_arr <= 0]
    payoff_ratio = abs(win_bps.mean() / loss_bps.mean()) if len(loss_bps) > 0 and loss_bps.mean() != 0 else float("inf")
    print(f"  Winners: N={len(win_bps)} mean={win_bps.mean():+.2f}  Losers: N={len(loss_bps)} mean={loss_bps.mean():+.2f}")
    print(f"  Payoff ratio (|mean win| / |mean loss|): {payoff_ratio:.2f}")

    # Test 8: Drawdown + losing streak
    print()
    print("=== Test 8: Drawdown + losing streak (chronological) ===")
    order = np.argsort(entry_ts)
    nets_chrono = net_bps_arr[order]
    cum = np.cumsum(nets_chrono)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd_bps = float(dd.max())
    streak = 0
    max_streak = 0
    for r in nets_chrono:
        if r <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Cum final={cum[-1]:+.0f} bps  Peak={running_max.max():+.0f}  Max DD={max_dd_bps:.0f} bps  Longest losing streak={max_streak}")

    summary = {
        "config": f"{SYMBOL} H={HORIZON} thr={THR_LONG} fees={MAKER_FEE*1e4:.0f}/{TAKER_FEE*1e4:.0f}bps q=1.0 train_frac={TRAIN_FRAC}",
        "baseline": {
            "n": N, "mean_bps": baseline_mean_bps,
            "win_pct": float(wins.mean() * 100),
            "long": n_long, "short": n_short,
        },
        "per_day_utc": per_day,
        "per_hour_utc": per_hour,
        "random_bootstrap": {
            "n_samples": N_BOOTSTRAP,
            "random_mean_bps": random_mean,
            "random_p5_bps": random_p5,
            "random_p95_bps": random_p95,
            "random_p99_bps": random_p99,
            "model_percentile_of_random": percentile_of_model,
            "edge_over_random_bps": edge_over_random,
        },
        "self_bootstrap_ci": {
            "ci_95_low": ci_low, "ci_95_high": ci_high,
            "ci_p5_one_sided": ci_p5, "crosses_zero": crosses_zero,
        },
        "trade_distribution": {
            "min_bps": float(net_bps_arr.min()),
            "p5": float(np.percentile(net_bps_arr, 5)),
            "median": float(np.median(net_bps_arr)),
            "p95": float(np.percentile(net_bps_arr, 95)),
            "max_bps": float(net_bps_arr.max()),
            "n_winners": int(len(win_bps)),
            "mean_win_bps": float(win_bps.mean()) if len(win_bps) else 0.0,
            "n_losers": int(len(loss_bps)),
            "mean_loss_bps": float(loss_bps.mean()) if len(loss_bps) else 0.0,
            "payoff_ratio": float(payoff_ratio),
        },
        "drawdown": {
            "max_dd_bps": max_dd_bps,
            "final_cum_bps": float(cum[-1]),
            "peak_cum_bps": float(running_max.max()),
            "longest_losing_streak": int(max_streak),
        },
    }
    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Summary JSON: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
