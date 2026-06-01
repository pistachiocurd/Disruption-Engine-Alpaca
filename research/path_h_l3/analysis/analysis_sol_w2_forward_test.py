"""
analysis_sol_w2_forward_test.py — Forward-test the SOL Phase 2 weights on
window-2 data.

Loads the existing per-symbol SOL TCN weights (trained on window 1, 2026-05-12
→ 2026-05-23) and runs them on the entire window-2 capture (2026-05-24 →
2026-06-01) without any retraining. Window 2 is 100% out-of-sample — there is
no train/val split here; all w2 ticks are evaluated.

This is the gating experiment per NEXT_PHASE_PLAN §3. Outputs:
  - Per-day fire-rate (does the model fire on multiple days, or only 1?)
  - Per-fire-day mean net bps (is per-fire-day economics positive on average?)
  - Threshold sweep (0.90 / 0.95 / 0.97 / 0.98) at standard retail fees
  - Pooled bootstrap CI across all w2 fire-day trades

Discriminates between three w2 scenarios:
  A. Regime overfit (~25% prior): fires on 0-2 days, all negative
  B. Rare-firer but real (~50% prior): fires on 4-7 days, mostly positive
  C. Robust signal (~20% prior): fires on most days, consistently positive
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from layer1_l3_sensors import FEATURE_COLS
from train_l3_directional import TCNSpikePredictor
from backtest_l3_directional import batched_tcn_predict

SEQ_LEN = 200
SIZE = 1.0
SEED = 42
N_BOOTSTRAP = 1000

# Standard retail Bitfinex fees (10 bps maker, 20 bps taker = 30 bps RT)
MAKER_FEE = 0.0010
TAKER_FEE = 0.0020

THRESHOLDS = [0.90, 0.95, 0.97, 0.98]


def simulate(preds, prices, ts_ms, thr_long: float, thr_short: float, rng, horizon: int):
    """q=1.0 fill model, matching the backtester."""
    pred_offset = SEQ_LEN - 1
    trades = []
    for i in range(len(preds)):
        t = i + pred_offset
        if t + horizon >= len(prices):
            break
        signal = float(preds[i])
        if signal > thr_long:
            side = "long"
        elif signal < thr_short:
            side = "short"
        else:
            continue
        entry_price = float(prices[t])

        upper = min(t + horizon + 1, len(prices))
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

        exit_t = t + horizon
        exit_price = float(prices[exit_t])
        if side == "long":
            gross = (exit_price - entry_price) * SIZE
        else:
            gross = (entry_price - exit_price) * SIZE
        net = gross - MAKER_FEE * SIZE * entry_price - TAKER_FEE * SIZE * exit_price
        trades.append({
            "ts_ms": int(ts_ms[t]),
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "net_pnl": net,
            "net_bps": (net / entry_price) * 10000.0,
            "win": int(net > 0),
            "signal": signal,
        })
    return trades


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="tSOLUSD",
                   help="Bitfinex symbol (tBTCUSD, tETHUSD, tSOLUSD)")
    p.add_argument("--horizon", type=int, default=300,
                   help="TCN training horizon used for the per-symbol weights")
    args = p.parse_args()

    symbol = args.symbol
    horizon = args.horizon
    sym_lower = symbol.lstrip("t").lower().replace("usd", "")  # tSOLUSD -> sol

    weights = ROOT / "calibration" / f"tcn_weights_l3_phase2_persym_{symbol}_H{horizon}.pt"
    stats_file = ROOT / "calibration" / f"l3_feature_stats_phase2_persym_{symbol}.json"
    w2_csv = ROOT / "calibration" / f"l3_ticks_w2_{symbol}.csv"
    out_json = ROOT / "calibration" / f"forward_test_w2_{sym_lower}_H{horizon}_summary.json"
    out_trades_csv = ROOT / "calibration" / f"forward_test_w2_{sym_lower}_H{horizon}_trades.csv"

    if not w2_csv.exists():
        raise SystemExit(f"Missing w2 CSV at {w2_csv} — run aggregate_mbo_events.py first")
    if not weights.exists():
        raise SystemExit(f"Missing weights at {weights}")
    if not stats_file.exists():
        raise SystemExit(f"Missing stats at {stats_file}")

    print(f"Forward-test: symbol={symbol}  H={horizon}")
    print(f"Loading {w2_csv}...")
    df = pl.read_csv(w2_csv)
    n = df.height
    print(f"  w2: {n:,} ticks")

    if n > 0:
        first_ts = int(df["timestamp_ms"][0])
        last_ts = int(df["timestamp_ms"][-1])
        first_dt = datetime.fromtimestamp(first_ts/1000, tz=timezone.utc)
        last_dt = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)
        span_days = (last_ts - first_ts) / 86400000
        print(f"  span: {first_dt:%Y-%m-%d %H:%M:%S} -> {last_dt:%Y-%m-%d %H:%M:%S}  ({span_days:.2f} days)")

    features = np.stack(
        [df[c].to_numpy().astype(np.float32) for c in FEATURE_COLS], axis=1,
    )
    prices = df["last_trade_price"].to_numpy().astype(np.float32)
    ts_ms = df["timestamp_ms"].to_numpy()

    with open(stats_file) as f:
        stats_blob = json.load(f)
    mean = np.array(stats_blob["mean"], dtype=np.float32)
    std = np.array(stats_blob["std"], dtype=np.float32)
    feats_norm = (features - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcn = TCNSpikePredictor(in_channels=len(FEATURE_COLS)).to(device).eval()
    tcn.load_state_dict(torch.load(weights, map_location=device))
    print(f"  device={device}  weights={weights.name}")

    preds = batched_tcn_predict(tcn, feats_norm, SEQ_LEN, device)
    print(f"  predictions: {len(preds):,}")

    # Per-tick UTC date for fire-rate diagnostic
    tick_dates = np.array(
        [datetime.fromtimestamp(int(t)/1000, tz=timezone.utc).date() for t in ts_ms]
    )
    unique_days = sorted(set(tick_dates))

    summary = {
        "config": f"{symbol} H={horizon} seq_len={SEQ_LEN} fees={MAKER_FEE*1e4:.0f}/{TAKER_FEE*1e4:.0f}bps q=1.0",
        "window": "w2 (2026-05-24 -> 2026-06-01 capture, forward-test using window-1 weights)",
        "n_ticks": int(n),
        "n_days": len(unique_days),
        "days": [str(d) for d in unique_days],
        "by_threshold": [],
    }

    all_trades_for_csv = []

    for thr in THRESHOLDS:
        print()
        print(f"=== thr = {thr:.2f}  (long > {thr}, short < {1-thr:.2f}) ===")
        rng = np.random.default_rng(SEED)
        trades = simulate(preds, prices, ts_ms, thr, 1.0 - thr, rng, horizon)
        N = len(trades)
        if N == 0:
            print(f"  N=0 — no signals fired across w2")
            summary["by_threshold"].append({
                "thr": thr, "n_total": 0, "long": 0, "short": 0,
                "win_pct": 0.0, "mean_bps": 0.0,
                "fire_days": 0, "per_day": [], "bootstrap_ci_95": [0.0, 0.0],
            })
            continue
        bps = np.array([t["net_bps"] for t in trades])
        wins = np.array([t["win"] for t in trades])
        sides = np.array([t["side"] for t in trades])
        entry_ts = np.array([t["ts_ms"] for t in trades])
        entry_dates = np.array(
            [datetime.fromtimestamp(t/1000, tz=timezone.utc).date() for t in entry_ts]
        )

        n_long = int((sides == "long").sum())
        n_short = int((sides == "short").sum())
        mean_bps = float(bps.mean())
        win_pct = float(wins.mean() * 100)

        print(f"  total: N={N}  L={n_long} S={n_short}  win={win_pct:.1f}%  mean={mean_bps:+.2f} bps")

        # Per-day breakdown
        per_day = []
        fire_days = 0
        print(f"  {'date':<12} {'N':>6} {'L/S':>9} {'win%':>6} {'mean_bps':>10}")
        for day in unique_days:
            tick_n = int((tick_dates == day).sum())
            mask = entry_dates == day
            d_n = int(mask.sum())
            if d_n == 0:
                per_day.append({
                    "date": str(day), "ticks": tick_n, "trades": 0,
                    "long": 0, "short": 0, "win_pct": 0.0, "mean_bps": 0.0,
                    "fired": False,
                })
                print(f"  {str(day):<12} {0:>6} {'-':>9} {'-':>6} {'-':>10}  (no fire)")
                continue
            fire_days += 1
            d_l = int((sides[mask] == "long").sum())
            d_s = int((sides[mask] == "short").sum())
            d_win = float(wins[mask].mean() * 100)
            d_mean = float(bps[mask].mean())
            per_day.append({
                "date": str(day), "ticks": tick_n, "trades": d_n,
                "long": d_l, "short": d_s, "win_pct": d_win, "mean_bps": d_mean,
                "fired": True,
            })
            print(f"  {str(day):<12} {d_n:>6} {d_l:>4}/{d_s:<4} {d_win:>5.1f}% {d_mean:>+9.2f}")
        print(f"  fire_days: {fire_days}/{len(unique_days)}")

        # Bootstrap CI on pooled mean
        rng_boot = np.random.default_rng(SEED + 1)
        boot_means = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng_boot.integers(0, N, size=N)
            boot_means[b] = float(bps[idx].mean())
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        crosses_zero = ci_low < 0 < ci_high
        print(f"  95% CI: [{ci_low:+.2f}, {ci_high:+.2f}]  crosses_zero={crosses_zero}")

        # Per-fire-day variance
        fire_means = np.array([d["mean_bps"] for d in per_day if d["fired"]])
        if len(fire_means) > 1:
            fire_std = float(fire_means.std(ddof=1))
            fire_mean = float(fire_means.mean())
            var_ratio = (fire_std / abs(fire_mean)) if abs(fire_mean) > 1e-9 else float("inf")
            pos_fire_days = int((fire_means > 0).sum())
            print(f"  per-fire-day:  mean={fire_mean:+.2f}  std={fire_std:.2f}  "
                  f"std/|mean|={var_ratio:.2f}  positive: {pos_fire_days}/{len(fire_means)}")
        else:
            fire_std = 0.0
            fire_mean = float(fire_means[0]) if len(fire_means) == 1 else 0.0
            var_ratio = 0.0
            pos_fire_days = int((fire_means > 0).sum()) if len(fire_means) >= 1 else 0

        summary["by_threshold"].append({
            "thr": thr, "n_total": N, "long": n_long, "short": n_short,
            "win_pct": win_pct, "mean_bps": mean_bps,
            "fire_days": fire_days, "total_days": len(unique_days),
            "per_fire_day_mean_bps": fire_mean,
            "per_fire_day_std_bps": fire_std,
            "per_fire_day_var_ratio": var_ratio,
            "per_fire_day_positive_count": pos_fire_days,
            "bootstrap_ci_95": [ci_low, ci_high],
            "ci_crosses_zero": crosses_zero,
            "per_day": per_day,
        })

        # Stash trades for the primary threshold (0.95) for downstream inspection
        if abs(thr - 0.95) < 1e-9:
            for t in trades:
                tcopy = dict(t)
                tcopy["thr"] = thr
                all_trades_for_csv.append(tcopy)

    # Save outputs
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print()
    print(f"Summary JSON: {out_json}")
    if all_trades_for_csv:
        pl.DataFrame(all_trades_for_csv).write_csv(out_trades_csv)
        print(f"Per-trade CSV (thr=0.95): {out_trades_csv}")

    # Scenario verdict
    print()
    print("=== Scenario verdict (per NEXT_PHASE_PLAN §3) ===")
    primary = next((t for t in summary["by_threshold"] if abs(t["thr"] - 0.95) < 1e-9), None)
    if primary is None or primary["n_total"] == 0:
        print("  No fires at thr=0.95 across w2 — Scenario A (regime overfit) candidate")
    else:
        fd = primary["fire_days"]
        td = primary["total_days"]
        ci = primary["bootstrap_ci_95"]
        pos = primary["per_fire_day_positive_count"]
        nfd = max(fd, 1)
        if fd <= 2 and primary["mean_bps"] <= 0:
            verdict = "A (regime overfit)"
        elif fd >= 4 and ci[0] > 0 and pos >= (nfd * 0.6):
            verdict = "C (robust signal)" if fd >= int(td * 0.7) else "B (rare-firer but real)"
        elif fd >= 2 and ci[0] > 0:
            verdict = "B (rare-firer but real) — possibly thin"
        elif fd >= 2 and ci[0] <= 0:
            verdict = "A/B boundary — fires but CI crosses zero"
        else:
            verdict = "A (regime overfit) — limited firing"
        print(f"  thr=0.95: fire_days {fd}/{td}, mean {primary['mean_bps']:+.2f} bps, "
              f"CI [{ci[0]:+.2f}, {ci[1]:+.2f}], positive fire-days {pos}/{nfd}")
        print(f"  Verdict: Scenario {verdict}")


if __name__ == "__main__":
    main()
