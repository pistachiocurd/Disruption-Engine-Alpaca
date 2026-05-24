"""
backtest_l3_directional.py - PnL backtest of the L3 directional TCN.

Mirrors backtest_directional.py's trading rule and fill model. Differences:

 1. Reads L3 event-clock tick CSVs (calibration/l3_ticks_t{BTC,ETH,SOL}USD.csv)
    instead of L2 feature_history_*.val.csv.
 2. Uses last_trade_price as a single price (no separate bid/ask in L3 ticks).
    Half-spread cost is baked into --maker-fee-rate / --taker-fee-rate so the
    user can tune to their venue's realistic spread + fees.
 3. Applies the saved L3 normalization stats (l3_feature_stats.json) before
    feeding the L3 feature stack to the TCN. Channel count is taken from
    `train_l3_directional.FEATURE_COLS` (re-exported from
    `layer1_l3_sensors`) so Phase 1.5's 11-channel vs Phase 2's 19-channel
    work without any backtester edits.
 4. Default fees (1 bp maker, 10 bps taker = 11 bps round-trip) are LOW —
    they approximate a Bitfinex volume-tier + LEO-discount stack, NOT
    standard public retail. Standard Bitfinex retail crypto is 10 bps maker
    / 20 bps taker = 30 bps RT. Always pass --maker-fee-rate 0.0010
    --taker-fee-rate 0.0020 to reproduce a realistic non-institutional
    retail account. The Phase 2 headline bps figures were generated under
    the 11-bps-RT defaults and overstate what a retail account would clear.
 5. Operates on the held-out 20% temporal val split per symbol (same as the
    training script's val partition) so the backtest is on unseen data.

Trading rule (unchanged from backtest_directional.py)
-----------------------------------------------------
At tick t with signal s = TCN(normalized_features[t-seq_len+1 : t+1]):
    s > long_threshold  -> post passive BID at last_trade_price[t]
    s < short_threshold -> post passive ASK at last_trade_price[t]
    otherwise           -> no action

Fill model
----------
Strict adverse-selection: a BID at price P fills iff last_trade_price[t'] < P
for some t' in [t+1, t+H]. Symmetric for asks. Conservative — captures
winner's curse. Use --queue-fill-prob Q to model some neutral fills.

Exit
----
At submit_t + H, force taker exit at last_trade_price[exit_t]. Unfilled
orders expire.

Usage
-----
    python backtest_l3_directional.py --horizon 300
    python backtest_l3_directional.py --horizon 300 --threshold 0.55 --queue-fill-prob 0.3
    python backtest_l3_directional.py --horizon 300 --threshold 0.55 \\
        --maker-fee-rate 0.0001 --taker-fee-rate 0.001
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

# research/path_h_l3/ → repo root is two levels up
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from layer2_alpha import TCNSpikePredictor
from train_l3_directional import (
    FEATURE_COLS,
    DEFAULT_CSV_DIR,
    DEFAULT_SYMBOLS,
    load_split,
)


@dataclass
class Stats:
    signals: int = 0
    filled: int = 0
    scratched: int = 0
    win_count: int = 0
    n_completed: int = 0
    gross_pnl: float = 0.0
    maker_rebate: float = 0.0
    taker_fee: float = 0.0
    net_pnl: float = 0.0
    long_signals: int = 0
    short_signals: int = 0
    long_filled: int = 0
    short_filled: int = 0


def batched_tcn_predict(
    tcn: TCNSpikePredictor,
    features: np.ndarray,  # already normalized
    seq_len: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Sliding-window TCN inference. Returns (N - seq_len + 1,) probs.
    Output index i corresponds to absolute tick (i + seq_len - 1)."""
    n = len(features)
    if n < seq_len:
        return np.array([], dtype=np.float32)
    n_windows = n - seq_len + 1
    out = np.empty(n_windows, dtype=np.float32)
    features_t = torch.from_numpy(features).to(device)
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        slice_end = end + seq_len - 1
        windows = features_t[start:slice_end].unfold(0, seq_len, 1)
        with torch.no_grad():
            preds = tcn(windows).cpu().numpy().flatten()
        out[start:end] = preds
    return out


def simulate_trading(
    prices: np.ndarray,
    predictions: np.ndarray,
    seq_len: int,
    long_threshold: float,
    short_threshold: float,
    hold_ticks: int,
    size: float,
    maker_fee_rate: float,
    taker_fee_rate: float,
    queue_fill_prob: float,
    rng: np.random.Generator,
) -> Stats:
    """Single price stream (no bid/ask separation); spread cost is baked
    into fee rates by the caller."""
    n = len(prices)
    stats = Stats()
    pred_offset = seq_len - 1

    for i in range(len(predictions)):
        t = i + pred_offset
        if t + hold_ticks >= n:
            break

        signal = predictions[i]
        if signal > long_threshold:
            side = "long"
            stats.long_signals += 1
        elif signal < short_threshold:
            side = "short"
            stats.short_signals += 1
        else:
            continue
        entry_price = float(prices[t])
        stats.signals += 1

        # Adverse-selection fill: market moved through our price.
        upper = min(t + hold_ticks + 1, n)
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

        # Queue-priority fill: with prob Q, also consider filling neutrally
        # at a random tick. Earlier of the two wins.
        queue_fill_t = -1
        if queue_fill_prob > 0.0 and rng.random() < queue_fill_prob:
            window_len = upper - (t + 1)
            if window_len > 0:
                queue_fill_t = t + 1 + int(rng.integers(window_len))

        if adverse_fill_t < 0 and queue_fill_t < 0:
            continue
        if adverse_fill_t < 0:
            fill_t = queue_fill_t
        elif queue_fill_t < 0:
            fill_t = adverse_fill_t
        else:
            fill_t = min(adverse_fill_t, queue_fill_t)

        stats.filled += 1
        if side == "long":
            stats.long_filled += 1
        else:
            stats.short_filled += 1

        exit_t = t + hold_ticks
        if exit_t >= n:
            stats.scratched += 1
            continue
        exit_price = float(prices[exit_t])

        if side == "long":
            gross = (exit_price - entry_price) * size
        else:
            gross = (entry_price - exit_price) * size

        maker_rebate = -maker_fee_rate * size * entry_price  # negative rate = rebate (received)
        taker_fee = taker_fee_rate * size * exit_price
        net = gross + maker_rebate - taker_fee

        stats.gross_pnl += gross
        stats.maker_rebate += maker_rebate
        stats.taker_fee += taker_fee
        stats.net_pnl += net
        stats.n_completed += 1
        if net > 0:
            stats.win_count += 1

    return stats


def aggregate(*statses: Stats) -> Stats:
    pool = Stats()
    for s in statses:
        for f in (
            "signals", "filled", "scratched", "win_count", "n_completed",
            "gross_pnl", "maker_rebate", "taker_fee", "net_pnl",
            "long_signals", "short_signals", "long_filled", "short_filled",
        ):
            setattr(pool, f, getattr(pool, f) + getattr(s, f))
    return pool


def print_stats(label: str, s: Stats, ref_notional: float | None = None) -> None:
    fill_rate = (s.filled / s.signals * 100) if s.signals > 0 else 0.0
    win_rate = (s.win_count / s.n_completed * 100) if s.n_completed > 0 else 0.0
    mean = (s.net_pnl / s.n_completed) if s.n_completed > 0 else 0.0
    ls = f"{s.long_filled}/{s.short_filled}"
    bps = ""
    if ref_notional is not None and s.n_completed > 0 and ref_notional > 0:
        # Mean net per trade as bps of notional.
        bps = f" ({mean / ref_notional * 10000:+.2f}bps)"
    print(
        f"{label:>10s} {s.signals:>7,d} {fill_rate:>7.1f}% {win_rate:>5.1f}% "
        f"{s.gross_pnl:>+13.2f} {s.maker_rebate:>+11.2f} {s.taker_fee:>11.2f} "
        f"{s.net_pnl:>+13.2f} {mean:>+9.4f}{bps} {ls:>9s}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--weights", type=Path,
                   default=DEFAULT_CSV_DIR / "tcn_weights_l3_H300.pt")
    p.add_argument("--stats-in", type=Path,
                   default=DEFAULT_CSV_DIR / "l3_feature_stats.json")
    p.add_argument("--horizon", type=int, default=300,
                   help="Hold ticks = training horizon. Default 300.")
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--threshold", type=float, default=None,
                   help="Symmetric threshold around 0.5 (overrides --thresholds for single-run mode).")
    p.add_argument("--thresholds", default="0.50,0.52,0.54,0.55,0.56,0.58,0.60",
                   help="Comma-separated thresholds to sweep when --threshold is not set.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--size", type=float, default=1.0)
    # LOW-FEE DEFAULTS (volume-tier + LEO discount). For standard public retail
    # Bitfinex use --maker-fee-rate 0.0010 --taker-fee-rate 0.0020 (30 bps RT).
    p.add_argument("--maker-fee-rate", type=float, default=0.00010,
                   help="Maker fee. Default 0.00010 (1 bp, low; standard retail BFX is 10 bps).")
    p.add_argument("--taker-fee-rate", type=float, default=0.00100,
                   help="Taker fee. Default 0.00100 (10 bps, low; standard retail BFX is 20 bps).")
    p.add_argument("--queue-fill-prob", type=float, default=0.0,
                   help="Q for queue-priority fills (0.0-1.0). 0=strict adverse-only.")
    p.add_argument("--zero-channels", default="",
                   help="Comma-separated FEATURE_COLS names to zero post-normalization "
                        "(inference-time ablation). Empty = no ablation.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load normalization stats
    with open(args.stats_in) as f:
        stats_blob = json.load(f)
    mean = np.array(stats_blob["mean"], dtype=np.float32)
    std = np.array(stats_blob["std"], dtype=np.float32)
    if stats_blob["feature_cols"] != FEATURE_COLS:
        raise SystemExit(f"feature_cols mismatch:\n  stats={stats_blob['feature_cols']}\n  expected={FEATURE_COLS}")

    # Load weights
    tcn = TCNSpikePredictor(in_channels=len(FEATURE_COLS)).to(device).eval()
    state = torch.load(args.weights, map_location=device)
    tcn.load_state_dict(state)
    print(f"[setup] device={device}  H={args.horizon}  seq_len={args.seq_len}")
    print(f"[setup] weights={args.weights}")
    print(f"[setup] fees: maker {args.maker_fee_rate*10000:+.2f}bps  taker {args.taker_fee_rate*10000:+.2f}bps")
    print(f"[setup] queue_fill_prob={args.queue_fill_prob}")

    # Parse channel-zero ablation arg once, before per-symbol loop.
    zero_idx: list[int] = []
    if args.zero_channels:
        names = [n.strip() for n in args.zero_channels.split(",") if n.strip()]
        for n in names:
            if n not in FEATURE_COLS:
                raise SystemExit(f"unknown channel '{n}'; valid: {FEATURE_COLS}")
            zero_idx.append(FEATURE_COLS.index(n))
        print(f"[ablation] zeroing channels: {names} (indices {zero_idx})")

    # Build per-symbol val predictions + price arrays once.
    per_sym: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for sym in args.symbols:
        path = args.csv_dir / f"l3_ticks_{sym}.csv"
        if not path.exists():
            print(f"  {sym}: missing ({path}) — skipping")
            continue
        features, prices = load_split(path, "val", args.train_frac)
        feats_norm = (features - mean) / std
        if zero_idx:
            feats_norm[:, zero_idx] = 0.0
        preds = batched_tcn_predict(tcn, feats_norm, args.seq_len, device)
        ref = float(np.mean(prices)) if len(prices) > 0 else 0.0
        per_sym[sym] = (prices, preds, ref)
        print(f"  {sym}: val_ticks={len(prices):,}  windows={len(preds):,}  mean_price={ref:,.2f}")

    if not per_sym:
        raise SystemExit("no symbols loaded")

    # Threshold list
    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [float(x) for x in args.thresholds.split(",")]

    for thr in thresholds:
        long_threshold = thr
        short_threshold = 1.0 - thr
        print()
        print(f"=== threshold = {thr:.3f}  (long>{long_threshold:.3f}, short<{short_threshold:.3f}) ===")
        print(
            f"{'coin':>10s} {'sigs':>7s} {'fill%':>7s} {'win%':>6s} "
            f"{'gross_pnl':>13s} {'rebate':>11s} {'taker_fee':>11s} "
            f"{'net_pnl':>13s} {'mean_$/bps':>9s} {'L/S':>9s}"
        )
        print("-" * 130)
        per_coin: list[Stats] = []
        for sym, (prices, preds, ref) in per_sym.items():
            s = simulate_trading(
                prices=prices,
                predictions=preds,
                seq_len=args.seq_len,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                hold_ticks=args.horizon,
                size=args.size,
                maker_fee_rate=args.maker_fee_rate,
                taker_fee_rate=args.taker_fee_rate,
                queue_fill_prob=args.queue_fill_prob,
                rng=np.random.default_rng(args.seed),
            )
            per_coin.append(s)
            print_stats(sym, s, ref_notional=ref * args.size)
        pool = aggregate(*per_coin)
        # Use weighted avg notional for pooled bps.
        total_notional = sum(prices.mean() * args.size for prices, _, _ in per_sym.values())
        avg_notional = total_notional / len(per_sym) if per_sym else 0.0
        print("-" * 130)
        print_stats("POOL", pool, ref_notional=avg_notional)

        if pool.n_completed > 0:
            print()
            print("  N completed:        ", f"{pool.n_completed:,}")
            print("  Adverse fill rate:  ", f"{pool.filled / pool.signals * 100:.1f}%" if pool.signals else "n/a")
            print("  Mean net per trade: ", f"${pool.net_pnl / pool.n_completed:+.4f}  "
                  f"({pool.net_pnl / pool.n_completed / avg_notional * 10000:+.2f}bps)")


if __name__ == "__main__":
    main()
