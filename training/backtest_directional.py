"""
backtest_directional.py — §14 Phase A: standalone PnL backtest of the
directional TCN signal against held-out val CSVs.

Maker-only entry, taker exit on hold-time expiry. Tracks adverse-selection
(fill-only-when-wrong proxy), maker rebates, and taker exit fees so the
simulated PnL reflects executed-alpha not paper-alpha.

Trading rule
------------
At tick t with signal s = TCN(features[t-seq_len+1 : t+1]):
    s > threshold              → submit passive BID at best_bid[t]
    s < 1 - threshold          → submit passive ASK at best_ask[t]
    otherwise                  → no action

Fill model (adverse-selection proxy, conservative — with optional queue-priority)
---------------------------------------------------------------------------------
Strict (default): a BID at price P fills iff best_bid[t'] < P for some
t' ∈ [t+1, t+H]. Symmetric for asks. The market having moved *through*
our level is the only signal we got filled — captures winner's curse
exactly. Strictest possible fill model.

With --queue-fill-prob Q (default 0): with probability Q, an order is
also considered filled "neutrally" at a uniformly-random tick within
[t+1, t+H] — modeling queue-priority hits where we get filled at our
price WITHOUT the market moving against us. Q=0.3-0.5 is realistic for
liquid venues. Q is applied per-order independently of adverse fills:
if the adverse condition triggers first chronologically, that wins;
otherwise we roll the Q dice at a random tick within the window.

Exit model
----------
At submit_t + H (regardless of fill_t, so the hold time within the
directional horizon shrinks if we filled late), force taker exit:
    LONG  → sell at best_bid[exit_t]  (pay taker)
    SHORT → buy  at best_ask[exit_t]  (pay taker)
Unfilled orders simply expire — no PnL impact.

Usage
-----
    python backtest_directional.py
    python backtest_directional.py --threshold 0.55 --hold-ticks 100 --size 1.0
    python backtest_directional.py --maker-fee-rate -0.0001 --taker-fee-rate 0.00045
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from layer2_alpha import TCNSpikePredictor  # noqa: E402

DEFAULT_COINS = ["BTC", "ETH", "SOL"]


@dataclass
class Stats:
    signals: int = 0
    filled: int = 0
    scratched: int = 0      # filled but ran past end of CSV before exit
    win_count: int = 0
    n_completed: int = 0
    gross_pnl: float = 0.0
    maker_rebate: float = 0.0
    taker_fee: float = 0.0
    net_pnl: float = 0.0
    # Per-side breakdown
    long_signals: int = 0
    short_signals: int = 0
    long_filled: int = 0
    short_filled: int = 0


def build_features_hl(df: pl.DataFrame) -> np.ndarray:
    """Build the 5-channel Path D feature stack matching train_stream.py."""
    return np.stack([
        df["ce_ratio"].to_numpy() / 10.0,
        df["obi"].to_numpy(),
        df["mlofi"].to_numpy(),
        df["vamp"].to_numpy() / 10.0,
        df["kyles_lambda"].to_numpy() * 100.0,
    ], axis=1).astype(np.float32)


def batched_tcn_predict(
    tcn: TCNSpikePredictor,
    features: np.ndarray,
    seq_len: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Run TCN over all sliding windows. Returns (N - seq_len + 1,) of probs.

    Result index i corresponds to absolute tick (i + seq_len - 1).
    """
    n = len(features)
    if n < seq_len:
        return np.array([], dtype=np.float32)

    n_windows = n - seq_len + 1
    out = np.empty(n_windows, dtype=np.float32)
    features_t = torch.from_numpy(features).to(device)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        slice_end = end + seq_len - 1
        # Unfold: (end-start, C, seq_len)
        windows = features_t[start:slice_end].unfold(0, seq_len, 1)
        with torch.no_grad():
            preds = tcn(windows).cpu().numpy().flatten()
        out[start:end] = preds

    return out


def simulate_trading(
    coin: str,
    best_bids: np.ndarray,
    best_asks: np.ndarray,
    predictions: np.ndarray,
    seq_len: int,
    long_threshold: float,
    short_threshold: float,
    hold_ticks: int,
    size: float,
    maker_fee_rate: float,
    taker_fee_rate: float,
    queue_fill_prob: float = 0.0,
    rng: np.random.Generator = None,
) -> Stats:
    """Returns aggregate Stats. See module docstring for the trading rule."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(best_bids)
    stats = Stats()
    pred_offset = seq_len - 1

    for i in range(len(predictions)):
        t = i + pred_offset
        if t + hold_ticks >= n:
            break  # not enough future ticks for a complete hold window

        signal = predictions[i]
        side = None
        entry_price = 0.0
        if signal > long_threshold:
            side = "long"
            entry_price = float(best_bids[t])
            stats.long_signals += 1
        elif signal < short_threshold:
            side = "short"
            entry_price = float(best_asks[t])
            stats.short_signals += 1
        else:
            continue

        stats.signals += 1

        # Scan forward for adverse-selection fill
        adverse_fill_t = -1
        upper = min(t + hold_ticks + 1, n)
        if side == "long":
            for tt in range(t + 1, upper):
                if best_bids[tt] < entry_price:
                    adverse_fill_t = tt
                    break
        else:
            for tt in range(t + 1, upper):
                if best_asks[tt] > entry_price:
                    adverse_fill_t = tt
                    break

        # Queue-priority fill: with prob Q, also consider filling at a
        # random tick within the window. Use the EARLIER of the two if
        # both apply.
        queue_fill_t = -1
        if queue_fill_prob > 0.0 and rng.random() < queue_fill_prob:
            window_len = upper - (t + 1)
            if window_len > 0:
                queue_fill_t = t + 1 + int(rng.integers(window_len))

        # Pick the earlier of the two fills (whichever happens first)
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

        if side == "long":
            exit_price = float(best_bids[exit_t])   # sell into bid (taker)
            gross = (exit_price - entry_price) * size
        else:
            exit_price = float(best_asks[exit_t])   # lift ask (taker)
            gross = (entry_price - exit_price) * size

        # Maker fee is on notional at entry. Negative rate = we receive a rebate.
        maker_rebate = -maker_fee_rate * size * entry_price
        # Taker fee on notional at exit. Positive = we pay.
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
        pool.signals += s.signals
        pool.filled += s.filled
        pool.scratched += s.scratched
        pool.win_count += s.win_count
        pool.n_completed += s.n_completed
        pool.gross_pnl += s.gross_pnl
        pool.maker_rebate += s.maker_rebate
        pool.taker_fee += s.taker_fee
        pool.net_pnl += s.net_pnl
        pool.long_signals += s.long_signals
        pool.short_signals += s.short_signals
        pool.long_filled += s.long_filled
        pool.short_filled += s.short_filled
    return pool


def print_header():
    print(
        f"{'coin':>6s} {'sigs':>7s} {'fill_rate':>9s} {'win%':>6s} "
        f"{'gross_pnl':>13s} {'rebate':>11s} {'taker_fee':>11s} "
        f"{'net_pnl':>13s} {'mean_$':>9s} {'L/S':>9s}"
    )
    print("-" * 110)


def print_stats(label: str, s: Stats):
    fill_rate = (s.filled / s.signals * 100) if s.signals > 0 else 0.0
    win_rate = (s.win_count / s.n_completed * 100) if s.n_completed > 0 else 0.0
    mean = (s.net_pnl / s.n_completed) if s.n_completed > 0 else 0.0
    ls = f"{s.long_filled}/{s.short_filled}"
    print(
        f"{label:>6s} {s.signals:>7,d} {fill_rate:>8.1f}% {win_rate:>5.1f}% "
        f"{s.gross_pnl:>+13.2f} {s.maker_rebate:>+11.2f} {s.taker_fee:>11.2f} "
        f"{s.net_pnl:>+13.2f} {mean:>+9.4f} {ls:>9s}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="calibration/tcn_weights_BTC_USDC_USDC.pt",
                   help="Path to TCN directional weights (default: live weights).")
    p.add_argument("--coins", default=",".join(DEFAULT_COINS),
                   help="Comma-separated coin symbols (default: BTC,ETH,SOL).")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Symmetric threshold around 0.5 (long if signal > th, short if signal < 1-th). "
                        "Overridden by --long-threshold/--short-threshold if either is set. Default 0.55.")
    p.add_argument("--long-threshold", type=float, default=None,
                   help="Long entry trigger (signal > this). If None, uses 1 - (1-threshold) = threshold.")
    p.add_argument("--short-threshold", type=float, default=None,
                   help="Short entry trigger (signal < this). If None, uses 1 - threshold. "
                        "Use --long-threshold and --short-threshold together to test asymmetric "
                        "triggers around a non-0.5 base rate (e.g., model's empirical prior).")
    p.add_argument("--hold-ticks", type=int, default=config.TCN_DIRECTIONAL_HORIZON_TICKS,
                   help="Hold ticks before forced taker exit. Default = model's training horizon.")
    p.add_argument("--size", type=float, default=1.0,
                   help="Position size in base units per signal. Default 1.0.")
    p.add_argument("--maker-fee-rate", type=float, default=-0.0001,
                   help="Maker fee rate (negative = rebate). Default -1 bp (HL maker rebate).")
    p.add_argument("--taker-fee-rate", type=float, default=0.00045,
                   help="Taker fee rate. Default +4.5 bps (HL taker fee).")
    p.add_argument("--queue-fill-prob", type=float, default=0.0,
                   help="Probability that an order also fills 'neutrally' "
                        "(no adverse movement) at a random tick within H. "
                        "Models queue-priority fills. 0.0 = strict adverse only "
                        "(default). Realistic range: 0.3-0.5.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for queue-fill-prob sampling.")
    p.add_argument("--seq-len", type=int, default=config.TCN_INPUT_LENGTH)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    long_threshold = args.long_threshold if args.long_threshold is not None else args.threshold
    short_threshold = args.short_threshold if args.short_threshold is not None else (1.0 - args.threshold)
    print(f"Device:        {device}")
    print(f"Weights:       {args.weights}")
    print(f"Long thresh:   {long_threshold:.4f}  (signal > this → long)")
    print(f"Short thresh:  {short_threshold:.4f}  (signal < this → short)")
    print(f"Hold ticks:    {args.hold_ticks}")
    print(f"Size:          {args.size} base units / signal")
    print(f"Maker fee:     {args.maker_fee_rate * 10000:+.2f} bps")
    print(f"Taker fee:     {args.taker_fee_rate * 10000:+.2f} bps")
    print()

    tcn = TCNSpikePredictor().to(device).eval()
    state = torch.load(args.weights, map_location=device)
    tcn.load_state_dict(state)

    print_header()
    per_coin: list[Stats] = []
    for coin in args.coins.split(","):
        coin = coin.strip()
        csv_path = Path(f"./calibration/feature_history_{coin}.val.csv")
        if not csv_path.exists():
            print(f"{coin}: missing ({csv_path})")
            continue

        df = pl.read_csv(csv_path)
        features = build_features_hl(df)
        best_bids = df["best_bid"].to_numpy().astype(np.float64)
        best_asks = df["best_ask"].to_numpy().astype(np.float64)

        preds = batched_tcn_predict(tcn, features, args.seq_len, device)
        stats = simulate_trading(
            coin=coin,
            best_bids=best_bids,
            best_asks=best_asks,
            predictions=preds,
            seq_len=args.seq_len,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            hold_ticks=args.hold_ticks,
            size=args.size,
            maker_fee_rate=args.maker_fee_rate,
            taker_fee_rate=args.taker_fee_rate,
            queue_fill_prob=args.queue_fill_prob,
            rng=np.random.default_rng(args.seed),
        )
        per_coin.append(stats)
        print_stats(coin, stats)

    pool = aggregate(*per_coin)
    print("-" * 110)
    print_stats("POOL", pool)

    # Decomposition
    print()
    print("=== PnL decomposition (pooled) ===")
    print(f"  Gross PnL (price moves only):   ${pool.gross_pnl:>+12.2f}")
    print(f"  Maker rebate received:          ${pool.maker_rebate:>+12.2f}")
    print(f"  Taker fees paid (exit):         ${-pool.taker_fee:>+12.2f}")
    print(f"  ----------------------------------------------------")
    print(f"  NET PnL:                        ${pool.net_pnl:>+12.2f}")
    if pool.n_completed > 0:
        print()
        print(f"  N completed trades:             {pool.n_completed:,}")
        print(f"  Mean net PnL per trade:         ${pool.net_pnl / pool.n_completed:+.4f}")
        print(f"  Rebate share of gross profit:   "
              f"{(pool.maker_rebate / pool.gross_pnl * 100) if pool.gross_pnl > 0 else float('nan'):+.1f}%")
        print(f"  Adverse-selection fill rate:    "
              f"{pool.filled / pool.signals * 100 if pool.signals > 0 else 0:.1f}%")


if __name__ == "__main__":
    main()
