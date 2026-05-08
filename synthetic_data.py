"""
synthetic_data.py - Generate a synthetic feature_history.csv with planted
shock events for TCN training.

Output format matches FeatureDumper exactly so train_tcn.py can consume it
without code changes. The generator plants pre-shock signatures (rising VPIN,
rising CE ratio, rising OBI) followed by a price jump within MAX_DIFFUSION_TICKS.

This is for plumbing testing ONLY. The TCN trained on this data will fire on
the synthetic patterns it learned, which are at best a stylized cartoon of
real crypto microstructure. Do NOT use these weights with real money.

Usage:
    python synthetic_data.py
    python synthetic_data.py --ticks 72000 --shocks 60 --out calibration/synthetic.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _make_shock_window(
    n_pre: int = 30,
    n_jump: int = 50,
    direction: int = 1,
    base_price: float = 80_000.0,
    move_pct: float = 0.008,
    rng: np.random.Generator | None = None,
) -> dict:
    """One shock window: pre-shock signature, then a price jump."""
    rng = rng or np.random.default_rng()

    # Pre-shock signature: VPIN rising, CE rising, OBI rising in shock direction.
    vpin_pre = np.linspace(0.55, 0.95, n_pre) + rng.normal(0, 0.02, n_pre)
    ce_pre = np.linspace(0.5, 6.0, n_pre) + rng.normal(0, 0.5, n_pre)
    obi_pre = np.linspace(0.05, 0.55, n_pre) * direction + rng.normal(0, 0.05, n_pre)

    # Price drift in the pre-shock window (small).
    drift_pre = rng.normal(0, 5e-5, n_pre)

    # Jump phase: VPIN sustained high, CE/OBI elevated, price moves move_pct.
    vpin_jump = 0.92 + rng.normal(0, 0.03, n_jump)
    ce_jump = rng.uniform(2.0, 8.0, n_jump)
    obi_jump = rng.uniform(0.3, 0.7, n_jump) * direction
    # Compress the price move into the first ~half of the jump phase.
    half = n_jump // 2
    move_per_tick = move_pct / max(half, 1)
    jump_returns = np.zeros(n_jump)
    jump_returns[:half] = move_per_tick * direction + rng.normal(0, move_per_tick * 0.2, half)
    jump_returns[half:] = rng.normal(0, 5e-5, n_jump - half)

    return {
        "vpin": np.concatenate([vpin_pre, vpin_jump]),
        "ce": np.concatenate([ce_pre, ce_jump]),
        "obi": np.concatenate([obi_pre, obi_jump]),
        "returns": np.concatenate([drift_pre, jump_returns]),
    }


def generate(
    n_ticks: int = 72_000,
    n_shocks: int = 60,
    base_price: float = 80_000.0,
    seed: int = 0,
) -> dict:
    """
    Generate a synthetic tick stream.

    Returns a dict of arrays (length n_ticks each) ready to write to CSV:
        timestamp_ms, ce_ratio, obi, liquidation_rate, vpin, regime, mahal_dist,
        best_bid, best_ask
    """
    rng = np.random.default_rng(seed)

    # Background regime: low VPIN, near-zero CE, near-zero OBI.
    vpin = np.clip(rng.normal(0.2, 0.08, n_ticks), 0.0, 1.0)
    ce = np.clip(rng.exponential(0.1, n_ticks), 0.0, 50.0)
    obi = rng.normal(0.0, 0.18, n_ticks)
    liq = np.zeros(n_ticks, dtype=np.float32)

    # Background log returns: BTC vol ~ 50 bps stdev / hour at 1s ticks.
    # We use 100ms ticks; rescale: sigma_per_tick = 50bps / sqrt(36000) ≈ 0.026 bps.
    log_returns = rng.normal(0, 2.6e-5, n_ticks)

    # Plant shocks at random positions, spaced out, away from edges.
    shock_starts: list[int] = []
    margin = 200
    min_spacing = 400
    available = list(range(margin, n_ticks - 200))
    rng.shuffle(available)
    for pos in available:
        if all(abs(pos - s) >= min_spacing for s in shock_starts):
            shock_starts.append(pos)
        if len(shock_starts) >= n_shocks:
            break
    shock_starts.sort()

    if len(shock_starts) < n_shocks:
        print(
            f"WARNING: only planted {len(shock_starts)} shocks "
            f"(asked for {n_shocks}); n_ticks too small for spacing.",
            file=sys.stderr,
        )

    for s in shock_starts:
        n_pre, n_jump = 30, 50
        direction = int(rng.choice([-1, 1]))
        window = _make_shock_window(
            n_pre=n_pre,
            n_jump=n_jump,
            direction=direction,
            base_price=base_price,
            move_pct=float(rng.uniform(0.005, 0.015)),
            rng=rng,
        )
        end = s + n_pre + n_jump
        if end > n_ticks:
            continue
        vpin[s:end] = np.clip(window["vpin"], 0.0, 1.0)
        ce[s:end] = np.clip(window["ce"], 0.0, 50.0)
        obi[s:end] = np.clip(window["obi"], -1.0, 1.0)
        log_returns[s:end] = window["returns"]

    # Cumulative log price → mid price.
    log_price = np.log(base_price) + np.cumsum(log_returns)
    mid = np.exp(log_price)
    # Tight half-spread: 1 bp.
    half_spread = mid * 0.0001 * 0.5
    best_bid = mid - half_spread
    best_ask = mid + half_spread

    # Timestamp: 100ms ticks starting now.
    base_ts = 1_700_000_000_000
    timestamp = base_ts + np.arange(n_ticks, dtype=np.int64) * 100

    # Regime label: 0 below 0.7 vpin, 1 if 0.7-0.9, 2 above.
    regime = np.where(vpin < 0.7, 0, np.where(vpin < 0.9, 1, 2))

    # mahal_dist is unused at training time; fill with 0.
    mahal = np.zeros(n_ticks)

    return {
        "timestamp_ms": timestamp,
        "ce_ratio": ce,
        "obi": obi,
        "liquidation_rate": liq,
        "vpin": vpin,
        "regime": regime,
        "mahal_dist": mahal,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "n_shocks_planted": len(shock_starts),
    }


def write_csv(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(data["timestamp_ms"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "timestamp_ms,ce_ratio,obi,liquidation_rate,vpin,regime,mahal_dist,"
            "best_bid,best_ask\n"
        )
        for i in range(n):
            f.write(
                f"{data['timestamp_ms'][i]},{data['ce_ratio'][i]:.6f},"
                f"{data['obi'][i]:.6f},{data['liquidation_rate'][i]:.6f},"
                f"{data['vpin'][i]:.6f},{int(data['regime'][i])},"
                f"{data['mahal_dist'][i]:.6f},{data['best_bid'][i]:.6f},"
                f"{data['best_ask'][i]:.6f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=72_000, help="number of ticks (~10Hz; 72k = 2h)")
    parser.add_argument("--shocks", type=int, default=60, help="number of planted shocks")
    parser.add_argument("--base-price", type=float, default=80_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="./calibration/synthetic_history.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    print(f"Generating {args.ticks} ticks with {args.shocks} planted shocks (seed={args.seed})...")
    data = generate(
        n_ticks=args.ticks,
        n_shocks=args.shocks,
        base_price=args.base_price,
        seed=args.seed,
    )
    out_path = Path(args.out)
    write_csv(data, out_path)
    print(f"Wrote {out_path}")
    print(f"  shocks planted:   {data['n_shocks_planted']}")
    print(f"  duration (mock):  {args.ticks * 0.1 / 60:.1f} min at 10Hz")
    print(f"  mid range:        {data['best_bid'].min():.2f} - {data['best_ask'].max():.2f}")


if __name__ == "__main__":
    main()
