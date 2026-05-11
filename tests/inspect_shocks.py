"""
tests/inspect_shocks.py — Path E label inspection.

Visualize + characterize the shock events that identify_shock_events
flags on HL crypto data. Tests whether the "shocks" the model is
trained against are coherent stress events with predictable pre-shock
signatures, or noise spikes the heuristic misfires on.

Outputs (per coin, into --out-dir):
  - shocks_<COIN>.png      — 16-event grid plot (mid Δ% + OBI overlay)
  - inspection_<COIN>.txt  — numeric summary (event count, magnitudes,
                             KS test of pre-shock OBI vs random)

Replicates the §13 KS test (D=0.13, p~1e-28) on the current 12-column
CSVs to verify the original finding holds on the re-harvested data.

Usage:
    python tests/inspect_shocks.py \
        --csv "calibration/feature_history_BTC.csv,calibration/feature_history_ETH.csv,calibration/feature_history_SOL.csv"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from train_tcn import build_labels


def inspect_coin(csv_path: Path, out_dir: Path, lookback: int = 10) -> dict:
    """All inspection for one coin. Returns a small dict of headline stats."""
    print(f"\n{'=' * 70}")
    print(f"Coin: {csv_path.name}")
    print(f"{'=' * 70}", flush=True)

    df = pl.read_csv(csv_path)
    n = len(df)
    data = {col: df[col].to_numpy() for col in df.columns}
    print(f"Total rows: {n:,}")

    # Build event list using the same path train_stream.py uses
    _, events = build_labels(data)
    indices = np.array(sorted([e.t_index for e in events if 0 <= e.t_index < n]))
    n_events = len(indices)
    print(f"Shock events: {n_events:,}  (1 per {n / max(n_events, 1):.0f} ticks)")

    summary_lines: list[str] = []
    summary_lines.append(f"Coin: {csv_path.name}")
    summary_lines.append(f"Total rows: {n:,}")
    summary_lines.append(f"Shock events: {n_events:,}")

    if n_events == 0:
        (out_dir / f"inspection_{csv_path.stem}.txt").write_text(
            "\n".join(summary_lines), encoding="utf-8",
        )
        return {"events": 0}

    bid = data["best_bid"]
    ask = data["best_ask"]
    mid = 0.5 * (bid + ask)
    obi = data["obi"]

    # Inter-event gap distribution
    if n_events > 1:
        gaps = np.diff(indices)
        line = (
            f"Inter-event ticks: "
            f"median={int(np.median(gaps))}, "
            f"P5={int(np.percentile(gaps, 5))}, "
            f"P50={int(np.percentile(gaps, 50))}, "
            f"P95={int(np.percentile(gaps, 95))}, "
            f"max={int(gaps.max())}"
        )
        print(line)
        summary_lines.append(line)

    # Event magnitude — mid % change over ±10 ticks
    LOOK = 10
    mags = []
    for idx in indices:
        if idx - LOOK < 0 or idx + LOOK >= n:
            continue
        pre_m = mid[idx - LOOK]
        post_m = mid[idx + LOOK]
        if pre_m > 0:
            mags.append((post_m - pre_m) / pre_m * 100.0)
    mags = np.array(mags)
    if len(mags):
        line = (
            f"Event magnitude |Δp/p| % over ±10 ticks: "
            f"mean={np.mean(np.abs(mags)):.4f}, "
            f"P50={np.median(np.abs(mags)):.4f}, "
            f"P95={np.percentile(np.abs(mags), 95):.4f}, "
            f"max={np.max(np.abs(mags)):.4f}"
        )
        print(line)
        summary_lines.append(line)
        line = (
            f"Event magnitude signed Δp/p % over ±10 ticks: "
            f"mean={mags.mean():+.4f}, std={mags.std():.4f}  "
            f"(buy:sell ratio {(mags > 0).sum()}:{(mags <= 0).sum()})"
        )
        print(line)
        summary_lines.append(line)

    # KS test — pre-shock OBI window vs random-window OBI
    pre_obi_parts = [obi[idx - lookback:idx] for idx in indices if idx - lookback >= 0]
    if pre_obi_parts:
        pre_obi = np.concatenate(pre_obi_parts)
        rng = np.random.default_rng(42)
        n_samples = len(pre_obi)
        # Random starts: anywhere except within lookback of an event
        random_starts = rng.integers(lookback, n, size=n_samples // lookback + 1)
        rand_obi = np.concatenate([obi[s - lookback:s] for s in random_starts])
        if len(rand_obi) > 0:
            ks_d, ks_p = stats.ks_2samp(pre_obi, rand_obi)
            line = f"KS test pre-shock OBI(last {lookback} ticks) vs random window:"
            print(line); summary_lines.append(line)
            line = f"  D = {ks_d:.4f},   p = {ks_p:.4e}   (§13 baseline: D=0.13, p~1e-28)"
            print(line); summary_lines.append(line)
            line = (
                f"  pre-shock OBI:  mean={pre_obi.mean():+.4f}, "
                f"std={pre_obi.std():.4f}, n={len(pre_obi):,}"
            )
            print(line); summary_lines.append(line)
            line = (
                f"  random   OBI:   mean={rand_obi.mean():+.4f}, "
                f"std={rand_obi.std():.4f}, n={len(rand_obi):,}"
            )
            print(line); summary_lines.append(line)

    # Plot grid — 16 example events
    n_plot = min(16, n_events)
    rng = np.random.default_rng(7)
    plot_idx = sorted(rng.choice(indices, size=n_plot, replace=False))
    PRE_WIN = 200
    POST_WIN = 50

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f"{csv_path.stem}: random sample of {n_plot} shock events "
        f"(red line = event, blue = mid Δ%, orange = OBI)",
        fontsize=12,
    )
    for ax, t in zip(axes.flat, plot_idx):
        lo = max(0, t - PRE_WIN)
        hi = min(n, t + POST_WIN)
        offsets = np.arange(lo, hi) - t
        local_mid = mid[lo:hi]
        local_obi = obi[lo:hi]
        if mid[t] > 0:
            mid_pct = (local_mid / mid[t] - 1) * 100
        else:
            mid_pct = np.zeros_like(local_mid)

        ax2 = ax.twinx()
        ax.plot(offsets, mid_pct, color="#1f77b4", lw=1.0)
        ax2.plot(offsets, local_obi, color="#ff7f0e", lw=0.7, alpha=0.55)
        ax.axvline(0, color="red", lw=0.7, alpha=0.6)
        ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
        ax.set_title(f"@ tick {t:,}", fontsize=8)
        ax.set_xlabel("Δ ticks", fontsize=7)
        ax.set_ylabel("mid Δ%", color="#1f77b4", fontsize=7)
        ax2.set_ylabel("OBI", color="#ff7f0e", fontsize=7)
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)

    png_path = out_dir / f"shocks_{csv_path.stem}.png"
    plt.savefig(png_path, dpi=85)
    plt.close(fig)
    print(f"Saved plot: {png_path}")
    summary_lines.append(f"Plot: {png_path}")

    (out_dir / f"inspection_{csv_path.stem}.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8",
    )
    return {
        "events": n_events,
        "abs_mag_p95_pct": float(np.percentile(np.abs(mags), 95)) if len(mags) else 0.0,
        "ks_d": float(ks_d) if pre_obi_parts and len(rand_obi) > 0 else 0.0,
        "ks_p": float(ks_p) if pre_obi_parts and len(rand_obi) > 0 else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True,
                        help="Comma-separated CSV paths to inspect.")
    parser.add_argument("--out-dir", default="./calibration/inspection",
                        help="Where to write PNG + text summaries.")
    parser.add_argument("--lookback", type=int, default=10,
                        help="KS test window length (ticks before each event). "
                             "§13's headline result used 10.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}")

    csv_paths = [Path(p.strip()) for p in args.csv.split(",")]
    headlines = {}
    for csv_path in csv_paths:
        headlines[csv_path.stem] = inspect_coin(csv_path, out_dir, lookback=args.lookback)

    print(f"\n{'=' * 70}\nHeadline summary\n{'=' * 70}")
    print(f"{'Coin':<32}  {'Events':>10}  {'|Δp|95%':>10}  {'KS D':>8}  {'KS p':>14}")
    for coin, stats_ in headlines.items():
        print(
            f"{coin:<32}  {stats_['events']:>10,}  "
            f"{stats_['abs_mag_p95_pct']:>10.4f}  "
            f"{stats_['ks_d']:>8.4f}  {stats_['ks_p']:>14.4e}"
        )


if __name__ == "__main__":
    main()
