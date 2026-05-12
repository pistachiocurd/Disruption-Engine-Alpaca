"""
test_features.py - Verify whether the H-tick pre-shock window distribution
of [ce_ratio, obi, liquidation_rate] is statistically distinguishable
from random non-shock windows.

Diagnostic, not a pytest test. Lives in tests/ alongside the unit-test
suite for convention. Run directly:

    python tests/test_features.py --csv calibration/feature_history_BTC.csv

If pre-shock and random distributions overlap heavily (KS D < 0.05, large
p), no model architecture can extract signal from this feature set on this
data — saves debug-loop time before chasing label-horizon / loss / data-
volume tweaks. If KS D is significant (typically > 0.10 on at least one
channel), the signal IS learnable; the problem is elsewhere (label
horizon, loss imbalance, training stability).

Reports:
  - per-channel mean +/- sd over the full window
  - per-channel mean +/- sd over the LAST 10 ticks (most temporally
    relevant for a leading classifier)
  - KS 2-sample distribution distance with approximate p-values
  - per-window summary stats (helps spot bursty vs steady signals)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python tests/test_features.py` from the project root without
# pytest activating conftest.py — explicit project-root insertion mirrors
# the conftest setup.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# train_tcn lives in training/ after the 2026-05-11 reorg.
_TRAINING = ROOT / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

import numpy as np
import polars as pl

import config
from train_tcn import build_labels, load_csv

HORIZON = config.TCN_LABEL_HORIZON_TICKS  # 30 by default
WINDOW = 60                                 # match TCN seq_len
RNG_SEED = 0


def collect_windows(features: np.ndarray, indices: np.ndarray, window: int) -> np.ndarray:
    """For each index i, gather features[i-window:i]. Skip indices < window."""
    valid = indices[indices >= window]
    out = np.empty((len(valid), window, features.shape[1]), dtype=np.float32)
    for k, i in enumerate(valid):
        out[k] = features[i - window:i]
    return out


def ks_2samp_simple(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Lightweight 2-sample Kolmogorov-Smirnov; returns (D, approx p)."""
    a = np.sort(a)
    b = np.sort(b)
    all_vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    n_e = (len(a) * len(b)) / (len(a) + len(b))
    # Approximate p-value (Kolmogorov distribution; one-tailed -> two-tailed)
    p = float(2 * np.exp(-2 * n_e * D * D))
    p = min(max(p, 0.0), 1.0)
    return D, p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=config.FEATURE_DUMP_PATH)
    parser.add_argument("--n-random", type=int, default=2000,
                        help="Number of random non-shock windows to sample for the "
                             "comparison baseline.")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help=f"Pre-shock label horizon (default: config "
                             f"TCN_LABEL_HORIZON_TICKS={HORIZON}).")
    parser.add_argument("--last-n", type=int, default=10,
                        help="Number of trailing ticks to compare separately as the "
                             "'most temporally relevant' slice (default 10).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return 1

    print(f"Loading {csv_path} ...")
    data = load_csv(csv_path)
    n = len(data["ce_ratio"])
    print(f"  rows: {n:,}")

    _, events = build_labels(data)
    n_shocks = len(events)
    print(f"  shock events: {n_shocks}")
    if n_shocks < 30:
        print(f"  WARNING: only {n_shocks} shocks; comparison will be noisy.")

    features = np.stack([
        np.asarray(data["ce_ratio"], dtype=np.float32) / 10.0,
        np.asarray(data["obi"], dtype=np.float32),
        np.asarray(data["liquidation_rate"], dtype=np.float32),
    ], axis=1)
    channel_names = ["ce_ratio/10", "obi", "liquidation_rate"]

    shock_indices = np.array([e.t_index for e in events])
    pre_shock_t = shock_indices - 1
    pre_shock_windows = collect_windows(features, pre_shock_t, WINDOW)
    print(f"\n  pre-shock windows collected: {len(pre_shock_windows)}")

    excl_mask = np.zeros(n, dtype=bool)
    for idx in shock_indices:
        lo = max(0, idx - args.horizon)
        hi = min(n, idx + 1)
        excl_mask[lo:hi] = True
    rng = np.random.default_rng(RNG_SEED)
    candidate_idxs = np.where(~excl_mask)[0]
    candidate_idxs = candidate_idxs[candidate_idxs >= WINDOW]
    if len(candidate_idxs) < args.n_random:
        print(f"  WARNING: only {len(candidate_idxs)} candidate non-shock "
              f"indices, requested {args.n_random}")
    sampled = rng.choice(candidate_idxs, size=min(args.n_random, len(candidate_idxs)),
                         replace=False)
    random_windows = collect_windows(features, sampled, WINDOW)
    print(f"  random non-shock windows collected: {len(random_windows)}")

    print()
    print("=== Per-channel comparison (mean +/- sd over the whole 60-tick window) ===")
    print(f"{'channel':<20} {'pre-shock mu+-sd':<22} {'random mu+-sd':<22} {'KS D':<8} {'~p':<10}")
    for c, name in enumerate(channel_names):
        a = pre_shock_windows[:, :, c].flatten()
        b = random_windows[:, :, c].flatten()
        D, p = ks_2samp_simple(a, b)
        marker = " ***" if p < 0.001 else (" *" if p < 0.05 else "")
        print(
            f"{name:<20} "
            f"{a.mean():>9.4f} +- {a.std():>7.4f}    "
            f"{b.mean():>9.4f} +- {b.std():>7.4f}    "
            f"{D:>6.4f}  {p:>9.2e}{marker}"
        )

    print()
    print(f"=== Per-channel comparison on the LAST {args.last_n} ticks of each window "
          f"(most temporally relevant) ===")
    print(f"{'channel':<20} {'pre-shock mu+-sd':<22} {'random mu+-sd':<22} {'KS D':<8} {'~p':<10}")
    for c, name in enumerate(channel_names):
        a = pre_shock_windows[:, -args.last_n:, c].flatten()
        b = random_windows[:, -args.last_n:, c].flatten()
        D, p = ks_2samp_simple(a, b)
        marker = " ***" if p < 0.001 else (" *" if p < 0.05 else "")
        print(
            f"{name:<20} "
            f"{a.mean():>9.4f} +- {a.std():>7.4f}    "
            f"{b.mean():>9.4f} +- {b.std():>7.4f}    "
            f"{D:>6.4f}  {p:>9.2e}{marker}"
        )

    print()
    print("=== Per-window summary (pre-shock vs random; mean of per-window aggregate) ===")
    for c, name in enumerate(channel_names):
        a_mean_per_win = pre_shock_windows[:, :, c].mean(axis=1)
        b_mean_per_win = random_windows[:, :, c].mean(axis=1)
        a_std_per_win = pre_shock_windows[:, :, c].std(axis=1)
        b_std_per_win = random_windows[:, :, c].std(axis=1)
        print(f"  {name}:")
        print(f"    pre-shock window-mean: mu={a_mean_per_win.mean():.4f} sd={a_mean_per_win.std():.4f}")
        print(f"    random   window-mean: mu={b_mean_per_win.mean():.4f} sd={b_mean_per_win.std():.4f}")
        print(f"    pre-shock window-std:  mu={a_std_per_win.mean():.4f}")
        print(f"    random   window-std:   mu={b_std_per_win.mean():.4f}")

    print()
    print("Interpretation:")
    print("  - KS D measures distribution distance. D ~= 0 = identical, D ~= 1 = disjoint.")
    print("  - Significant D + small p-value (< 0.001 marked '***') = the model")
    print("    HAS distinguishable signal to learn. Architecture/data should work.")
    print("  - D < 0.05 + large p = pre-shock and random distributions overlap.")
    print("    No NN can extract what isn't statistically there. Need richer features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
