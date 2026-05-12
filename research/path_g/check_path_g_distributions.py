"""
check_path_g_distributions.py — Verify the recomputed Path G π-groups
are living in the dynamic range across all coins.

Healthy = not saturated at ±1 on most rows, not stuck near zero, has
non-trivial std. After running recompute_path_g_pi_groups.py this is
the smoke test before launching Phase 4 training.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

DEFAULT_COINS = ["BTC", "ETH", "SOL", "HYPE"]
PI_COLS = ["fo_market", "sr", "pi_kappa", "pi_vamp_dim"]


def percentile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    return sorted_vals[min(int(q * n), n - 1)]


def main(coins):
    print(
        f"{'coin':5s} {'pi':14s} "
        f"{'mean':>8s} {'std':>6s} {'p05':>7s} {'p95':>7s}  "
        f"{'sat':>6s} {'near0':>6s}"
    )
    print("-" * 80)
    for c in coins:
        p = _ROOT / "calibration" / f"feature_history_{c}.csv"
        if not p.exists():
            print(f"{c}: missing")
            continue
        with open(p) as f:
            rdr = csv.reader(f)
            h = next(rdr)
            rows = list(rdr)
        n = len(rows)
        for pi_name in PI_COLS:
            i = h.index(pi_name)
            vals = [float(r[i]) for r in rows]
            mean = sum(vals) / n
            std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
            sat = sum(1 for v in vals if abs(v) > 0.99) / n
            near0 = sum(1 for v in vals if abs(v) < 0.001) / n
            svals = sorted(vals)
            p05 = percentile(svals, 0.05)
            p95 = percentile(svals, 0.95)
            print(
                f"{c:5s} {pi_name:14s} "
                f"{mean:>+8.3f} {std:>6.3f} {p05:>+7.3f} {p95:>+7.3f}  "
                f"{sat:>6.1%} {near0:>6.1%}"
            )
        print()


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_COINS)
