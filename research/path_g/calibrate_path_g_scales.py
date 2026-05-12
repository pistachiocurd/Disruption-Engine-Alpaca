"""
calibrate_path_g_scales.py — Recompute PATH_G_TANH_SCALES from harvested CSVs.

Pools raw scale columns (tau_c_s, L_c, D_c, V_c, kappa_c) + timestamp/vamp/best_bid/best_ask
across all coin CSVs found in ./calibration/, computes the raw pi-group ratios
that the tanh saturator wraps, and prints the empirical p95(|raw|)/2 to use as
PATH_G_TANH_SCALES in layer1_sensors.py.

Usage:
    python calibrate_path_g_scales.py
    python calibrate_path_g_scales.py BTC ETH SOL HYPE     # specific coins
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

DEFAULT_COINS = ["BTC", "ETH", "SOL", "HYPE"]


def cols(rows, h, *names):
    idx = [h.index(n) for n in names]
    return [[float(r[i]) for r in rows] for i in idx]


def p95abs(x):
    a = sorted(abs(v) for v in x if v != 0 and v == v)
    return a[int(0.95 * len(a))] if a else 0.0


def main(coins):
    all_fo, all_sr, all_pk, all_pv = [], [], [], []

    for c in coins:
        p = _ROOT / "calibration" / f"feature_history_{c}.csv"
        if not p.exists():
            print(f"{c}: missing ({p})")
            continue
        with open(p) as f:
            rdr = csv.reader(f)
            h = next(rdr)
            data = list(rdr)
        tau, Lc, Dc, Vc, kap, bb, ba, vamp, ts = cols(
            data, h,
            "tau_c_s", "L_c", "D_c", "V_c", "kappa_c",
            "best_bid", "best_ask", "vamp", "timestamp_ms",
        )
        dt_s = [0.0] + [(ts[i] - ts[i-1]) / 1000.0 for i in range(1, len(ts))]
        fo = [Dc[i] * dt_s[i] / (Lc[i] ** 2)        for i in range(len(data)) if Lc[i] > 0]
        sr = [dt_s[i] / tau[i]                       for i in range(len(data)) if tau[i] > 0]
        pk = [kap[i] * Vc[i] * tau[i] / Lc[i]        for i in range(len(data)) if Lc[i] > 0]
        pv = [(vamp[i] / 1e4) * ((bb[i] + ba[i]) / 2) / Lc[i]
              for i in range(len(data)) if Lc[i] > 0]
        all_fo += fo
        all_sr += sr
        all_pk += pk
        all_pv += pv
        print(
            f"{c:5s}: rows={len(data):>9}  "
            f"fo p95={p95abs(fo):.4g}  "
            f"sr p95={p95abs(sr):.4g}  "
            f"pk p95={p95abs(pk):.4g}  "
            f"pv p95={p95abs(pv):.4g}"
        )

    print()
    print("POOLED PATH_G_TANH_SCALES (raw p95 / 2):")
    print(f'    "fo_market":   {p95abs(all_fo) / 2:.4g},')
    print(f'    "sr":          {p95abs(all_sr) / 2:.4g},')
    print(f'    "pi_kappa":    {p95abs(all_pk) / 2:.4g},')
    print(f'    "pi_vamp_dim": {p95abs(all_pv) / 2:.4g},')


if __name__ == "__main__":
    coins = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_COINS
    main(coins)
