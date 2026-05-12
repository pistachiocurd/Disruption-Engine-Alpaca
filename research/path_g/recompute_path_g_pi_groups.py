"""
recompute_path_g_pi_groups.py — Recompute the four tanh-bounded π-group
columns (fo_market, sr, pi_kappa, pi_vamp_dim) in each
feature_history_<COIN>.csv using the *current* PATH_G_TANH_SCALES in
layer1_sensors.py. The raw scale columns (tau_c_s, L_c, D_c, V_c,
kappa_c) and all Path D columns are unchanged — they're ground truth
from the harvest; only the tanh-bounded π-groups depend on the
calibration scales.

Atomic: writes to <path>.tmp then os.replace() over the original.
Crash-safe — if the script dies midway, the original CSV is untouched.

Usage:
    python recompute_path_g_pi_groups.py
    python recompute_path_g_pi_groups.py BTC ETH SOL HYPE
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

# Allow running from anywhere; resolve repo root from this file's location.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from layer1_sensors import PATH_G_TANH_SCALES  # type: ignore  # noqa: E402
from config import EPSILON  # type: ignore  # noqa: E402

DEFAULT_COINS = ["BTC", "ETH", "SOL", "HYPE"]


def main(coins):
    fo_s = PATH_G_TANH_SCALES["fo_market"]
    sr_s = PATH_G_TANH_SCALES["sr"]
    pk_s = PATH_G_TANH_SCALES["pi_kappa"]
    pv_s = PATH_G_TANH_SCALES["pi_vamp_dim"]
    print(f"Using PATH_G_TANH_SCALES: fo={fo_s} sr={sr_s} pk={pk_s} pv={pv_s}\n")

    for c in coins:
        in_path = _ROOT / "calibration" / f"feature_history_{c}.csv"
        if not in_path.exists():
            print(f"{c}: missing ({in_path})")
            continue
        tmp_path = in_path.with_suffix(".csv.tmp")
        print(f"{c}: rewriting {in_path} ...", end=" ", flush=True)
        rewritten = 0
        with open(in_path, newline="") as fin, open(tmp_path, "w", newline="") as fout:
            rdr = csv.reader(fin)
            wtr = csv.writer(fout)
            header = next(rdr)
            wtr.writerow(header)
            idx = {n: i for i, n in enumerate(header)}
            prev_ts = None
            for row in rdr:
                ts = int(row[idx["timestamp_ms"]])
                tau = float(row[idx["tau_c_s"]])
                Lc = float(row[idx["L_c"]])
                Dc = float(row[idx["D_c"]])
                Vc = float(row[idx["V_c"]])
                kap = float(row[idx["kappa_c"]])
                vamp_bps = float(row[idx["vamp"]])
                bb = float(row[idx["best_bid"]])
                ba = float(row[idx["best_ask"]])

                dt_s = (
                    max((ts - prev_ts) / 1000.0, EPSILON)
                    if prev_ts is not None
                    else EPSILON
                )
                Lc_safe = max(Lc, EPSILON)
                tau_safe = max(tau, EPSILON)
                mid = (bb + ba) / 2.0 if (bb > 0.0 or ba > 0.0) else 0.0

                fo_raw = Dc * dt_s / (Lc_safe * Lc_safe)
                sr_raw = dt_s / tau_safe
                pk_raw = kap * Vc * tau_safe / Lc_safe
                pv_raw = (vamp_bps / 1e4) * mid / Lc_safe

                row[idx["fo_market"]] = f"{math.tanh(fo_raw / fo_s):.6f}"
                row[idx["sr"]] = f"{math.tanh(sr_raw / sr_s):.6f}"
                row[idx["pi_kappa"]] = f"{math.tanh(pk_raw / pk_s):.6f}"
                row[idx["pi_vamp_dim"]] = f"{math.tanh(pv_raw / pv_s):.6f}"

                wtr.writerow(row)
                rewritten += 1
                prev_ts = ts
        os.replace(tmp_path, in_path)
        print(f"done ({rewritten:,} rows)")


if __name__ == "__main__":
    coins = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_COINS
    main(coins)
