"""
transform_l3_features_v2_lite.py

Offline, no-lookahead feature-drift fix for the L3 directional pipeline.

Why this exists
---------------
Phase 2 failed forward testing because the most load-bearing raw channels
(`hidden_trade_rate`, `lifespan_bid_p50_ms`, `lifespan_ask_p50_ms`) shifted
between train and forward-test windows. A model trained on raw magnitudes can
therefore learn "old regime" values instead of reusable market structure.

This script creates a v2-lite CSV that keeps the existing 19-column schema
so train_l3_directional.py/backtest_l3_directional.py do not need refactors.
It replaces drift-prone raw magnitudes with no-lookahead rolling percentile
ranks. For each row t, percentile_rank(x_t) is computed against rows
[t-window, t), never against future rows.

Output columns are intentionally kept under the same names for compatibility.
The output JSON records which columns were transformed so this is auditable.

Usage
-----
python research/path_h_l3/transform_l3_features_v2_lite.py \
  --csv-dir research/path_h_l3/calibration \
  --symbols tBTCUSD tETHUSD tSOLUSD \
  --window 20000

Then train on the transformed CSVs:
python research/path_h_l3/train_l3_directional.py \
  --csv-dir research/path_h_l3/calibration \
  --symbols tBTCUSD_v2lite tETHUSD_v2lite tSOLUSD_v2lite \
  --horizon 300 --seq-len 60 --epochs 1 \
  --weights-out research/path_h_l3/calibration/tcn_weights_l3_v2lite_H300.pt \
  --stats-out research/path_h_l3/calibration/l3_feature_stats_v2lite_H300.json \
  --metrics-out research/path_h_l3/calibration/l3_directional_metrics_v2lite_H300.json
"""
from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right, insort
from collections import deque
from pathlib import Path
from typing import Iterable

DRIFT_PRONE_PERCENTILE_COLS = [
    "hidden_trade_rate",
    "lifespan_bid_p50_ms",
    "lifespan_ask_p50_ms",
    # Size and queue-rate channels are also regime-sensitive; transform them
    # in v2-lite but leave already bounded ratio/bps channels raw.
    "top_bid_size",
    "top_ask_size",
    "queue_depletion_bid_per_s",
    "queue_depletion_ask_per_s",
]

NOISY_DROP_TO_ZERO_COLS = [
    # NEXT_PHASE_PLAN flags p95 lifespan as net-noise. Keep schema-compatible
    # columns but neutralize them so existing trainer width stays 19.
    "lifespan_bid_p95_ms",
    "lifespan_ask_p95_ms",
]


def rolling_percentile_no_lookahead(values: Iterable[float], window: int, min_periods: int) -> list[float]:
    """Return percentile ranks against prior rolling history only.

    If there is not enough prior history, emit neutral 0.5. Using 0.5 says
    "not unusually high or low yet" and avoids contaminating early rows.
    """
    hist_sorted: list[float] = []
    hist_queue: deque[float] = deque()
    out: list[float] = []

    for raw in values:
        x = float(raw)
        if len(hist_sorted) < min_periods:
            out.append(0.5)
        else:
            out.append(bisect_right(hist_sorted, x) / len(hist_sorted))

        insort(hist_sorted, x)
        hist_queue.append(x)
        if len(hist_queue) > window:
            old = hist_queue.popleft()
            idx = bisect_right(hist_sorted, old) - 1
            # Walk left in the rare case of floating duplicate placement.
            while idx > 0 and hist_sorted[idx] != old:
                idx -= 1
            if hist_sorted[idx] == old:
                hist_sorted.pop(idx)
            else:
                raise RuntimeError("rolling percentile internal removal failed")

    return out


def safe_ratio(a: float, b: float) -> float:
    denom = float(a) + float(b)
    return 0.5 if denom <= 1e-12 else float(a) / denom


def transform_file(input_path: Path, output_path: Path, window: int, min_periods: int) -> dict:
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"no header in {input_path}")
        fieldnames = list(reader.fieldnames)

    missing = [c for c in DRIFT_PRONE_PERCENTILE_COLS + NOISY_DROP_TO_ZERO_COLS if c not in fieldnames]
    if missing:
        raise ValueError(f"{input_path} missing expected columns: {missing}")

    # Add stable bid-vs-ask ratio information while staying schema-compatible:
    # overwrite bid columns with bid/(bid+ask), ask columns with ask/(bid+ask).
    ratio_pairs = [
        ("arrival_rate_bid_per_s", "arrival_rate_ask_per_s"),
        ("cancel_rate_bid_per_s", "cancel_rate_ask_per_s"),
    ]
    for bid_col, ask_col in ratio_pairs:
        if bid_col in fieldnames and ask_col in fieldnames:
            for r in rows:
                bid = float(r[bid_col])
                ask = float(r[ask_col])
                bid_ratio = safe_ratio(bid, ask)
                r[bid_col] = f"{bid_ratio:.8g}"
                r[ask_col] = f"{1.0 - bid_ratio:.8g}"

    for col in DRIFT_PRONE_PERCENTILE_COLS:
        vals = [float(r[col]) for r in rows]
        ranks = rolling_percentile_no_lookahead(vals, window=window, min_periods=min_periods)
        for r, rank in zip(rows, ranks):
            r[col] = f"{rank:.8g}"

    for col in NOISY_DROP_TO_ZERO_COLS:
        for r in rows:
            r[col] = "0.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "window_events": window,
        "min_periods": min_periods,
        "percentile_rank_columns": DRIFT_PRONE_PERCENTILE_COLS,
        "zeroed_noise_columns": NOISY_DROP_TO_ZERO_COLS,
        "ratio_pairs_overwritten_for_compatibility": ratio_pairs,
        "no_lookahead": True,
        "schema_compatible_with_v1_trainer": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, default=Path(__file__).parent / "calibration")
    p.add_argument("--symbols", nargs="+", default=["tBTCUSD", "tETHUSD", "tSOLUSD"])
    p.add_argument("--window", type=int, default=20000, help="Rolling event window for percentile ranks.")
    p.add_argument("--min-periods", type=int, default=1000, help="Prior rows required before emitting ranks.")
    p.add_argument("--suffix", default="_v2lite")
    p.add_argument("--manifest-out", type=Path, default=None)
    args = p.parse_args()

    manifest = []
    for sym in args.symbols:
        inp = args.csv_dir / f"l3_ticks_{sym}.csv"
        out = args.csv_dir / f"l3_ticks_{sym}{args.suffix}.csv"
        if not inp.exists():
            print(f"[skip] missing {inp}")
            continue
        info = transform_file(inp, out, args.window, args.min_periods)
        manifest.append(info)
        print(f"[ok] {sym}: {info['rows']:,} rows -> {out.name}")

    manifest_path = args.manifest_out or (args.csv_dir / "feature_stack_v2lite_manifest.json")
    with manifest_path.open("w") as f:
        json.dump({"files": manifest}, f, indent=2)
    print(f"[ok] wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
