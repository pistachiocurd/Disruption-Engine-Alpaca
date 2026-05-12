"""
check_shocks.py - Quick status check on the per-symbol feature CSV.

Reports:
  - row count (and approx duration at 10 Hz)
  - shock event count (per identify_shock_events)
  - whether you have enough data to train the TCN (>= MIN_CALIBRATION_EVENTS)

Usage:
    python check_shocks.py
    python check_shocks.py --csv path/to/other.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from train_tcn import build_labels, load_csv  # noqa: E402  (sibling)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=config.FEATURE_DUMP_PATH,
        help=f"CSV to inspect (default: {config.FEATURE_DUMP_PATH})",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    data = load_csv(csv_path)
    n = len(data["vpin"])
    hours = n / 36_000.0  # 10 ticks/sec

    _, events = build_labels(data)
    n_shocks = len(events)

    print()
    print(f"  CSV:           {csv_path}")
    print(f"  Symbol:        {config.SYMBOL}")
    print(f"  Rows:          {n:,}  (~{hours:.2f} hours at 10 Hz)")
    print(f"  Shock events:  {n_shocks}")
    print(f"  Threshold:     {config.MIN_CALIBRATION_EVENTS}")
    print()
    if n_shocks >= config.MIN_CALIBRATION_EVENTS:
        print(f"  Ready to train the TCN.  Run:  python train_tcn.py")
    else:
        needed = config.MIN_CALIBRATION_EVENTS - n_shocks
        print(f"  Need {needed} more shocks before train_tcn.py is recommended.")
        print(f"  Keep collecting; refit OOD daily with: python fit_ood_from_csv.py")
    print()


if __name__ == "__main__":
    main()
