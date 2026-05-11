"""
split_csv_time.py — Time-ordered split of a feature_history CSV into
train and val files. Used for §14 per-symbol training/validation
protocol where train/val must be temporally disjoint (no leakage).

Usage:
    python split_csv_time.py <coin> [train_frac]
    # default train_frac=0.8

Writes:
    calibration/feature_history_<coin>.train.csv
    calibration/feature_history_<coin>.val.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main(coin: str, train_frac: float = 0.8) -> None:
    src = Path(f"./calibration/feature_history_{coin}.csv")
    if not src.exists():
        print(f"missing: {src}")
        sys.exit(1)

    train_path = src.with_suffix(".train.csv")
    val_path = src.with_suffix(".val.csv")

    with open(src, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = list(rdr)

    n = len(rows)
    split = int(n * train_frac)

    with open(train_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows[:split])

    with open(val_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows[split:])

    print(f"{coin}: total={n:,}  train={split:,} ({train_frac:.0%})  "
          f"val={n-split:,} ({1-train_frac:.0%})")
    print(f"  wrote {train_path}")
    print(f"  wrote {val_path}")


if __name__ == "__main__":
    coin = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    main(coin, frac)
