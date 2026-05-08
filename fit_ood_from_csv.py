"""
fit_ood_from_csv.py - Produce calibration/latest.json from a feature_history.csv
written by the engine when DUMP_FEATURES=1.

Workflow:
    1. Run the engine for a few hours with DUMP_FEATURES=1.
    2. Stop it (Ctrl+C). Inspect the CSV - should have ~36k rows/hour at 10Hz.
    3. Run this script. It fits mu, Sigma over [ce_ratio, obi, liquidation_rate].
    4. Restart the engine. It auto-loads calibration/latest.json on startup.

Usage:
    python fit_ood_from_csv.py
    python fit_ood_from_csv.py --csv path/to/file.csv --out calibration/latest.json
    python fit_ood_from_csv.py --trim-quantile 0.999

The script preserves any existing alpha_calibration_c in the output file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config
from calibration import fit_ood_distribution, load_calibration, save_calibration


def _read_csv(path: Path) -> np.ndarray:
    """Read [ce_ratio, obi, liquidation_rate] columns. Returns (N, 3) array."""
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            ce_idx = header.index("ce_ratio")
            obi_idx = header.index("obi")
            lr_idx = header.index("liquidation_rate")
        except ValueError as exc:
            raise SystemExit(
                f"required column missing from {path} header: {exc}\n"
                f"  found: {header}"
            )
        for line_no, line in enumerate(f, start=2):
            parts = line.strip().split(",")
            if len(parts) <= max(ce_idx, obi_idx, lr_idx):
                continue
            try:
                rows.append([
                    float(parts[ce_idx]),
                    float(parts[obi_idx]),
                    float(parts[lr_idx]),
                ])
            except ValueError:
                continue
    return np.array(rows, dtype=np.float64)


def _trim_outliers(data: np.ndarray, quantile: float) -> np.ndarray:
    """Drop rows with any feature outside [1-q, q] quantile range. Robustness aid."""
    if not 0.5 < quantile < 1.0:
        return data
    lo = np.quantile(data, 1.0 - quantile, axis=0)
    hi = np.quantile(data, quantile, axis=0)
    mask = np.all((data >= lo) & (data <= hi), axis=1)
    return data[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=config.FEATURE_DUMP_PATH,
        help=(
            "Input CSV produced by the engine with DUMP_FEATURES=1. "
            f"Default is per-symbol: {config.FEATURE_DUMP_PATH}"
        ),
    )
    parser.add_argument(
        "--out",
        default=config.OOD_CALIBRATION_PATH,
        help=(
            "Output JSON loaded by engine.py on startup. "
            f"Default is per-symbol: {config.OOD_CALIBRATION_PATH}"
        ),
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10_000,
        help="Refuse to fit on fewer than this many rows. ~10k = ~17min at 10Hz.",
    )
    parser.add_argument(
        "--trim-quantile",
        type=float,
        default=0.0,
        help="If >0, drop rows outside [1-q, q] quantile of each feature. "
             "Try 0.999 to clip the worst 0.1 pct as outliers.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        print(
            "Run the engine with DUMP_FEATURES=1 first:\n"
            "  $env:DUMP_FEATURES = '1'; python engine.py",
            file=sys.stderr,
        )
        sys.exit(1)

    data = _read_csv(csv_path)
    if len(data) < args.min_rows:
        print(
            f"ERROR: only {len(data)} valid rows in {csv_path}; "
            f"need >= {args.min_rows}. Let the engine run longer.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.trim_quantile > 0:
        before = len(data)
        data = _trim_outliers(data, args.trim_quantile)
        print(f"Trimmed outliers: {before} → {len(data)} rows "
              f"(quantile band [{1 - args.trim_quantile:.4f}, {args.trim_quantile:.4f}])")

    mu, sigma = fit_ood_distribution(data)

    # Preserve any existing alpha_calibration_c so this script doesn't clobber
    # the diffusion calibration when the operator hasn't run that yet.
    alpha_c = config.ALPHA_CALIBRATION_C
    if out_path.exists():
        try:
            existing = load_calibration(out_path)
            alpha_c = float(existing.get("alpha_calibration_c", alpha_c))
        except Exception as exc:
            print(f"WARNING: could not read existing {out_path}: {exc}", file=sys.stderr)

    save_calibration(out_path, alpha_c, mu, sigma)

    diag = np.diag(sigma)
    print(f"Wrote {out_path}")
    print(f"  rows used:                {len(data)}")
    print(f"  alpha_calibration_c:      {alpha_c:.6f}  (preserved)")
    print(f"  mu [ce, obi, liq]:        [{mu[0]:.4f}, {mu[1]:.4f}, {mu[2]:.4f}]")
    print(f"  Sigma diag:               [{diag[0]:.4f}, {diag[1]:.4f}, {diag[2]:.4f}]")
    print(f"  Sigma off-diag (ce/obi, ce/liq, obi/liq): "
          f"[{sigma[0,1]:.4f}, {sigma[0,2]:.4f}, {sigma[1,2]:.4f}]")
    print()
    print("Restart the engine to load the new calibration.")


if __name__ == "__main__":
    main()
