"""
tests/diagnose_tcn.py — Path B / P0.5 diagnostic recipe.

Step 1 implemented: logistic regression on **un-windowed** raw ticks.
Tests whether the signal that the TCN tries to extract from a 60-tick
window also exists at the instant tick level. If logistic AUROC ≈ 0.55,
the TCN's F2 ≈ 0.12 ceiling is consistent with the dataset's
intrinsic instant-level signal strength — confirming the data ceiling
explanation (see §13.8 of LAYER2_TRAINING.md). If AUROC ≥ 0.65, the
TCN is leaving signal on the table.

Usage:
    python tests/diagnose_tcn.py \
        --csv "calibration/feature_history_BTC.csv,calibration/feature_history_ETH.csv,calibration/feature_history_SOL.csv"

Output: AUROC + AUPRC for the multivariate fit, plus per-feature
univariate AUROCs. Interpretation footer prints the conclusion.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from train_tcn import build_labels


def compute_shock_labels(data: dict, horizon: int) -> np.ndarray:
    """Replicates train_stream.py:StreamingTCNDataset's --label-source=shock
    logic. Marks the H ticks BEFORE each identify_shock_events event as
    positive. Returns int8 label array of length n.
    """
    n = len(data["ce_ratio"])
    labels = np.zeros(n, dtype=np.int8)
    _, events = build_labels(data)
    for e in events:
        if 0 <= e.t_index < n:
            lo = max(0, e.t_index - horizon)
            labels[lo:e.t_index] = 1
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True,
        help="Comma-separated CSV paths (multi-coin pooled).",
    )
    parser.add_argument(
        "--label-horizon", type=int, default=config.TCN_LABEL_HORIZON_TICKS,
        help=f"H ticks before each shock to mark positive. "
             f"Default {config.TCN_LABEL_HORIZON_TICKS}.",
    )
    parser.add_argument(
        "--features",
        default="ce_ratio,obi,mlofi,vamp,kyles_lambda",
        help="Comma-separated feature columns (must exist in CSV).",
    )
    parser.add_argument(
        "--max-rows-per-coin", type=int, default=None,
        help="Optional subsample for speed. None = use all rows.",
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Inverse regularization for sklearn LogisticRegression.",
    )
    args = parser.parse_args()

    feature_names = [f.strip() for f in args.features.split(",")]
    csv_paths = [p.strip() for p in args.csv.split(",")]

    # --- Load + label per coin --------------------------------------------------
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    t0 = time.time()
    for path in csv_paths:
        print(f"Loading {path}...", flush=True)
        df = pl.read_csv(path, n_rows=args.max_rows_per_coin)
        n = len(df)
        # Sanity-check expected columns exist
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise SystemExit(
                f"CSV {path} is missing columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        data = {col: df[col].to_numpy() for col in df.columns}
        labels = compute_shock_labels(data, horizon=args.label_horizon)
        X = np.stack([data[f] for f in feature_names], axis=1).astype(np.float32)
        # Drop rows where any feature is NaN (rare but possible at chunk edges)
        valid = np.all(np.isfinite(X), axis=1)
        X = X[valid]
        labels = labels[valid]
        all_X.append(X)
        all_y.append(labels)
        pos = int(labels.sum())
        print(
            f"  {path}: {n:,} rows ({len(X):,} valid), "
            f"{pos:,} positives ({100.0 * pos / max(len(X), 1):.4f}%)",
            flush=True,
        )

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    base_rate = float(y.mean())
    print(
        f"\nPooled: {len(X):,} rows, {int(y.sum()):,} positives "
        f"({100.0 * base_rate:.4f}%). Load + label time: {time.time() - t0:.1f}s",
        flush=True,
    )

    # --- Multivariate fit -------------------------------------------------------
    print(f"\nFitting LogisticRegression(C={args.C}, class_weight=balanced)...", flush=True)
    t1 = time.time()
    clf = LogisticRegression(
        C=args.C, max_iter=2000, class_weight="balanced",
        solver="lbfgs", n_jobs=-1,
    )
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    auroc = float(roc_auc_score(y, probs))
    auprc = float(average_precision_score(y, probs))
    print(f"Fit time: {time.time() - t1:.1f}s", flush=True)

    print()
    print(f"=== Multivariate logistic ({len(feature_names)} features) ===")
    print(f"AUROC                : {auroc:.4f}")
    print(f"AUPRC                : {auprc:.4f}")
    print(f"AUPRC / base_rate    : {auprc / max(base_rate, 1e-9):.2f}x  "
          f"(random = 1.00x; TSLA-equity reached ~25x at convergence)")
    print()
    print("Feature coefficients (raw, not standardized — interpret signs only):")
    for fname, coef in zip(feature_names, clf.coef_[0]):
        print(f"  {fname:>16}: {coef:+.4f}")

    # --- Per-feature univariate -------------------------------------------------
    print()
    print(f"=== Univariate logistic (per-feature AUROC) ===")
    per_feature_auroc: dict[str, float] = {}
    for i, fname in enumerate(feature_names):
        clf_i = LogisticRegression(
            C=args.C, max_iter=2000, class_weight="balanced",
            solver="lbfgs", n_jobs=-1,
        )
        clf_i.fit(X[:, i:i + 1], y)
        p_i = clf_i.predict_proba(X[:, i:i + 1])[:, 1]
        a_i = float(roc_auc_score(y, p_i))
        per_feature_auroc[fname] = a_i
        print(f"  {fname:>16}: AUROC = {a_i:.4f}")

    # --- Interpretation footer --------------------------------------------------
    print()
    print(f"=== Interpretation ===")
    print(f"Base rate: {base_rate:.4f} ({100.0 * base_rate:.4f}% positives)")
    print(f"Random baseline AUROC: 0.5000")
    if auroc >= 0.65:
        print(f"AUROC {auroc:.3f} ≥ 0.65 — MEANINGFUL instant-level signal exists.")
        print("The TCN's F2 ≈ 0.12 ceiling is NOT a data ceiling; the model")
        print("is leaving signal on the table. Look for label noise, training")
        print("instability, or representation issues.")
    elif auroc >= 0.55:
        print(f"AUROC {auroc:.3f} ∈ [0.55, 0.65) — WEAK instant-level signal.")
        print("The TCN's F2 ≈ 0.12 is roughly consistent with this. Most of")
        print("the available signal is being extracted; further improvements")
        print("require new features (Path G), more data (H1), or richer labels.")
    else:
        print(f"AUROC {auroc:.3f} < 0.55 — instant-level signal at NOISE FLOOR.")
        print("Confirms the data ceiling explanation: this feature set + shock")
        print("label scheme does not admit much more than what we have.")
        print("Forward path is Path G (new features) or H1 (more data) or")
        print("re-examining the label definition (Path E label-inspection).")


if __name__ == "__main__":
    main()
