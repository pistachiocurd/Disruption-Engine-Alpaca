"""
train_stream.py — Streaming TCN trainer + threshold tuner.

Differs from train_tcn.py in three ways:

1. Loads the feature_history CSV in polars batches via an IterableDataset
   rather than pulling the whole file into memory. Lets us train on
   multi-GB harvests that wouldn't fit in RAM.

2. Builds positive labels directly from the `regime` column when present,
   falling back to identify_shock_events for older CSVs without a regime
   tag. The regime column is denser and works on curated / discontinuous
   files (the per-batch event detector chokes on cluster boundaries
   because it needs continuous price/VPIN context to validate candidates).

3. After training (or with --tune-only on existing weights), runs a
   threshold sweep on either the train CSV or a separately-pointed
   --val-csv, picks an operating point per `--threshold-criterion`, and
   writes the threshold + full sweep table to disk. The engine reads
   tcn_threshold_<SYMBOL>.json["threshold"] at startup.

Default operating-point criterion is F-β with β=2 — recall-biased,
because for an arbitrage engine a missed shock costs more than a wasted
mandate (you can't capture alpha on a shock you didn't see).

Usage:
    python train_stream.py                                  # train + auto-pick threshold
    python train_stream.py --tune-only                      # re-pick threshold, ~30s, no retrain
    python train_stream.py --tune-only --val-csv other.csv  # held-out / cross-symbol eval
    python train_stream.py --tune-only --threshold-criterion min-precision --min-precision 0.75
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import config
from layer2_alpha import TCNSpikePredictor
from train_tcn import build_labels


class StreamingTCNDataset(IterableDataset):
    """
    Yields (window, label) pairs from a feature_history CSV without loading
    the whole file. Each chunk is read by polars, converted to a numpy
    dict, labeled (regime column preferred, identify_shock_events as
    fallback), and emitted as overlapping seq_len-tick windows.

    A seq_len-tick suffix is carried into the next chunk so windows
    spanning the chunk boundary are still produced — without this we'd
    drop seq_len-1 windows at every boundary.
    """

    def __init__(self, csv_path, seq_len=60):
        self.csv_path = csv_path
        self.seq_len = seq_len

    def __iter__(self):
        reader = pl.read_csv_batched(self.csv_path)

        carry_over_features = None
        carry_over_labels = None

        batches = reader.next_batches(1)
        while batches:
            df_chunk = batches[0]
            data_chunk = {col: df_chunk[col].to_numpy() for col in df_chunk.columns}

            # Leading classifier: mark the H ticks BEFORE each shock as
            # positive so the model fires on the pre-shock signature, not
            # on the shock itself when it's already too late to act.
            n = len(df_chunk)
            labels = np.zeros(n, dtype=np.float32)
            horizon = config.TCN_LABEL_HORIZON_TICKS

            if "regime" in data_chunk:
                regime = data_chunk["regime"].astype(np.int8)
                is_start = np.zeros(n, dtype=bool)
                if n > 0:
                    is_start[0] = regime[0] == 1
                    is_start[1:] = (regime[1:] == 1) & (regime[:-1] == 0)
                for idx in np.where(is_start)[0]:
                    lo = max(0, idx - horizon)
                    labels[lo:idx] = 1.0
            else:
                _, events = build_labels(data_chunk)
                for e in events:
                    if 0 <= e.t_index < n:
                        lo = max(0, e.t_index - horizon)
                        labels[lo:e.t_index] = 1.0

            # ce_ratio is divided by 10 to bring its dynamic range into
            # rough parity with obi (∈ [-1, 1]) and liquidation_rate.
            # AlphaEngine._push_features applies the same /10 at inference,
            # so trained weights and live inputs share scale.
            features = np.stack([
                data_chunk["ce_ratio"] / 10,
                data_chunk["obi"],
                data_chunk["liquidation_rate"],
            ], axis=1).astype(np.float32)

            if carry_over_features is not None:
                features = np.vstack([carry_over_features, features])
                labels = np.concatenate([carry_over_labels, labels])

            for i in range(self.seq_len, len(features)):
                X = features[i - self.seq_len:i].T  # (3, seq_len)
                y = labels[i]
                yield torch.from_numpy(X), torch.tensor(y)

            carry_over_features = features[-self.seq_len:]
            carry_over_labels = labels[-self.seq_len:]

            batches = reader.next_batches(1)


def evaluate(model, loader, criterion, device):
    """Single eval pass over `loader`. Returns dict of loss + classification counts."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    tp = fp = fn = 0
    n_pos_pred = 0
    n_pos_true = 0
    n_total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h = model.input_proj(X_batch)
                for blk in model.blocks:
                    h = blk(h)
                logits = model.head(h[:, :, -1]).squeeze(-1)
                loss = criterion(logits, y_batch)
            total_loss += loss.item()
            n_batches += 1
            preds = torch.sigmoid(logits.float()) >= 0.5
            y_bool = y_batch >= 0.5
            tp += int((preds & y_bool).sum().item())
            fp += int((preds & ~y_bool).sum().item())
            fn += int((~preds & y_bool).sum().item())
            n_pos_pred += int(preds.sum().item())
            n_pos_true += int(y_bool.sum().item())
            n_total += y_bool.numel()
    return {
        "avg_loss": total_loss / max(n_batches, 1),
        "tp": tp, "fp": fp, "fn": fn,
        "n_pos_pred": n_pos_pred, "n_pos_true": n_pos_true,
        "n_total": n_total,
    }


def collect_predictions(model, loader, device):
    """One inference pass over `loader`. Returns (probs, labels) as flat np arrays."""
    model.eval()
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h = model.input_proj(X_batch)
                for blk in model.blocks:
                    h = blk(h)
                logits = model.head(h[:, :, -1]).squeeze(-1)
            probs_all.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels_all.append(y_batch.cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)


def threshold_sweep(probs, labels):
    """Compute prec/rec/F1 across a 0.05..0.99 threshold grid. Returns list of dicts."""
    thresholds = np.concatenate([
        np.linspace(0.05, 0.45, 9),
        np.linspace(0.50, 0.90, 17),
        np.linspace(0.92, 0.99, 8),
    ])
    y_bool = labels >= 0.5
    rows = []
    for thr in thresholds:
        pred = probs >= thr
        tp = int(np.logical_and(pred, y_bool).sum())
        fp = int(np.logical_and(pred, ~y_bool).sum())
        fn = int(np.logical_and(~pred, y_bool).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({
            "thr": float(thr), "tp": tp, "fp": fp, "fn": fn,
            "pred_pos": tp + fp, "prec": prec, "rec": rec, "f1": f1,
        })
    return rows


def pick_threshold(rows, criterion, min_precision=None, min_recall=None, beta=1.0):
    """Pick the row whose threshold best satisfies `criterion`."""
    if criterion == "f1":
        return max(rows, key=lambda r: r["f1"])
    if criterion == "fbeta":
        # F_β = (1+β²)·prec·rec / (β²·prec + rec). β > 1 weights recall higher.
        b2 = beta ** 2
        def fbeta(r):
            num = (1 + b2) * r["prec"] * r["rec"]
            den = b2 * r["prec"] + r["rec"]
            return num / max(den, 1e-9)
        return max(rows, key=fbeta)
    if criterion == "min-precision":
        eligible = [r for r in rows if r["prec"] >= min_precision]
        # If nothing meets the floor, fall back to the highest-precision row so
        # we still save *something* — but flag it in the chosen row.
        if not eligible:
            best = max(rows, key=lambda r: r["prec"])
            best = {**best, "fallback": True}
            return best
        return max(eligible, key=lambda r: r["rec"])
    if criterion == "min-recall":
        eligible = [r for r in rows if r["rec"] >= min_recall]
        if not eligible:
            best = max(rows, key=lambda r: r["rec"])
            best = {**best, "fallback": True}
            return best
        return max(eligible, key=lambda r: r["prec"])
    raise ValueError(f"Unknown threshold criterion: {criterion}")


def _format_metrics(label, avg_loss, tp, fp, fn, n_pos_true, n_pos_pred, n_total):
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    pos_pct = 100.0 * n_pos_true / max(n_total, 1)
    pred_pct = 100.0 * n_pos_pred / max(n_total, 1)
    return (
        f"  {label}: avg_loss={avg_loss:.4f}  "
        f"pos {n_pos_true}/{n_total} ({pos_pct:.4f}%)  "
        f"pred_pos {n_pos_pred}/{n_total} ({pred_pct:.4f}%)  "
        f"prec@0.5 {prec:.3f}  rec@0.5 {rec:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Optional held-out CSV for per-epoch validation. "
             "Should cover a different time window than the training CSV.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--threshold-criterion",
        choices=["f1", "fbeta", "min-precision", "min-recall"],
        default="fbeta",
        help="How to pick the saved engine threshold from the post-train sweep. "
             "Default 'fbeta' with --beta=2 biases toward recall (correct for "
             "an arbitrage engine where missed shocks cost more than false fires).",
    )
    parser.add_argument("--beta", type=float, default=2.0,
                        help="β for --threshold-criterion=fbeta. β>1 weights recall higher.")
    parser.add_argument("--min-precision", type=float, default=0.7,
                        help="Used when --threshold-criterion=min-precision.")
    parser.add_argument("--min-recall", type=float, default=0.7,
                        help="Used when --threshold-criterion=min-recall.")
    parser.add_argument(
        "--tune-only", action="store_true",
        help="Skip training; load existing weights and just run the threshold sweep. "
             "Lets you re-pick the operating point without retraining.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Neural Network on: {device}")

    dataset = StreamingTCNDataset(config.FEATURE_DUMP_PATH, seq_len=60)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )

    val_loader = None
    if args.val_csv:
        val_dataset = StreamingTCNDataset(args.val_csv, seq_len=60)
        # num_workers=0 on val: IterableDataset doesn't shard by worker_info,
        # so num_workers>0 would yield each sample twice and inflate counts.
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=0,
        )
        print(f"Validation CSV: {args.val_csv}")
    else:
        print("No --val-csv provided; reporting train-set metrics only.")

    model = TCNSpikePredictor().to(device)

    # pos_weight=10 nudges gradients toward catching shocks even though the
    # base class balance is closer to 4% positive on the curated CSV. Dial
    # down to ~3 if precision matters more than recall.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))

    scaler = torch.amp.GradScaler('cuda')

    weights_path = Path(config.TCN_WEIGHTS_PATH)

    if args.tune_only:
        if not weights_path.exists():
            raise SystemExit(
                f"--tune-only requires existing weights at {weights_path}. "
                "Run train_stream.py without --tune-only first."
            )
        print(f"--tune-only: loading weights from {weights_path}, skipping training.")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        epochs = 0
    else:
        epochs = args.epochs
        print("Starting Streaming Training Loop...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches_processed = 0
        # Running TP/FP/FN over training-batch predictions for the epoch.
        # Not held-out validation — but reveals class collapse (rec=0 means the
        # model is just predicting "no shock" everywhere, regardless of low loss).
        tp = fp = fn = 0
        n_pos_pred = 0
        n_pos_true = 0
        n_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Re-do the forward without TCNSpikePredictor.forward()'s final
            # sigmoid so BCEWithLogitsLoss receives raw logits — feeding it
            # already-sigmoided values produces a double sigmoid that
            # collapses gradients (loss freezes at ln 2 ≈ 0.6931).
            # bfloat16 autocast for throughput; safe because bf16 has fp32's
            # exponent range so GradScaler is a no-op here.
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h = model.input_proj(X_batch)
                for blk in model.blocks:
                    h = blk(h)
                logits = model.head(h[:, :, -1]).squeeze(-1)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batches_processed += 1

            with torch.no_grad():
                preds = torch.sigmoid(logits.float()) >= 0.5
                y_bool = y_batch >= 0.5
                tp += int((preds & y_bool).sum().item())
                fp += int((preds & ~y_bool).sum().item())
                fn += int((~preds & y_bool).sum().item())
                n_pos_pred += int(preds.sum().item())
                n_pos_true += int(y_bool.sum().item())
                n_total += y_bool.numel()

            if batches_processed % 25 == 0:
                print(f"  Epoch {epoch+1} | Batch {batches_processed} | Current Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, batches_processed)
        print(f"--- End of Epoch {epoch+1} ---")
        print(_format_metrics(
            "train", avg_loss, tp, fp, fn, n_pos_true, n_pos_pred, n_total,
        ))

        if val_loader is not None:
            v = evaluate(model, val_loader, criterion, device)
            print(_format_metrics(
                "val  ", v["avg_loss"], v["tp"], v["fp"], v["fn"],
                v["n_pos_true"], v["n_pos_pred"], v["n_total"],
            ))
        
    if not args.tune_only:
        torch.save(model.state_dict(), weights_path)
        print(f"\nSaved neural network weights to {weights_path}")

    # Pick the engine operating point from data, not the placeholder
    # config.TURBULENCE_THRESHOLD. Use val data if provided; otherwise
    # re-iterate the train CSV with a single-worker loader to dodge the
    # IterableDataset double-yield that num_workers=2 introduces.
    if val_loader is not None:
        sweep_loader = val_loader
        sweep_source = "val"
        sweep_path = args.val_csv
    else:
        sweep_dataset = StreamingTCNDataset(config.FEATURE_DUMP_PATH, seq_len=60)
        sweep_loader = DataLoader(
            sweep_dataset, batch_size=args.batch_size,
            pin_memory=True, num_workers=0,
        )
        sweep_source = "train"
        sweep_path = config.FEATURE_DUMP_PATH

    print(f"\n=== Threshold sweep ({sweep_source}: {sweep_path}) ===")
    probs, labels = collect_predictions(model, sweep_loader, device)
    rows = threshold_sweep(probs, labels)

    print(f"{'thr':>6} {'pred_pos':>10} {'tp':>7} {'fp':>7} {'fn':>7} "
          f"{'prec':>6} {'rec':>6} {'f1':>6}")
    for r in rows:
        print(f"{r['thr']:>6.2f} {r['pred_pos']:>10} {r['tp']:>7} {r['fp']:>7} "
              f"{r['fn']:>7} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}")

    chosen = pick_threshold(
        rows,
        criterion=args.threshold_criterion,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        beta=args.beta,
    )
    fallback_note = " [FALLBACK: floor not reachable]" if chosen.get("fallback") else ""
    criterion_label = (
        f"{args.threshold_criterion} (beta={args.beta})"
        if args.threshold_criterion == "fbeta"
        else args.threshold_criterion
    )
    print(
        f"\nChosen ({criterion_label}{fallback_note}): "
        f"thr={chosen['thr']:.3f}  prec={chosen['prec']:.3f}  "
        f"rec={chosen['rec']:.3f}  f1={chosen['f1']:.3f}"
    )

    # When sweeping on val data, route saves to a separate stem so we don't
    # clobber the prod threshold (the train-time-tuned operating point the
    # engine actually consumes for the current SYMBOL). Lets multiple eval
    # runs — e.g. cross-symbol PLTR, temporal hold-out — coexist on disk.
    prod_thresh_path = Path(config.TCN_THRESHOLD_PATH)
    if sweep_source == "val":
        val_stem = Path(args.val_csv).stem
        thresh_path = prod_thresh_path.parent / f"tcn_threshold_eval_{val_stem}.json"
        sweep_csv_path = prod_thresh_path.parent / f"tcn_threshold_sweep_eval_{val_stem}.csv"
    else:
        thresh_path = prod_thresh_path
        sweep_csv_path = prod_thresh_path.parent / f"tcn_threshold_sweep_{config.SYMBOL_SLUG}.csv"

    with open(thresh_path, "w") as f:
        json.dump({
            "threshold": chosen["thr"],
            "precision": chosen["prec"],
            "recall": chosen["rec"],
            "f1": chosen["f1"],
            "criterion": args.threshold_criterion,
            "beta": args.beta if args.threshold_criterion == "fbeta" else None,
            "sweep_source": sweep_source,
            "sweep_csv_path": str(sweep_path),
            "sweep_n_predictions": int(len(probs)),
            "sweep_n_positives": int((labels >= 0.5).sum()),
            "fallback": bool(chosen.get("fallback", False)),
        }, f, indent=2)
    print(f"Saved threshold configuration to {thresh_path}")
    with open(sweep_csv_path, "w") as f:
        f.write("threshold,pred_pos,tp,fp,fn,precision,recall,f1\n")
        for r in rows:
            f.write(
                f"{r['thr']:.4f},{r['pred_pos']},{r['tp']},{r['fp']},{r['fn']},"
                f"{r['prec']:.6f},{r['rec']:.6f},{r['f1']:.6f}\n"
            )
    print(f"Saved threshold sweep table to {sweep_csv_path}")

if __name__ == "__main__":
    main()