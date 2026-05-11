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
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import config
from layer2_alpha import MambaSpikePredictor, TCNSpikePredictor, TransformerSpikePredictor
from train_tcn import build_labels

# LibAUC provides AUCMLoss + PESG optimizer for direct AUC optimization
# under extreme class imbalance — the recommended remediation per Gemini
# Deep Research for the HL TCN F2-stuck-at-0.046 failure mode (see
# ~/.claude/plans/tcn-failure-investigation.md, P0 in TODO.md, §13.3 of
# LAYER2_TRAINING.md). Optional dependency: only required when --loss=aucm.
try:
    from libauc.losses import AUCMLoss
    from libauc.optimizers import PESG
    _HAS_LIBAUC = True
except ImportError:
    _HAS_LIBAUC = False


class MultiCsvTCNDataset(IterableDataset):
    """Concatenates multiple StreamingTCNDataset instances for multi-coin
    training. Each per-coin CSV is processed independently — its own
    sliding-window labels, its own carry-over across chunk boundaries —
    so coin transitions don't introduce a cross-symbol price discontinuity
    in the windowed features. Yields (X, y) pairs in coin-by-coin order.

    Use multi-coin training only when the per-coin shock counts are
    individually thin. The implicit assumption is that pre-shock
    microstructure signatures are coin-agnostic enough that aggregating
    across coins gives the model more positives without adding noise.
    The cross-symbol matrix in §7 of LAYER2_TRAINING.md (TSLA->NVDA F1
    near in-domain) is the empirical justification for that assumption
    — but it was measured on equity-IEX data; verify on crypto perps.
    """

    def __init__(self, csv_paths: list, seq_len: int = 60,
                 label_source: str = "regime", label_horizon=None,
                 pretext: bool = False):
        self.datasets = [
            StreamingTCNDataset(p, seq_len=seq_len,
                                label_source=label_source,
                                label_horizon=label_horizon,
                                pretext=pretext)
            for p in csv_paths
        ]

    def __iter__(self):
        for ds in self.datasets:
            yield from ds


class ShuffledBufferDataset(IterableDataset):
    """Reservoir-style shuffle buffer wrapping another IterableDataset.

    REQUIRED for AUCM training (--loss=aucm). The base streaming pipeline
    (StreamingTCNDataset / MultiCsvTCNDataset) yields windows in temporal
    order. Shock-derived labels cluster (H ticks before each shock = H
    consecutive positives), so most batches contain zero positives — and
    AUCMLoss returns 0 with a UserWarning when a batch has no positives.
    The model then gets useful gradient on <5%% of batches and never
    escapes the trivial 'predict near zero' minimum, reproducing the
    §13.3 F2=0.046 noise floor even with the correct loss.

    With buffer_size=500K and 2.35%% positive density, each 2048-batch
    draws ~48 positives ± 7 (σ from binomial), so the probability of a
    zero-positive batch is astronomically small. AUCM's pairwise margin
    gets a full gradient signal on every batch.

    Not needed for BCE/Focal — those losses produce non-zero gradients
    even on all-negative batches (via pos_weight or focal modulation).
    """

    def __init__(self, base: IterableDataset, buffer_size: int = 500_000, seed: int = 42):
        self.base = base
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        buffer = []
        for sample in self.base:
            if len(buffer) < self.buffer_size:
                buffer.append(sample)
            else:
                idx = int(rng.integers(0, self.buffer_size))
                yield buffer[idx]
                buffer[idx] = sample
        rng.shuffle(buffer)
        yield from buffer


class FocalLossWithLogits(nn.Module):
    """Binary focal loss with logits. Lin et al. 2017 (RetinaNet paper).

        FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    where p_t = p if y=1 else (1-p), and α_t = α if y=1 else (1-α).

    γ (focusing): down-weights easy examples. γ=0 reduces to weighted BCE.
        γ=2 is the canonical RetinaNet value.
    α (balancing): weights positive class. α=0.25 weights negatives 3:1
        (counterintuitive but counters the modulating factor's bias toward
        the rare class). For sparse positives in market microstructure,
        α=0.5–0.75 often works better — tune empirically.

    Replaces BCEWithLogitsLoss(pos_weight=...) when the rare-class learning
    bottleneck (§5 lesson 6 of LAYER2_TRAINING.md) makes pos_weight scaling
    unstable — losses spike on positive-heavy batches and gradients flip
    between "predict everything" and "predict nothing". Focal loss is more
    stable because it scales gradients by per-sample confidence rather
    than uniformly per-class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = sigmoid(logit) for positives, (1 - sigmoid(logit)) for negatives
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma
        loss = focal_factor * ce
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


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

    def __init__(self, csv_path, seq_len=60, label_source="regime", label_horizon=None,
                 pretext=False):
        """label_source: 'regime' (default — uses regime==1 transitions) or
        'shock' (uses identify_shock_events from calibration.py — price+VPIN
        based, doesn't depend on HMM calibration). Use 'shock' on harvests
        where the HMM is cold-start; use 'regime' once HMM emissions are
        fitted offline.

        label_horizon: number of ticks before each shock to mark positive.
        None (default) = use config.TCN_LABEL_HORIZON_TICKS. Override when
        tests/test_features.py shows signal concentrated in fewer ticks.

        pretext: Path E / P3.7 — if True, the dataset emits next-tick
        log-return (in basis points, scaled by 1e4) as the target instead
        of binary shock labels. Used by `--pretext` in train_stream.py to
        pretrain the encoder on a dense self-supervised target (~6.5M
        supervisory steps per epoch, vs ~181K shock labels). The
        label_source/label_horizon args are ignored in pretext mode.
        """
        self.csv_path = csv_path
        self.seq_len = seq_len
        if label_source not in ("regime", "shock", "directional"):
            raise ValueError(
                f"label_source must be 'regime', 'shock', or 'directional', got {label_source!r}"
            )
        self.label_source = label_source
        self.label_horizon = label_horizon
        self.pretext = pretext

    def __iter__(self):
        reader = pl.read_csv_batched(self.csv_path)

        carry_over_features = None
        carry_over_labels = None

        batches = reader.next_batches(1)
        while batches:
            df_chunk = batches[0]
            data_chunk = {col: df_chunk[col].to_numpy() for col in df_chunk.columns}

            n = len(df_chunk)
            labels = np.zeros(n, dtype=np.float32)

            if self.pretext:
                # Path E / P3.7 — emit next-tick log-return in bps as the
                # target. labels[i] = log(mid[i+1] / mid[i]) * 1e4. The
                # last position has no future tick in this chunk; left
                # at 0 (statistical noise — millions of valid samples
                # per chunk swamp it).
                bid = data_chunk["best_bid"].astype(np.float64)
                ask = data_chunk["best_ask"].astype(np.float64)
                mid = 0.5 * (bid + ask)
                mid_safe = np.maximum(mid, 1e-9)
                if n > 1:
                    log_ret = np.log(mid_safe[1:] / mid_safe[:-1]) * 10_000.0
                    labels[:-1] = log_ret.astype(np.float32)
            else:
                # Leading classifier: mark the H ticks BEFORE each shock as
                # positive so the model fires on the pre-shock signature, not
                # on the shock itself when it's already too late to act.
                if self.label_horizon is not None:
                    horizon = self.label_horizon
                elif self.label_source == "directional":
                    horizon = config.TCN_DIRECTIONAL_HORIZON_TICKS
                else:
                    horizon = config.TCN_LABEL_HORIZON_TICKS

                if self.label_source == "directional":
                    # Pivot 2 Option C (§14): label[t] = 1 iff mid[t+H] > mid[t].
                    # Balanced ~50/50 target; tests whether L2 features carry
                    # ANY leading info about price direction over a tradeable
                    # horizon. BCE loss expected (AUCM is overkill for a
                    # balanced target).
                    mid = (data_chunk["best_bid"].astype(np.float64)
                           + data_chunk["best_ask"].astype(np.float64)) / 2.0
                    if n > horizon:
                        labels[:n - horizon] = (mid[horizon:] > mid[:-horizon]).astype(np.float32)
                    # Last `horizon` ticks: no future available — leave as 0.
                    # Negligible bias for horizon << n (e.g., 100 / 1.7M = 6e-5).
                else:
                    use_regime = (self.label_source == "regime") and ("regime" in data_chunk)
                    if use_regime:
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
            #
            # Path D (P3.6 / §13.4) — Hyperliquid uses the expanded
            # 5-channel crypto-native feature set; everything else stays
            # on the original 3 channels. Feature scaling factors below
            # are mirrored in AlphaEngine._push_features so trained
            # weights stay consistent at inference.
            if config.EXCHANGE_ID == "hyperliquid":
                base_pathD = [
                    data_chunk["ce_ratio"] / 10,
                    data_chunk["obi"],
                    data_chunk["mlofi"],
                    data_chunk["vamp"] / 10.0,           # bps / 10 → ~[-5, 5]
                    data_chunk["kyles_lambda"] * 100.0,  # dimensionless × 100
                ]
                if config.USE_PATH_G_FEATURES:
                    # Path G π-groups are already tanh(-1, +1) — no scaling
                    # needed to match the Path D channels' magnitude.
                    base_pathD.extend([
                        data_chunk["fo_market"],
                        data_chunk["sr"],
                        data_chunk["pi_kappa"],
                        data_chunk["pi_vamp_dim"],
                    ])
                features = np.stack(base_pathD, axis=1).astype(np.float32)
            else:
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


def evaluate(model, loader, criterion, device, loss_name="bce"):
    """Single eval pass over `loader`. Returns dict of loss + classification counts.

    loss_name: 'bce' / 'focal' / 'aucm'. AUCM expects probabilities, not
    logits — same convention as the training loop.
    """
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
                logits = model.forward_logits(X_batch)
                if loss_name == "aucm":
                    loss = criterion(torch.sigmoid(logits.float()), y_batch)
                else:
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
                logits = model.forward_logits(X_batch)
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
        "--csv",
        default=None,
        help=f"Training CSV path(s). Single path or comma-separated list "
             f"for multi-coin training (each coin's CSV is iterated "
             f"independently — see MultiCsvTCNDataset). "
             f"Default: config.FEATURE_DUMP_PATH "
             f"({config.FEATURE_DUMP_PATH})",
    )
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
    parser.add_argument(
        "--label-source",
        choices=["regime", "shock", "directional"],
        default="regime",
        help="Source of positive labels. 'regime' uses regime==1 transitions "
             "(default; matches docs; requires fitted HMM emissions). 'shock' "
             "uses identify_shock_events (price+VPIN; HMM-independent — use "
             "this when the HMM is cold-start, e.g. on a freshly harvested "
             "venue without offline calibration). 'directional' labels "
             "label[t] = 1 iff mid[t+H] > mid[t] — balanced ~50/50 target "
             "for testing leading feature signal independent of label "
             "engineering (Pivot 2 Option C, §14).",
    )
    parser.add_argument(
        "--bce-pos-weight",
        type=float,
        default=10.0,
        help="Positive-class weight for --loss=bce (default 10.0, tuned for "
             "~1-4%% positive density shock prediction). Set to 1.0 for "
             "balanced targets like --label-source=directional.",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "focal", "aucm"],
        default="bce",
        help="Loss function. 'bce' is BCEWithLogitsLoss(pos_weight=10) "
             "(default; matches existing TSLA training). 'focal' is binary "
             "focal loss (Lin et al. 2017) — use this when training is "
             "unstable from the rare-class learning bottleneck. 'aucm' is "
             "LibAUC's AUC-Margin loss (Yuan et al. 2023) paired with the "
             "PESG optimizer — use this to escape the trivial 'predict near "
             "zero everywhere' minimum that point-wise BCE/Focal fall into "
             "at <1%% positive density (the HL perp shock case; see §13.3 "
             "of LAYER2_TRAINING.md, P0 in TODO.md, and "
             "~/.claude/plans/tcn-failure-investigation.md).",
    )
    parser.add_argument(
        "--aucm-margin",
        type=float,
        default=1.0,
        help="Margin for --loss=aucm. Default 1.0 (LibAUC standard).",
    )
    parser.add_argument(
        "--aucm-lr",
        type=float,
        default=0.1,
        help="Learning rate for PESG optimizer (paired with --loss=aucm). "
             "Default 0.1 — PESG dynamics differ from Adam (whose default "
             "1e-3 is too small for PESG's minimax update rule).",
    )
    parser.add_argument(
        "--aucm-shuffle-buffer",
        type=int,
        default=500_000,
        help="Reservoir-shuffle buffer size when --loss=aucm. Required to "
             "break the streaming temporal order — without it, ~95%% of "
             "batches have zero positives (shock labels cluster H ticks "
             "before each shock) and AUCMLoss returns 0 with a UserWarning, "
             "leaving the model at the noise-floor F2=0.046 baseline. "
             "Default 500K samples (~360 MB) — large enough that every "
             "2048-batch contains positives with overwhelming probability.",
    )
    parser.add_argument(
        "--model",
        choices=["tcn", "transformer", "mamba"],
        default="tcn",
        help="Model architecture. 'tcn' (default) is the Bai-Kolter-Koltun "
             "causal 1D conv stack (TCNSpikePredictor). 'transformer' is a "
             "causal Transformer encoder. 'mamba' is a hand-rolled "
             "pure-PyTorch Mamba (Gu & Dao 2023) selective-scan model with "
             "input-dependent step size — Path C / P3.5's direct test of "
             "the H2 hypothesis that HL's bursty irregular cadence is the "
             "binding constraint after Path A (loss) and Path D "
             "(features). See §13.6 of LAYER2_TRAINING.md.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="γ (focusing) for --loss=focal. γ=0 is weighted BCE; γ=2 is "
             "RetinaNet default. Higher γ = more aggressive down-weighting "
             "of easy examples.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.5,
        help="α (positive-class weight) for --loss=focal. α=0.25 is "
             "RetinaNet default (counterintuitively biased toward "
             "negatives; the modulating factor already favors hard "
             "positives). For sparse market microstructure positives, "
             "α=0.5–0.75 often works better. Tune empirically.",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=None,
        help="Override config.TCN_LABEL_HORIZON_TICKS (default 30). The "
             "H ticks before each shock are marked positive. Shorter H = "
             "tighter signal-to-label match (use when tests/test_features.py "
             "shows signal concentrated in last ~10 ticks). Longer H = "
             "more positive labels but more label-noise from windows "
             "where the actual signature isn't yet visible.",
    )
    parser.add_argument(
        "--pretext",
        action="store_true",
        help="SSL pretext pretraining mode (Path E / P3.7). Trains the "
             "encoder to predict next-tick log-return (in bps) via MSE "
             "instead of the shock binary label. Pairs with a follow-up "
             "invocation using --load-pretrained PATH + --freeze-encoder "
             "+ --loss aucm to fine-tune the classification head on top "
             "of the SSL-pretrained encoder. Skips the post-train "
             "threshold sweep (no classification target). Weights are "
             "written to --pretrain-weights-path (default "
             "./calibration/pretrain_weights_<SYMBOL_SLUG>.pt) so they "
             "don't overwrite production classifier weights.",
    )
    parser.add_argument(
        "--load-pretrained",
        default=None,
        metavar="PATH",
        help="Load model weights from PATH at startup (strict=False, so "
             "the head can be re-initialized for fine-tune). Used in "
             "Path E phase 2: after --pretext produced encoder weights, "
             "load them here and add --freeze-encoder to fine-tune the "
             "classification head on top.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze all parameters except the head (Linear(d_hidden, 1)). "
             "Path E fine-tune: with the encoder frozen, AUCM only adjusts "
             "the head's linear projection — much faster, and protected "
             "from heuristic-label noise corrupting the encoder.",
    )
    parser.add_argument(
        "--pretrain-weights-path",
        default=None,
        metavar="PATH",
        help="Override default save path for --pretext mode. Default: "
             "./calibration/pretrain_weights_<SYMBOL_SLUG>.pt",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Neural Network on: {device}")

    train_csv_arg = args.csv or config.FEATURE_DUMP_PATH
    csv_paths = [p.strip() for p in str(train_csv_arg).split(",") if p.strip()]
    horizon_used = args.label_horizon if args.label_horizon is not None else config.TCN_LABEL_HORIZON_TICKS
    if len(csv_paths) == 1:
        train_csv = csv_paths[0]
        print(f"Training CSV: {train_csv}")
        dataset = StreamingTCNDataset(
            train_csv, seq_len=60,
            label_source=args.label_source, label_horizon=args.label_horizon,
            pretext=args.pretext,
        )
    else:
        train_csv = csv_paths  # list, used for sweep below
        print(f"Training CSVs ({len(csv_paths)}, multi-coin): {csv_paths}")
        dataset = MultiCsvTCNDataset(
            csv_paths, seq_len=60,
            label_source=args.label_source, label_horizon=args.label_horizon,
            pretext=args.pretext,
        )
    if args.pretext:
        print("PRETEXT MODE (Path E): target = next-tick log-return (bps); MSE loss.")
    else:
        print(f"Label source: {args.label_source}, horizon: {horizon_used} ticks")

    if args.loss == "aucm":
        dataset = ShuffledBufferDataset(
            dataset, buffer_size=args.aucm_shuffle_buffer,
        )
        print(
            f"Wrapping training dataset in ShuffledBufferDataset "
            f"(buffer={args.aucm_shuffle_buffer:,}) — AUCM requires positives "
            f"in every batch, which the temporal-order streaming pipeline "
            f"does not provide."
        )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )

    val_loader = None
    if args.val_csv:
        val_dataset = StreamingTCNDataset(
            args.val_csv, seq_len=60,
            label_source=args.label_source, label_horizon=args.label_horizon,
        )
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

    if args.model == "mamba":
        model = MambaSpikePredictor().to(device)
        print(f"Model: MambaSpikePredictor (Path C / P3.5)")
    elif args.model == "transformer":
        model = TransformerSpikePredictor().to(device)
        print(f"Model: TransformerSpikePredictor (Path C / P3.5)")
    else:
        model = TCNSpikePredictor().to(device)
        print(f"Model: TCNSpikePredictor")

    # Path E phase 2: load pretrained encoder weights (strict=False so the
    # head can be re-initialized for the new classification task) and/or
    # freeze the encoder (everything except the head's parameters).
    if args.load_pretrained:
        state = torch.load(args.load_pretrained, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(unexpected)
        print(
            f"Loaded pretrained from {args.load_pretrained}: "
            f"{n_loaded} tensors restored; missing={list(missing)[:5]}"
            f"{'...' if len(missing) > 5 else ''}; "
            f"unexpected={list(unexpected)[:5]}"
            f"{'...' if len(unexpected) > 5 else ''}"
        )
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"Encoder FROZEN. Trainable: {n_trainable:,} / {n_total:,} "
            f"({100.0 * n_trainable / max(n_total, 1):.2f}%)"
        )

    # Loss function: BCE+pos_weight is the historical default; focal is the
    # documented next step (§11.4 of LAYER2_TRAINING.md) when the rare-class
    # learning bottleneck makes BCE unstable. See --loss flag help.
    if args.pretext:
        # Path E pretraining: dense regression on next-tick log-return.
        # Adam at the standard 1e-3 — no PESG dynamics needed for MSE.
        criterion = nn.MSELoss()
        # If --freeze-encoder is set, only the head's parameters need an
        # optimizer; but it's harmless and cleaner to give Adam all
        # params — frozen ones simply have requires_grad=False and the
        # optimizer's step is a no-op for them.
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=1e-3,
        )
        print("Loss: MSE on next-tick log-return (bps); optimizer: Adam (lr=1e-3)")
    elif args.loss == "aucm":
        if not _HAS_LIBAUC:
            raise SystemExit(
                "--loss=aucm requires libauc. Install: pip install libauc>=1.4 "
                "(see requirements.txt and P0 in TODO.md)."
            )
        # AUCM loss + PESG optimizer go together; PESG's minimax inner-loop
        # tracks running statistics of positive/negative score means that
        # the AUCMLoss instance owns. Don't pair AUCM with a standard
        # optimizer — the dual variables won't update.
        criterion = AUCMLoss(margin=args.aucm_margin).to(device)
        optimizer = PESG(
            (p for p in model.parameters() if p.requires_grad),
            loss_fn=criterion,
            lr=args.aucm_lr,
            margin=args.aucm_margin,
            epoch_decay=2e-3,
            weight_decay=1e-4,
        )
        print(
            f"Loss: AUCM via LibAUC (margin={args.aucm_margin}); "
            f"optimizer: PESG (lr={args.aucm_lr})"
        )
    elif args.loss == "focal":
        criterion = FocalLossWithLogits(
            gamma=args.focal_gamma, alpha=args.focal_alpha,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f"Loss: focal (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    else:
        # pos_weight nudges gradients toward catching the rare class. Default
        # 10 is tuned for ~1-4% positive density shock prediction. Override
        # to 1.0 for balanced targets (e.g. --label-source=directional).
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.bce_pos_weight]).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f"Loss: BCE with pos_weight={args.bce_pos_weight}")

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
                logits = model.forward_logits(X_batch)
                if args.pretext:
                    # Path E pretraining — MSE on raw model output as
                    # predicted next-tick log-return (bps). y_batch is a
                    # float tensor of bps; cast logits to fp32 for stable
                    # MSE under bf16 autocast.
                    loss = criterion(logits.float(), y_batch)
                elif args.loss == "aucm":
                    # AUCMLoss expects probabilities in [0, 1], not raw
                    # logits. Use fp32 sigmoid to avoid bf16 precision loss
                    # in the AUCM minimax inner-loop.
                    loss = criterion(torch.sigmoid(logits.float()), y_batch)
                else:
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
            v = evaluate(model, val_loader, criterion, device, loss_name=args.loss)
            print(_format_metrics(
                "val  ", v["avg_loss"], v["tp"], v["fp"], v["fn"],
                v["n_pos_true"], v["n_pos_pred"], v["n_total"],
            ))
        
    if not args.tune_only:
        if args.pretext:
            pretrain_path = Path(
                args.pretrain_weights_path
                or f"./calibration/pretrain_weights_{config.SYMBOL_SLUG}.pt"
            )
            pretrain_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), pretrain_path)
            print(f"\nSaved PRETRAIN weights to {pretrain_path}")
            print("Pretext mode: skipping threshold sweep (no classification target).")
            return
        weights_path.parent.mkdir(parents=True, exist_ok=True)
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
        if isinstance(train_csv, list):
            sweep_dataset = MultiCsvTCNDataset(
                train_csv, seq_len=60,
                label_source=args.label_source, label_horizon=args.label_horizon,
            )
            sweep_path = ",".join(train_csv)
        else:
            sweep_dataset = StreamingTCNDataset(
                train_csv, seq_len=60,
                label_source=args.label_source, label_horizon=args.label_horizon,
            )
            sweep_path = train_csv
        sweep_loader = DataLoader(
            sweep_dataset, batch_size=args.batch_size,
            pin_memory=True, num_workers=0,
        )
        sweep_source = "train"

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

    thresh_path.parent.mkdir(parents=True, exist_ok=True)
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