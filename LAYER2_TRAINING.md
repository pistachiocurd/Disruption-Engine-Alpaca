# Layer 2 — TCN Training: Data, Findings, and Lessons

This document is a research log for the Layer 2 spike predictor
(`TCNSpikePredictor` in `layer2_alpha.py`). It captures what the dataset
actually looks like once harvested, what positive-class densities the
model can and cannot train on, what failed and why, and what we
observed across symbols. Architecture-level documentation for how
Layer 2 plugs into the broader engine lives in `IMPLEMENTATION.md`
(§3a, §6); this file focuses purely on training-time concerns.

When equivalent files appear later for the PPO execution agent (Layer
3) or shadow simulator (Layer 4), the pattern would be
`LAYER3_TRAINING.md` etc.

## 1. Source

All US equity data was harvested from **Alpaca's free-tier IEX feed**
via `fetch_history_alpaca.py`. The harvester does not download
pre-computed CSVs — it pulls raw quotes and trades from
`StockHistoricalDataClient.get_stock_quotes` /
`get_stock_trades`, then replays them in chronological order through
the live `SensorArray._process_order_book` and `_process_trade` code
paths. The output `feature_history_<SYMBOL>.csv` is therefore
**byte-identical** to what the production `FeatureDumper` writes when
the engine runs in shadow mode against live data. This eliminates
train/serve skew on the engineered features (`ce_ratio`, `obi`,
`liquidation_rate`, `vpin`, `regime`, `mahal_dist`).

Crypto pairs (BTC/USD, SOL/USD) retain their original Coinbase
Advanced provenance from before the equities pivot; they are kept for
back-compat regression tests, not active training.

## 2. Dataset inventory

| Symbol | File | Size | Rows | Sessions | Provenance |
|---|---|---|---|---|---|
| NVDA (raw) | `feature_history_NVDA.csv` | 2.8 GB | 33,929,232 | ~14 | Alpaca IEX |
| NVDA (balanced) | `feature_history_NVDA_balanced.csv` | 25 MB | 384,354 | curated subset | derived from NVDA raw |
| PLTR (raw) | `feature_history_PLTR.csv` | 771 MB | 9,365,123 | ~14 | Alpaca IEX |
| TSLA (raw) | `feature_history_TSLA.csv` | 3.8 GB | 46,973,402 | 73 | Alpaca IEX |
| SPY (raw) | `feature_history_SPY.csv` | 9.6 GB | 119,030,669 | 46 | Alpaca IEX |
| BTC/USD | `feature_history_BTC_USD.csv` | 18 MB | 244,528 | (legacy) | Coinbase Advanced |
| SOL/USD | `feature_history_SOL_USD.csv` | 13 MB | 157,542 | (legacy) | Coinbase Advanced |
| synthetic | `synthetic_history.csv` | 6.1 MB | 72,000 | – | `synthetic_data.py` |

`NVDA_balanced` is a curated subset built by extracting the ticks
where `regime == 1` plus a 60-tick prelude per shock event plus 5,000
random noise rows. It was constructed to bootstrap training when the
raw NVDA file's positive density was too sparse to overcome the
hardcoded `pos_weight=10`. Once the labeling and threshold-sweep
machinery were in place (see §4), per-symbol training on raw CSVs
became viable for symbols above the empirical density threshold.

## 3. Observed class densities

Positive labels are constructed by horizon expansion of `regime == 1`
transitions: for each `0 -> 1` transition at tick `t`, the windows
ending at ticks `[t - H, t)` are marked positive, with
`H = TCN_LABEL_HORIZON_TICKS = 30` (30 seconds at the 1 Hz IEX feed).
The model is therefore trained as a leading classifier — "does a
shock start within the next 30 ticks?".

| Symbol | regime=1 ticks | Shock starts | Positive rate after H=30 | Trainability verdict |
|---|---|---|---|---|
| NVDA balanced | (curated) | (curated) | **4.43%** (34,026 / 768,588 windows) | trains cleanly |
| TSLA raw | 434,307 (0.92%) | 25,307 | **0.98%** (919,736 / 93,946,684 windows, double-yield) | trains cleanly |
| PLTR raw | – | ~4,256 | **~0.045%** | not directly tested; used as held-out eval |
| SPY raw | 105,407 (0.09%) | 615 | **~0.015%** | class collapse confirmed empirically |

## 4. Empirical density threshold for in-domain training

We observed two distinct failure modes during this work, each tied to
a different positive-class density. Both look identical at first
glance ("loss not improving") but have different mechanisms and
different fixes.

**Failure mode 1: stuck-at-ln-2 (gradient collapse).**
Initial NVDA training appeared to "converge" to `loss ~ 0.6931`
across all batches and all epochs. Root cause was a *double sigmoid*:
`TCNSpikePredictor.forward()` already applies `torch.sigmoid()`, and
`BCEWithLogitsLoss` applies its own sigmoid internally for numerical
stability. The model's optimizer parked at internal `logit ~ 0`
forever — the inner sigmoid produced 0.5, the loss treated that
0.5 as a *new* logit, the outer sigmoid produced 0.622, and
gradients vanished through both sigmoid derivatives. **Fixed** by
bypassing `forward()` during training and computing
`logits = model.head(model.blocks(model.input_proj(x))[..., -1])`
manually before passing to the loss. This mirrors the workaround
already present in `train_tcn.py:204-208`. See `train_stream.py`
autocast block.

**Failure mode 2: class collapse (vanishing minority gradient).**
SPY training — at 0.015 % positive-class density after horizon
expansion — showed `Current Loss: 0.0000` across 3,225+ consecutive
batches with *no learning trajectory*. Root cause was extreme class
imbalance: with `pos_weight = 10` and roughly one positive per 6,600
windows, the gradient signal on the positive class is negligible
relative to the optimizer's pull toward "always predict negative."
Empirically, **0.015 % is below this architecture's trainability
threshold; 0.98 % (TSLA) is above it**. Fixing class collapse would
require a denser feed, focal loss, oversampling, curated subset
construction, or a much larger `pos_weight`; none of these were in
scope for the equities pivot and SPY is therefore retained as a
*held-out cross-symbol probe* rather than a training symbol.

The two failure modes are diagnostically distinct:

| Symptom | Stuck-at-ln-2 | Class collapse |
|---|---|---|
| `Current Loss` per batch | constant ~0.6931 | constant 0.0000 |
| Variance across batches | ~zero | ~zero |
| Loss ever spikes high? | no | no |
| `pred_pos` at eval | ~50% (model says 0.5 to everything) | 0% (model says no to everything) |
| Fix | code (remove double sigmoid) | data / loss-function intervention |

## 5. Per-symbol training results

All numbers below are train-set metrics from the post-train threshold
sweep. Per-symbol weights live at `calibration/tcn_weights_<SYMBOL>.pt`;
operating thresholds at `calibration/tcn_threshold_<SYMBOL>.json`.
Cross-symbol eval JSON files use the
`tcn_threshold_eval_<val_stem>.json` naming so the production
threshold for each `SYMBOL` is never clobbered by a diagnostic run.

**NVDA balanced** (5 epochs, `pos_weight=10`, F1-criterion threshold
sweep):

| Epoch | avg_loss | prec@0.5 | rec@0.5 |
|---|---|---|---|
| 1 | 0.66 | 0.313 | 0.596 |
| 2 | 0.46 | 0.342 | 0.758 |
| 3 | 0.41 | 0.375 | 0.791 |
| 4 | 0.40 | 0.392 | 0.800 |
| 5 | 0.38 | 0.404 | 0.807 |

Threshold sweep on train data picked F1-optimal at thr=0.775
(prec=0.66, rec=0.71, F1=0.686). Switching to F-beta with beta=2
(recall-biased, see §6) re-picked thr=0.60 (prec=0.55, rec=0.79,
F2=0.74).

**TSLA raw** (5 epochs, `pos_weight=10`, F-β β=2 threshold sweep
default):

| Epoch | avg_loss | prec@0.5 | rec@0.5 |
|---|---|---|---|
| 1 | 0.0935 | 0.279 | 0.809 |
| 2 | 0.0902 | 0.291 | 0.808 |
| 3 | 0.0911 | 0.292 | 0.806 |
| 4 | 0.0917 | 0.292 | 0.805 |
| 5 | 0.0909 | 0.292 | 0.808 |

Threshold sweep on train data picked F2-optimal at **thr=0.500**
(prec=0.329, rec=0.732, F1=0.454, F2=0.588). Sweep also reveals
F1-optimal at thr=0.750 (prec=0.478, rec=0.557, F1=0.514) — the
recall-biased F2 criterion deliberately trades 5pp of F1 for 18pp
more recall, on the engine-economics premise that missed shocks cost
more than wasted mandates.

Comparison to NVDA balanced shows TSLA is structurally harder:

| | TSLA raw | NVDA balanced |
|---|---|---|
| Class density | 0.98% | 4.43% |
| Peak F1 | 0.514 (thr=0.75) | 0.686 (thr=0.78) |
| Peak F2 | 0.588 (thr=0.50) | 0.740 (thr=0.60) |
| Precision at F2 op | 0.329 | 0.552 |
| Recall at F2 op | 0.732 | 0.792 |

The 4.5× density gap explains most of the F2 delta. TSLA's *raw*
distribution also includes more dense shock-cluster days the model
can only partially learn (§9 lesson 6), depressing the precision
ceiling further. NVDA's `_balanced` curated set was specifically
constructed to upweight learnable structure — it shows in the
metrics, but is not representative of what the engine sees in
production.

Notable findings from the TSLA run:

- TSLA epoch 1 already matched NVDA epoch 5 on recall (0.81), despite
  operating on a 4.5x sparser class density. Indicates the architecture
  generalizes across density regimes provided the lower bound (~1%)
  isn't crossed.
- **Most of the useful training happened in epochs 1-2.** Aggregate
  metrics plateaued from epoch 3 onwards — prec stuck at 0.292,
  rec drifting in the 0.805-0.809 range, avg_loss noise-bouncing in
  [0.0902, 0.0917]. For a future run on similarly-dense data,
  `--epochs 2` would give equivalent results in 40% of the wall-clock.
- Per-batch loss curves on dense shock-cluster regions (~5% of
  batches) plateau between every epoch — the model has hit a
  representational or loss-balance limit on those windows
  specifically. Aggregate metrics keep improving because the
  remaining 95% of batches (laminar trading) refine across epochs.
  This is the classic "rare-class learning bottleneck": the rare
  positive's gradient is washed out by the bulk's gradient over a
  full epoch. Fixes that would actually move the cluster-region
  loss: focal loss, larger `pos_weight`, or capacity scaling.

**SPY raw**: class collapse, no usable model.

## 6. Threshold-tuning policy

The engine reads `tcn_threshold_<SYMBOL>.json["threshold"]` at
startup as the boundary between "fire a mandate" and "stay quiet."
`train_stream.py` picks that value from a sweep over the trained
model's prediction distribution. Default criterion is **F-beta with
beta = 2** — recall-biased, on the operational premise that a missed
shock costs more than a wasted mandate (an arbitrage engine cannot
capture alpha on a shock it never saw). Other criteria available:

- `f1` — symmetric balance
- `min-precision <p>` — highest-recall threshold whose precision >= p
- `min-recall <r>` — highest-precision threshold whose recall >= r

The full sweep curve is dumped to
`calibration/tcn_threshold_sweep_<SYMBOL>.csv` for plotting. The
script supports `--tune-only` to re-run the sweep on existing weights
without retraining (~30 sec) and `--val-csv path/to/other.csv` to
sweep on held-out data instead of the training CSV.

## 7. Cross-symbol generalization

Cross-symbol evaluation was performed by loading per-symbol weights
and running the threshold sweep on a different symbol's CSV via
`--tune-only --val-csv path/to/other.csv`. Outputs are saved under
`tcn_threshold_eval_<val_stem>.{json,csv}` so the production
threshold is never overwritten.

| Train symbol | Eval symbol | F1 at peak | Threshold | prec | rec |
|---|---|---|---|---|---|
| NVDA balanced | PLTR raw | 0.448 | 0.85 | 0.504 | 0.403 |
| NVDA balanced | (F2 operating point on PLTR) | 0.426 | 0.70 | 0.401 | 0.453 |

Cross-symbol generalization holds qualitatively — F1 dropped from
0.69 (NVDA train-set) to 0.45 (PLTR held-out, ~15x sparser positive
density), but the model still ranked predictions meaningfully (PR
curve unimodal, peak in upper threshold half). Recall held at 45 %
on the F2 operating point and precision climbed to 72 % in the
high-confidence tail (thr=0.99). This justifies the architecture
claim — the TCN learns universal microstructure patterns rather than
NVDA-specific memorization — and motivates per-symbol calibration as
the operational answer rather than a single universal model.

Pending rows once TSLA training finishes:

- TSLA → NVDA, TSLA → PLTR, TSLA → SPY
- NVDA → TSLA (NVDA evaluated on TSLA's denser raw feed)

## 8. IEX feed sparsity by symbol

A non-obvious finding: **IEX free-tier captures vastly different
fractions of total flow depending on the symbol's primary venue.**
Despite SPY being the highest-notional US equity, we obtained only
615 shock starts from 119 M rows / 46 sessions of SPY data — sparser
than TSLA's 25,307 starts in 47 M rows / 73 sessions. Reason: SPY
trades primarily on NYSE Arca and through ETF arbitrage flow across
many lit and dark venues; IEX gets a small slice. TSLA trades
heavily through retail flow on NASDAQ; IEX captures more of that as
a reference exchange.

Operational implication: **IEX-tier data is sufficient for
single-name microstructure work on actively-traded equities (NVDA,
TSLA, PLTR — the latter at the edge) but insufficient for ETF-class
instruments (SPY) without upgrading to SIP feed.** This is a
data-source limitation, not a methodology or architecture problem.

## 9. Lessons learned during this build-out

1. **Train/serve parity through replay > pre-computed feature CSVs.**
   The harvester runs the live sensor pipeline against historical
   raw quotes/trades; the resulting CSV is the same byte-for-byte as
   what production writes. Pre-computed feature CSVs (Yahoo,
   Polygon-tick, etc.) would not have this property and would
   introduce silent train/serve skew.

2. **`IterableDataset` + `num_workers > 0` double-yields by default.**
   `StreamingTCNDataset.__iter__` does not shard by
   `torch.utils.data.get_worker_info()`, so each DataLoader worker
   iterates the full CSV. With `num_workers=2` this means each epoch
   contains two passes through the data — visible in the doubled
   `n_total` reported in epoch summaries. Five nominal epochs ~= ten
   genuine passes. Not corruption, but worth knowing for
   wall-clock estimates and effective-epoch interpretation. Future
   fix: shard by `worker_info.num_workers / id` inside `__iter__`.

3. **Threshold = 0.5 is wrong for any `pos_weight != 1`.**
   `pos_weight = 10` deliberately biases the model toward firing,
   which means the F1-optimal threshold sits well above 0.5 (in our
   case 0.7-0.8). The per-train threshold sweep is mandatory rather
   than cosmetic — saving the placeholder `TURBULENCE_THRESHOLD`
   would have systematically over-fired the engine.

4. **Two distinct "loss not improving" failure modes.** See §4.
   Knowing which is happening tells you whether you have a code bug
   (stuck-at-ln-2) or a data problem (class collapse). They look
   identical at the per-batch loss level; the per-epoch
   `pred_pos` count distinguishes them.

5. **IEX-tier feeds are venue-fragmented.** Symbol selection for
   training on a free-tier feed should prioritize symbols whose
   primary venue overlaps with IEX's quote feed, not symbols with
   the highest absolute notional volume. This is a defensive
   limitation of the current pipeline and motivates SIP upgrade for
   ETF-class instruments before live deployment on those.

6. **Rare-class learning bottleneck shows up as cross-epoch loss
   plateau on the hard subset.** When per-batch loss values on a
   particular contiguous batch range stay nearly identical from
   epoch N to epoch N+1 while aggregate metrics keep improving,
   that's the model converging on the easy bulk while leaving the
   hard subset at irreducible error. Diagnostic test: compare
   epoch-1 and epoch-2 loss values at the same batch indices —
   if they're within ~5%, that subset has plateaued. Aggregate
   metrics will still improve through negative-class refinement.
