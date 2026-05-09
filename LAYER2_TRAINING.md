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
`--tune-only --val-csv path/to/other.csv`. Outputs save under
`tcn_threshold_eval_<val_stem>.{json,csv}` so the production
threshold is never overwritten.

Note one footgun in this naming scheme: the eval-stem path is keyed
on the val CSV only, not the trained SYMBOL, so two different
trained models evaluating the same val CSV will overwrite each
other on disk. The numbers are still captured in this table, but
re-running an earlier eval is the only way to recover its sweep CSV.

| Train | Eval | Criterion | thr | prec | rec | F1 |
|---|---|---|---:|---:|---:|---:|
| NVDA bal. | PLTR raw | F1-max | 0.85 | 0.504 | 0.403 | 0.448 |
| NVDA bal. | PLTR raw | F2-op  | 0.70 | 0.401 | 0.453 | 0.426 |
| TSLA raw  | NVDA bal. | F1-max | 0.75 | 0.680 | 0.702 | 0.691 |
| TSLA raw  | NVDA bal. | F2-op  | 0.70 | 0.596 | 0.765 | 0.670 |
| TSLA raw  | PLTR raw | F1-max | 0.78 | 0.513 | 0.422 | 0.463 |
| TSLA raw  | PLTR raw | F2-op  | 0.725 | 0.434 | 0.461 | 0.447 |
| NVDA bal. | TSLA raw | F1-max | 0.80 | 0.464 | 0.543 | 0.501 |
| NVDA bal. | TSLA raw | F2-op  | 0.45 | 0.331 | 0.720 | 0.453 |

For reference, in-domain F1 maxes from §5: NVDA-on-NVDA 0.686 at
thr=0.775; TSLA-on-TSLA 0.514 at thr=0.75.

Three findings come out of the matrix:

1. **Cross-symbol transfer is near-lossless between trainable
   symbols.** TSLA→NVDA F1=0.691 essentially matches NVDA-on-NVDA
   F1=0.686 — the TSLA-trained model is as good at predicting NVDA
   shocks as the NVDA-trained model. NVDA→TSLA F1=0.501 vs
   TSLA-on-TSLA F1=0.514 — same pattern, ~1 F1 point of degradation.
   The TCN learns microstructure features, not symbol-specific
   memorization.

2. **The bottleneck is density, not symbol identity.** Both
   NVDA-trained and TSLA-trained models cap out at F1≈0.45 on PLTR
   (≈0.045 % positive density, ~20× sparser than TSLA, ~100× sparser
   than NVDA). Whichever model you point at PLTR, the held-out F1 is
   the same — because the limiting factor is how rare the positive
   class is in the eval data, not how the model was trained.

3. **F1-optimal threshold is remarkably stable across pairs.** All
   four F1-max points cluster at thr ∈ [0.75, 0.85], independent of
   train or eval symbol. This means a single per-deployment threshold
   in this band would underperform per-pair tuning by < 0.02 F1.
   Practical consequence: the engine's per-symbol threshold JSON is
   the right operational unit, but transferring a TSLA threshold to
   NVDA in a pinch loses very little.

The asymmetry in the matrix — TSLA→NVDA actually beating NVDA-on-NVDA
by 0.005 F1 — is small enough to be noise from the validation split
choice (NVDA used the curated `_balanced` CSV, TSLA used raw).
Restating it without overclaim: cross-symbol transfer in either
direction lands within noise of the in-domain baseline when both
ends are above the trainable density threshold.

SPY is excluded from this matrix. SPY's positive density (~0.015 %)
sits below the trainable floor, no SPY weights exist, and the
SPY val CSV (9.6 GB) takes ~1 hour to sweep without changing the
qualitative picture documented above.

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

## 10. End-to-end pipeline verification (2026-05-09)

This section records the smoke test that exercised Layers 1 (replay
substitute) → 2 (alpha) → 3 (PPO execution) → 4 (shadow training) +
dashboard end-to-end against the trained TSLA model. It was run
against the Saturday-closed market via a `ReplaySensorArray` that
yields `PhysicsState` from `feature_history_TSLA_balanced.csv`,
gated by a single env var `REPLAY_CSV`. With `REPLAY_CSV` unset the
engine reverts to its live Alpaca path with no code differences.

Files added/touched:
- `test_replay.py` — `ReplaySensorArray` + 4 pytest tests, ~150 lines
- `engine.py` — three branches behind `config.REPLAY_CSV`
  (`_init_exchange` short-circuit, `SessionManager` skip,
  `SensorArray` swap)
- `config.py` — `REPLAY_CSV` env var; `MAX_SESSION_DRAWDOWN_PCT`
  made env-overridable
- `layer2_alpha.py:312-318` — `_push_features` now divides
  `ce_ratio` by 10 to match `train_stream.py:101-105` preprocessing
- `engine.py:369` — coerce `config.ALPACA_DATA_FEED` string to
  `DataFeed` enum (current `alpaca-py` requires the enum)

The replay smoke test surfaced **four pre-existing bugs that would
have blocked the live RTH test cold**, plus exposed two replay-only
synthesis quirks. Each is named for the slide deck.

### Bug 1: train/inference scaling mismatch on `ce_ratio`

`train_stream.py:101-105` had divided `ce_ratio` by 10 during
training, but `AlphaEngine._push_features` fed it raw at inference.
The TSLA model, trained on `ce_ratio` values of ~0–5, saw ~0–50 at
inference. Sigmoid output collapsed to ~0 across all windows;
turbulence never crossed any threshold; no mandates ever fired.

Diagnostic: a standalone script (`_diag_tcn_scale.py`) confirmed
both `forward()`-fp32 and manual-bf16 paths produced identical
distributions over 5000 windows of the training CSV — ruling out
autocast as the cause and isolating the scale mismatch.

Fix: align inference to training by dividing `ce_ratio / 10.0` in
`AlphaEngine._push_features`. After the fix, the diagnostic
distribution showed p99=0.551, max=0.773 — matching what the
training-time threshold sweep had computed.

### Bug 2: synthesized 1-level book gave Q=0 → all mandates suppressed

`HeatEquationSolver.solve` rejects mandates when `Q < MIN_ORDER_SIZE`
(layer2_alpha.py:228). `Q = |bid_total - ask_total|`. Our initial
replay synthesized equal-size depth on both sides, so Q was always
0 and the solver suppressed every mandate even when turbulence
exceeded threshold.

Fix: skew `ReplaySensorArray`'s synthesized depth by the harvested
OBI signal so `bid_total = total · (1 + obi)/2`, giving
`Q = total · |obi|`. With `REPLAY_FAKE_DEPTH_SIZE = 1000` and
realistic TSLA OBI in [0.1, 0.5], `Q` clears `MIN_ORDER_SIZE = 1.0`
easily. Live mode's real OB depth makes this irrelevant.

### Bug 3: drawdown formula had unbounded leverage at small peaks

Original `engine.py:647-651`:
```python
drawdown = (self.session_peak_pnl - self.session_pnl) / self.session_peak_pnl
return drawdown > config.MAX_SESSION_DRAWDOWN_PCT
```

When `session_peak_pnl` is small (e.g., $0.91 after a winning
episode) and `session_pnl` swings to a negative value (e.g.,
-$8.59 after a loss), drawdown ratio = 10.44, far exceeding any
percentage threshold including 1.0. The 2 % production gate would
trip after a handful of episodes any time the engine started cold —
which is exactly what we observed during the smoke test.

Fix: hybrid gate (added in this session, see `engine.py` and
`test_engine.py`):

1. **Absolute USD cap** — `MAX_SESSION_DRAWDOWN_USD` (default
   $1000), always active. Trips when `peak - current > USD_cap`
   regardless of peak size. Primary safety stop.

2. **Percentage cap** — `MAX_SESSION_DRAWDOWN_PCT` (default 2 %),
   active only once `peak >= DRAWDOWN_PCT_MIN_PEAK_USD` (default
   $100). Below that floor the ratio is pathological and we rely
   on the USD cap.

Six unit tests in `test_engine.py` lock in the behavior — including
the exact tiny-peak case from the smoke-test log ($0.91 peak,
-$8.59 pnl) which now correctly does not trip.

### Bug 4: Alpaca SDK `DataFeed` enum requirement

`engine.py:369` had `data_stream = StockDataStream(key, secret,
feed=feed)` with `feed = "iex"`. Current `alpaca-py` accepts the
str-equality comparison inside `StockDataStream.__init__` (so the
"only IEX/SIP supported" check passes) but then crashes on
`feed.value` when constructing the websocket URL.

Fix: `feed = DataFeed(config.ALPACA_DATA_FEED.lower())` before
passing to the constructor. After the fix, the live boot path
produced the expected sequence on Saturday: Alpaca init,
SessionManager calendar fetch, websocket connect, subscribe to
TSLA trades + quotes, then "market closed — pausing strategy" as
designed.

### Replay-only synthesis quirks (not bugs, documented for fidelity)

- **1-level synthesized book.** `LocalMatchingEngine.attempt_fill`
  walks only top-of-book for cross detection, so 1 level is
  sufficient for fills. Market-order depth walks consume the full
  level and stop, which is unrealistic for large sizes but fine
  for verifying pipeline plumbing. Live mode gets the real ~10-level
  L2 from Alpaca's IEX feed.
- **`viscosity` synthesized as `max(spread, 1e-4)`.** Live mode
  gets a smoothed value from `StudentTKalmanFilter`. Synthesized
  value is in the right scale and avoids the heat solver's
  divide-by-zero at zero spread.

### What the smoke test demonstrated (defense-narrative)

After all four fixes, the replay engine ran continuously against
TSLA harvested data with sustained mandate generation, no
kill-switch trips, and visible activity at every layer:

- Layer 2 firing mandates at the trained threshold (0.5)
- Layer 3 PPO routing via market+limit order splits with sizes
  varying by mandate (17, 25, 36 shares)
- Layer 4 ShadowSimulator collecting per-step replay entries
- LocalMatchingEngine producing fills with bps-level fee accounting
- Dashboard updating physics state, recent episodes, recent orders
  in real time

Forty-plus episodes in 16 seconds of wall clock. Sustained
operation past the previously-tripping drawdown gate. Layer 4
replay buffer accumulating toward its push threshold of 50,000.

The Monday RTH command is the same as the Saturday swap-back
rehearsal:

```powershell
Remove-Item Env:REPLAY_CSV -ErrorAction SilentlyContinue
Remove-Item Env:MAX_SESSION_DRAWDOWN_PCT -ErrorAction SilentlyContinue
$env:SYMBOL = "TSLA"
python -u engine.py 2>&1 | Tee-Object -FilePath engine_smoke_TSLA.log
```

The drawdown gate restores to 2 % when the env var is unset, so
production safety holds for the live test.
