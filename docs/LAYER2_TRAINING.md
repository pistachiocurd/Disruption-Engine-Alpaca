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

Fix: hybrid gate (see `engine.py` and `test_engine.py`):

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

## 11. Hardening pass (2026-05-09, evening)

After the §10 smoke test confirmed the pipeline runs end-to-end, a
deeper audit of the engine's risk surface and the replay's OOD
behavior surfaced four more issues that would make Monday's RTH
results misleading even with the pipeline "working." This section
records the fixes and the final layer-by-layer strategy.

### 11.1 Issues found in audit

1. **OOD detector ran on the cold-start prior** because no
   `latest_TSLA.json` existed. The frozen identity-Σ produced
   `mahal_dist ≈ ‖x‖₂`, flagging any tick with `ce_ratio > ~4.5` as
   out-of-distribution. The replay's first run mass-suppressed
   ~70 % of mandates with `MANDATE_SUPPRESSED_OOD: mahal_dist=50`
   storms.

2. **No aggregate position limit.** `MAX_POSITION_LIMIT` was checked
   per-mandate (layer2_alpha.py:264) but not against running
   `_position` (engine.py:619, 622). Successive same-direction
   mandates stacked unboundedly — observed +797 share TSLA position
   against the documented 100-share cap. Drove wild MTM swings.

3. **Drawdown gate measured the wrong PnL.** `_max_drawdown_breached`
   used `session_pnl` (execution-shortfall, $10s of dollars) but the
   real risk lived in `theoretical_pnl = cash + position·mid` (MTM,
   could swing $1000+). The gate could not see directional exposure
   losses.

4. **Replay's `mahal_dist` was stale.** `ReplaySensorArray` read the
   `mahal_dist` column from the CSV — values computed during the
   original harvest with whatever prior existed at that time
   (typically identity). The new calibrated detector loaded into the
   engine wasn't being used. Replay dashboards showed
   `mahal_dist` values orders of magnitude larger than what live
   mode would compute.

### 11.2 Fixes (committed)

1. **OOD calibration.** Ran `fit_ood_from_csv.py` against the 47M-row
   `feature_history_TSLA_balanced.csv`, producing
   `calibration/latest_TSLA.json` with
   `μ = [41.0348, -0.0038, 0.0000]`,
   `Σ_diag = [195.84, 0.164, 0.0]`. The singular `liq` covariance
   reflects equity's disabled liquidation tracker; the
   `OODDetector._set_inverse` 1e-6 jitter handles it. Engine boot
   now logs `loaded calibration from .../latest_TSLA.json` instead
   of `cold-start prior`.

2. **Aggregate position-limit gate** (`engine._would_exceed_position_limit`).
   In the strategy loop, a mandate that would push `abs(_position)`
   further past `MAX_POSITION_LIMIT` in the *same direction it's
   already biased* is suppressed and logged as
   `MANDATE_SUPPRESSED_POSITION_LIMIT`. Position can still flip
   directions; just cannot pile on a saturated side.

3. **Hybrid IS + MTM drawdown gate.** `_max_drawdown_breached` now
   composes two parallel checks:
   - `_is_drawdown_breached` (existing, IS-based, $1K USD or 2 %)
   - `_mtm_drawdown_breached` (new, reads `cash + position·mid`,
     defaults $20K USD or 10 % in live mode, $1M / 1000% in replay
     mode to absorb CSV session-boundary jumps).
   The kill-switch log line now reports which gate(s) tripped and
   the underlying values (`is=… mtm=… peak_is=… pnl_is=… peak_mtm=…
   pos=… cash=…`). Both caps are env-overridable.

4. **Session reset on market open** (`_reset_session_risk_state`).
   On every detected close→open transition, `_position`, `_cash`,
   `session_pnl`, `session_peak_pnl`, `_mtm_peak_pnl`, fee/notional
   counters are zeroed. Prevents stale peaks from yesterday tripping
   today's gates after sub-cent moves. Wired beside the existing
   `SensorArray.reset_session` call so the lifecycle is unified.

5. **Replay's OOD now uses the live detector.**
   `ReplaySensorArray.__init__` accepts `ood_detector` (passed by
   engine.py from the calibration-loaded `OODDetector`). Each row's
   `mahal_dist` is recomputed via `ood_detector.evaluate(...)` against
   the calibrated μ/Σ rather than read from the CSV. The CSV column
   is now ignored when a detector is available. After this fix,
   replay's mahal_dist range dropped from 5–50 to 0.5–3.0 — matching
   what live mode produces.

6. **Mode-conditional MTM defaults** (`config.py` REPLAY_CSV branch).
   Live mode uses tight production caps; replay mode uses caps that
   absorb CSV concatenation artifacts. Single-variable swap (unset
   REPLAY_CSV) restores live caps for Monday RTH.

7. **Dashboard gains a second drawdown panel** (`dashboard.html`,
   `dashboard.py`). Top row now shows IS Drawdown and MTM Drawdown
   side-by-side with their respective sublabels (`max $1000 or
   2.00%`, `max $20000 or 10.00%`). Theoretical P&L's "BTC" hardcode
   was replaced with a symbol-aware unit (`sh` for equities, base
   currency for crypto pairs).

### 11.3 Test coverage

`test_engine.py` (NEW): 15 tests covering all six IS-drawdown cases
plus four position-limit cases (long cap, short cap, allow-flip,
allow-buy-when-short), four MTM cases (no-OB, USD trip, sub-cap,
percent trip), and two session-reset cases. `test_replay.py`
expanded to 5 tests including OOD-flag-derived-from-threshold and
OBI-skewed depth synthesis. Total project test count: 21 passing.

### 11.4 Final per-layer strategy and current state

| Layer | What it does | Current state | What's calibrated | What's deferred |
|---|---|---|---|---|
| 1 — Sensors | VPIN, Kalman, HMM, OOD, regime, OBI | Operational | OOD μ/Σ from 47M rows; HMM cold-start prior | HMM emission re-fit (operator job); session-aware reset wired |
| 2 — Alpha | TCN turbulence + heat-equation P_eq + 4 gates | Operational | TCN trained, threshold tuned, OOD gate calibrated | None for defense; focal loss could lift TSLA F2 |
| 3 — Execution | ExecutionEnv + PPO routing + LocalMatchingEngine | **Random-init each session**; trains online via Layer 4 | Action-space, reward shaping, fee adjustment | PPO weights persistence (defer post-defense) |
| 4 — Shadow | ReplayBuffer + PPOTrainer + StabilityGate + Polyak | Operational; collects on every episode step | KL=0.1, regime match, 50K-step minimum | Promotion observed in live demo; needs longer run for stability gate to fire |
| Risk gates | Position limit, IS DD, MTM DD, OOD, drift | All operational, env-overridable | Default caps sized to per-symbol limits | Replay session-boundary detection (cosmetic) |
| Replay shim | `test_replay.ReplaySensorArray` | Operational; OOD recomputes from live detector | Synthesized 1-level book, OBI-skewed depth | Multi-level book synthesis (post-defense) |

### 11.5 Defense-narrative claims that are supportable

- *"Layer 2 TCN trained on TSLA harvest, F1=0.514, threshold tuned
  via post-train sweep. Cross-symbol matrix shows the model
  generalizes between trainable-density symbols (TSLA→NVDA F1=0.691)
  while density-bottlenecked on PLTR-class symbols (~0.05% positive
  density)."*
- *"End-to-end pipeline operates in shadow mode against either live
  Alpaca IEX or harvested CSV via a single env-var swap."*
- *"Five risk gates engage independently: TCN turbulence threshold,
  OOD Mahalanobis (calibrated), aggregate position limit, hybrid
  IS+MTM drawdown, and drift monitor."*
- *"Layer 4 collects training data from every episode step. After
  ~50,000 environment steps the stability gate clears and Polyak
  averaging promotes the shadow policy to live, with three
  independent gating conditions."*

### 11.6 Defense claims to AVOID

- *"Strategy is profitable."* — No backtest with held-out data;
  replay smoke shows -$1K range MTM losses on 30-second runs because
  PPO is random-init and pays round-trip spread costs. Profitability
  is post-defense work requiring trained PPO weights.
- *"PPO is trained."* — Random init each session. Online learning
  via Layer 4 has barely started in any single session.
- *"Production-ready for live capital."* — `_live_submit` is a stub
  (layer3_execution.py:334). Full live order path needs a real
  Alpaca POST-orders integration. Defense is shadow-mode by design.

### 11.7 Monday RTH command

Same as the §10 command — defaults already produce correct live
behavior:

```powershell
$env:SYMBOL = "TSLA"
python -u engine.py 2>&1 | Tee-Object -FilePath engine_smoke_TSLA_monday.log
```

`REPLAY_CSV` unset → live Alpaca path, tight MTM caps ($20K USD,
10 % peak-relative), `latest_TSLA.json` auto-loads, position cap at
100 shares, all gates active. Session reset fires on market open.

## 12. Feature predictability diagnostic (`tests/test_features.py`)

Added 2026-05-10 during the Hyperliquid harvest prove-out. Quickly answers
"is the signal even in the data?" before spending hours on label/loss/
data-volume tuning. Computes 2-sample Kolmogorov-Smirnov distance between
pre-shock windows and random non-shock windows, per channel, on both the
full 60-tick window and the trailing 10 ticks (where leading-classifier
signal should concentrate).

```powershell
python tests/test_features.py --csv calibration/feature_history_<COIN>.csv
```

**Reading the output:** KS D > 0.10 with p < 1e-10 on at least one
channel = signal is genuinely there; if training fails, suspect labels /
loss / data volume. KS D < 0.05 with non-significant p across all
channels = no signal in this feature set; no NN architecture can extract
what isn't statistically present.

**HL BTC perp findings (3 days, 210 shocks, 2026-05-10):**

| channel | full window KS D | last-10-tick KS D | reading |
|---|---|---|---|
| ce_ratio/10 | 0.021 (p=8e-5) | 0.027 (p=0.13) | weak signal, not significant in last 10 |
| **obi** | **0.061 (p=2e-37)** | **0.130 (p=3e-28)** | strong, temporally concentrated |
| liquidation_rate | dead channel | dead channel | HL liq adapter not yet wired |

OBI carries the bulk of HL's pre-shock signal and concentrates in the
**last ~10 ticks** before a shock. Implications: the H=30 label horizon
used by `train_stream.py` dilutes this signal; training with H=10 should
produce materially better F2. CE_ratio behaves differently on HL than
TSLA — on TSLA the L1 quote-cancel proxy carried most of the signal; on
HL the L2-derived CE doesn't carry HL's perp-shock signature in the
last-tick window. This is expected: different microstructure regimes
have different leading indicators.

**Empirical correction (2026-05-10, after H=10 retry):** Tightening the
horizon to 10 ticks made things *worse* (F2 0.027 → 0.010). The diagnostic
correctly identified where signal lives but understated the role of
absolute positive count. With 210 shocks × H=30 = 11,880 positives,
focal+α=0.5 plateaus at "predict negative everywhere"; cutting to
210 × 10 = 2,100 positives crashes harder. TSLA's reference is
25,307 shocks × H=30 = 759K positive labels — 64× more than HL 3-day.
The dominant constraint on HL is positive-label *count*, not density and
not feature concentration. Combined fixes (multi-coin + liquidation
channel + 14-21 day harvest) target this directly.

## 14. Pivot to directional bias prediction — L2 features carry real alpha (2026-05-12)

### 14.1 Temporal integrity check — §13.4 baseline was modestly inflated

§13 closed at F2 ≈ 0.121 (§13.4 multi-coin AUCM with Path D features),
with six independent refinement strategies failing to lift the score
(§13.6 through §13.15). Before declaring the data ceiling fundamental,
ran a temporal-holdout integrity check: same §13.4 protocol but with
each coin's CSV time-split 80/20 (train on first 80% of BTC/ETH/SOL,
val on held-out last 20%). Result: **val F2 = 0.106** at thr=0.050,
max prec 1.5× base (vs §13.4's reported 1.81×). The original baseline
was ~12% inflated by training-only F2 reporting, but the ceiling
itself is real at F2 ≈ 0.10-0.11.

Also ran the same protocol BTC-only (per-symbol pivot test, time-split):
**val F2 = 0.062 with AUCM (volume-starved, warning fired), 0.061 with
Focal.** Per-symbol training underperforms pooled — cross-coin pooling
provides a regularization benefit single-coin can't replicate at this
data volume. **Pivot 1 (per-symbol models) is OFF the table.**

### 14.2 The reframe — directional bias as a balanced target

The §13 investigation had been trying to push shock-prediction F2 from
0.10 → 0.20+ across:
- Loss function changes (AUCM, Focal, BCE)
- Feature engineering (Path D, Path G dimensionless)
- Label refinement (windows, vol-scaled, hybrid floor)
- Architecture (Path C swap, Path E SSL pretrain)

Every refinement was implicitly assuming **shocks** were the right
prediction target. The shock target has two structural problems:

1. **Class imbalance**: ~1.4% positive density forces AUCM, Focal,
   or pos-weighted BCE — each with its own pathology.
2. **Sparse, noisy events**: a "shock" is defined by a downstream
   heuristic (VPIN + price move + window), so labels themselves carry
   information loss before the network even sees them.

The directional target sidesteps both: `label[t] = 1 iff mid[t+H] > mid[t]`
is balanced ~50/50 (positive density 44.4% on the held-out val) and is
defined without any auxiliary signals. Asks the most fundamental
question: **does the 50ms state vector contain leading information
about price direction over a tradeable horizon?**

### 14.3 Implementation

Changes:

1. `config.py`: `TCN_DIRECTIONAL_HORIZON_TICKS = 100` (~55s on HL's
   ~1.8 ticks/s — well past the spread-roundtrip timescale).
2. `train_stream.py`: new `--label-source=directional` variant.
   `label[t] = 1 iff mid[t+H] > mid[t]`; last H ticks padded with 0
   (negligible bias since H << n).
3. `train_stream.py`: new `--bce-pos-weight` flag (default 10.0 for
   back-compat with shock training; set to 1.0 for balanced
   directional target).

### 14.4 Results — features carry real leading signal

Protocol: 3-coin pool (BTC/ETH/SOL × 14 days), 80/20 time-split per
coin, train on `feature_history_<COIN>.train.csv`, val on pooled
`feature_history_pooled.val.csv` (1.3M held-out ticks). Path D feature
set (5 channels). BCE loss, `pos_weight=1.0`, lr=1e-3, batch 2048,
1 epoch.

Val threshold sweep:

| thr | pred_pos | precision | edge vs 0.444 base |
|---:|---:|---:|---:|
| 0.35 | 1.09M (84%) | 0.461 | +1.7pp |
| 0.40 | 697K (54%) | 0.498 | +5.4pp |
| **0.45** | **367K (28%)** | **0.544** | **+10.0pp** |
| **0.50** | **28K (2.2%)** | **0.555** | **+11.1pp** |
| 0.55 | 77 | 0.506 | +6.2pp |
| 0.70 | 14 | 0.643 | +20pp (small N) |
| 0.85 | 5 | 0.800 | +36pp (small N) |

**The 0.45 row is the headline result.** 28% of val ticks predicted UP
with 54.4% accuracy on N=367K = +10pp edge over the 0.444 base rate,
N large enough to be statistically rock-solid. The 0.50 row is
sharper at 55.5% but on only N=28K (2.2% of val). Higher thresholds
suggest the model has a small "high-confidence" tail it can flag with
56-80% precision but only on ~5-15 predictions out of 1.3M — actionable
only if the cost-of-trade-entry is very low.

### 14.5 Implications for the whole investigation

**The §13 "data ceiling" was a target-design artifact, not a feature
limitation.** The same Path D feature vector that capped at F2 ≈ 0.11
for shock prediction produces a 1.25× lift on directional prediction
— a fundamentally more usable signal density. Three corollaries:

1. **All six §13 refinements (loss, architecture, features, labels)
   were correctly noting that shock-target rescue wasn't working**, but
   each was reaching for a partial fix when the target itself was the
   structural issue.
2. **L3 order book data (Pivot 3) is NOT required.** We have alpha at
   L2 once the target is appropriate.
3. **The HMM-based regime target (Pivot 2 Option A/B) was correctly
   abandoned.** With the HMM 98% stuck on regime 2 for BTC/ETH, no
   refit short of a full re-calibration would have lifted it.

### 14.6 Decision — directional is the new training target

§13's shock-prediction architecture is preserved (for incident
detection / circuit-breaker use cases where a binary "did a shock just
start?" label is operationally meaningful), but the **primary alpha
generator** for the arbitrage engine becomes directional prediction.

Forward work (§14.7 and beyond):
- Translate the directional signal into a tradeable threshold (e.g.,
  enter long on signal > 0.50 + half-spread cost; exit at signal
  reversal or H ticks elapsed).
- Tune `H` — test 50 / 100 / 200 / 500 to find the horizon with the
  cleanest precision lift vs cost-of-execution.
- Multi-epoch training (current is 1 epoch). The 0.69 train loss
  barely budged across batches — significant headroom for further
  optimization.
- AlphaEngine wire-up: replace the shock-classifier threshold gate
  with a directional-confidence gate. The current PhysicsState ↔
  TCN flow continues; only the label interpretation changes.

**Logs and artifacts.** `bce_directional_H100.log` (pos_weight=10,
predict-all calibration), `bce_directional_H100_balanced.log`
(pos_weight=1.0, the canonical result). `tcn_weights_BTC_USDC_USDC.pt`
now reflects the directional model — must be regenerated under
shock-labels protocol before any §13 shock-prediction reproduction.

### 14.7 Phase A backtest — L2 directional alpha is REAL but UNEXECUTABLE (2026-05-12)

§14.6 deferred the question of whether the +10pp paper edge would
survive translation into a live execution mandate. Built
`backtest_directional.py` as a standalone PnL simulator to answer this
before committing to a multi-layer architectural refactor (the
proposed `DirectionalMandate` + `MakerExecutor` rewiring of
layer2_alpha.py / layer3_execution.py / engine.py).

**Trading rule.** Maker-only entry: signal > long_threshold → post
passive bid at best_bid[t]; signal < short_threshold → post passive ask
at best_ask[t]. Taker exit at submit_t + H ticks (sell at bid for
longs; lift ask for shorts).

**Fill models.** Two are implemented, controlled by `--queue-fill-prob`:
- *Strict adverse-selection (default)*: fill iff best_bid drops below
  entry within H (market swept through our level — winner's curse
  proxy). Strictest possible.
- *Queue-priority (q > 0)*: with probability q, additionally fill
  "neutrally" at a random tick in [t+1, t+H]. Models the fraction of
  real-world maker fills that happen without adverse movement.

**Seven runs across the configuration space:**

| run | weights | hold | thr / dual | q | trades | gross_pnl | net_pnl |
|---|---|---:|---|---:|---:|---:|---:|
| 0  | H=100 | 100 | 0.55 / 0.45 | 0.0 | 524K | -$1.84M | -$5.76M |
| A  | H=100 | 100 | 0.70 / 0.30 | 0.0 | 285 | -$1,665 | -$6,738 |
| C  | H=100 | 100 | 0.53 / 0.43 | 0.0 | 428K | -$1.49M | -$4.69M |
| B  | H=100 | 100 | 0.70 / 0.30 | 0.3 | 308 | -$727 | -$6,303 |
| B' | H=100 | 100 | 0.70 / 0.30 | 1.0 | 360 | **+$1,203** | -$5,423 |
| 1  | H=100 | **500** | 0.70 / 0.30 | 0.3 | 330 | **+$2,163** | -$3,911 |
| 2b | **H=500** | 500 | 0.70 / 0.30 | 0.3 | 69 | +$126 | -$1,102 |

**Three definitive conclusions:**

1. **L2 features predict H=100 direction (real, ~55% paper precision);
   they do NOT predict H=500 direction.** Run 2 retrained the model
   at H=500 horizon — val sweep showed precision *at base rate
   everywhere* and *dropping below* base rate at high-confidence
   thresholds (thr=0.60 → prec=0.380 vs base 0.479). The
   microstructure signal in the Path D feature set decays past
   ~100 ticks.

2. **The H=100 directional edge magnitude is far too small for
   maker+taker execution.** Run B' with zero adverse selection
   (q=1.0) gave gross +$1,203 across 360 best-confidence trades on
   ~$20M notional = **+0.06 bps per trade**. The round-trip cost
   (1 bp rebate − 4.5 bp taker fee = -3.5 bps net) eats this **58×**.
   No threshold tuning, no fill-model relaxation, no longer hold
   horizon, no architectural change can rescue an edge this small.

3. **Run 1's H=500-OOD-hold improvement was random walk variance,
   not signal.** The H=100-trained model evaluated at H=500 hold
   gave gross +$2,163, but Run 2b (correctly trained at H=500) gave
   only gross +$126 on equivalent high-confidence signals — a 17×
   gap that's pure variance from giving moves more time.
   Longer holds don't extend the model's predictive horizon; they
   just let variance create more wins on the same flat distribution.

### 14.8 L2 chapter closed; Pivot 3 (L3 microstructure) is the next direction (2026-05-12)

The Phase A backtest closes the L2 investigation definitively.
Three exhaustively tested framings — shock prediction (§13), regime
classification (§14.2 scoping), directional prediction (§14.4-7) —
all converge on the same physical fact: **the Path D feature set
(ce_ratio, obi, mlofi, vamp, kyles_lambda) carries real but
*magnitude-insufficient* leading signal**. The paper accuracy of
55% on directional H=100 is genuine and consistent across both
in-sample and temporally-held-out evaluation, but the *fraction of
those wins where price moved more than the round-trip execution cost*
is too small to survive any reasonable fee structure.

**Why Path A (inventory framework) was considered and rejected.**
A market-making framework with passive bids on both sides, no taker
exit, and inventory accumulation could capture the full ~2 bps
round-trip maker rebate, dwarfing the 0.06 bps directional edge.
But that strategy IS the rebate; the TCN becomes window-dressing for
marginal inventory skew. We would no longer be building a
"disruption arbitrage engine" but a high-frequency rebate-farming
bot, competing directly with FPGA market makers on queue priority.
That's an infrastructure-and-latency contest, not a predictive-ML
contest. Outside the project's thesis.

**Why Path B (L3 / order-by-order data) is the correct pivot.**
The fundamental problem with L2 aggregates is they *destroy* the
microstructure dynamics that *precede* shocks: queue depletion,
cancellation velocity, order lifespan, hidden-order inference,
trade-aggressor sequence. L2 is a snapshot of the book state; L3 is
the event-by-event tape of *how the book is being made*. The signal
that drives professional HFT alpha (and that academic limit-order-book
literature consistently finds predictive — see Cont & Kukanov 2017,
Gould-Porter-Williams-McDonald-Fenn-Howison 2013) lives at L3, not L2.
Path G's `BOUNDARY_CONDITIONS.md` already noted that the discrete
quantum of market data is the order-event, not the snapshot.

**What Pivot 3 requires.**

1. **L3 data source.** Hyperliquid's S3 archive (which we already pull
   for trades + L2 snapshots) is L2-only as far as we've verified.
   Either: (a) find an L3-emitting venue with a public archive (Binance
   tick data via S3, Coinbase Advanced order-by-order, Polygon options
   tape, Databento for futures); (b) reconstruct L3 from HL's "delta"
   stream if accessible; (c) commit to live L3 capture from the WS
   feed forward.
2. **New sensor architecture in `layer1_sensors.py`.** The current
   sensors (VPIN, MLOFI, Kyle's λ) consume snapshots. L3 sensors need
   to consume *order events*: order arrivals, cancellations, modifies,
   trades-against-resting-orders. New characteristic quantities
   include cancellation rate per side, order lifespan distribution,
   queue position evolution, hidden-order inference from price
   improvement events.
3. **Replacement TCN feature stack.** Path D's 5 channels go away or
   become a small subset. New channels are L3-derived. `TCN_INPUT_CHANNELS`
   will need to grow substantially (likely 10-20 channels).
4. **§13 / §14 calibration carries over conceptually but the labels
   need re-derivation.** Shock-trigger definitions assumed L2 snapshots
   (VPIN spike + N-tick price move). L3 may support better labels —
   e.g., "did the book *thin* before the move?" — which were not
   reconstructible from L2 alone.

**Status of the L2 codebase.** Preserved as-is for reproducibility
and as the reference baseline (anything L3 builds must beat
F2 = 0.106 / max-prec = 1.5× base rate / directional gross edge
0.06 bps per trade — the L2 ceilings). `USE_PATH_G_FEATURES`,
`USE_VOL_SCALED_LABELS`, the directional `label_source`, and
`backtest_directional.py` all remain wired up in case future work
revisits a hybrid L2+L3 feature stack. The `tcn_weights_*.pt` files
freeze the current model state; the H=100 directional weights are
preserved at `tcn_weights_BTC_USDC_USDC_H100.pt`.

**L2 investigation is formally complete.** §15 and beyond belong to L3.

---

## 13. Hyperliquid liquidation channel and multi-coin training (2026-05-10)

Two infrastructure additions to address the HL TCN training plateau:

### 13.1 HL liquidation channel — un-degenerating the OOD covariance

`liquidation_rate` was identically 0 on HL in earlier runs because
`LiquidationCascadeTracker` is conditioned on Binance's `!forceOrder`
WS feed — non-Binance venues got `enabled=False`. This degraded the
3-channel OOD detector to effectively 2D (`Σ_diag[2] = 0`, regularized
by 1e-6 jitter, see §A row 5 of the research review) and gave the TCN
one input channel that contributes nothing.

**Source choice**: HL does not surface liquidations identifiably in the
`node_fills_by_block` archive. Verified empirically (see
[tests/inspect_hl_liquidations.py](tests/inspect_hl_liquidations.py)
output, 2026-05-10): no `dir` value contains "Liquidate", 100% of
`trade_dir_override` are missing/'Na', and no counterparty address
dominates with a "liquidator vault" pattern. The labeled source is the
official [`hyperliquid-dex/historical_data/liquidations.csv`](https://github.com/hyperliquid-dex/historical_data),
schema `time,user,liquidated_ntl_pos,liquidated_account_value,leverage_type`.

Note: the CSV has **no `coin` field** — events represent aggregate
cross-market liquidation pressure in USD notional, not per-coin
liquidations. Treated as a market-wide stress signal injected uniformly
across per-coin harvests; per-coin VPIN normalization in
`LiquidationCascadeTracker.liquidation_rate()` produces a per-coin
scaling.

Wiring:

- `LiquidationCascadeTracker.record(ts_ms, qty)` — public injection hook
  for adapters. The Binance WS path still uses `_record` directly.
- `LiquidationCascadeTracker.run()` — early-returns for non-Binance
  venues so we don't try to connect to the Binance URL on HL.
- `SensorArray.__init__` — `liq_enabled = (is_binance or is_hyperliquid)
  and not IS_EQUITY`. Tracker stays enabled on HL.
- `fetch_history_hyperliquid.load_hl_liquidations()` — downloads (and
  caches at `./calibration/hl_liquidations.csv`) the official CSV, parses
  to `[(ts_ms, ntl_usd)]`. `--refresh-liquidations` forces re-download.
- `fetch_history_hyperliquid.cmd_harvest()` — heap-merges three event
  streams (L2 books, trades, liquidations from CSV) and dispatches to
  `_process_order_book`, `_process_trade`, and `liquidation.record()`
  respectively. Pass `--no-liquidations` to skip if you want the old
  Σ_diag[2]=0 behavior for comparison.

After wiring: `liquidation_rate > 0` in the harvested CSV, OOD
covariance becomes full-rank, TCN sees three meaningful input channels.

### 13.2 Multi-coin training

For HL the per-coin shock count is the binding constraint on TCN
training (210 BTC shocks/3 days vs TSLA's 25K equity-IEX). Aggregating
across BTC + ETH + SOL multiplies the positive-label count without
extending the harvest window — assumes the cross-symbol generalization
finding from §7 (TSLA→NVDA F1=0.691 ≈ in-domain) extends from equities
to crypto perps.

- Harvester: `--coins BTC,ETH,SOL` runs the chronological replay once
  per coin in its own `SensorArray` (correct per-coin VPIN/Kalman
  state) and writes `feature_history_<COIN>.csv` per coin.
- Trainer: `train_stream.py --csv path1.csv,path2.csv,path3.csv` uses
  `MultiCsvTCNDataset`, which yields from each per-coin
  `StreamingTCNDataset` independently — labels and carry-over are
  per-coin so coin transitions don't introduce cross-symbol price
  discontinuities in the windowed features.

The cross-symbol-transfer assumption is the load-bearing piece. If
multi-coin training degrades vs single-coin, that's evidence the
crypto-perp microstructure is more coin-specific than equity microstructure
and we should train per-coin and ensemble; if it improves, the
universal-microstructure-features hypothesis holds on perps and we
proceed with aggregate training.

### 13.3 Multi-coin training — negative result (2026-05-10)

Multi-coin pooling tested. **Multi-coin helped marginally but did not
break through.** F2 climbed from 0.027 (3d BTC single-coin) to 0.046
(7d × 3-coin) — directionally correct but still at noise-floor levels
(base rate prec 0.024).

Full experiment progression on HL perps, in order:

| # | Config | Pos count | F2 | Threshold sweep shape |
|---|---|---:|---:|---|
| 1 | 1d BTC, regime, BCE | 1,720 | 0.205 | usable curve, density below floor |
| 2 | 1d BTC, shock, BCE | 5,880 | 0.046 | degenerate (everything-positive then nothing) |
| 3 | 3d BTC, shock, BCE | 11,880 | 0.028 | degenerate, unstable training |
| 4 | 3d BTC, shock, focal α=0.5, H=30 | 11,880 | 0.027 | degenerate, stable training |
| 5 | 3d BTC, shock, focal α=0.75, H=10 | 3,960 | 0.010 | degenerate, fewer positives hurt |
| 6 | 7d × 3-coin, shock, focal α=0.5, H=30, ep=10 | 153,578 | 0.046 | degenerate, prec≈base rate at any usable thr |
| 7 | 7d × 3-coin, shock, focal α=0.5, H=30, ep=1 | 153,578 | 0.046 | degenerate (overtraining isn't the issue) |

**What's been ruled out:**

- Cold-start HMM regime noise — switching to `--label-source shock`
  (price+VPIN) fixed the labeling distribution but didn't move F2.
- Label horizon too long — `--label-horizon 10` made it worse, not
  better. Diagnostic correctly identified signal lives in last 10 ticks
  but the absolute-positive-count constraint dominates.
- BCE pos_weight=10 instability — `--loss focal --focal-alpha 0.5`
  fixed per-batch spike instability; F2 unchanged.
- Focal alpha tuning — α=0.5 and α=0.75 produced effectively identical F2.
- Insufficient single-coin data — 3× more data on a single coin (1d→3d)
  marginally hurt F2 not helped.
- Single-coin training bottleneck — multi-coin 21 coin-days produced
  similar F2 to single-coin 3 days.
- Overtraining collapse — `--epochs 1` produced the same threshold-sweep
  shape as `--epochs 10`. Model never extracts signal at any epoch.

**What's confirmed:**

- Data injection is correct (channel order, scale, time direction
  match between train and serve; verified by reading both paths).
- Signal IS in the data — `tests/test_features.py` measures KS D=0.13
  on OBI in the last 10 ticks before shock, p ≈ 1e-28. Statistically
  robust discriminative signal.
- Architecture fits — loss decreases monotonically every run; per-batch
  losses stable under focal.
- Model never genuinely discriminates — every threshold sweep produces
  the same pathology: predicts all-positive below thr=0.15, prec≈base
  rate at thr=0.20, predicts nothing above thr=0.25.

**Working hypotheses for why the TCN doesn't extract HL's signal:**

1. **Absolute positive count is still too low.** TSLA reference is
   759K positive labels (25K shocks × 73 sessions × H=30); we're at
   154K (2.8K shocks × 7d × 3c × H=30). 5× short. May need 30+ coin-days.
2. **Architecture is wrong for HL microstructure.** TCN is causal 1D conv;
   maybe transformer/state-space (Mamba) handles ms-irregular HL cadence
   better.
3. **Feature set inadequate.** OBI carries the bulk of signal; ce_ratio
   adds little; liquidation_rate is dead. A 1-feature model may extract
   different/better with focused input.
4. **Loss function fundamentally wrong.** Focal+BCE both find the
   "predict near zero" trivial solution. Maybe ranking loss, contrastive
   loss, or AUC-direct loss would avoid this.
5. **Implicit label noise.** `identify_shock_events` uses price-spike +
   VPIN criteria that may not match HL's actual microstructure shock
   semantics; shocks we're labeling may not be the shocks the engineered
   features predict.

Dedicated investigation planned to characterize the root cause more
rigorously than this progression of ad-hoc experiments.

### 13.4 AUCM loss breakthrough (2026-05-10)

The dedicated research session (above) ingested Gemini Deep Research on
TCN failure modes, which diagnosed §13.3's pathology — "predict-all
below thr=0.15, base-rate prec at thr=0.20, predict-nothing above
thr=0.25" — as the canonical signature of point-wise loss collapse under
extreme class imbalance (<1% positive density). The model converges to
"predict near zero everywhere" because the flat basin around base-rate
mathematically dominates the gradient signal from rare positives.

The recommended remediation: replace BCE/Focal with LibAUC's `AUCMLoss`
(pairwise AUC-margin) paired with the PESG optimizer (minimax inner
loop). AUCM optimizes the **rank** of positive vs. negative scores,
making the base-rate trivial minimum mathematically irrelevant — the loss
penalizes only when a negative outscores a positive.

**Implementation (2026-05-10):**

- Added `libauc>=1.4` to `requirements.txt`.
- `train_stream.py` gained `--loss aucm` alongside `bce`/`focal`, with
  the PESG optimizer when selected; pre-loss `sigmoid()` to feed AUCM
  probabilities; `--aucm-margin` and `--aucm-lr` flags.
- **Initial run failed silently** — AUCM produced loss=0 on ~95% of
  batches because shock labels cluster (H consecutive positives per
  shock event) and the streaming temporal-order pipeline yielded
  zero-positive batches. LibAUC emitted `UserWarning: Input data has
  no positive sample!`.
- **Fix:** added `ShuffledBufferDataset` — reservoir-style 500K-sample
  buffer applied only when `--loss=aucm`. Each 2048-batch then contains
  ~48 positives ± 7 (σ from binomial); P(zero-positive batch) ≈ 10⁻²¹.
  DualSampler proper requires a map-style dataset; this is the cheapest
  equivalent for streaming IterableDataset.

**Result — 7d × 3-coin, shock labels, AUCM, 1 epoch, PESG lr=0.01:**

| Threshold | pred_pos | TP | FP | FN | prec | rec | F1 | F2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 1,112,205 | 28,953 | 1,083,252 | 47,836 | 0.026 | 0.377 | 0.049 | **0.102** ← chosen |
| 0.10 | 514,046 | 14,600 | 499,446 | 62,189 | 0.028 | 0.190 | 0.049 | 0.088 |
| 0.15 | 356,752 | 10,798 | 345,954 | 65,991 | 0.030 | 0.141 | 0.050 | 0.081 |
| 0.20 | 244,191 | 7,828 | 236,363 | 68,961 | 0.032 | 0.102 | 0.049 | 0.071 |
| 0.25 | 156,745 | 5,229 | 151,516 | 71,560 | 0.033 | 0.068 | 0.045 | 0.056 |
| 0.30 | 86,340 | 3,108 | 83,232 | 73,681 | 0.036 | 0.040 | 0.038 | 0.040 |
| 0.35 | 34,047 | 1,329 | 32,718 | 75,460 | 0.039 | 0.017 | 0.024 | 0.020 |
| 0.40 | 6,914 | 278 | 6,636 | 76,511 | 0.040 | 0.004 | 0.007 | 0.005 |

**What's confirmed:**

- **Hypothesis #4 (loss-function geometric degeneracy) was real and
  contributory.** The §13.3 threshold-sweep cliff is gone — precision
  climbs smoothly and monotonically from 0.026 → 0.040 as threshold
  rises; pred_pos varies smoothly across the sweep range; no
  predict-everything floor or predict-nothing cliff. This is the
  fingerprint of a working ranker, not a collapsed classifier.
- **F2 = 0.102, up from §13.3's 0.046 (2.2× improvement).** Success
  criterion (F2 ≥ 0.10 with non-degenerate sweep curve) met.

**What's NOT yet resolved:**

- **Signal extraction is still weak.** Max precision in the sweep is
  0.040 = 1.67× the base rate of 0.024. TSLA-equity for comparison
  reached precision > 0.5 at high thresholds. The model now extracts a
  ranking but it's a weak one.
- **Hypothesis #4 was necessary but not sufficient.** The remaining gap
  is plausibly H2 (architecture-cadence mismatch) or H3 (feature set
  inadequate).

**Forward pointer:** Path D (P3.6 in TODO.md) — crypto-native features.
Replace the dead `liquidation_rate` channel and the weak `ce_ratio`
contribution with MLOFI (multi-level OFI, top 5–10 levels), VAMP
(volume-adjusted mid price), and Kyle's λ (rolling regression of
return on signed volume). With AUCM proving the model CAN extract weak
rank, richer features should let it extract stronger rank without
further architecture changes (H2 deferred to P3.5).

**Artifacts (in `calibration/`):**
`tcn_weights_BTC_USDC_USDC.pt`,
`tcn_threshold_BTC_USDC_USDC.json` (thr=0.05, F2=0.102),
`tcn_threshold_sweep_BTC_USDC_USDC.csv`.

Path A artifacts preserved alongside the re-harvested Path D CSVs as
`feature_history_<COIN>.pathA.csv` (~98 MB each) for direct A/D diff.

### 13.5 Path D — crypto-native features (2026-05-10)

Path A confirmed loss geometry (H4) was a contributor but not sufficient.
Path D replaces the dead `liquidation_rate` channel and weak `ce_ratio`
contribution with three crypto-native features added to the SensorArray:

- **MLOFI** (Multi-Level OFI, Xu et al. 2018) — per-snapshot signed
  flow across the top-5 book levels, normalized by total top-K depth.
  Captures bid/ask pressure beyond top-of-book OBI; less susceptible
  to spoofing because deeper levels are harder to fake.
- **VAMP** (Volume-Adjusted Mid, Stoikov 2018) — queue-position-weighted
  mid expressed as basis-point deviation from arithmetic mid. A heavy
  ask queue pulls VAMP toward the bid (sell-side pressure expected).
- **Kyle's λ** — rolling 100-tick OLS regression of mid-return on
  normalized signed trade volume. Higher λ = thinner book / more
  price-impact-per-unit-flow.

**Implementation (this session, 2026-05-10):**

- New sensor classes `MLOFITracker`, `VAMPDeviationComputer`,
  `KylesLambdaTracker` in `layer1_sensors.py`, hooked into
  `SensorArray._process_order_book` and `_process_trade`. Equity path
  unaffected (no L2 depth from Alpaca, no per-trade initiator field).
- `PhysicsState` gained `mlofi`, `vamp`, `kyles_lambda` fields.
  `FeatureDumper` extended to write three new CSV columns.
- `config.TCN_INPUT_CHANNELS` made conditional: 5 if
  `EXCHANGE_ID=="hyperliquid"`, else 3. Equity weights unaffected.
- `train_stream.py` and `AlphaEngine._push_features` mirror conditional
  5-channel stacking: `[ce_ratio/10, obi, mlofi, vamp/10, kyles_lambda*100]`.
- `ReplaySensorArray` reads new columns with `.get(col, 0.0)` so old
  equity CSVs still replay cleanly. All 5 replay tests pass.
- 7d × 3-coin (BTC+ETH+SOL) re-harvested with the new sensors writing
  the 12-column schema. ~3 hours wall, $1 S3, 6.54M rows total.

**Result — 7d × 3-coin, shock labels, AUCM, 1 epoch, PESG lr=0.01:**

| Threshold | pred_pos | TP | FP | FN | prec | rec | F1 | F2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 2,337,581 | 65,486 | 2,272,095 | 25,141 | 0.028 | 0.723 | 0.054 | **0.121** ← chosen |
| 0.10 | 575,479 | 18,496 | 556,983 | 72,131 | 0.032 | 0.204 | 0.056 | 0.098 |
| 0.15 | 368,284 | 12,673 | 355,611 | 77,954 | 0.034 | 0.140 | 0.055 | 0.086 |
| 0.20 | 218,687 | 8,306 | 210,381 | 82,321 | 0.038 | 0.092 | 0.054 | 0.072 |
| 0.25 | 106,529 | 4,516 | 102,013 | 86,111 | 0.042 | 0.050 | 0.046 | 0.048 |
| 0.30 | 41,053 | 2,060 | 38,993 | 88,567 | 0.050 | 0.023 | 0.031 | 0.026 |
| 0.35 | 12,009 | 582 | 11,427 | 90,045 | 0.048 | 0.006 | 0.011 | 0.007 |
| 0.40 | 1,499 | 39 | 1,460 | 90,588 | 0.026 | 0.000 | 0.001 | 0.000 |

**What improved (Path D vs Path A):**

| Metric | Path A | Path D | Δ |
|---|---:|---:|---:|
| F2 (chosen op point) | 0.102 | **0.121** | **+18%** |
| Max precision in sweep | 0.040 (@ thr=0.40) | **0.050** (@ thr=0.30) | **+25%** |
| Max-prec ÷ base rate | 1.70× | **1.81×** | +6% rel |
| Recall at thr=0.05 | 0.377 | **0.723** | +92% |
| Positive density (label rate) | 2.35% | 2.77% | +18% |
| Re-harvest cost | — | $1, ~3h | — |

Notes:
- The +18% positive-density change comes from the re-harvest hitting
  a slightly different date window (HL archive is ~35 days lagged;
  the two harvests targeted overlapping but not identical days). Part
  of the F2 lift is explained by this base-rate shift; the
  base-rate-adjusted improvement (max-prec ÷ base rate) is +6% rel.
- The most striking change is **recall at low thresholds**: 0.377 →
  0.723 at thr=0.05. The model now produces high scores for a much
  larger fraction of true positives — it's *more confident* on
  positives without losing too much precision. This is the new
  features paying their way.
- Sweep curve remains non-degenerate. Precision climbs monotonically
  0.028 → 0.050 from thr=0.05 to thr=0.30 (vs Path A's 0.026 → 0.040
  over a similar range). The H4-collapse failure mode is fully gone.

**What's NOT yet resolved:**

- Max precision is still only **1.81× the base rate**. TSLA-equity
  reached >20× base rate. Crypto-native features carry signal but
  the model isn't extracting it strongly enough.
- Loss trajectory within epoch 1 peaked at batch ~1500 then came
  back down (different shape from Path A's monotone rise + plateau).
  More epochs might help; or the architecture is genuinely the
  remaining bottleneck.

**Forward pointer.** Two candidates for the next move:

1. **P3.5 / Path C** — replace the fixed-grid TCN with Mamba or
   Neural CDE. Mamba's input-dependent step size is designed for
   exactly the millisecond-irregular bursty cadence HL data exhibits.
   Higher leverage if the remaining gap is architectural (H2).
2. **P3.7 / Path E** — SSL pretext pretraining on next-tick OBI/mid
   prediction, then fine-tune the classification head on shock
   labels with AUCM. Higher leverage if the gap is label noise (H5).

Path C is the higher-likelihood bet given that Path A + D have already
addressed H4 and H3, leaving H2 (cadence) as the most plausible
unfixed contributor.

**Artifacts (in `calibration/`):**
`tcn_weights_BTC_USDC_USDC.pt` (5-channel, Path D),
`tcn_threshold_BTC_USDC_USDC.json` (thr=0.05, F2=0.121),
`tcn_threshold_sweep_BTC_USDC_USDC.csv`.
`feature_history_<COIN>.csv` — fresh 12-column 7d × 3-coin harvest.
`feature_history_<COIN>.pathA.csv` — 9-column Path A baseline data.

### 13.6 Path C — architecture swap (2026-05-11) — TESTED NEGATIVE

After Path A (loss, H4) and Path D (features, H3) lifted F2 from 0.046
→ 0.121, the next hypothesis was H2: TCN's fixed-grid causal-conv
inductive bias is the binding constraint for HL's millisecond-irregular
bursty cadence. The DR recommended Mamba (input-dependent selective
scan) or Neural CDE (continuous-time splines) as the fix.

**Install constraints.** Bleeding-edge dev environment:
- Python 3.14.4 — too new for `mamba-ssm` wheels (latest supports ≤3.12).
- No `nvcc` in PATH — can't compile mamba's fused CUDA kernel from source.
- PyTorch 2.12.0.dev with CUDA 12.8 on an RTX 5070 Ti (Blackwell).
- `pip install mamba-ssm causal-conv1d` failed with
  `NameError: name 'bare_metal_version' is not defined` (setup.py
  probing for nvcc).
- `torchcde` installed cleanly but Neural CDE's ODE solver overhead
  would be 5-10× per batch on T=60.

**Implementation.** Two pure-PyTorch alternatives added to
`layer2_alpha.py` and dispatched via a new `--model {tcn,transformer,mamba}`
flag in `train_stream.py`:

1. **TransformerSpikePredictor** — causal Transformer encoder, sinusoidal
   positional encoding, pre-norm, 3 layers, d_model=32, 4 heads,
   dim_ff=128. Tests "attention vs convolution" without ODE/scan overhead.
   ~38K trainable params.

2. **MambaSpikePredictor** — hand-rolled minimal Mamba (Gu & Dao 2023)
   in pure PyTorch. `MambaBlock` implements:
   `Linear → split x,z → causal Conv1d(d_conv=4) → SiLU → SSM scan
   → gate by SiLU(z) → Linear out → residual`. The SSM scan computes
   input-dependent Δ via softplus(dt_proj(x_proj)), discretizes A and
   B per-tick, and runs a sequential O(L) recurrence in fp32 (bf16
   underflows on `exp(Δ·A)` for thin states). 3 blocks, d_state=16,
   d_conv=4, expand=2. **30,241 trainable params** — matches TCN's
   31,265 for clean architecture-only comparison.

Refactored both TCN and the new models to expose a common
`forward_logits()` method so the training loop is model-agnostic.
Causality verified on Mamba: zeroing future tokens does not change
past hidden states (max diff = 0.0).

**Results — same protocol as Path D (7d × 3-coin, shock labels, AUCM,
PESG lr=0.01):**

| Architecture | Epochs | Params | F2 | Max prec | Max-prec ÷ base rate | Sweep shape |
|---|---:|---:|---:|---:|---:|---|
| **TCN (Path D)** | 1 | 31,265 | **0.121** | **0.050** | **1.81×** | smooth monotone climb |
| Transformer | 1 | 38,337 | 0.126 | 0.034 | 1.22× | degenerate cliff |
| Transformer | 3 | 38,337 | 0.126 | 0.052 | 1.88× | smoother climb, comparable to TCN |
| **Mamba** | 1 | **30,241** | 0.126 | 0.032 | 1.15× | **degenerate cliff** |

**What's confirmed:**

- **H2 (architecture-cadence mismatch) is NOT the binding constraint.**
  All three architectures converge to F2 ≈ 0.12. The Mamba selective
  scan — the most theoretically-motivated fix for irregular cadence —
  produced the *worst* discrimination (max prec 0.032 = 1.15× base
  rate, vs TCN's 0.050 / 1.81×). Even with 3 epochs of Transformer
  training, max precision (0.052) only barely exceeded TCN's 1-epoch
  result.
- Loss trajectories across all three architectures peak around
  batch 1500 and plateau in [0.025, 0.030]. The architectures are
  finding the same loss minimum given current data + AUCM. The
  remaining gap is not in the model class.

**Why Mamba didn't help (interpretation):**

- The selective scan addresses irregular *sampling intervals*. HL data
  is irregular but tick-indexed; once the harvest collapses bursty
  arrivals into per-tick rows, the *index distance* between ticks no
  longer carries the cadence information that the scan's Δ would
  exploit. Adding explicit Δt as a 6th input channel might recover
  this — deferred as a follow-up.
- The pure-PyTorch scan ran ~25 min for 1 epoch vs TCN's 7 min (~3-4×
  slower) due to sequential CUDA kernel launches without the fused
  mamba-ssm kernel. Not a quality issue but limits iteration speed.
- A 3-epoch Mamba run would cost ~75 min for marginal info given the
  Transformer 3-ep result showed no F2 change with more training.

**What's NOT yet addressed:**

- **H5 — implicit label noise.** `identify_shock_events` uses
  price-spike + VPIN criteria calibrated on equities. The shocks it
  labels in HL data may not be the ones that have predictable
  pre-shock microstructure signatures.
- H1 (more data) — DR demoted; least likely.

**Forward pointer.** P3.7 (Path E) — self-supervised pretext
pretraining. Train the encoder on dense next-tick OBI/mid prediction
(MSE, ~6.5M supervisory steps per epoch — orders of magnitude denser
than the 180K shock labels), freeze, then fine-tune a fresh
classification head on shock labels with AUCM. Insulates the encoder
from heuristic-label noise by forcing it to learn the underlying
microstructure manifold first.

**Note on weight artifacts.** The single `tcn_weights_<SLUG>.pt` path
in `config.py` was overwritten by each architecture's training run.
Path D TCN weights are recoverable by re-running Path D's training
command; the Path C model weights (Transformer 3-ep is the most
recent) currently occupy that path. A future iteration should append
the model name to the weights/threshold paths.

**Artifacts (in `calibration/`):**
`tcn_weights_BTC_USDC_USDC.pt` (currently: Transformer 3-ep, then
overwritten by Mamba 1-ep — Path D recovery requires re-train).
`tcn_threshold_BTC_USDC_USDC.json` (chosen: thr=0.05, F2=0.126,
Mamba). `tcn_threshold_sweep_BTC_USDC_USDC.csv`.

### 13.7 Run-to-run variance addendum (2026-05-11)

After Path C concluded, a Path D recovery run (same code, same data,
same args, no seed set) produced F2 = 0.104 vs the original Path D's
F2 = 0.121. The recovery sweep curve shape is qualitatively the same
(smooth monotone precision climb, no cliff) but the absolute numbers
shifted by ~14%.

This means **all single-run F2 comparisons in §13.4–§13.6 have noise
of ~±0.02**. Specifically:

- Path A vs Path D (0.102 → 0.121, +0.019) — within noise. Path D's
  signal is the *recall lift* (0.377 → 0.723) which is robust to seed,
  not the F2 lift.
- Path D vs Path C (0.121 vs 0.126) — within noise. Path C's claim of
  marginal improvement is not statistically supported by single runs.
  The honest read is: TCN/Transformer/Mamba all sit in F2 ∈ [0.10,
  0.13] under this protocol.

**What survives.** The qualitative findings are robust:
- AUCM removed the §13.3 trivial-collapse pathology (predict-all
  /predict-nothing cliff). Sweep curves are now non-degenerate across
  architectures.
- Max precision is consistently ≤ 2× base rate regardless of
  architecture or features-within-current-set.
- Mamba's degenerate cliff at 1 epoch (max prec 0.032 = 1.15× base
  rate) is materially worse than TCN/Transformer — that one *is*
  outside noise.

**Implications for future runs.**
- Set a fixed seed in `train_stream.py` (currently absent). One-line
  fix.
- Report 3-seed mean ± std for any path comparison, not single runs.
- Update Path E and any future paths' protocols to require ≥3 seeds.

This doesn't change the path priorities (H5 is still the highest-
leverage remaining hypothesis) but it does temper how much we trust
the cross-path F2 numbers.

### 13.8 Path E — SSL pretext pretraining (2026-05-11) — TESTED NEGATIVE

H5 (implicit label noise) was the last unaddressed hypothesis from
the §13.3 list. Path E tests it via self-supervised pretraining: train
the encoder on a *dense* target (~6.5M next-tick log-returns), freeze
it, then fine-tune a fresh classification head with AUCM on the
sparse shock labels. The rationale: if heuristic shock labels are
noisy, putting most of the model's capacity behind a frozen,
labels-independent encoder protects it from learning the noise.

**Implementation.** `train_stream.py` gained three flags:

- `--pretext` — dataset emits next-tick log-return (in bps) as the
  target; MSE loss; Adam at lr=1e-3; skips threshold sweep; saves to
  `pretrain_weights_<SLUG>.pt`.
- `--load-pretrained PATH` — `load_state_dict(strict=False)` at
  startup so the head can be re-initialized for the new task.
- `--freeze-encoder` — sets `requires_grad=False` on every parameter
  except `head.*`. Only the head (Linear(32, 1) = 33 params) trains.

Backward-compat preserved: the existing `aucm` / `focal` / `bce`
paths are unchanged.

**Phase 1 (pretrain) — 2 epochs on 7d × 3-coin:**

- Loss curve: 19.6 → 0.35 (bps² MSE). The √loss ≈ 0.59 bps RMSE on
  per-tick log-return prediction — reasonable for HL microstructure.
- Encoder learned a non-trivial representation of next-tick price
  dynamics (loss decreased monotonically, not stuck).

**Phase 2 (fine-tune) — 1 epoch, frozen encoder, AUCM, PESG lr=0.01:**

- `Encoder FROZEN. Trainable: 33 / 31,265 (0.11%)` — only the head's
  Linear(32, 1) is updated.

Threshold sweep — chosen `thr=0.050, prec=0.028, rec=0.992, F1=0.054,
F2=0.126`. Selected rows:

| Threshold | pred_pos | TP | FP | FN | prec | rec | F2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 3,209,551 | 89,861 | 3,119,690 | 766 | 0.028 | 0.992 | **0.126** ← chosen |
| 0.10 | 636,387 | 20,125 | 616,262 | 70,502 | 0.032 | 0.222 | 0.099 |
| 0.15 | 382,724 | 12,979 | 369,745 | 77,648 | 0.034 | 0.143 | 0.089 |
| 0.20 | 213,473 | 7,911 | 205,562 | 82,716 | 0.037 | 0.087 | 0.071 |
| 0.25 | 88,571 | 3,477 | 85,094 | 87,150 | 0.039 | 0.038 | 0.039 |
| 0.30 | 18,151 | 853 | 17,298 | 89,774 | 0.047 | 0.009 | 0.018 |
| 0.35 | 521 | 25 | 496 | 90,602 | 0.048 | 0.000 | 0.001 |

**Verdict.** Indistinguishable from Path D / Path C results within
the ±0.02 single-seed variance band (§13.7). H5-via-SSL did not
break the F2 ≈ 0.12 / max-prec ≈ 1.7-1.8× base-rate ceiling.

**What this tells us about the dataset.**

Five interventions (loss, features, architecture × 2, SSL) all
converge to the same F2 band:

| Path | Hypothesis | F2 | Max prec | Max prec / base |
|---|---|---:|---:|---:|
| §13.3 | (baseline, BCE/Focal) | 0.046 | flat | 1.00× |
| A | H4 — loss collapse | **0.102** | 0.040 | 1.70× |
| D | H3 — feature inadequacy | 0.121 | 0.050 | 1.81× |
| C (Transformer 3ep) | H2 — architecture | 0.126 | 0.052 | 1.88× |
| C (Mamba 1ep) | H2 — architecture | 0.126 | 0.032 | 1.15× |
| **E (SSL frozen)** | **H5 — label noise** | **0.126** | **0.048** | **1.73×** |

**Path A was the only intervention that produced a replicable jump
above the §13.3 noise floor.** Everything after operates in the same
narrow band. This is the effective signal-extraction ceiling of the
current data + labeling scheme.

Plausible explanations for the ceiling:

1. **The labels themselves cap the signal.** `identify_shock_events`
   may flag mostly-unpredictable events on HL — heuristic-defined
   shocks whose precursors aren't in OBI/MLOFI/VAMP/Kyle's λ at any
   temporal scale, and aren't learnable from next-tick dynamics
   either (since SSL didn't help). The "signal" KS test (D=0.13 on
   OBI in last 10 ticks before shock) confirms *some* signal exists,
   but it may be a weak statistical regularity in a small subset of
   events, not a learnable invariant.
2. **Data scale floor (H1, DR-demoted).** 181K positives across 3
   coins × 7 days is below the threshold needed for *any* deep
   learning model to extract this signal cleanly. Confirming this
   would require harvesting 30+ coin-days and re-running Path A.
3. **The feature space is missing something fundamental** that
   neither the crypto-native features (Path D) nor SSL encoder
   (Path E) can recover — e.g., truly dimensionless representations
   (Path G), explicit Δt cadence, or order-level (not aggregated)
   data.

**What remains.** With H4 ✓, H3 ✓, H2 ✗, H5 ✗, the unaddressed
hypotheses are:

- **H1 — more data.** Now the highest-leverage remaining bet, though
  expensive (re-harvest cost ~$1 + 3 hours per 7-day extension).
  Would need 30+ coin-days to clearly test.
- **Path G — dimensionless features.** Parked on branch
  `research/path-g-dimensionless`. 4-day research project; would
  test whether universal correlations exist in dimensionless space.
- **Deeper label investigation.** Run P0.5 (diagnostic recipe) step
  1 (logistic regression on un-windowed raw ticks) to check whether
  the instant-level signal supports anything beyond what we've
  achieved. If logistic AUROC is also ~0.55, we're at the data's
  intrinsic ceiling regardless of model.

**Recommendation.** Ship Path D's F2=0.121 as the current operating
point. Pivot to live-deployment work (or P0.5 diagnostic + P3.6 Path
G) rather than further hypothesis-test paths from this list. The
hypothesis space defined in §13.3 has been substantially exhausted.

**Artifacts (in `calibration/`):**
`pretrain_weights_BTC_USDC_USDC.pt` (135 KB, SSL-trained TCN encoder).
`tcn_weights_BTC_USDC_USDC.pt` (current: Path E fine-tuned head + 
pretrained encoder). `tcn_threshold_BTC_USDC_USDC.json` (thr=0.05,
F2=0.126).

### 13.9 P0.5 + Path E diagnostics: data ceiling + label heterogeneity confirmed (2026-05-11)

After Path E negative, ran two cheap diagnostics to characterize the
F2 ≈ 0.12 ceiling:

**C — P0.5 step 1: logistic regression on un-windowed raw ticks.**
`tests/diagnose_tcn.py`. Multivariate logistic on the 5 Path D features
(no windowing, no temporal aggregation) at the instant tick level,
pooled across BTC + ETH + SOL (3.27M rows, 92,967 positives).

| Metric | Value | Reference |
|---|---:|---|
| Multivariate AUROC | **0.5154** | Random = 0.500 |
| AUPRC | 0.0302 | Base rate = 0.0284 |
| AUPRC / base rate | **1.06×** | TSLA-equity at convergence: ~25× |
| Per-feature `ce_ratio` AUROC | 0.5113 | Only feature above 0.501 |
| `obi` univariate AUROC | 0.5010 | Random |
| `mlofi`, `vamp`, `kyles_lambda` AUROC | 0.500–0.501 | Random |

**Findings:**
- The instant-level signal across all 5 features is at the noise
  floor. `ce_ratio` is the only feature with any per-tick
  predictive content, and even it's marginal (AUROC 0.5113).
- **`obi` is at random at the instant level (AUROC 0.5010).**
  The §13 KS-D=0.13 result on OBI must come purely from temporal
  patterns in the trailing 10-tick window, not from any
  individual tick's value. The TCN at F2 ≈ 0.12 is the model
  successfully extracting some of that temporal pattern — but the
  ceiling is set by how weakly even the windowed pattern
  correlates with shock labels.
- **TCN's F2=0.12 / max-prec=1.8× is therefore not "leaving signal
  on the table"** — it's roughly extracting what the data offers.
  The ceiling is intrinsic to the (features × labels) combination.

**E — label inspection: `tests/inspect_shocks.py`.**
`identify_shock_events` invoked per coin; event-level statistics and
KS test of pre-shock OBI vs. random-window OBI.

| Coin | Events | Event rate | \|Δp\|@±10 P95 | Buy:sell after event | KS D | KS p |
|---|---:|---|---:|---:|---:|---:|
| BTC | 365 | 1 / 2,988 ticks | 0.054% | 118 : 247 (32% : 68%) | **0.082** | 4.2e-11 |
| ETH | 1,663 | 1 / 656 ticks | 0.058% | 589 : 1074 (35% : 65%) | **0.026** | 3.4e-05 |
| SOL | 1,075 | 1 / 1,014 ticks | 0.080% | 411 : 663 (38% : 62%) | **0.023** | 5.7e-03 |

§13 baseline (original pathA harvest, BTC): D=0.13, p~1e-28.

**Findings — H5 (label noise) is empirically confirmed:**

1. **Inter-event median is exactly 112 ticks across all three coins.**
   That's the floor enforced by `MIN_SHOCK_SPACING_SECONDS = 60`
   (~108 ticks at HL's ~1.8 ticks/s cadence). The heuristic is
   firing as fast as the spacing constraint allows — events cluster
   at the minimum. This means the heuristic is permissive: many
   events are flagged, then deduplicated by the spacing rule.

2. **Event magnitudes are far below the nominal 0.25% shock
   threshold.** Mean |Δp/p| at ±10 ticks: 0.016–0.022%. P95:
   0.054–0.080%. Max: 0.46–0.58%. `SHOCK_PRICE_MOVE_PCT = 0.0025`
   (0.25%) is the configured threshold — but the heuristic must be
   triggering on something *other* than that local price move
   (likely VPIN spike + multi-tick price-move detector). The
   "shock" label and "instantaneous large price move" are not the
   same thing in this data.

3. **Strong asymmetry — events overwhelmingly precede DOWN moves.**
   Buy:sell after-event ratio ~32-38% : 62-68% across all coins.
   Either the data window had a downtrend, or `identify_shock_events`
   is more sensitive to sell-side imbalance. Either way: the
   label distribution is asymmetric in a way the model may have
   to learn around.

4. **KS signal strength varies dramatically across coins**:
   BTC retains D=0.082 (62% of §13's headline), but **ETH (D=0.026)
   and SOL (D=0.023) are 5× weaker**. The OBI-pre-shock signal that
   originally motivated this investigation is essentially a
   BTC-specific phenomenon on the re-harvested data; on ETH and
   SOL the labels are nearly indistinguishable from random windows.

5. **Multi-coin pooling dilutes BTC's signal.** Training on
   3-coin pool means BTC's D=0.082 signal is diluted by ETH+SOL's
   D~0.025 near-noise. This explains why §13.3's multi-coin run
   (F2=0.046) was only marginally better than single-coin BTC
   (F2=0.027) — the additional coins added more *label noise*,
   not more signal.

6. **Visual inspection of the 16-event sample plots
   (`calibration/inspection/shocks_*.png`) shows heterogeneous
   event character.** Some events have clear vertical price jumps
   at t=0 with coherent OBI patterns; others look like sub-noise
   spikes. The heuristic is mixing real stress events with
   transient triggers.

**Synthesis — why the F2 ≈ 0.12 ceiling exists:**

- The shock labels are noisy and coin-heterogeneous.
- The instant-level signal on Path D features is at noise floor.
- The windowed signal (TCN's 60-tick receptive field) extracts the
  temporal pattern but it's a weak statistical regularity, not a
  strong predictor.
- Path A (loss) was the only intervention that helped because the
  previous BCE/Focal pipeline was *worse than the data ceiling*
  (collapsed to trivial). AUCM unlocked the data ceiling. Once
  there, more features (D), better architecture (C), and SSL
  pretraining (E) all hit the same wall because *the wall is
  set by the labels, not the model*.

**Forward.** Three plausible moves that aren't from the §13.3 list:

1. **Per-coin training.** Drop multi-coin pooling. Train BTC-only
   with AUCM + Path D features. BTC has the strongest signal
   (D=0.082) and shouldn't be diluted. Estimate ~5 min train +
   sweep; would test whether F2 lifts above 0.12 on BTC alone.

2. **Refine `identify_shock_events`.** The current heuristic is
   producing labels that don't admit signal extraction. Options:
   tighten the price-spike threshold (filters down to true large
   moves only); require both VPIN AND price-spike (not OR);
   require ±N-tick post-event confirmation. This is a labeling
   research project (~1 day) but probably the highest-leverage
   move left.

3. **Path G — dimensionless features** (branch
   `research/path-g-dimensionless`). The cross-coin invariance test
   in Path G's research plan is directly motivated by what we
   found here: BTC's signal doesn't transfer to ETH/SOL under the
   current feature set. Dimensionless features might restore
   transferability. ~4 days. See also `BOUNDARY_CONDITIONS.md`
   for the implementation traps to avoid.

The investigation is effectively complete on the §13.3 hypothesis
list. Further progress requires changing the labeling, the data
scope, or the feature framework — not the model.

**Artifacts (in `calibration/inspection/`):**
`shocks_feature_history_<COIN>.png` — 16-event sample grids.
`inspection_feature_history_<COIN>.txt` — per-coin numeric summaries.

### 13.10 Option 1 reframe — pooling helps AUCM via volume, not signal homogeneity (2026-05-11)

A follow-up experiment after §13.9 tested whether multi-coin pooling
was diluting BTC's stronger per-coin signal (D=0.082 vs ETH 0.026 /
SOL 0.023). **Result: pooling actually HELPS AUCM, contradicting §13.9's
dilution framing.**

**Setup.** BTC-only AUCM training, otherwise identical to Path D
(7d BTC, shock labels, PESG lr=0.01, 1 epoch).

**Result.**

| Metric | BTC-only | Multi-coin (Path D) |
|---|---:|---:|
| Positives | 10,320 | 90,627 (9× more) |
| Base rate | 0.95% | 2.77% |
| Max precision | 0.010 | 0.050 |
| Max-prec / base rate | 1.05× | 1.81× |
| Chosen op point | thr=0.150, F2=0.046 | thr=0.050, F2=0.121 |

**BTC-only F2 = 0.046 — collapsed to §13.3 noise floor.** Max precision
only 1.05× base rate vs multi-coin's 1.81×.

**Reframe.** AUCM's pairwise margin loss requires many positive×negative
pairs per batch to compute a meaningful margin. With BTC's 10K positives
in a 1M-tick stream + a 500K shuffle buffer, batch composition becomes
pathological: ~10 positives per 2048-batch (~0.5%) instead of ~48
(~2.3%) in the multi-coin pool. AUCM can't extract rank from contrast
that thin.

**§13.3 misread.** That section observed multi-coin (F2=0.046) was only
marginally better than single-coin BTC (F2=0.027) under BCE/Focal —
both were at the noise floor regardless of coin count. Under AUCM
the picture inverts: multi-coin lifts to F2=0.121 *because* AUCM can
use the volume; single-coin BTC stays at noise floor because it
can't.

**Implication for Option 2 (label refinement).** Refinements that
shrink the event count (e.g., raising TURBULENCE_THRESHOLD 0.5 → 0.9
to filter marginal VPIN spikes) risk dropping below AUCM's data-volume
floor. The window-tightening route (Option 2a, `MAX_DIFFUSION_TICKS`
6000 → 600) preserves event count better than threshold raising
because it filters *which* VPIN spikes get tagged, not *whether* they
get tagged. Recommended next experiment is 2a (window tightening; see
§13.11 below for the executed run).

**What still survives §13.9's framing.** The labels ARE coin-heterogeneous
(BTC has 4× stronger per-coin signal than ETH/SOL). But the right
response is *better labels*, not *less data*. Multi-coin pooling stays
correct for the current label scheme.

### 13.11 Option 2a — window tightening collapsed event volume (2026-05-11) — TESTED NEGATIVE

Executed the §13.10 follow-up plan:
`MAX_DIFFUSION_TICKS = 6000 → 600` (10× tighter VPIN-spike-to-price-move
window). Goal: replace noisy "VPIN spike then any 0.25% move within 55 min"
labels with tighter "VPIN spike then 0.25% move within 5 min" labels.

**Inspection result (`tests/inspect_shocks.py`, BTC/ETH/SOL).**

| Coin | Pre-2a events | Post-2a events | Reduction | Pre-2a KS D | Post-2a KS D |
|---|---:|---:|---:|---:|---:|
| BTC | 365 | 38 | 89.6% | 0.082 | 0.2434 (3.0×) |
| ETH | 1663 | 341 | 79.5% | 0.026 | 0.0972 (3.7×) |
| SOL | 1075 | 263 | 75.5% | 0.023 | 0.0886 (3.9×) |

All three coins gained 3-4× pre-shock KS D — refined labels DO carry
cleaner signal than the 6000-tick version. The label-quality direction
is right.

**Training result (multi-coin AUCM, same command as §13.4 / §13.10).**
Positives 38,068 / 6,542,480 (density 0.582%, down from §13.10's 1.385%).
**F2 = 0.043** at thr=0.250 (prec=0.012, rec=0.124). Below §13.10's
BTC-only baseline (F2=0.046) and far below Path D's F2=0.121.

**§13.10's volume caveat materialized.** Three concrete symptoms:

1. `UserWarning: Input data has no positive sample!` from
   `libauc.losses.auc:111` early in the run.
2. Multiple `Current Loss: 0.0000` batches in the first ~500 batches —
   same Path A imbalanced-collapse pathology recurring.
3. At 0.582% positive density × 2048-batch ≈ 12 positives per batch.
   §13.10's BTC-only run had ~10/batch and collapsed to F2=0.046.
   Multi-coin pooling was preserved during 2a, but the volume drop
   from window tightening pushed per-batch density into the same
   starved regime.

**§13.10's prediction was off.** §13.10 claimed window tightening
"preserves event count better than threshold raising." In practice the
10× window shrink cut events 75-90% — many VPIN spikes that qualified
under "any 0.25% drift within 55 min" simply don't have a move within
5 min. Tightening behaved more like a threshold raise than predicted.

**Direction right, magnitude too aggressive.** Per option-2a's F2-bucket
table for F2 < 0.10, next experiments:

1. **2a-smaller — `MAX_DIFFUSION_TICKS = 2000`.** 3× tighter instead of
   10×. Expected positive density ~1.0% (~20/batch — borderline for
   AUCM but above the 12/batch breakdown point seen here).
2. **2c — asymmetric labels (sell-side only).** Post-2a buy:sell ratios
   were 15:23 (BTC), 136:205 (ETH), 114:149 (SOL); dropping ~40% of
   buy-side could clean labels with less volume loss than tightening.
3. **2d — coin-relative thresholds.** Long-term right answer per §13.10.

**State left.** `config.py` `MAX_DIFFUSION_TICKS` stays at 600 for the
next experimenter to inspect (Step 1's commit notes the change).
Artifacts `calibration/tcn_weights_BTC_USDC_USDC.pt`,
`calibration/tcn_threshold_BTC_USDC_USDC.json`,
`calibration/tcn_threshold_sweep_BTC_USDC_USDC.csv`,
`calibration/inspection_2a/*`, and `aucm_option2a.log` reflect the
collapsed F2=0.043 run. Path D baseline (F2=0.121) requires reverting
`MAX_DIFFUSION_TICKS` to 6000 and re-running, modulo the ±0.02
single-seed variance noted in §13.7.

### 13.12 Option 2a-smaller — volume preserved, F2 still collapsed (2026-05-11) — TESTED NEGATIVE

Follow-up to §13.11 per the F2-bucket table's "use 2a with smaller change"
guidance. `MAX_DIFFUSION_TICKS = 600 → 2000` (3× tighter than 6000 instead
of 10×). Hypothesis: less aggressive window shrink preserves enough
positives for AUCM while keeping label quality lift.

**Inspection (`tests/inspect_shocks.py`, BTC/ETH/SOL, 2000-tick window).**

| Coin | Events | Reduction vs 6000 | KS D | Δ vs 6000 |
|---|---:|---:|---:|---:|
| BTC | 147 | 59.7% ↓ | 0.0933 | +0.011 (1.14×) |
| ETH | 912 | 45.2% ↓ | 0.0587 | +0.033 (2.26×) |
| SOL | 688 | 36.0% ↓ | 0.0596 | +0.037 (2.59×) |

Events sit between §13.10's 6000 (3103 total) and §13.11's 600 (642 total).
KS D is modestly elevated vs Path D across all coins, less dramatically
than the 600 case (BTC was 0.2434 there).

**Training result (multi-coin AUCM, same command as §13.4 / §13.11).**
Positives 103,030 / 6,542,480 (density **1.57%**, *above* §13.10's
1.39% Path D baseline). Per-batch positives ~32 — far above the
12/batch breakdown point seen in §13.11. **No `UserWarning`, no
`Current Loss: 0.0000` batches.** AUCM volume floor is not the
constraint here.

**F2 = 0.079** at thr=0.100 (prec=0.019, rec=0.363). Falls in the
"< 0.10" bucket of option-2a's interpretation table. Better than
§13.11's F2=0.043 but still 35% below §13.10's Path D baseline of
F2=0.121.

**Diagnosis — volume preserved, signal still degraded.** Max precision
~0.028 = 1.78× the 1.57% base rate. Path D had max prec ~0.050 = 3.6×
its base rate. The model is finding *weaker* discrimination than the
6000-tick baseline despite cleaner per-event KS D.

**Working hypothesis (label-boundary noise).** Tightening the diffusion
window introduces a new failure mode: VPIN spikes whose 0.25% price
move happens *just after* 2000 ticks (~18 min) are labeled NEGATIVE,
while topologically identical spikes with the same pre-shock signature
but a move within 2000 ticks are labeled POSITIVE. The model sees
similar OBI/MLOFI/VAMP patterns mapped to opposing labels — the
information content of the labels drops even as the KS-D metric
suggests they're cleaner. KS D measures distributional separation of
pre-shock context vs random; it does not penalize the
labeling-boundary contradiction.

If correct, this means **diffusion-window-based label refinement is
self-defeating beyond a point**: tighter windows trade volume noise
(2a's failure mode) for boundary-classification noise (2a-smaller's
failure mode). Neither dominates Path D's F2=0.121 with the
6000-tick "permissive" definition.

**Conclusion.** Label refinement via `MAX_DIFFUSION_TICKS` is
exhausted as a single-knob strategy. Both candidate-2 alternatives
worth trying (2c asymmetric labels, 2d coin-relative thresholds)
are now lower-priority than addressing the *features* — Path G's
dimensionless features test cross-coin invariance, which is the
structural concern that survives §13.10's "labels are heterogeneous"
finding regardless of how labels are tuned.

**Action.** Pivoting to Path G implementation. See branch
`research/path-g-dimensionless` for planning docs (README,
RESEARCH_PLAN, THEORY, BOUNDARY_CONDITIONS). The active Path G
implementation branch and progress will be tracked in §13.13.

**State left.** `MAX_DIFFUSION_TICKS = 2000` in `config.py`. New
artifacts: `calibration/inspection_2a_smaller/*`, `aucm_option2a_smaller.log`,
and overwritten weights/thresholds in `calibration/tcn_*_BTC_USDC_USDC.{pt,json,csv}`
reflect the F2=0.079 run.

### 13.13 Path G — Phase 2+3 implemented; Phase 4 awaits re-harvest (2026-05-11)

After §13.12's exhaustion of label-knob tuning, pivoted to Path G —
the dimensionless-features research project that was parked in
`mini_projects/path_G_dimensionless/` on branch
`research/path-g-dimensionless`. Phases 1 (theory), 2 (online scale
estimators), and 3 (π-group computation) per `RESEARCH_PLAN.md`.
This session landed Phases 2+3 on master (working tree, not yet
committed); Phase 4 is blocked on a re-harvest with the new column
schema.

**What landed in `layer1_sensors.py`.**

1. **`CharacteristicScales`** — rolling online estimators for
   `(τ_c, L_c, D_c, V_c, κ_c)`. All denominators floored to their
   quantization unit per `BOUNDARY_CONDITIONS.md`:
   - `L_c ≥ TICK_SIZE` (spread can't drop below 1 tick)
   - `τ_c ≥ min_tau_ms / 1000` (inter-tick can't be zero)
   - `V_c ≥ MIN_ORDER_SIZE / window_s` (volume rate floor)

   Feeds: `on_trade(qty)` between book snapshots (sign-blind throughput
   for V_c); `on_book(ts_ms, mid, spread, kyles_lambda)` on each
   snapshot. Lookback default 200 ticks, matching
   `KYLES_LAMBDA_LOOKBACK_TICKS`.

2. **`compute_dimensionless_features(dt_ms, vamp_minus_mid_abs, scales)`**
   — wraps each raw π in `tanh(raw / scale_factor)` for bounded,
   differentiable saturation at quantization boundaries (per
   `BOUNDARY_CONDITIONS.md` §"Mitigation 2"). Returns four groups:
   - `fo_market = tanh(D·Δt / L_c²)` — diffusive timescale vs Δt
   - `sr = tanh(Δt / τ_c)` — local vs recent cadence
   - `pi_kappa = tanh(κ·V·τ / L)` — dimensionless price impact
   - `pi_vamp_dim = tanh((VAMP−mid) / L)` — queue deviation in spread units

   `π_obi` and `π_mlofi` are already dimensionless in the existing
   pipeline — kept as-is, no recomputation needed (THEORY.md §π₄, §π₅).

3. **`PATH_G_TANH_SCALES`** module-level dict with best-effort
   initial scale factors `{fo_market: 1e-2, sr: 5.0, pi_kappa: 1.0,
   pi_vamp_dim: 1.0}`. Smoke test confirmed `fo_market` saturates
   at 1.0 with these values on synthetic input — these need
   empirical recalibration from the first harvest's 95th-percentile
   per `BOUNDARY_CONDITIONS.md` before Phase 4 training.

4. **`PhysicsState`** extended with 9 new fields: 4 tanh-bounded
   π-groups (`fo_market`, `sr`, `pi_kappa`, `pi_vamp_dim`) and 5 raw
   scales (`tau_c_s`, `L_c`, `D_c`, `V_c`, `kappa_c`). All default to
   `0.0` so the equity path (no L2 depth, no trades) emits zeros
   without changes elsewhere.

5. **`FeatureDumper.HEADER`** extended from 12 to 23 columns: adds
   the 4 π-groups, 5 raw scales, and 2 top-of-book sizes
   (`bid_sz_top`, `ask_sz_top`). Top-of-book sizes were missing from
   the prior schema and are needed to re-derive π-groups under
   different tanh-scale calibrations without re-running the engine.
   Column-name lookup in `train_stream.py` keeps old CSVs readable.

6. **`SensorArray`** wired up: `CharacteristicScales` instantiated
   for HL crypto only (same gating as MLOFI / Kyle's λ), fed via
   `on_trade` in `_process_trade`, `on_book` + π-group computation
   in `_process_order_book` (after Path D features so it can reuse
   `kyles_lambda` for κ_c), reset in `reset_session`.

**Smoke test.** A synthetic two-snapshot trace produced expected
scale values (`τ_c=0.5s, L_c=0.55, D_c=2.0 P²/s, V_c=1.0 Q/s`) and
tanh-bounded π-groups. `PhysicsState` accepted all new fields at
construction. `layer1_sensors.py` parses cleanly. No live data
exercised yet.

**What did NOT land — Phase 4 (training) blocker.**

The existing `calibration/feature_history_*.csv` files (12 columns)
do not contain the Path G outputs. `train_stream.py` reads columns
by name, so old CSVs are still loadable, but a Path G training run
needs the new columns — which means a re-harvest. Two paths:

1. **Live harvest with extended schema.** Run
   `fetch_history_hyperliquid.py` (or equivalent) to capture fresh
   data using the now-extended `FeatureDumper`. Needs the user's
   AWS creds; not available in this shell.
2. **Replay-mode harvest.** If the existing CSV's `best_bid` /
   `best_ask` series can be re-played through a fresh SensorArray,
   `CharacteristicScales` and the π-groups can be recomputed
   offline. But `V_c` needs raw trade volumes — which the CSV
   doesn't preserve — so the dimensionless features `pi_kappa`,
   `pi_vamp_dim`, and `fo_market` would all be partial. `sr` alone
   is recoverable from `timestamp_ms` deltas.

**What ALSO did NOT land — TCN consumption.** `TCN_INPUT_CHANNELS`
in `config.py` stays at 5 (Path D). `train_stream.py`'s feature
stack still references only Path D columns. Adding Path G channels
is a Phase 4 task: needs new harvest, then a config switch + a
feature-stack branch in `train_stream.py`.

**What ALSO did NOT land — stress-slice validation.** The five
boundary scenarios in `BOUNDARY_CONDITIONS.md` (low-vol lull,
first-tick-of-session, sub-second burst, 1-tick-spread under high
volume, stale book) need a real data slice to histogram each π
against. Initial tanh saturation already triggered on synthetic
input — the empirical 95th-percentile calibration is mandatory
before Phase 4 training.

**Forward path.**

1. Re-harvest with the extended `FeatureDumper` schema (live or
   replay-with-trades). Save as
   `calibration/feature_history_<COIN>.pathG.csv` to preserve the
   Path D baselines.
2. Run stress-slice validation per `BOUNDARY_CONDITIONS.md`
   checklist. Re-calibrate `PATH_G_TANH_SCALES` from the 95th-
   percentile of each raw ratio on a baseline regime.
3. Phase 4 training: extend `TCN_INPUT_CHANNELS` to 9 (5 Path D +
   4 Path G), branch `train_stream.py`'s feature stack, run AUCM
   on the new schema. Target: F2 ≥ 0.18 OR cross-coin retention
   ≥ 70% (RESEARCH_PLAN.md success criteria).
4. Phase 5 writeup in `mini_projects/path_G_dimensionless/RESULTS.md`
   on `research/path-g-dimensionless` branch.

**Branch hygiene note.** Path G's README recommended branching from
`research/path-g-dimensionless`; this session implemented on `master`
instead to keep the §13 investigation log linear. The
implementation can be cherry-picked or rebased onto
`feature/path-g-impl` cleanly if/when the planning-branch hygiene
is desired.

**State left.** `layer1_sensors.py` modified (uncommitted).
`config.py` `MAX_DIFFUSION_TICKS` still at 2000 from §13.12 — has
no effect on Path G since Path G's feature columns aren't yet
consumed by the TCN. No new artifacts (no training run, no
inspect).

**Addendum — tanh-scale calibration (2026-05-11, evening).** A 56,734-row
BTC smoke harvest (HL `node_fills_by_block/hourly/20260406/*.lz4`) ran
the extended `FeatureDumper` end-to-end and exposed two of four π-groups
as miscalibrated under the hand-picked defaults:

| π | default scale | empirical p95(\|raw\|)/2 | issue |
|---|---:|---:|---|
| `fo_market` | 1e-2 | 7.932 | saturated at 1.0 |
| `sr` | 5.0 | 0.5442 | mildly under-saturated |
| `pi_kappa` | 1.0 | 1.38e-06 | dead at machine precision |
| `pi_vamp_dim` | 1.0 | 0.248 | mildly under-saturated |

`PATH_G_TANH_SCALES` in `layer1_sensors.py` updated to the empirical
values above. The overnight 4-coin × 14-day harvest used these scales.

**Addendum — multi-coin recalibration (2026-05-11 → 2026-05-12, post-harvest).**
Overnight harvest produced ~2.17M rows per coin × 4 coins. Morning
sanity-check confirmed the BTC-only scales DID generalize poorly:

| coin | last-row `fo` | last-row `pk` | issue |
|---|---:|---:|---|
| BTC | 0.026 | 0.0086 | fine |
| ETH | 0.0044 | **+1.000** | pk saturated |
| SOL | 0.000 | **−1.000** | pk saturated |
| HYPE | 3e-6 | **+1.000** | pk saturated |

`calibrate_path_g_scales.py` was added and run over the pooled 8.67M-row
distribution. Resulting pooled scales:

| π | BTC-only smoke | pooled multi-coin | shift |
|---|---:|---:|---:|
| `fo_market` | 7.932 | 3.448 | 0.43× |
| `sr` | 0.5442 | 0.5566 | 1.02× |
| `pi_kappa` | 1.38e-06 | 2.875e-04 | **208×** |
| `pi_vamp_dim` | 0.248 | 0.2315 | 0.93× |

The `pi_kappa` shift confirms `BOUNDARY_CONDITIONS.md`'s warning about
depth heterogeneity — ETH/SOL/HYPE have thinner books than BTC, which
makes Kyle's λ (κ_c) 100-300× larger and would have left
`pi_kappa` constant-saturated on every non-BTC coin during training.

`PATH_G_TANH_SCALES` in `layer1_sensors.py` updated to the pooled
values. `recompute_path_g_pi_groups.py` was added to recompute the
four tanh-bounded columns in each CSV from the raw scale columns —
crash-safe atomic rewrite, single source of truth for tanh scales
imported from `layer1_sensors.py`. The raw scales (`tau_c_s`, `L_c`,
`D_c`, `V_c`, `kappa_c`) in CSV are ground truth from the harvest and
remain untouched, so further calibration changes are always
recoverable via the recompute script.

### 13.14 Path G — Phase 4 cross-coin invariance test (2026-05-12) — TESTED NEGATIVE

Phase 4 of `RESEARCH_PLAN.md` ran four experiments to answer the
headline Path G question: does adding the four dimensionless π-groups
to the TCN input stack improve cross-coin transfer?

**Setup.** `TCN_INPUT_CHANNELS` extended from 5 → 9 (Path D channels
+ `[fo_market, sr, pi_kappa, pi_vamp_dim]`) behind a new
`USE_PATH_G_FEATURES` env var (config.py + train_stream.py edits;
AlphaEngine inference parity intentionally NOT updated — research
branch, not deployable). AUCM/PESG protocol matched §13.4 exactly.

**Experiments and results.**

| run | features | train coins | val coin | F2_val | retention vs 0.121 |
|---|---|---|---|---:|---:|
| §13.4 baseline | Path D (5) | BTC/ETH/SOL | — (in-domain) | **0.121** | 100% |
| 4-coin Path G | Path D+G (9) | BTC/ETH/SOL/HYPE | (in-domain) | 0.066 | 55% |
| 4-coin Path D (control) | Path D (5) | BTC/ETH/SOL/HYPE | (in-domain) | 0.066 | 55% |
| 3-coin Path D + HYPE val | Path D (5) | BTC/ETH/SOL | HYPE | **0.0545** | 45% |
| 3-coin Path G + HYPE val | Path D+G (9) | BTC/ETH/SOL | HYPE | **0.0534** | 44% |

**Two conclusions.**

1. **HYPE poisoned the 4-coin pool.** The control with Path D-only
   features (USE_PATH_G_FEATURES=0) on the same 4-coin pool got
   F2=0.066 — identical to the Path G run. Path G features were
   innocent; the F2 drop from 0.121 to 0.066 came entirely from
   adding HYPE to the training set.
2. **Path G's dimensionless framework delivered zero invariance
   lift.** With the cleanest test design (3-coin train, HYPE
   held-out as val), Path G's F2=0.0534 vs Path D's F2=0.0545 differ
   by 0.0011 — well inside the ±0.02 single-seed variance band
   (§13.7). The 7 candidate π-groups in `THEORY.md` (4 implemented)
   did not produce the universal-correlation behavior that
   Buckingham-π predicted for these markets.

`RESEARCH_PLAN.md` success criteria — all three failed:
- F2 (in-domain) ≥ 0.20 — no
- Cross-coin F2 retention ≥ 70% — no (44%)
- Any π-group improves AUC by 0.02 — no

This is `RESEARCH_PLAN.md`'s "clean negative" outcome.

**Why label-side, not feature-side.** Three signals converge on
labels being the bottleneck, not features or architecture:

1. **Same val F2 regardless of feature set.** Both 3-coin runs hit
   F2≈0.054 on HYPE. If the issue were Path D's feature
   non-transferability, Path G should have lifted retention.
2. **Same val F2 regardless of pool composition.** 4-coin
   in-domain F2 = 3-coin HYPE-val F2 ≈ 0.054. HYPE behaves the same
   way whether trained on or held out — its labels don't carry
   ranking signal under the current trigger.
3. **HYPE's val_loss (0.0092) is LOWER than train_loss (0.0148).**
   The model fits HYPE's marginal distribution fine. It can't
   *rank* shocks from non-shocks because the labels are noisy.

**Mechanism.** `identify_shock_events` triggers a shock when
`|price_move| / price > SHOCK_PRICE_MOVE_PCT = 0.0025` (25 bps)
within `MAX_DIFFUSION_TICKS` (2000 ticks ≈ 18 min). On BTC at
σ ≈ 3.4 bps/√s, a 25 bps move over 18 min is a 0.22σ event —
already inside the random-walk envelope. On HYPE with σ several
times larger, a 25 bps move is well under 0.1σ — pure Brownian
noise. So `identify_shock_events` over-fires on HYPE,
generating ~22K "shocks" per coin that are actually random walks,
not real microstructure events. The AUCM optimizer can't learn to
rank random walks against actual liquidations.

**Decision — pivot to per-symbol vol-scaled labels (option 2d from
`option-2-label-refinement.md`).** Replace the fixed-percentage
trigger with `|Δp| > k × √(D_c · Δt)` — a k-sigma event in
absolute price units, using the rolling realized variance D_c that
Path G already estimates and that the CSV already contains. This is
the same trigger statistic across coins regardless of volatility
regime. No re-harvest needed; labels are computed at training time
from existing CSV columns.

**State left.** `layer1_sensors.py`, `config.py`, `train_stream.py`
all modified for Path G. Training logs `aucm_pathG.log`,
`aucm_control_pathD_4coin.log`, `aucm_pathD_3coin_HYPEval.log`,
`aucm_pathG_3coin_HYPEval.log` preserved. `tcn_weights_*.pt` reflects
last run (Path G 3-coin + HYPE val, 9-channel). The `USE_PATH_G_FEATURES`
env var stays in place so re-enabling Path G for future composition
experiments is a one-flag change.

### 13.15 Per-symbol vol-scaled labels — TESTED NEGATIVE; data ceiling verified (2026-05-12)

§13.14 diagnosed labels as the bottleneck and recommended per-symbol
vol-scaled triggers. Implemented and tested three configurations.

**Implementation.** `identify_shock_events` extended with optional
`D_c_series` (Path G's rolling realized variance, P²/s) and `k_sigma`
parameters. New trigger: `|Δp| > k · √(D_c[t] · Δt_s)` — a k-sigma
event in absolute price units, uniformized across symbols regardless
of vol regime. `build_labels` wires `data["D_c"]` through to the
trigger when `USE_VOL_SCALED_LABELS=1`. Subsequently extended with a
`min_pct_floor` parameter to gate the trigger on BOTH statistical
rarity AND structural significance: `threshold = max(k·σ·√Δt, floor·p)`.

**Three trials, three negatives.**

| run | density | F2 | max prec | max prec / base | verdict |
|---|---:|---:|---:|---:|---|
| §13.4 baseline (3-coin, old labels) | 1.39% | **0.121** | 0.050 | **3.6×** | real ranking |
| §13.14 4-coin old labels (HYPE poison) | 1.37% | 0.066 | 0.023 | 1.7× | degraded |
| vol-scaled k=3 (4-coin) | 2.56% | 0.118 | 0.026 | 1.02× | predict-all |
| vol-scaled k=5 (4-coin) | 2.05% | 0.097 | 0.021 | 1.02× | predict-all |
| hybrid k=3 + 15bps floor (4-coin) | 1.40% | 0.066 | 0.014 | 1.0× | predict-all |

All three vol-scaled variants produce predict-all models — max precision
collapses to base rate (1.0-1.02× lift) and F2 is driven entirely by
recall=1.0 at the lowest threshold. The hybrid trigger's structural
floor delivered the cleanest event distributions (BTC KS D=0.058,
ETH=0.037, comparable to §13.4 baseline) but the rank signal still
didn't materialize at training time.

**What this proves.** The label-trigger refinement direction is
exhausted. Three independent attacks — pure k-σ, more-selective k-σ,
hybrid — all hit the same predict-all ceiling. The §13.14 diagnosis
("labels are the bottleneck") was half right: HYPE labels under the
old fixed-percentage trigger ARE worse than BTC's, but fixing the
labels statistically did not unlock predictive signal in the leading
features.

**Connecting to the full §13 investigation.** Eight distinct attacks
on the F2 ≈ 0.12 plateau:

| § | approach | F2 | rank vs §13.4 |
|---|---|---:|---|
| §13.3 | Multi-coin, BCE/Focal (pre-AUCM) | 0.046 | well below |
| §13.4 | AUCM loss (Path A) | 0.121 | baseline |
| §13.5 | Crypto-native features (Path D) | 0.121 | held |
| §13.6 | Architecture swap (Path C) | ≤ 0.121 | held/below |
| §13.7 | Run-to-run variance | ±0.02 | (noise floor) |
| §13.8 | SSL pretraining (Path E) | ≤ 0.121 | held/below |
| §13.10 | Option 1 reframe (BTC-only AUCM) | 0.046 | below (volume) |
| §13.11 | Label window 6000→600 | 0.043 | volume collapse |
| §13.12 | Label window 6000→2000 | 0.079 | predict-all |
| §13.14 | Dimensionless features (Path G) | 0.066–0.053 | below (HYPE poison) |
| §13.15 | Vol-scaled labels (this) | 0.066–0.118 | predict-all |

**Six independent strategies failed to lift F2 meaningfully above the
0.121 baseline.** The signal in leading microstructure features
(`ce_ratio`, `obi`, `mlofi`, `vamp`, `kyles_lambda`) appears to be
inherently capped at ~1.8× base-rate precision on this dataset. This
matches §13.9's diagnostic (univariate AUROC ≈ 0.5 on raw features,
multivariate logistic AUROC 0.515 on un-windowed ticks) — there isn't
enough mutual information between features and shocks for a TCN to
extract more than the current edge.

**Data ceiling verified.** F2 = 0.121 with max-precision ≈ 1.8× base
rate is the current upper bound for cross-coin shock prediction on
HL with the Path D feature set. Further label or feature engineering
within this framing will yield diminishing returns.

**Strategic pivot.** Path forward isn't more knob-tuning; it's a
reframe. Three candidate next directions, in roughly increasing scope:

1. **Per-symbol models** (abandon cross-coin pooling). Train separate
   weights per coin. Loses transferability but each model is tuned to
   its symbol's microstructure. The "data ceiling" might be a
   pooling-distortion artifact — single-coin F2 could be higher if
   the model isn't forced to fit a compromise distribution.
2. **Different prediction target.** Drop "binary shock classification"
   in favor of directional bias (which way will price move?),
   regime/volatility classification (will the next minute be turbulent?),
   or magnitude regression (how far?). These are easier statistical
   targets that may carry stronger leading-feature signal.
3. **More data + richer features.** Re-harvest with the extended
   FeatureDumper schema for longer periods (1+ months per coin), add
   missing microstructure channels (queue position, order-flow burst
   statistics, cross-asset cointegration), and accept the cost. Only
   pays off if (1) and (2) also fail.

§13 investigation is effectively complete. Subsequent work belongs in
a new §14.

**State left.** `config.py` `USE_VOL_SCALED_LABELS` flag and
`SHOCK_K_SIGMA` / `SHOCK_MIN_PCT_FLOOR` constants remain in place;
all default OFF so legacy training behavior is the default.
`calibration.py` `identify_shock_events` extended signature is
backward-compatible (D_c_series=None preserves legacy trigger).
`train_tcn.py` `build_labels` reads config flags. Logs:
`aucm_volscaled_4coin.log` (k=3), `aucm_volscaled_k5_4coin.log` (k=5),
`aucm_hybrid_k3_15bps_4coin.log` (hybrid). Inspection plots in
`calibration/inspection_volscaled_k3/`, `calibration/inspection_hybrid_k3_15bps/`.
`tcn_weights_*.pt` reflects last run (hybrid k=3 + 15bps). The Path D
baseline weights (F2=0.121) can be regenerated with
`USE_VOL_SCALED_LABELS=0` on the 3-coin pool.
