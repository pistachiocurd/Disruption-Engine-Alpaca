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

Dedicated research session planned (see plan file
`~/.claude/plans/tcn-failure-investigation.md`) to characterize the
root cause more rigorously than this progression of ad-hoc experiments.
