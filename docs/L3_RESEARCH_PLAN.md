# L3 (Order-by-Order) Microstructure — Research Plan

> Status: Planning. Created 2026-05-12 after L2 closure
> ([LAYER2_TRAINING.md §14.7-§14.8](LAYER2_TRAINING.md)).
> Successor to the L2 investigation; this document defines the architecture,
> data acquisition, phased work, and decision points for L3.

## 1. Context — Why pivot

§14.7's Phase A backtest definitively closed the L2 feature thesis:
- L2 features (CE ratio, OBI, MLOFI, VAMP, Kyle's λ) carry real directional
  signal — ~55% paper accuracy at H=100 ticks, +10pp over random.
- That signal MAGNITUDE is 0.06 bps gross edge per trade vs a 5.5 bps
  round-trip execution cost. The alpha doesn't survive any reasonable
  execution framework.
- Eight refinement strategies (§13.6-§13.15) all hit the same ceiling.
  Architecture, loss, labels, dimensionless reformulation — none lifted
  the floor.

The diagnosis: **L2 snapshots aggregate away the dynamics that PRECEDE
book-state changes.** OBI sees the state of the book; it does not see
*how the book is being assembled*. Queue depletion, cancellation cascades,
order lifespan, hidden-order inference — these are L3 events, invisible
in L2.

Academic + industry literature consistently identifies L3 as the layer
where leading microstructure signal lives: Cont & Kukanov (2017) on
optimal order placement, Gould et al. (2013) limit-order-book review,
Bouchaud-Bonart-Donier-Gould (2018) on HFT scaling laws,
Cartea-Jaimungal-Penalva (2015) algorithmic and HFT. These are the
references Path G's THEORY.md already cited; the dimensional analysis
was *correct* — we just applied it at the wrong data layer.

## 2. Thesis — What L3 carries that L2 destroys

L3 ("market-by-order") sees individual events:

| Event | L3 sees | L2 sees | What L2 loses |
|---|---|---|---|
| Order arrival | Side, price, size, order_id, timestamp | Δ in book level | Order identity, lifespan, queue position |
| Cancellation | order_id removed before fill | Δ in book level | *Why* the level shrunk |
| Modification | Replace event with new size/price | Δ in book level | Adverse signal: traders updating in response to flow |
| Trade | Which specific resting order(s) got hit | Aggregate trade print | Aggressor identity by queue position |
| Hidden order | Price improvement without explicit order | Nothing | Hidden liquidity provider activity |

Candidate L3-only features (initial list, to be refined in implementation):

1. **Cancellation velocity per side** — `cancel_rate_bid / cancel_rate_ask`.
   Cascading cancels precede shocks (informed traders pulling liquidity).
2. **Order arrival rate imbalance** — `arrival_rate_bid - arrival_rate_ask`,
   signed flow at arrival not trade.
3. **Queue depletion velocity at top level** — how fast best_bid_size is
   shrinking via cancels (not fills).
4. **Mean order lifespan per side** — short-lived orders indicate
   high-frequency spoof/withdraw behavior.
5. **Trade-aggressor sequence autocorrelation** — runs of buy-aggressor
   trades suggest momentum / informed flow.
6. **Hidden-order inference** — count of trades at price levels with no
   visible resting orders. Big at fragmented venues.
7. **Order-flow imbalance at deeper levels** — events at top-K levels not
   just touch.
8. **Cancellation-to-fill ratio** — per side, indicates spoofing.

These are *first-pass candidates*. The actual feature set comes out of
Phase 2 (sensor implementation) once we see real distributions.

## 3. Data acquisition — venue options

The critical first decision is which venue to capture L3 from. Options
ordered by tractability:

| Venue | L3 quality | Cost | Archive | Live capture | Notes |
|---|---|---|---|---|---|
| **Coinbase Advanced** | Order-by-order | Free | None | Yes (WS) | Capture forward; clean schema |
| **Binance** | Book delta stream | Free | Yes (paid tick data) | Yes (WS) | Reconstruct L3 from deltas; complex |
| **Hyperliquid** | L2 only (verified) | Free | Yes (S3) | N/A | Out — L2 is what we just exhausted |
| **Kraken** | Full L3 WS | Free | None | Yes | Smaller venue; thinner data |
| **Databento** | Full MBO (industry standard) | $$$ | Yes | Yes | Clean format; futures + US equities |
| **Polygon** | L3 with WAQ for equities | $$ | Yes | Yes | Good US equity option |

**Decision (2026-05-12): paid historical vendor (Databento or Tardis).**

The strategic risk is finding out whether L3 carries enough alpha to
beat the 0.06 bps L2 ceiling. A live capture forces a 7-14 day wait
before any feature engineering can start; a paid historical MBO slice
removes that bottleneck and lets Phase 2 begin tomorrow.

- *Primary*: Databento (clean MBO format, industry standard for
  this kind of research)
- *Secondary*: Tardis.dev (cheaper for crypto, similar MBO depth)
- *Fallback*: Coinbase Advanced WS live capture (only if paid vendor
  data is genuinely inaccessible)

The cost discipline: 7 days × 3 coins of MBO data is the smallest
slice that can statistically prove a >5 bps predictive edge in
cancellation velocity / queue depletion. Expand the window only if
the 7-day proof-of-concept shows green.

## 4. Architectural changes

L3 is a different data layer; significant codebase work required.

### 4.1 Data layer

New file: `fetch_history_coinbase_l3.py`
- Subscribe to Coinbase Advanced WS `level3` channel for BTC-USD, ETH-USD,
  SOL-USD
- Persist events as one row per event (not per snapshot)
- New schema: `timestamp_ms, event_type, order_id, side, price, size,
  remaining_size, is_aggressor`
- Output: `calibration/l3_events_<COIN>.csv` (rotating daily files;
  raw event log)

### 4.2 Sensor layer

New file: `layer1_l3_sensors.py` (or extend `layer1_sensors.py`)

New sensor classes that consume `OrderEvent` objects:
- `OrderBookReconstructor` — maintains full L3 book state from events;
  emits L2-like snapshot on demand for backward compat
- `OrderArrivalRateTracker` — rolling per-side arrival rate
- `CancellationVelocityTracker` — rolling per-side cancel rate +
  cancel/fill ratio
- `OrderLifespanTracker` — distribution stats over rolling window
- `HiddenOrderDetector` — count of trades at price levels with no
  matching arrival event since last clear
- `QueueDepletionTracker` — top-level size decay rate
- `AggressorSequenceTracker` — autocorrelation of trade aggressor side

All sensors output scalar features per-tick (where "tick" is now defined
as either a wall-clock interval or an event-count interval — TBD in
Phase 2).

### 4.3 Feature aggregation

The L3 sensors emit instantaneous values per event. The TCN needs
per-tick samples on a regular grid. Two aggregation modes to test:

- **Wall-clock**: emit feature row every N ms (e.g., 500 ms)
- **Event-count**: emit feature row every N events (e.g., 100)

Wall-clock is the natural extension of the L2 pipeline. Event-count
adapts to activity bursts. Both worth testing.

New extended `FeatureDumper` writes a row per aggregation interval with
the L3-derived features + reconstructed L2 features (OBI, MLOFI etc.
recomputed from the L3 book for cross-comparison).

### 4.4 Model layer

`config.py`:
- New constants `TCN_L3_INPUT_CHANNELS` (target: 10-15 channels)
- Mode flag `LAYER1_MODE = "l2_snapshots" | "l3_events"` (env-overridable)

`train_stream.py`:
- Feature stack branch on `LAYER1_MODE`
- L3 mode: stack the new L3 features (plus optionally a subset of
  reconstructed L2 features for comparison)
- Reuses everything else (BCE/AUCM, val protocol, threshold sweep,
  directional/shock label-source)

`backtest_directional.py`:
- No change needed; runs over any feature CSV the trained model can
  consume

### 4.5 Branch strategy

`research/path-h-l3` (next research-branch letter after Path G).
Mirror Path G's discipline: planning files immutable on the research
branch; implementation on `feature/path-h-impl` forked from it.

## 5. Phased plan

Conservative estimates; ambitious estimates in parentheses.

### Phase 1 — Data acquisition (1 day, paid historical slice)

- Purchase 7-day MBO slice from Databento (primary) or Tardis.dev
  (secondary) for BTC/ETH/SOL
- Inspect vendor schema; map to canonical `OrderEvent` format
  (timestamp_ms, event_type, order_id, side, price, size, aggressor_side)
- Wire vendor-specific parser in `aggregate_mbo_events.py`
- Verify event counts, type distribution per coin, ordering integrity
- **Deliverable**: raw vendor file + canonical event stream replayable
  through the aggregator

### Phase 2 — Sensor implementation + replay (3 days / 2 days)

- Implement 6-7 L3 sensors in `layer1_l3_sensors.py`
- Implement `OrderBookReconstructor` (most complex piece — full L3 book
  state maintenance from events)
- Replay captured L3 stream through sensors, build feature CSV
- Unit tests for each sensor against known fixtures
- **Deliverable**: feature CSV with ~10 L3-derived columns plus
  L3-reconstructed L2 columns for cross-comparison

### Phase 3 — Distribution check + harvest (1 day)

- Run distributional checks per coin: are sensors active? saturated?
  dead?
- Apply boundary mitigations (tanh saturation, denominator floors)
  matching Path G's `BOUNDARY_CONDITIONS.md` discipline
- Sanity-check pre-shock KS-D on L3 features (looking for the 0.15+
  values that L2 features couldn't deliver outside artifacts)
- **Deliverable**: cleaned per-coin feature distributions, calibration
  constants

### Phase 4 — Training + comparison (2 days / 1 day)

- Train TCN at L3 mode on 80/20 time-split
- Directional target at H=100 (matches L2 directional comparison)
- Compare:
  - Val F2 vs L2 baseline F2=0.106
  - Val directional gross edge per trade vs L2 0.06 bps
  - Per-feature ablation: which L3 features carry signal?
- **Deliverable**: `LAYER3_TRAINING.md` §15 with full results

### Phase 5 — Backtest + decision (1 day)

- Run `backtest_directional.py` with L3-trained model
- Measure gross edge after adverse selection (q=0.30) and upper-bound
  (q=1.0) fill models
- Decision point: does the L3 alpha clear the 5.5 bps execution cost?
  - **YES → Phase 6**: inventory framework, AlphaEngine wire-up
  - **NO → write up negative result**, consider Pivot 4 options
    (longer horizons, different venues, different models)

### Phase 6 (conditional on Phase 5 success) — Engine integration

Same scope as the L2 Phase B that was deferred at §14.8:
- `DirectionalMandate` dataclass in `layer2_alpha.py`
- `AlphaEngine.evaluate_directional` skipping the heat solver
- `MakerExecutor` in `layer3_execution.py` for continuous market-making
- Shadow replay PnL validation

Estimated 1 week if Phase 5 is positive.

**Total Phase 1-5: 9 days / 6 days ambitious.** Faster if data
acquisition reuses an existing capture process or paid feed.

## 6. Success criteria

L3 must clear EITHER threshold to justify continued investment:

**Signal threshold:**
- F2 (directional, H=100, val) ≥ 0.15 — beats L2 baseline 0.106 by
  at least 40% relative
- OR per-feature ablation shows at least 2 L3 features lift AUC by
  ≥ 0.02 individually

**Execution threshold:**
- Gross edge per trade ≥ 2 bps (vs L2's 0.06 bps; ≥ 30× lift)
- OR fill-model-robust positive net PnL at queue_fill_prob=0.30 in
  `backtest_directional.py`

If neither threshold clears, L3 features add complexity without
proportional alpha — same negative outcome as L2 but at a different
data layer. Documented as §15 NEGATIVE.

If both threshold clear, Phase 6 is justified and the architecture
pivot pays off.

## 7. Kickoff decisions (resolved 2026-05-12)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Venue | **Databento / Tardis** (paid historical MBO) | Skips 7-14 day live-capture wait; offline feature engineering starts immediately. Coinbase WS is fallback only. |
| 2 | Asset set | **BTC/ETH/SOL only** | Apples-to-apples vs L2 baseline; no confounders from new assets |
| 3 | Window | **7 days** | Mathematically sufficient to detect >5 bps edge in cancel velocity / queue depletion; minimizes paid-data cost; expand if green |
| 4 | Cadence | **Event-count clock** (initial K=100 events/tick) | Microstructure time is elastic: 500ms = 0 info in quiet regimes, 1000 state changes in shocks. Event clock normalizes information density per sample. Architecturally the most important override from §6 defaults. |
| 5 | Branch | **`research/path-h-l3`** | Matches Path G discipline; L2 codebase preserved on master as reference baseline; implementation forks from this research branch as `feature/path-h-impl` |

**Why event-clock is the most important shift.** L2 fed the TCN a
fixed wall-clock cadence; the entire predict-all collapse pattern in
§13.11-§13.15 partly stemmed from quiet-regime ticks dominating the
training distribution (the network learned the prior, not the signal).
The event clock guarantees that every TCN input window contains a
constant amount of *market activity*, not a constant amount of *wall
time*. Activity = information; the network now learns from informative
samples by construction.

## 8. Risks

- **Data volume.** L3 streams can be 10-100× larger than L2 snapshots.
  Coinbase BTC-USD at peak might emit 10K events/sec. 7 days × 86400 s
  × 5K avg events/s = 3 billion events. Storage and processing scale.
  Mitigation: aggregate features at capture time, not just on replay.
- **Cross-venue heterogeneity.** L3 schemas differ across venues. If
  we ever extend beyond Coinbase, expect another sensor refactor.
- **Latency reality.** Real L3 trading is microsecond-scale; FPGAs
  dominate. Our edge has to come from *different features* (not
  faster execution of the same features) — the directional and
  longer-horizon framings remain candidate angles.
- **HFT competition.** If L3 alpha at our resolution is real and
  large, professionals have already monetized it. Realistic
  expectation: we find signal that survives execution costs but is
  smaller than what the FPGAs capture at higher frequency. That's
  still a meaningful research result.
- **Signal-still-too-small.** Path G's parallel: dimensional analysis
  was sound but the dimensionless features added complexity without
  proportional alpha. L3 features could repeat this if our resolution
  is too coarse or our feature design is naive. Mitigation: aggressive
  per-feature ablation in Phase 4.

## 9. Connection to prior work

What from §13-§14 carries forward:

- **Training infrastructure**: `train_stream.py`, AUCM / BCE / Focal
  loss options, threshold sweep, val protocol — all reusable as-is
- **`backtest_directional.py`**: works with any feature CSV the model
  consumes; only the schema changes
- **Boundary discipline**: Path G's `BOUNDARY_CONDITIONS.md`
  (denominator floors, tanh saturation, stress-slice histograms) is
  *more* important at L3 because event-rate quantities have wider
  dynamic range than snapshot ratios
- **Temporal integrity protocol** (§14.1, 80/20 time-split val):
  mandatory from day 1 to avoid the §13.4-style in-sample inflation

What does NOT carry forward:

- **Path D feature scaling** (`ce_ratio/10`, `vamp/10.0`,
  `kyles_lambda*100.0`) — these scalings were Path D-specific; L3
  features need their own calibration
- **HMM regime** — calibrated on L2 features; either re-fit on L3
  or drop entirely (see §14.2 Friction A — HMM was stuck at regime 2
  on BTC/ETH anyway)
- **`identify_shock_events`** — built for L2-snapshot VPIN + price
  triggers; L3 has richer trigger possibilities (e.g., "did the
  cancel rate spike right before the move?") that may produce
  cleaner labels. Consider reframe.

## 10. Decision summary

This document is the planning artifact. The decision to commit to L3 is
the user's. My recommended sequence:

1. User answers the five open decisions in §7 (especially venue choice)
2. Phase 1 (data acquisition) begins
3. Decision gate at end of Phase 5: alpha clears 2 bps or it doesn't

Total commitment to first decision gate: 9 days of focused work.
