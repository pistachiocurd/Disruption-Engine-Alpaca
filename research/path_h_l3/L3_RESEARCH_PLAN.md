# L3 (Order-by-Order) Microstructure — Research Plan

> Status: Phase 1 (capture) in progress on Bitfinex public WS.
> Successor to the L2 investigation closed in
> [LAYER2_TRAINING.md §14.7-§14.8](../../docs/LAYER2_TRAINING.md);
> this document defines the architecture, data acquisition, phased
> work, and decision points for L3.

## 1. Context — Why pivot

§14.7's Phase A backtest definitively closed the L2 feature thesis:
- L2 features (CE ratio, OBI, MLOFI, VAMP, Kyle's λ) carry real directional
  signal — ~55% paper accuracy at H=100 ticks, +10pp over random.
- That signal magnitude is 0.06 bps gross edge per trade vs a 5.5 bps
  round-trip execution cost. The alpha doesn't survive any reasonable
  execution framework.
- Eight refinement strategies (§13.6–§13.15) all hit the same ceiling.
  Architecture, loss, labels, dimensionless reformulation — none lifted
  the floor.

The diagnosis: **L2 snapshots aggregate away the dynamics that PRECEDE
book-state changes.** OBI sees the state of the book; it does not see
*how the book is being assembled*. Queue depletion, cancellation
cascades, order lifespan, hidden-order inference — these are L3 events,
invisible in L2.

Academic + industry literature consistently identifies L3 as the layer
where leading microstructure signal lives: Cont & Kukanov (2017) on
optimal order placement, Gould et al. (2013) limit-order-book review,
Bouchaud–Bonart–Donier–Gould (2018) on HFT scaling laws,
Cartea–Jaimungal–Penalva (2015) algorithmic and HFT. These are the
references Path G's `THEORY.md` already cited; the dimensional analysis
was correct — it was applied at the wrong data layer.

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
2. **Order arrival rate imbalance** — `arrival_rate_bid − arrival_rate_ask`,
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
Phase 2 (sensor implementation) once real distributions are observed.

## 3. Data acquisition — venue decision history

| Venue | L3 quality | Cost | Archive | Live capture | Notes |
|---|---|---|---|---|---|
| Coinbase Exchange `full` | Order-by-order | Free until 2026-Q1 | None | Was yes; now HMAC-auth only | Retail Exchange signup closed; institutional API ~$5k |
| Coinbase Advanced Trade | L2 deltas only | Free | None | Yes | Not L3 — same data layer §13 just exhausted |
| Binance | Book delta stream | Free | Yes (paid tick data) | Yes (WS) | Reconstruct L3 from deltas; complex |
| Hyperliquid | L2 only | Free | Yes (S3) | N/A | Out — L2 is what §13 exhausted |
| Kraken | Full L3 WS | Free | None | Yes | Smaller venue; thinner data |
| Databento | Full MBO (industry standard) | $$$ | Yes | Yes | Crypto-spot is gated; high price floor |
| Polygon | L3 with WAQ for equities | $$ | Yes | Yes | Good US equity option but out of L3-crypto scope |
| Tardis.dev | Historical L3 replay | $$ | Yes | N/A | ~$1k/quarter base tier; no free anonymous access (2026-05-12) |
| **Bitfinex public** | **Raw book (`book` prec=R0) + trades** | **Free** | **None (live capture only)** | **Yes (WS)** | **No authentication; selected venue** |

### Decision history (2026-05-12)

1. *First revision — Tardis 1st-of-month free tier.* Smoke test returned
   HTTP 404 on `/data-feeds/` for all anonymous probes; the
   tardis-dev/tardis-node README confirms no free account tier of any
   kind. Dropped.
2. *Second revision — EC2 Coinbase Exchange `full` channel.* Probe
   returned `level2/level3/full channels now require authentication`.
   exchange.coinbase.com retail signup is gated to either "Continue to
   Advanced Trading" (no L3) or "Start Business Application"
   (institutional API ~$5k). Dropped.
3. *Final — Bitfinex public L3.* `wss://api-pub.bitfinex.com/ws/2` emits
   true order-by-order events via `book(prec=R0)` (adds/modifies/
   cancels per `ORDER_ID`) plus `trades` (aggressor side from `AMOUNT`
   sign). No auth required. Together they cover the §2 feature
   candidates except taker order-id linkage (a minor downgrade,
   partially recoverable by trade↔delete event correlation at matching
   price/timestamp).

The harvester (`harvest_bitfinex_l3.py`) and diagnostic probe
(`probe_bitfinex_ws.py`) are the executable artifacts of this decision;
`download_tardis_l3_free.py` is preserved as a working Tardis client
in case the paid tier becomes accessible later.

## 4. Architectural changes

L3 is a different data layer; significant codebase work required.

### 4.1 Data layer

`harvest_bitfinex_l3.py` (in this directory) subscribes to:
- `book(prec=R0)` per symbol — raw adds / modifies / cancels keyed by
  `ORDER_ID`
- `trades` per symbol — executed trades; aggressor side via `AMOUNT` sign

Both channels' raw messages are interleaved into one gzipped JSONL file
per UTC date: `l3_data/bitfinex_l3_<YYYYMMDD>.jsonl.gz`. Each line is
the raw WS frame; the `chanId → (channel, symbol)` mapping is
reconstructed offline by reading the subscription-ack messages (also
written verbatim).

### 4.2 Sensor layer

New file: `layer1_l3_sensors.py` at the project root (does not extend
`layer1_sensors.py` — different input contract).

New sensor classes that consume `OrderEvent` objects (canonical schema
in `aggregate_mbo_events.py`):

- `OrderBookReconstructor` — maintains full L3 book state from events;
  emits L2-like snapshot on demand for backward compatibility
- `OrderArrivalRateTracker` — rolling per-side arrival rate
- `CancellationVelocityTracker` — rolling per-side cancel rate +
  cancel/fill ratio
- `OrderLifespanTracker` — distribution stats over rolling window
- `HiddenOrderDetector` — count of trades at price levels with no
  matching arrival event since last clear
- `QueueDepletionTracker` — top-level size decay rate
- `AggressorSequenceTracker` — autocorrelation of trade aggressor side

All sensors output scalar features per-tick (where "tick" is defined
either by wall-clock interval or event-count interval — see §4.3).

### 4.3 Feature aggregation

The L3 sensors emit instantaneous values per event. The TCN needs
per-tick samples on a regular grid. Two aggregation modes are
implemented in `aggregate_mbo_events.py`:

- **Wall-clock**: emit feature row every N ms (e.g., 500 ms)
- **Event-count**: emit feature row every N events (e.g., 100)

Event-count is the initial choice — it adapts to activity bursts where
wall-clock under-samples shocks and over-samples quiet regimes. See §7
for the rationale.

The extended `FeatureDumper` writes a row per aggregation interval
with the L3-derived features + reconstructed L2 features (OBI, MLOFI,
etc. recomputed from the L3 book for cross-comparison).

### 4.4 Model layer

`config.py`:
- New constants `TCN_L3_INPUT_CHANNELS` (target: 10–15 channels)
- Mode flag `LAYER1_MODE = "l2_snapshots" | "l3_events"` (env-overridable)

`training/train_stream.py`:
- Feature stack branch on `LAYER1_MODE`
- L3 mode: stack the new L3 features (plus optionally a subset of
  reconstructed L2 features for comparison)
- Reuses everything else (BCE / AUCM / Focal, val protocol, threshold
  sweep, directional / shock label-source)

`training/backtest_directional.py`:
- No change needed; runs over any feature CSV the trained model can
  consume

### 4.5 Branch / directory strategy

Initially developed on `research/path-h-l3` (next research-branch letter
after Path G). Code consolidated into `research/path_h_l3/` on `master`
once the Bitfinex pivot stabilized. The `research/path-h-l3` branch is
retained until the active long-capture finishes (it owns the running
harvester's worktree).

## 5. Phased plan

Conservative estimates; ambitious estimates in parentheses.

### Phase 1 — Data acquisition (3–4 days wall, ~0 active work)

- Launch `harvest_bitfinex_l3.py` against the 3-symbol default set
  (`tBTCUSD`, `tETHUSD`, `tSOLUSD`) on a local workstation or EC2
  (`DEPLOY_HARVESTER.md`)
- Output: `l3_data/bitfinex_l3_<YYYYMMDD>.jsonl.gz` (UTC-date rotated)
- Target: ~15–40 M events across the 3 symbols × 3–4 days, multi-GB
  compressed
- **Deliverable**: 3–4 days of raw Bitfinex L3 events on disk

### Phase 1.5 — Parser wire-up (~30 min)

- Wire `parse_bitfinex_l3` into `aggregate_mbo_events.py`'s
  `VENDORS` dict (mapping in this directory's `README.md` under
  "Outstanding work")
- Bitfinex interleaves book + trades for all symbols in one file; the
  parser builds a `chanId → (channel, symbol)` map from the
  `subscribed` events at the top of the stream, then partitions events
  and emits one CSV per symbol
- **Deliverable**: canonical `OrderEvent` stream replayable through the
  aggregator

### Phase 2 — Sensor implementation + replay (3 days / 2 days)

- Implement 6–7 L3 sensors in `layer1_l3_sensors.py`
- Implement `OrderBookReconstructor` first (most complex piece — full
  L3 book state maintenance from events)
- Replay captured L3 stream through sensors, build feature CSV
- Unit tests for each sensor against known fixtures
- **Deliverable**: feature CSV with ~10 L3-derived columns plus
  L3-reconstructed L2 columns for cross-comparison

### Phase 3 — Distribution check (1 day)

- Run distributional checks per symbol: are sensors active? saturated?
  dead?
- Apply boundary mitigations (tanh saturation, denominator floors)
  matching Path G's `BOUNDARY_CONDITIONS.md` discipline
- Sanity-check pre-shock KS-D on L3 features (looking for the 0.15+
  values that L2 features couldn't deliver outside artifacts)
- **Deliverable**: cleaned per-symbol feature distributions, calibration
  constants

### Phase 4 — Training + comparison (2 days / 1 day)

- Train TCN in L3 mode on 80/20 time-split
- Directional target at H=100 (matches L2 directional comparison)
- Compare:
  - Val F2 vs L2 baseline F2 = 0.106
  - Val directional gross edge per trade vs L2 0.06 bps
  - Per-feature ablation: which L3 features carry signal?
- **Deliverable**: `docs/EXECUTION_TRAINING.md` extension or new
  `docs/LAYER3_TRAINING.md` with full results

### Phase 5 — Backtest + decision (1 day)

- Run `training/backtest_directional.py` with L3-trained model
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

**Total Phase 1–5: ~9 days / 6 days ambitious**, of which 3–4 days is
unattended capture.

## 6. Success criteria

L3 must clear EITHER threshold to justify continued investment.

**Signal threshold:**
- F2 (directional, H=100, val) ≥ 0.15 — beats L2 baseline 0.106 by
  at least 40% relative
- OR per-feature ablation shows at least 2 L3 features lift AUC by
  ≥ 0.02 individually

**Execution threshold:**
- Gross edge per trade ≥ 2 bps (vs L2's 0.06 bps; ≥ 30× lift)
- OR fill-model-robust positive net PnL at queue_fill_prob=0.30 in
  `training/backtest_directional.py`

If neither threshold clears, L3 features add complexity without
proportional alpha — same negative outcome as L2 but at a different
data layer. Documented as `docs/LAYER3_TRAINING.md` §15 NEGATIVE.

If either threshold clears, Phase 6 is justified and the architecture
pivot pays off.

## 7. Kickoff decisions (resolved 2026-05-12)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Venue | **Bitfinex public WS** (`book(R0)` + `trades`) | Coinbase Exchange went HMAC-auth-only; Tardis paid-only at ~$1k/qtr; Databento crypto gated. Bitfinex public is the only known free venue still emitting true L3. |
| 2 | Asset set | **BTC/ETH/SOL** (`tBTCUSD`, `tETHUSD`, `tSOLUSD`) | Apples-to-apples vs L2 baseline; no confounders from new assets |
| 3 | Window | **3–4 days continuous capture** | Targets ~15–40 M events; covers EU/US session overlap and overnight regimes |
| 4 | Cadence | **Event-count clock** (initial K=100 events/tick) | Microstructure time is elastic: 500 ms = 0 info in quiet regimes, 1000 state changes in shocks. Event clock normalizes information density per sample. Architecturally the most important override from §6 defaults. |
| 5 | Branch / dir | **`research/path_h_l3/`** on `master` | L2 codebase preserved on `master` as reference baseline; the `research/path-h-l3` branch is retained until long-capture completes (it owns the live harvester's worktree). |

**Why event-clock is the most important shift.** L2 fed the TCN a
fixed wall-clock cadence; the entire predict-all collapse pattern in
§13.11–§13.15 partly stemmed from quiet-regime ticks dominating the
training distribution (the network learned the prior, not the signal).
The event clock guarantees that every TCN input window contains a
constant amount of *market activity*, not a constant amount of *wall
time*. Activity = information; the network learns from informative
samples by construction.

## 8. Risks

- **Data volume.** L3 streams can be 10–100× larger than L2 snapshots.
  Bitfinex peak rates ~200/s × 3 symbols = ~600/s; 4 days × 86400 s ×
  600/s ≈ 200 M events upper bound. Mitigation: gzip compression at
  capture; aggregate features at parse time, not on every replay.
- **Cross-venue heterogeneity.** L3 schemas differ across venues. If
  this work ever extends beyond Bitfinex, expect another sensor refactor.
- **Latency reality.** Real L3 trading is microsecond-scale; FPGAs
  dominate. Any edge here has to come from *different features* (not
  faster execution of the same features) — the directional and
  longer-horizon framings remain candidate angles.
- **HFT competition.** If L3 alpha at this resolution is real and
  large, professionals have already monetized it. Realistic
  expectation: signal that survives execution costs but is smaller
  than what FPGAs capture at higher frequency. That's still a
  meaningful research result.
- **Signal-still-too-small.** Path G's parallel: dimensional analysis
  was sound but the dimensionless features added complexity without
  proportional alpha. L3 features could repeat this if the resolution
  is too coarse or the feature design is naive. Mitigation: aggressive
  per-feature ablation in Phase 4.
- **No taker order-id on trades.** Bitfinex trades emit `[TRADE_ID,
  TS_MS, AMOUNT, PRICE]` with no `maker_order_id` or `taker_order_id`.
  Trade↔book-event linkage is approximate (matching price/timestamp)
  rather than exact. Affects only the hidden-order and aggressor-by-
  queue-position features; the cancellation-velocity / lifespan
  features are unaffected.

## 9. Connection to prior work

Carries forward from §13–§14:

- **Training infrastructure**: `training/train_stream.py`,
  AUCM / BCE / Focal loss options, threshold sweep, val protocol — all
  reusable as-is
- **`training/backtest_directional.py`**: works with any feature CSV
  the model consumes; only the schema changes
- **Boundary discipline**: Path G's `BOUNDARY_CONDITIONS.md`
  (denominator floors, tanh saturation, stress-slice histograms) is
  *more* important at L3 because event-rate quantities have wider
  dynamic range than snapshot ratios
- **Temporal-integrity protocol** (§14.1, 80/20 time-split val):
  mandatory from day 1 to avoid §13.4-style in-sample inflation

Does NOT carry forward:

- **Path D feature scaling** (`ce_ratio/10`, `vamp/10.0`,
  `kyles_lambda*100.0`) — these were Path D-specific; L3 features need
  their own calibration
- **HMM regime** — calibrated on L2 features; either re-fit on L3 or
  drop entirely (§14.2 Friction A — HMM was stuck at regime 2 on
  BTC/ETH anyway)
- **`identify_shock_events`** — built for L2-snapshot VPIN + price
  triggers; L3 has richer trigger possibilities (e.g., "did the
  cancel rate spike right before the move?") that may produce cleaner
  labels. Reframe candidate.

## 10. Next decision point

The active long-capture is the gating step; everything downstream is
deterministic engineering until Phase 5. The next decision point is
Phase 5's decision gate: does the L3 alpha clear the 2 bps gross-edge
threshold or doesn't it?

Total commitment to first decision gate: ~9 days of focused work after
the 3–4 day capture completes.
