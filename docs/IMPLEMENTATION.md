# Implementation Documentation

This document explains how the engine is realized in code. It is intended
to be read by anyone modifying the system after handoff. Every non-obvious
design choice is justified here so future changes preserve the invariants
that make the system safe to run with capital at risk.

The system supports two market venues behind a single set of layered
abstractions:

- **US equities via Alpaca** (default). NBBO + trade ticks; regular hours
  09:30–16:00 ET; session-aware state management; IEX-only feed by default
  with optional SIP upgrade.
- **Crypto via ccxt.pro** (original path). Full-depth L2 books; 24/7;
  optional Binance forced-liquidation feed.

The branch is selected automatically by `config.IS_EQUITY` (derived from
the shape of `SYMBOL`). Where the two paths diverge, this document calls
it out explicitly.

---

## 1. System Topology

Persistent asyncio tasks running forever:

| Task            | Module             | Cadence            | Purpose                                                       |
| --------------- | ------------------ | ------------------ | ------------------------------------------------------------- |
| Sensor loops    | `layer1_sensors`   | per WS event       | Translate raw quotes/trades/liquidations into `PhysicsState`. |
| Strategy loop   | `engine.py`        | 50 ms tick         | Read state → Layer 2 alpha → Layer 3 execution.               |
| Shadow trainer  | `layer4_shadow`    | 30 s training pass | Continuous PPO updates + stability-gated weight push.         |
| Session manager | `engine.SessionManager` | daily refresh | Equities only — calendar cache, market-hours gate.            |

Sensor streams differ by venue:

- **Equities (Alpaca)**: `StockDataStream.subscribe_quotes` and
  `subscribe_trades` register handlers; the SDK drives its own loop. NBBO
  updates are repackaged into the canonical `ob`-shaped dict so
  `_process_order_book` is identical to the crypto path.
- **Crypto (ccxt.pro)**: `watch_order_book` and `watch_trades` polled in
  separate coroutines, plus the Binance `!forceOrder@arr` raw WS for
  liquidations.

Each ingestion stream has an independent reconnection loop so that one
dropping cannot kill the other two.

A separate offline job recalibrates on a fixed cadence (default 48 h).
That job is in `calibration.py` and writes per-symbol files to
`./calibration/latest_{SYMBOL_SLUG}.json`, which the engine reloads on
next restart. There is no live recalibration: `KALMAN_DOF`, the HMM ν per
state, and the OOD covariance are all computed offline and frozen during
operation.

```
                  Equities path (Alpaca)            Crypto path (ccxt.pro)
                  ──────────────────────            ──────────────────────
              ┌── StockDataStream ──┐         ┌── ccxt.pro WebSockets ────┐
              │  subscribe_quotes ──┐         │  watch_order_book ──┐     │
              │  subscribe_trades ──┐         │  watch_trades ──────┐     │
              └─────────────────────┘         │  raw forceOrder@arr ┐     │
                                              └───────────────────────────┘
                     │                                   │
                     ▼                                   ▼
       ┌─────────────────────────────────────────────────────────────┐
       │ Layer 1: SensorArray                                         │
       │   VPINToxicityTracker (Lee-Ready on equities) ·              │
       │   StudentTKalmanFilter · StudentTHMM ·                       │
       │   LiquidationCascadeTracker (Binance only) · OODDetector ·   │
       │   CEInferenceEngine (crypto, L2 diff) /                      │
       │   QuoteCancellationProxy (equities, L1 quote-rate proxy)     │
       │   reset_session() called by engine on each market open       │
       └────────────────────────────┬─────────────────────────────────┘
                                    │  PhysicsState (every tick)
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │ Layer 2: AlphaEngine                                         │
       │   TCNSpikePredictor → turbulence_index                       │
       │   HeatEquationSolver → P_eq + sensitivity check              │
       │   ── GATES: ood_flag, attrition uncertainty, MIN_ORDER_SIZE  │
       └────────────────────────────┬─────────────────────────────────┘
                                    │  TradeMandate
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │ Layer 3: ExecutionEnv (Gymnasium) + PPOAgent (live)          │
       │   42-dim obs (NO directional info) → action ∈ [-1, 1]        │
       │   reward = -IS_fee_adj - η·var - γ_inv·inv_pen               │
       └──────────────┬───────────────────────────────────┬───────────┘
                      │                                   │
                      ▼                                   ▼
        LocalMatchingEngine                  Alpaca / ccxt.create_order()
        (SHADOW_MODE = True)                 (SHADOW_MODE = False)

       ┌─────────────────────────────────────────────────────────────┐
       │ Layer 4: ShadowSimulator (background, GPU)                   │
       │   24 h ReplayBuffer → PPO updates → StabilityGate            │
       │   Polyak τ=0.05 → live agent                                 │
       │   Replay buffer is pruned on each market open (equities)     │
       └─────────────────────────────────────────────────────────────┘

       ┌─────────────────────────────────────────────────────────────┐
       │ engine.SessionManager (equities only)                        │
       │   Pulls /v2/calendar daily; gates the strategy loop;         │
       │   triggers SensorArray.reset_session() on each open with     │
       │   SESSION_WARMUP_SECONDS of OOD-gate quarantine.             │
       └─────────────────────────────────────────────────────────────┘
```

---

## 2. Spec → Code Mapping

| Spec component                              | File / class                                        |
| ------------------------------------------- | --------------------------------------------------- |
| §1.1 VPIN tracker                           | `layer1_sensors.VPINToxicityTracker`                |
| Lee-Ready trade-side classification         | `VPINToxicityTracker.add_trade_classified()`        |
| §1.2 Student-t Kalman                       | `layer1_sensors.StudentTKalmanFilter`               |
| §1.3 Student-t HMM                          | `layer1_sensors.StudentTHMM`                        |
| §1.4 Liquidation cascade tracker            | `layer1_sensors.LiquidationCascadeTracker`          |
| §1.5 OOD detector                           | `layer1_sensors.OODDetector`                        |
| C/E inference (crypto, L2 diff)             | `layer1_sensors.CEInferenceEngine`                  |
| C/E proxy (equities, L1 quote-rate)         | `layer1_sensors.QuoteCancellationProxy`             |
| Per-sensor session reset                    | `*.reset_session()` on VPIN, Kalman, HMM, CE        |
| §1→2 calibration handoff                    | `calibration.fit_alpha_calibration_c()`             |
| Shock event definition                      | `calibration.identify_shock_events()`               |
| Drift monitor                               | `calibration.CalibrationDriftMonitor`               |
| Historical harvester (equities, same-day)   | `fetch_history_alpaca.py`                           |
| §2.1 TCN spike predictor                    | `layer2_alpha.TCNSpikePredictor`                    |
| §2.2 Heat equation solver                   | `layer2_alpha.HeatEquationSolver`                   |
| Attrition-sensitivity P_eq guard            | `HeatEquationSolver.solve()` (lambda ± δ)           |
| TradeMandate                                | `layer2_alpha.TradeMandate`                         |
| §3 ExecutionEnv (state, action, reward)     | `layer3_execution.ExecutionEnv`                     |
| §3 PPO agent (split CNN encoders)           | `layer3_execution.PPOAgent`                         |
| Fee-adjusted IS                             | `ExecutionEnv._step_is_fee_adj()`                   |
| Terminal forced fill (2x penalty)           | `ExecutionEnv._force_terminal_fill()`               |
| Shadow Live Router                          | `matching_engine.LocalMatchingEngine` + `engine.py` |
| §4 ShadowSimulator                          | `layer4_shadow.ShadowSimulator`                     |
| Stability gate (KL + regime + steps)        | `layer4_shadow.StabilityGate`                       |
| Polyak averaging                            | `layer4_shadow.polyak_update()`                     |
| HPO over (η, γ_inv, terminal_mult)          | `hpo.py`                                            |
| Kill switches (drawdown, exception, signal) | `engine.py` (see §3.4 for gate inventory)           |
| Aggregate position limit                    | `engine.Engine._would_exceed_position_limit()`      |
| IS-based drawdown gate                      | `engine.Engine._is_drawdown_breached()`             |
| MTM-based drawdown gate                     | `engine.Engine._mtm_drawdown_breached()`            |
| Hybrid drawdown composer                    | `engine.Engine._max_drawdown_breached()`            |
| Session risk-state reset                    | `engine.Engine._reset_session_risk_state()`         |
| Equity market-hours gate + state reset      | `engine.SessionManager` + `SensorArray.reset_session()` + `Engine._reset_session_risk_state()` |
| Replay-mode driver                          | `test_replay.ReplaySensorArray` (gated by `config.REPLAY_CSV`) |

---

## 3. Key Invariants

These must NOT be violated by future changes. Each invariant has a structural
or test-based guarantee.

### 3.1 The matching engine cannot place real orders

`matching_engine.py` ends with an import-time assertion that scans
`sys.modules` for `ccxt`, `ccxt.pro`, and `ccxtpro`. If any of these have been
imported into the same module's transitive graph, the assertion fires. The
file imports nothing exchange-related and is expected to remain that way.

`tests/test_matching_engine.py::TestStructuralIsolation` re-validates this by
scanning the source for ccxt references.

### 3.2 ExecutionEnv state space contains no directional information

The 42-dim observation vector is, by construction, side-symmetric. It exposes
top-10 bid prices/volumes, top-10 ask prices/volumes, and two scalars
(`remaining_vol_frac`, `time_remaining_frac`). It does **not** expose:

- `P_eq`
- `mandate.direction`
- `vpin`, `viscosity`, `regime`, or any other physics-state value

This is enforced by the layout in `ExecutionEnv._build_observation()`. The
PPO agent has no path to directional signal — its only job is to fill
volume cheaply.

### 3.3 Calibration parameters are frozen at runtime

- `KALMAN_DOF`, the per-state `HMM_DOF_*`, the OOD covariance — all read once
  at engine startup and never updated during operation.
- The drift monitor pauses mandate generation rather than auto-recalibrating.
- The only way to update calibration is to run the offline job and restart
  the engine (or trigger a recalibration that writes to disk and signals the
  engine to reload).

### 3.4 Mandate generation has independent gates at every layer

Layer-2 gates (`AlphaEngine.evaluate()` and `HeatEquationSolver.solve()`):

1. `physics_state.ood_flag` (Mahalanobis > `OOD_THRESHOLD = 4.5` against the
   calibrated μ/Σ) → suppressed.
2. `turbulence_index < TURBULENCE_THRESHOLD` → no solver call.
3. P_eq sensitivity range > confidence band → suppressed.
4. `target_size < MIN_ORDER_SIZE` (per-mandate) → suppressed.

Engine-level gates (in `_strategy_loop` between mandate generation and
execution):

5. `CALIBRATION_STALE` (drift monitor MAPE > 0.35) → strategy loop skips.
6. **Aggregate position limit** (`_would_exceed_position_limit`): if the
   mandate would push `abs(_position)` further past `MAX_POSITION_LIMIT` in
   the same direction it's already biased, skip. Position can flip directions;
   it cannot pile on a saturated side. The Layer-2 per-mandate cap (gate 4)
   only bounds individual mandate sizes — successive same-direction mandates
   would otherwise stack unboundedly without this aggregate gate.
7. **Hybrid drawdown** (`_max_drawdown_breached` calls both
   `_is_drawdown_breached` and `_mtm_drawdown_breached`): trips kill switch
   when *either*
   - IS-based drawdown (accumulated execution-shortfall costs) exceeds
     `MAX_SESSION_DRAWDOWN_USD` OR exceeds `MAX_SESSION_DRAWDOWN_PCT` of
     `session_peak_pnl` once peak ≥ `DRAWDOWN_PCT_MIN_PEAK_USD`
   - MTM-based drawdown (`cash + position·mid` from current order-book
     snapshot) exceeds `MAX_MTM_DRAWDOWN_USD` OR exceeds
     `MAX_MTM_DRAWDOWN_PCT` of `_mtm_peak_pnl` once peak ≥
     `DRAWDOWN_PCT_MIN_PEAK_USD`.

   Two gates because IS measures "how expensive are our fills" and MTM measures
   "is the held position bleeding directionally." They diverge by orders of
   magnitude under normal trading; either alone would miss a real risk class.
   The kill-switch log includes which gate(s) tripped and the underlying
   values for forensics.
8. Unhandled exception → kill switch trips, all orders cancelled.

A failure of any one gate stops the trade. There is no "best 2 of 3" logic.

The IS drawdown formula uses a min-peak floor (`DRAWDOWN_PCT_MIN_PEAK_USD`)
to suppress the percentage check below tiny peaks — `(peak − pnl) / peak` has
unbounded leverage when `peak` is small (a $0.91 peak followed by a $9 loss
is a 1000% "drawdown" trip on a 2% gate). The absolute USD cap is always
active and is the primary safety stop; the percentage cap is secondary and
only meaningful at production-relevant peak sizes.

### 3.5 The shadow agent never directly replaces the live agent

Promotions are always Polyak-averaged with τ = 0.05. There is no hard cutover
path. Even after the stability gate passes, the live policy moves only 5%
toward the shadow weights per push.

### 3.6 Live order flow requires two explicit positive flags

`SHADOW_MODE = False` AND `EXCHANGE_LIVE = "true"`. Each is a separate
deliberate gate; flipping only one keeps the system in paper mode.

### 3.7 Equity sensor state must be reset at every market open

After a 16.5-hour overnight gap the live observation stream has a
discontinuity that the filters cannot interpret correctly:

- The Kalman filter would receive the first morning quote as a 1-tick
  spread innovation. Even with Student-t robustness the velocity term
  spikes.
- The HMM would score the first morning observation against yesterday's
  forward variable α — yesterday's regime is no longer informative.
- The VPIN buckets carry partial yesterday-evening volume into today.
- The OOD detector trips on every feature simultaneously.

`engine.Engine._strategy_loop` calls `SensorArray.reset_session()` at
every detected close→open transition, which uniforms the HMM α,
re-initializes the Kalman state, clears the VPIN buckets, and resets the
C/E proxy. The OOD gate is suspended for `SESSION_WARMUP_SECONDS` (default
30s) to absorb residual transients. Flag this in any future modification
that touches the strategy loop or sensor lifecycle.

This invariant is crypto-irrelevant (24/7 markets); it is enforced only
when `IS_EQUITY` is true.

### 3.8 Engine risk state must be reset alongside sensor state at session open

Beyond the sensor reset documented in §3.7, the engine itself maintains
per-session risk state — `_position`, `_cash`, `session_pnl`,
`session_peak_pnl`, `_mtm_peak_pnl`, fee/notional accumulators — that must
**also** be reset on close→open transitions, otherwise:

- Yesterday's `session_peak_pnl` of e.g. $1500 carrying into today means a
  first-trade $30 loss reads as a $1530 drawdown, false-tripping the IS gate
  before the day's strategy has run.
- Yesterday's `_position` and `_cash` carrying into today means today's MTM
  is computed against today's mid using yesterday's positions — a
  meaningless number that will likely trip the MTM gate immediately.
- The drift monitor's rolling window includes shocks from yesterday's
  closing regime that are no longer informative.

`engine.Engine._reset_session_risk_state` is called immediately after
`SensorArray.reset_session` on every detected market-open transition. It
zeros the engine-side counters listed above and calls `drift_monitor.reset`
if available. Crypto path: not invoked (24/7).

This is the engine-side counterpart to §3.7. They were missing from the
original implementation; the symptom in replay mode (where market-open never
fires) was an MTM kill-switch trip whenever the harvested CSV stitched
across a session boundary with a $50+ TSLA mid jump.

### 3.9 Replay-mode mandate suppression

`ReplaySensorArray` (test_replay.py) is the engine's off-market driver.
When `config.REPLAY_CSV` is non-empty, `engine.py` swaps the live
`SensorArray` for the replay shim, and the strategy loop runs against the
historical CSV exactly as it would against a live feed — Layers 2/3/4 can't
distinguish.

Two replay-specific synthesis decisions:

- **Order book is synthesized 1-level**, with bid_size and ask_size skewed
  by the harvested OBI signal so `Q = |bid_total − ask_total|` is meaningful
  (the matching engine walks only top-of-book for cross detection, so 1
  level suffices for fill detection; market orders walk to depletion which
  is unrealistic for large fills but fine for verifying pipeline plumbing).
- **OOD `mahal_dist` is recomputed** from the loaded `OODDetector` rather
  than read from the CSV. The CSV's `mahal_dist` column was produced under
  whatever μ/Σ existed at harvest (typically identity prior, hence inflated
  values); replay using the live calibrated detector matches what production
  will compute.

Risk-gate defaults loosen automatically when `REPLAY_CSV` is set
(`MAX_MTM_DRAWDOWN_USD: 20000 → 1000000`,
`MAX_MTM_DRAWDOWN_PCT: 0.10 → 10.0`). Reason: replay never hits
"market open," so `_reset_session_risk_state` never fires across CSV
session-boundary mid jumps, and tight production caps would trip
non-deterministically based on which CSV segment the replay is in. Single
unset of `REPLAY_CSV` restores production caps.

---

## 3a. Equities Pivot — Design Notes

This section documents decisions specific to the equities path. Crypto
behavior is unchanged.

### 3a.1 Why Lee-Ready instead of `trade["side"]`

Crypto exchanges report buyer/seller initiator on every print. US equity
prints (NASDAQ/NYSE/IEX) do not — especially internalized prints from
wholesalers, which carry no initiator information. `VPINToxicityTracker`
therefore exposes two ingestion paths:

- `add_trade(trade)` — original; reads `trade["side"]`. Used by crypto.
- `add_trade_classified(trade, mid_at_trade, last_trade_price)` — Lee-Ready
  / tick-test classifier. Used by equities.

The midpoint is cached on every NBBO update by `_process_order_book` and
read by `_process_trade`. When no quote has arrived yet, the classifier
falls back to the tick test against the previous print, then to a "buy"
default — degenerate but bounded, and averages out across the bucket.

### 3a.2 Why `QuoteCancellationProxy` instead of `CEInferenceEngine`

`CEInferenceEngine` diffs full-depth L2 books to attribute level-by-level
volume reduction to either trades (executions) or vanished resting orders
(cancellations). This is impossible from L1 NBBO alone, and L1 is all
Alpaca's free tier provides.

`QuoteCancellationProxy` reinterprets the C/E ratio as
`(NBBO updates − matched trades) / matched trades` over a rolling
`CE_ROLLING_WINDOW_SEC` window. Each NBBO update with no co-incident
trade is treated as inferred cancellation pressure. The output is bounded
by `CE_RATIO_MAX` to match the L2 engine's clamp, and exposes the same
`.ce_ratio` property and `.reset_session()` method, so HMM, OOD, and
FeatureDumper see the same interface in both modes.

The HMM and OOD detector keep their input shapes (4 emission features,
3-feature TCN obs vector) — but the `ce_ratio` distribution differs from
the L2-derived one, so all three of `StudentTHMM` priors,
`OODDetector(μ, Σ)`, and the TCN must be retrained on equity data. That
retraining is what `fetch_history_alpaca.py` enables.

### 3a.3 `liquidation_rate` is dead weight on equities

There is no equity analog to the Binance `!forceOrder` feed. The channel
is preserved at constant 0.0 for schema compatibility with the existing
9-column `feature_history.csv` and the existing 4-feature HMM emission
shape. The TCN will down-weight the constant channel during retraining;
the OOD detector handles a constant feature trivially (covariance entry
goes to zero, regularized away by the 1e-6 jitter in
`OODDetector._set_inverse`).

If a meaningful equity-specific signal becomes available (LULD-band
proximity, halt indicator), it should reuse this slot rather than expand
the schema — that keeps the trained models and downstream tools
unchanged.

### 3a.4 Historical harvester drives the live code path

`fetch_history_alpaca.py` does **not** reimplement feature computation.
It instantiates `SensorArray` with a `FeatureDumper`, then drives
`_process_order_book` and `_process_trade` directly via a heap-merged
chronological replay of Alpaca's historical quote and trade endpoints.

This guarantees the resulting CSV is byte-identical to what the live
engine writes — there is zero risk of train/serve skew because there is
no separate "training feature" code path.

The harvester also calls `sensors.reset_session()` between calendar
sessions, mirroring the live behavior so the trained model sees the
same warm-start dynamics it will see in production.

### 3a.5 Fee model

Equity defaults: `TAKER_FEE = MAKER_FEE = 0.0` (zero-commission broker),
plus asymmetric regulatory fees applied only on sells:

- `SEC_FEE_BPS_ON_SELL = 0.00229` (~$22.90 per $1M principal)
- `FINRA_TAF_PER_SHARE_SELL = 0.000166` (capped at $8.30 per execution)

The PPO IS reward is dominated by spread + impact in this regime, not
by fees. The fee-adjusted IS calculation in
`ExecutionEnv._step_is_fee_adj()` still applies; the asymmetric sell-only
surcharges should be added at the per-fill accumulator (one-line branch)
before going live. Crypto fees remain Coinbase Advanced spot defaults
(60/40 bps) for the crypto path.

### 3a.6 Data tier — IEX vs SIP

`ALPACA_DATA_FEED` defaults to `iex` (free tier). IEX represents
roughly 2–3% of consolidated tape volume; the NBBO observed is IEX's
top-of-book, not the real consolidated NBBO. VPIN absolute levels and
quote-update rates will be lower than what a SIP feed would show.

This is consistent train-vs-live as long as both pipelines use the same
feed — and they will, because both `fetch_history_alpaca.py` and the live
`StockDataStream` read `ALPACA_DATA_FEED` from config. Upgrading to SIP
(Algo Trader Plus subscription) requires re-harvesting and re-training
because the feature distribution shifts.

---

## 4. Calibration Pipeline

The offline calibration job is the most failure-sensitive part of the system.
Run it on a clean Python process with archived data.

### 4.1 Inputs

- L2 order book snapshots aligned to a tick stream.
- Trade history (with `side` field).
- The VPIN series computed from those trades using the same `VPINToxicityTracker`
  configuration as production.

### 4.2 Procedure

```python
from calibration import (
    identify_shock_events,
    fit_alpha_calibration_c,
    fit_ood_distribution,
    save_calibration,
)

# 1. Shock events under the strict three-condition definition.
events = identify_shock_events(ob_snapshots, vpin_series)
assert len(events) >= 30, "insufficient shocks; widen window or wait for more data"

# 2. Exponentially weighted OLS for the diffusivity scalar.
c_hat = fit_alpha_calibration_c(events)
assert c_hat is not None

# 3. OOD covariance from the [ce_ratio, obi, liquidation_rate] history
#    over the SAME window as the shock identification.
mu, sigma = fit_ood_distribution(feature_history)

# 4. Persist
save_calibration(Path("calibration/latest.json"), c_hat, mu, sigma)
```

### 4.3 What the live drift monitor does

For each completed live shock (vpin_at_trigger, observed_T_actual), append
to a rolling N-window. Once the window is full, compute MAPE of
`predicted T = 1 / (current_c × vpin)` vs observed. If MAPE > 0.35,
`is_stale = True` and the strategy loop suspends mandate generation. The
monitor never tries to "self-heal"; it surfaces the problem and waits for the
next offline recalibration.

### 4.4 When calibration is rejected

`fit_alpha_calibration_c()` returns `None` when fewer than
`MIN_CALIBRATION_EVENTS` (30) events are available. The engine's behavior is
to keep using the previously-saved `c` and log a warning. Operator must
decide whether to enlarge the calibration window or pause trading.

### 4.5 Same-day equity calibration via the historical harvester

The crypto pipeline above assumes an existing archive of L2 + trade data.
For the equities path the engine ships a same-day pipeline driven by
`fetch_history_alpaca.py`:

```powershell
# 1. Replay 14 trading sessions through the live SensorArray code path,
#    writing the canonical 9-column feature_history CSV.
python harvesters/fetch_history_alpaca.py --symbol NVDA --days 14

# 2. Verify shock density.
python training/check_shocks.py

# 3. Fit OOD μ, Σ over [ce_ratio, obi, liquidation_rate] from the CSV.
#    Writes calibration/latest_NVDA.json.
python training/fit_ood_from_csv.py

# 4. Train the TCN supervised against shock labels derived from the same
#    CSV. Writes calibration/tcn_weights_NVDA.pt and tcn_threshold_NVDA.json.
python training/train_tcn.py
```

The harvester drives `SensorArray._process_*` directly so the CSV is
byte-identical to live `FeatureDumper` output. There is no separate
"training feature" code path, hence no train/serve skew risk. The
per-session `reset_session()` call between calendar days mirrors live
behavior.

For the equity path, `ALPHA_CALIBRATION_C` is a placeholder in
`CALIBRATION_REGISTRY`; it should be refit by the EWLS step against
identified shocks once the OOD calibration and TCN are in place.

---

## 5. Reward Engineering: Why Fee-Adjusted IS

The reward is the central piece of fee discipline. Every fill is recorded
in fee-adjusted units before being plugged into the IS calculation:

```
buy taker:     fee_adj = fill * (1 + 0.0005)    →  IS = (fee_adj - p_dec) / p_dec
buy maker:     fee_adj = fill * (1 + 0.0001)
sell taker:    fee_adj = fill * (1 - 0.0005)    →  IS = (p_dec - fee_adj) / p_dec
sell maker:    fee_adj = fill * (1 - 0.0001)
```

On crypto venues the taker fee (e.g. 60 bps on Coinbase Advanced spot,
5 bps on Binance Futures) typically dominates or exceeds the spread. An
agent trained without fees in the reward systematically prefers taker
orders because they fill instantly. Fee-adjusted training makes the maker
route attractive whenever inventory urgency permits, which is exactly the
maker-vs-taker tradeoff a human execution desk runs.

On US equities through a zero-commission broker (Alpaca), `TAKER_FEE` and
`MAKER_FEE` are both 0.0 — but spread + impact still drive the IS, so
the reward shaping still does useful work. Asymmetric regulatory fees
(SEC, FINRA TAF) apply only on sells; see §3a.5.

The full reward composes three terms:

```
r = -IS_fee_adj  -  ETA · fill_variance  -  GAMMA_INV · inventory_time_penalty
```

- `fill_variance` discourages erratic execution paths (better-quality fills
  matter more than minimum-cost fills if minimum-cost is wildly inconsistent).
- `inventory_time_penalty = rem_frac · (1 - time_rem_frac)²` grows quadratically
  near the end of the window, structurally discouraging "passive-until-panic".

Terminal forced fill applies a `TERMINAL_PENALTY_MULTIPLIER × IS_fee_adj`
penalty if the window expires with volume outstanding. This combines two
disincentives: bad IS (we crossed the spread at the worst moment) AND the
maximum-fee taker rate.

`ETA`, `GAMMA_INV`, `TERMINAL_PENALTY_MULTIPLIER` are tuned via Optuna in
`hpo.py`. The HPO objective is

```
objective = mean_IS + 0.5 · fill_price_std + 2.0 · terminal_penalty_rate
```

minimized. The winning trial is rejected if held-out validation degrades by
more than 20% relative to training — an explicit overfitting guard.

---

## 6. Heat Equation Solver: What It Actually Computes

The brief frames the equilibrium price calculation in PDE language
(∂u/∂t = α ∂²u/∂x² + Q). In the implementation, the post-shock equilibrium
price `P_eq` is computed by **walking the absorbing side of the book** with
the attrition-discounted depth at each level:

```python
adjusted_depth_i = raw_depth_i · exp(-λ · vpin · d_i_bps)
```

We accumulate `adjusted_depth_i` from best-of-book outward in the direction
of the shock until the cumulative absorbs `Q`. The price at that crossover
point is `P_eq`. The PDE form is preserved as the diffusion-timescale
calculation (used for the CFL stability check that sets `Δt`), but the trade
decision needs only `P_eq` itself.

The attrition discount captures phantom liquidity: HFT firms pull quotes
under toxicity. `λ = 0.15` is the crypto default; equity books are typically
deeper and a value closer to 0.10 fits, but the production tuning lives in
`config.LAMBDA_ATTRITION` and should be set per-symbol against actual
calibration data — placeholder values in the registry should not be trusted
without an HPO pass.

### 6.1 The sensitivity guard is the load-bearing safety check

A single `λ` cannot perfectly capture phantom liquidity. The solver re-runs
at `λ - δ` and `λ + δ`. If `(P_eq_high - P_eq_low) / P_current > 15 bps`,
the answer is too sensitive to attrition assumptions and the mandate is
suppressed. This is more important than it looks: it prevents trading in
exactly the regimes where the book is most fragile and the model is least
trustworthy.

---

## 7. Layer 4: How Continuous Training Stays Safe

Three independent conditions must all pass before any weight push:

1. **KL divergence** between live and shadow policies < 0.1.
   This is a closed-form computation between the two diagonal Normal
   distributions, evaluated over a sample of 256 recent observations.
   High KL means the shadow has wandered too far from the live policy and
   would cause large execution-style shifts on push.

2. **Regime match.** The dominant HMM regime in the shadow training window
   must match the current live regime. Pushing weights trained on Turbulent
   data into a Laminar live market would systematically over-aggress.

3. **Step minimum.** ≥ 50 000 PPO gradient steps since the last successful
   promotion. Prevents rapid back-and-forth pushes that destabilize live
   behavior.

Even when all three pass, the update is Polyak-averaged with τ = 0.05. The
live policy never moves more than 5 % toward the shadow per promotion.

---

## 8. Operational Runbook

### 8.1 First-time bring-up — Equities (Alpaca, default)

```
1. Install alpaca-py (in requirements.txt; pip install -r requirements.txt).
2. Set EXCHANGE_API_KEY, EXCHANGE_SECRET (Alpaca paper keys).
3. Set SYMBOL=NVDA (or SPY). Default is NVDA.
4. Run pytest. Confirm all tests pass.
5. Harvest history and train the TCN today (avoids days of waiting on
   live capture). See §4.5.
6. Run engine.py with SHADOW_MODE=True, EXCHANGE_LIVE unset (paper).
7. Confirm logs show during market hours (09:30–16:00 ET):
     - alpaca initialized: feed=iex paper=True
     - SessionManager: cached N sessions
     - SensorArray.reset_session at first market open
     - shadow simulator started on cuda (or cpu fallback)
     - strategy loop reports market open
8. Let it run across at least one market close → next-day open and confirm:
     - "market closed — pausing strategy and cancelling open orders" fires
     - "market open — resetting sensor state" fires the next morning
     - no OOD storm in the first SESSION_WARMUP_SECONDS
9. After ~10 sessions of clean shadow operation (≈ crypto's 72h), proceed.
```

### 8.1b First-time bring-up — Crypto (Coinbase / Binance)

```
1. Set EXCHANGE_ID=coinbaseadvanced (or binance), SYMBOL=BTC/USD.
2. Set EXCHANGE_API_KEY, EXCHANGE_SECRET (testnet keys).
3. Run pytest. Confirm all tests pass.
4. Run engine.py with SHADOW_MODE=True, EXCHANGE_LIVE unset (testnet).
5. Confirm logs show:
     - sensors task connected to all three streams
     - shadow simulator started on cuda (or cpu fallback)
     - strategy loop started
6. Let it run 72 hours. Confirm:
     - no unhandled exceptions
     - liquidation tracker reconnects successfully across drops
     - drift monitor remains inactive
     - shadow simulator promotes weights at least once via Polyak
```

### 8.2 Going live (deliberate, two-step)

```
1. Flip SHADOW_MODE = False in config.py. Restart.
   Now mandates → real testnet orders.
2. Run for >= 24 h on testnet. Confirm fills match LocalMatchingEngine
   estimates within reasonable tolerance.
3. Set EXCHANGE_LIVE=true in environment. Restart.
   Now mandates → real live orders.
4. Watch the drawdown counter. The 2% drawdown kill switch will trip
   automatically; test that it fires correctly with a deliberately bad
   small trade before scaling capital up.
```

### 8.3 Recalibration (every 48 h)

```
1. Pull the last 48–72 h of L2 + trade data.
2. Run calibration.py end-to-end. Confirm >= 30 shock events.
3. Diff the new c_hat against the production CALIBRATION_REGISTRY value.
   If the change is > 30%, do not deploy automatically — investigate first.
4. Save to ./calibration/latest.json.
5. SIGHUP the engine (or restart) to pick up the new file.
```

### 8.4 HPO (every 2 weeks, or after a drift event)

```
1. Run hpo.py with N_TRIALS=50 over the most-recent week of mandates.
2. Validate the winning trial on the SUBSEQUENT week's data.
3. If validation degradation > 20%, reject and keep current defaults.
4. Otherwise, update ETA / GAMMA_INV / TERMINAL_PENALTY_MULTIPLIER in
   config.py and restart.
```

### 8.5 Kill-switch recovery

```
- Drawdown trip:    TRADING_ENABLED=False is set automatically.
                    Restart only after manual review of the loss source.
- Exception trip:   Same. The traceback is in logs/engine.log.
- Drift trip:       Strategy loop self-pauses. Run recalibration.
- Manual:           SIGINT / SIGTERM → graceful shutdown, all orders cancelled.
- Market close (eq): Strategy loop pauses; open orders cancelled; sensor
                    state reset on next open. Self-recovering — not a kill.
```

---

## 9. Known Gaps for the Operator

The following are explicit handoff items where the spec calls for real
training data that the codebase scaffolds but doesn't ship:

- **`calibration.fit_hmm_emissions()`** is implemented but expects a
  pre-labelled state sequence. Producing those labels (offline EM or a
  hand-tuned heuristic) is the operator's job. The default priors in
  `StudentTHMM.from_default_priors()` are a cold-start placeholder.
- **`hpo.run_hpo`** is now wired via `make_csv_data_provider(csv_path, symbol,
  start_frac, end_frac)` (see hpo.py `__main__`). The CSV streamer
  delivers `(mandate, env, state_advancer)` tuples to `evaluate_objective`
  and `_run_episode` drives the env tick-by-tick. A real HPO study still
  needs ≥1 week of mandates per slice (train / eval / holdout); the
  default `--n-trials 3` invocation is for plumbing verification only.
- **TCN training data.** The TCN ships with random-init weights; it must
  be supervised-trained on labeled shock events from the calibration
  dataset before mandate generation produces a tradeable signal. The
  recommended target is the same `T_actual` proxy used by the heat solver
  calibration: positive label if a shock occurred within the next
  `MAX_DIFFUSION_TICKS`, else negative.
- **PPO checkpoint flow.** Live and shadow PPOAgent are mirrored at engine
  boot (initial KL ≈ 0). If `PPO_LIVE_CHECKPOINT` env var (or the default
  `./calibration/ppo_weights_<SYMBOL>.pt`) exists, the live agent loads
  it and the shadow agent is initialized as a copy. `train_ppo.py`
  produces the checkpoint; `eval_ppo.py` compares it against naive
  baselines. See [EXECUTION_TRAINING.md](EXECUTION_TRAINING.md).

---

## 10. Performance Notes

- Inference budget per tick (CPU, single thread): TCN forward ~ 1 ms,
  heat solver ~ 0.3 ms, PPO act() ~ 0.5 ms. Total well under the 50 ms
  loop budget.
- The shadow trainer is GPU-bound. On RTX 5070 Ti, a single PPO update
  over a 1–2 k step batch completes in 100–300 ms. The 30 s training
  interval gives plenty of headroom.
- The replay buffer uses a `deque` keyed on wall-clock time. Memory cost
  for 24 h at one transition every 50 ms is roughly 1.7 M entries × ~0.5 KB
  ≈ 850 MB. If memory is constrained, decimate the push rate in
  `engine._execute_mandate()`.

---

## 11. What to Look At First If Things Go Wrong

| Symptom                                       | First place to look                         |
| --------------------------------------------- | ------------------------------------------- |
| No mandates ever fire                         | `physics_state.ood_flag` always True? Check whether `latest_<SYMBOL>.json` exists — without calibration the cold-start identity prior makes Mahalanobis ≈ ‖x‖₂ and any large `ce_ratio` trips OOD. Run `fit_ood_from_csv.py`. Also verify TCN trained and `TURBULENCE_THRESHOLD` not too high. |
| `MANDATE_SUPPRESSED_OOD` floods the log       | OOD calibration missing or stale. Boot log will say `cold-start prior` instead of `loaded calibration from ...`. Refit with `fit_ood_from_csv.py`. |
| `MANDATE_SUPPRESSED_POSITION_LIMIT` floods    | Working as designed — Layer 2 keeps firing same-direction mandates but aggregate position is at cap. Check whether the OBI signal is genuinely one-sided or whether something upstream is biased. Position will unstick when OBI flips. |
| Mandates fire but suppressed by attrition guard | `LAMBDA_SENSITIVITY_DELTA` too aggressive, or book genuinely fragile |
| Lots of forced terminal fills                 | `BASE_EXECUTION_WINDOW` too short for current vpin levels; tune `GAMMA_INV` higher |
| Shadow agent never promotes                   | KL stuck > 0.1 (training data covers a different regime than live), or replay too small |
| `MAX DRAWDOWN BREACHED` with `is=True`        | Accumulated execution-cost runaway. Inspect `mean_IS` distribution in `_recent_episodes`; agent may be paying full spread on every fill. |
| `MAX DRAWDOWN BREACHED` with `mtm=True`       | Directional exposure loss. Check `pos` and `peak_mtm` in the log line. Either the position is too big, or mid moved sharply. In replay, often a CSV session-boundary jump — confirm with mid in surrounding ticks. |
| Drift monitor flips frequently                | Microstructure regime change (fee schedule? new HFT entrant?). Recalibrate immediately. |
| Replay shows reasonable mahal_dist but live mode floods OOD | Live `OODDetector` somehow not loading `latest_<SYMBOL>.json` — verify calibration path and JSON schema. Boot log should report the loaded α value. |

---

## 12. Where the Math Lives

For traceability, here are the locations in code where the central equations
live:

| Equation                                            | Location                                       |
| --------------------------------------------------- | ---------------------------------------------- |
| `VPIN = |V_buy - V_sell| / V_total`                 | `VPINToxicityTracker.vpin`                     |
| Student-t reweight `w = (ν+1)/(ν+d²)`               | `StudentTKalmanFilter.update()`                |
| Student-t log-pdf                                   | `student_t_logpdf()`                           |
| Forward filter `α' ∝ (Aᵀα) ⊙ p(o\|s)`               | `StudentTHMM.update()`                         |
| Mahalanobis `d = √((x−μ)ᵀΣ⁻¹(x−μ))`                 | `OODDetector.evaluate()`                       |
| EWLS slope `c = Σwᵢxᵢyᵢ / Σwᵢxᵢ²`                  | `fit_alpha_calibration_c()`                    |
| Drift MAPE                                          | `CalibrationDriftMonitor.record()`             |
| Attrition `adjusted = raw · exp(-λ · vpin · d_bps)` | `HeatEquationSolver._solve_equilibrium_price()`|
| CFL `Δt = 0.5 · Δx² / α`                            | `HeatEquationSolver.solve()`                   |
| Fee-adjusted IS                                     | `ExecutionEnv._step_is_fee_adj()`              |
| GAE                                                 | `PPOTrainer._compute_gae()`                    |
| Polyak `θ ← τθ_shadow + (1-τ)θ`                     | `polyak_update()`                              |


---

**Layer 2 (TCN) training findings, dataset inventory, density thresholds, and per-symbol results live in [LAYER2_TRAINING.md](LAYER2_TRAINING.md).** That document is updated as new training runs land, while this one stays focused on engine architecture and runbook material.
