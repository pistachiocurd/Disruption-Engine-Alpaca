# Next Phase Plan — Post-Phase-2 Follow-up + Layer Progression

> Created 2026-05-24 after Phase 2 follow-up analysis.
>
> **UPDATE 2026-06-01**: Window-2 forward-test resolved §3 in the
> **Scenario A (regime overfit)** case. Phase 2 thesis falsified on
> out-of-sample data. Feature drift diagnostic identified the failure
> mechanism. §1-§9 below are preserved as historical context — the
> active forward plan is now **§11 (Phase 3: drift-robust feature
> engineering)** at the bottom of this document.
>
> Read in this order:
> 1. [HANDOFF.md](HANDOFF.md) — current operational state
> 2. §11 below — the actual forward plan
> 3. §1-§9 — historical context (most still relevant; Scenario A
>    "next actions" at §3 are now what's being executed)

## 1. Where we are (2026-05-24)

The Phase 2 L3 thesis produced one candidate signal worth taking
further: **SOL at H=300, thr=0.95-0.98** trained on the 11-day
Bitfinex L3 corpus. Per-symbol training cleared all original
L3_RESEARCH_PLAN §6 success criteria.

Three rounds of follow-up analysis have eroded the headline number's
strength without falsifying the signal:

| Layer | Headline | After correction |
|---|---|---|
| Phase 2 commit | "All three coins net-positive" | True only at the 11-bps-RT fee assumption used in the backtest defaults — not standard retail |
| Standard retail fees (30 bps RT) | "BTC −7, ETH −14, SOL +7.9" at thr=0.90 | Confirmed empirically; only SOL clearly survives. Threshold sweep shows BTC crosses positive at thr=0.98 (+0.06 bps); ETH never crosses |
| Val-window concentration | "SOL +11.97 over 423 trades, 100th percentile vs random" | All 423 trades on day 3 of 3 val days. Effectively single-fire-day evidence |
| Train fire-rate diagnostic | (newly investigated) | Model fires on 9/10 training days. Structural rare-firer, not pathological one-day overfit |
| Threshold sweep on non-fire val days | (newly investigated) | Lower thresholds on May 21-22 fire and lose money. Threshold is doing real noise filtering |

**Current honest framing**: SOL is a calibrated rare-firer-but-real
*candidate*. We have positive per-trade evidence from exactly one
real-world fire-day (2026-05-23). The model's training-set pattern
shows fluent rare-firing across regimes, which is consistent with
the val behavior — May 21-22 weren't fire-days, May 23 was. But we
need multiple independent fire-days in window-2 to estimate the
per-fire-day distribution and confirm the strategy generalizes.

The L3 thesis (microstructure features carry leading signal
invisible at L2) is not falsified. The implementation produces a
calibrated rare-firer; whether it generalizes is open.

## 2. Refined success criteria (post-analysis)

Phase 2 met L3_RESEARCH_PLAN §6 on paper. That criterion was set
before we understood the val-window concentration risk and before
fee assumptions were nailed down. Refined criteria for "go live":

| Criterion | Original §6 | Refined |
|---|---|---|
| Lift over baseline | ≥ 1.5× | unchanged |
| Gross edge | ≥ 2 bps/trade | unchanged |
| Net after standard retail fees (30 bps RT) | not measured | ≥ 5 bps/trade on at least 60% of fire-days |
| Multi-fire-day generalization | not measured | net positive on ≥ 4 of w2's fire-days, with 95% CI on pooled mean > 0 |
| Per-fire-day variance | not measured | std/mean ≤ 1.5 across w2 fire-days (signals consistency, not lucky outliers) |
| Per-fire-day N | not measured | ≥ 30 trades per fire-day for valid in-day inference |

The first two are unchanged. The last four are the new bars and are
the things w2 data is needed to evaluate.

## 3. Three scenarios for w2 outcome — and what each unlocks

Probabilities are subjective priors based on what we've seen so far.

### Scenario A — Regime overfit (~25%)

**Symptoms in w2 forward-test:**
- Model fires on 0-2 of 12 w2 days
- On fire-days, per-trade bps is consistently negative or barely above 0
- Bootstrap CI on pooled w2 fire-day trades crosses zero or is negative

**Conclusion**: SOL signal was a one-day regime exploit (May 23
specific). The L3 microstructure features as currently engineered
produce a regime-detector, not transferable directional alpha.

**Next actions:**
- DO NOT proceed to L3/L4 work — would be training an execution
  policy on noise
- Re-examine L2 architecture:
  - Different label horizons (try H=50, 1000, 3000)
  - Different feature engineering (drop the p95 lifespan channels
    flagged in ablation; try alternative aggressor / hidden-order
    formulations)
  - Different model architecture (transformer, gated recurrent,
    or simpler — current TCN may be overparameterized)
  - Different instruments (ETH/BTC didn't survive standard retail
    fees either; consider DOGE, LINK, or other mid-cap altcoins
    where microstructure inefficiencies are more likely)
- Possibly: revisit the venue assumption. Bitfinex L3 is the only
  free L3 source we have; if it's structurally a quiet venue (low
  flow → low signal), pivot to a higher-flow venue (Kraken? Binance
  via reconstruction?) may be necessary
- Document the negative result rigorously — single-day concentration
  is a discoverable failure mode that future similar projects should
  check for

**L3/L4 work: PAUSED** until L2 produces a new candidate signal
that survives the refined §2 criteria.

### Scenario B — Rare-firer but real (~50%)

**Symptoms in w2 forward-test:**
- Model fires on 4-7 of 12 w2 days
- Per-fire-day mean bps is positive on most (≥ 60%) of fire-days
- Pooled 95% CI on w2 fire-day trades is positive
- Per-fire-day variance is moderate (some +25, some +5, occasional
  small negative)

**Conclusion**: SOL signal is real but event-driven. Not the "always-on"
pipeline framed in the original Phase 6 plan — closer to "rare
high-confidence opportunity detector."

**Next actions:**
- Keep L2 as-is (SOL TCN). Maintenance only.
- **Layer 3 (execution policy) becomes the central next phase.**
  This is exactly the layer where context-dependent decision-making
  lives: given an L2 signal, given current regime, given recent
  fill quality and inventory — should we trade, and how?
- L3 PPO should learn to:
  - **Filter** L2 signals by additional features (vol regime, time-of-day,
    recent flow, basis vs HL perp). Some L2 fires may be unprofitable
    in certain regimes; L3 learns to skip those
  - **Size** by confidence. Currently size=1.0 always; PPO should
    scale with the L2 probability (and possibly the L2 prediction's
    distance from threshold)
  - **Order type** — currently maker-limit entry hardcoded; PPO
    should pick taker market when the signal is strong enough and
    fill probability matters
  - **Exit timing** — currently force-exit at +H ticks; PPO should
    consider early-exit on adverse moves, late-exit if conviction
    strengthens during the hold
- Per-fire-day economics enrichment: train L3 not just on the L2
  signal stream but also on features that distinguish good fire-days
  from bad ones (realized vol over last 1h/4h, basis to HL perp,
  funding rate, recent aggressor imbalance)

**L3/L4 work: GREEN-LIGHTED**, with focus on regime/context detection
as the primary L3 learning target.

### Scenario C — Robust signal (~20%)

**Symptoms in w2 forward-test:**
- Model fires on most days (≥ 8 of 12)
- Per-fire-day mean bps is consistently positive (≥ 80% of fire-days)
- Pooled 95% CI is well above zero
- Per-fire-day variance is low (tight cluster around the mean)

**Conclusion**: SOL signal is closer to always-on than expected.
The May 23 single-day val pattern was an artifact of val happening
to span 2 non-fire days; the rare-firer label understates the model.

**Next actions:**
- Full speed ahead on the original Phase 6 plan
- L3 PPO simpler — main job is sizing + order type, less need for
  regime gating
- L4 validation and live deployment can proceed in parallel with
  any further L2 refinement

**L3/L4 work: FULL SPEED**

### Truly catastrophic (~5%)

**Symptoms**: Strategy collapses on w2. Model fires often but
loses money on every fire-day; or signal completely degrades.

**Conclusion**: Something is fundamentally wrong — possibly with
the val split (look for in-sample leakage between train and val we
missed), the feature engineering (sensors might be reading vendor
artifacts not market structure), or the data itself (Bitfinex L3
schema might have shifted in w2).

**Next actions**: pause everything, audit the pipeline end-to-end
before continuing.

## 4. Layer architecture progression

The project's layer architecture (separate from "L3 = market data
level"):

```
┌─ Live Bitfinex L3 WS ─────────────────────────────────┐
│                                                       │
│  ▼                                                    │
│  Layer 1 (sensors)                                    │
│  - layer1_l3_sensors.py                               │
│  - 19-channel feature vector per tick                 │
│  - WORKING                                            │
│                                                       │
│  ▼                                                    │
│  Layer 2 (signal model)                               │
│  - train_l3_directional.py + per-symbol TCN weights   │
│  - Sigmoid output ∈ [0, 1] per tick                   │
│  - CANDIDATE (SOL); w2 pending                        │
│                                                       │
│  ▼                                                    │
│  Layer 3 (execution policy)                           │
│  - training/train_ppo.py (on master)                  │
│  - Decides: enter/skip, maker/taker, size, exit       │
│  - SCAFFOLDED; previous PPO attempt (equity context)  │
│    collapsed to deterministic always-taker because    │
│    L2 signal magnitude was zero                       │
│                                                       │
│  ▼                                                    │
│  Layer 4 (shadow validation)                          │
│  - training/validate_layer4.py (on master)            │
│  - KL-gated promotion of new L3 policies             │
│  - STRUCTURALLY VALIDATED; awaits real L3 policy      │
│                                                       │
│  ▼                                                    │
│  Order placement                                      │
│  - ccxt.pro Hyperliquid (engine.py supports today)    │
│  - Bitfinex L3 → HL SOL-PERP cross-venue              │
│  - Not yet wired for L3-signal-driven trading         │
│                                                       │
└────────────────────────────────────────────────────────┘
```

What's working: L1 (sensor stack with 28 unit tests green). L2
(per-symbol weights trained, threshold calibrated). Engine has ccxt
support for Hyperliquid via `EXCHANGE_ID=hyperliquid`. L4 validator
passes its structural checks (KL bounded, regimes non-degenerate,
gate fires, promotion observed, polyak math exact).

What's the gap: L2 generalization (w2). L3 trained on real L2 signal
(prior PPO attempt was on equity where L2 was structurally zero —
PPO had nothing to learn from). L3 → engine wiring. Live L2 inference
loop (currently aggregate_mbo_events.py is batch-only; live needs
a streaming variant).

## 5. Layer 3 design (assumes scenario B or C from §3)

### 5.1 What L3 PPO should learn

The current backtest hardcodes a 5-decision execution policy. L3
PPO should learn each:

| Decision | Current | L3 should learn |
|---|---|---|
| Enter or skip | Threshold cross | Conditional on L2 signal AND regime features (vol, time-of-day, flow, basis) |
| Maker vs taker entry | Maker limit always | Pick taker market when L2 signal is very strong and waiting risks missing the move |
| Size | Fixed 1.0 | Scale with L2 confidence + recent fill quality + portfolio risk |
| Exit timing | Force taker at +H ticks | Early-exit on adverse moves; late-exit if L2 signal strengthens during hold |
| Capital allocation | Implicit (1.0 size) | Reserve capital for high-EV fire-days; idle on low-confidence days |

### 5.2 State / observation space

L3 PPO observes:
- L2 signal probability (current tick)
- L2 signal recent history (window of last K signals — captures
  whether the model is in a fire-day or quiet regime)
- Realized vol features (1h, 4h, 24h realized over recent ticks)
- Time-of-day (UTC hour, encoded as sin/cos)
- Recent flow features (aggressor imbalance, hidden-trade rate from
  L1 sensors — overlap with L2 input is fine; PPO sees them at a
  different abstraction)
- Inventory state (current position, recent PnL, drawdown)
- Cross-venue basis (if cross-venue execution is in scope)

### 5.3 Action space

Initial design (discrete, can extend to continuous later):
- **Enter**: {long-maker, long-taker, short-maker, short-taker, skip}
- **Size**: {0.25, 0.5, 1.0, 2.0} — multipliers of base size
- **Exit-trigger**: {hold-to-H, exit-on-flip, exit-on-adverse-N-bps}

### 5.4 Reward function

Net PnL per closed trade in bps of notional. Include:
- Realized fees (matches the L2 backtest accounting)
- Holding-time penalty (small) to discourage gratuitous late exits
- Capital-allocation reward (small bonus for skip when subsequent
  realized move was negative — i.e., correctly avoided)

### 5.5 What to avoid (lessons from prior PPO collapse)

The c71850c commit notes: *"The trained PPO converged to deterministic
always-taker behavior under the L2 mandate stream + 1-level synthesized
replay book. The training pipeline is structurally correct; the env
signal is just too low-magnitude for PPO to learn maker/taker
discrimination at equity zero-fee."*

To avoid the same trap:
1. **Use the real SOL L2 signal as the mandate stream** — not a
   synthetic or weakened version. The whole point is that SOL fire-days
   produce +10-25 bps signal; PPO should learn to act on this.
2. **Use realistic Hyperliquid fees in the env** — −1 bps maker
   rebate, 4.5 bps taker. Maker/taker discrimination is a non-trivial
   ~6 bps decision; PPO should be able to learn it.
3. **Replay the actual L3 book state during fills** — not a 1-level
   synthesis. The L3 sensor pipeline already maintains book state;
   reuse it in the env.
4. **Curriculum learning**: start with L3 PPO seeing only fire-days
   (where signal exists) to bootstrap, then mix in non-fire-days
   for skip-discipline training.

### 5.6 Infrastructure reuse

From master (`training/train_ppo.py`, `training/eval_ppo.py`,
`training/validate_layer4.py`):
- PPO training loop, policy network, advantage estimation — reusable
- Replay buffer + env wrapper interface — reusable
- KL gate + polyak averaging — reusable

What needs to be written:
- New env wrapper: `BitfinexL3SolEnv` that:
  - Iterates the SOL L2 signal stream chronologically
  - Maintains live L3 book state for accurate fill simulation
  - Augments observation with the §5.2 state features
  - Computes reward per §5.4
- Reward shaping experiments (the L2 signal alone may not be enough
  for stable PPO convergence; may need intermediate shaped rewards)

Estimated effort: 1-2 weeks for env wrapper + initial PPO training
loop + tuning. Another 1 week for shadow-mode integration with L4.

## 6. Layer 4 (shadow validation) — gating L3 promotion

L4 already passes its structural checks (per c71850c). When L3 PPO
produces a candidate policy:

1. **Polyak shadow training**: L4 maintains a Polyak-averaged shadow
   policy that mirrors the live policy and is incrementally updated
   from L3 training output. Shadow runs in parallel with live with
   zero capital allocation.
2. **KL bound**: shadow → live promotion is gated by KL(shadow || live)
   < threshold. Prevents reckless policy swaps.
3. **Regime check**: shadow must produce non-degenerate actions across
   multiple regimes (not collapsed to "always-taker" or "always-skip").
4. **Gate fires**: promotion criteria must actually trigger in offline
   eval before live promotion.

The L4 validator script tests all of this on synthetic policies today.
The same flow will apply to the SOL L3 policy when it's trained.

## 7. Live deployment path

### 7.1 Why Hyperliquid (not Bitfinex)

Bitfinex is **not US-accessible for trading** — KYC blocks US persons
from opening accounts. Bitfinex L3 WS data feeds are public (no auth),
so we can keep using them for the signal source, but execution must
happen elsewhere.

Hyperliquid is the right execution venue:
- US-accessible via web3 wallet (no KYC)
- ~5 bps taker, −1 bps maker rebate (≈ 4 bps round-trip — much
  cheaper than any retail spot venue)
- Already wired in engine.py: `EXCHANGE_ID=hyperliquid`
- SOL-PERP is the natural instrument (basis to SOL-spot is negligible
  at minute scale, which is our H=300 hold horizon)

Tradeoff: cross-venue (Bitfinex signal → HL execution) and
cross-instrument (spot signal → perp execution). Both are minor at
the bps level for SOL given its tight cross-venue arb and tiny
funding accrual over short holds. Verify empirically in shadow mode.

### 7.2 Deployment sequence

1. **Live L2 inference loop**: wire `harvest_bitfinex_l3.py`'s WS
   client + parser into a streaming inference path. Reuse the
   `L3SensorArray` orchestrator from `layer1_l3_sensors.py` (it's
   already designed for stateful event processing). Emit feature
   vectors at each tick-boundary (every 100 events, matching
   `aggregate_mbo_events.py --tick-events 100`) and run the SOL TCN
   forward to produce a signal stream.
2. **L3 policy load**: trained PPO weights loaded into engine.py
   alpha layer. Same pattern as the existing PPO_LIVE_CHECKPOINT
   support already in engine.py.
3. **Order routing**: L3 action → ccxt.pro Hyperliquid order via
   existing engine paths. SOL-PERP symbol mapping; size in USD
   notional respecting account limits.
4. **Shadow first, then capital**: run L4-validated L3 policy in
   shadow for 1-2 weeks, comparing simulated fills against what
   actual HL fills would have been. Promote to live with small
   capital allocation, scale up after another 1-2 weeks of green
   shadow + live agreement.

### 7.3 Risk controls (out of scope for L2 work but needed before live)

- Max position size per symbol
- Max daily loss (kill switch)
- Max consecutive losing trades (pause and inspect)
- Bitfinex WS disconnection handling — what happens to live positions
  if signal source goes dark mid-trade?
- Hyperliquid disconnection handling — order cancellation policy
- Funding rate cost accounting — perp positions accrue funding;
  short holds at H=300 ticks (minutes) accrue trivially but worth
  measuring

## 8. Timeline (depending on w2 outcome)

| Phase | Scenario B | Scenario C | Scenario A |
|---|---|---|---|
| w2 harvest | 2 weeks wall-clock | same | same |
| w2 forward-test | 1-2 days analysis | same | same |
| L3 env wrapper + PPO training | 2-3 weeks | 1-2 weeks (less regime complexity needed) | N/A |
| L4 validation of L3 policy | 1 week | same | N/A |
| Live L2 inference loop wiring | 1 week | same | N/A |
| Shadow mode (no capital) | 1-2 weeks | 1 week | N/A |
| Live with small capital | 1-2 weeks ramp | same | N/A |
| **Total to scaled live** | **~7-10 weeks** | **~5-7 weeks** | **N/A — back to L2 reset, ~unknown** |

These are best-case estimates assuming work proceeds without major
discoveries. Realistic estimates are 1.5-2× longer.

## 9. Decision gates

Concrete go/no-go points to avoid sliding into L3/L4 work prematurely:

| Gate | Trigger | Pass | Fail |
|---|---|---|---|
| **G1: w2 ready** | w2 harvest reaches 10+ days | Aggregate + forward-test | Wait |
| **G2: Generalization** | Forward-test on w2 complete | §3 scenario B or C | §3 scenario A → L2 reset |
| **G3: L3 env validates** | PPO env wrapper written | env produces non-degenerate signal stream + accurate fill model | Fix env before training |
| **G4: L3 PPO convergence** | PPO training runs | Policy non-degenerate, beats hardcoded backtest baseline | Reward shaping or hyperparam pass |
| **G5: L4 promotion** | Trained L3 candidate ready | KL bounded, regimes non-degenerate, gate fires | Re-train or relax KL bound carefully |
| **G6: Shadow agreement** | Live shadow runs ≥ 1 week | Simulated vs would-be-real fills agree within tolerance | Diagnose execution model mismatch |
| **G7: Live with small capital** | Shadow green for ≥ 2 weeks | Allocate $1k initial | Extend shadow |
| **G8: Scale capital** | Small-capital live ≥ 2 weeks green | Scale 5-10× | Stay small, investigate |

## 10. What to avoid

Anti-patterns from this project's history that the new phase should
not repeat:

1. **Headline-first reporting**. Phase 2 was announced as "thesis
   confirmed" before the val-window concentration was understood.
   Going forward: report a metric with its context (single-day vs
   multi-day, single-regime vs multi-regime, fee assumption).
2. **In-sample optimism**. The threshold sweep, ablation, and
   bootstrap CI all produced positive in-sample numbers that
   *all* turned out to be conditional on the same May 23 data.
   Going forward: any in-sample test must be reported with the
   independent-day count it actually pulls from.
3. **Premature L3 work**. The earlier PPO attempt on equity collapsed
   because there was no L2 signal to act on. Don't start L3 PPO until
   the L2 candidate clears G2 (w2 generalization).
4. **Skipping shadow mode**. Even with a working L3 policy, shadow
   first. Live trading on a strategy that was only validated offline
   is reckless.
5. **Over-engineering the unprofitable case**. If w2 puts us in
   scenario A, don't try to "fix" the SOL model with ever-more-clever
   ablations or threshold tunings — reset the L2 architecture or
   pivot instruments.

---

## 11. Phase 3 — Drift-robust feature engineering (added 2026-06-01)

> Phase 2 was falsified on w2: Scenario A confirmed. The failure
> mechanism is documented (feature distribution drift on the top
> load-bearing channels). This section is the new active forward
> plan. The pre-2026-06-01 sections above (§1-§10) are historical
> context for why we're here.

### 11.1 What's being changed and why

The diagnostic (`analysis/analysis_sol_w2_feature_drift.py`) shows that
the three most load-bearing channels (per the prior ablation) are also
the three biggest drifters between w1 and w2:

| Channel | Ablation load | KS-D | Drift mode |
|---|---:|---:|---|
| `hidden_trade_rate` | +29.19 bps | **0.239** | −0.52σ mean shift + 38% variance compression |
| `lifespan_bid_p50_ms` | +34.49 bps | **0.137** | +0.37σ mean shift + 16% variance expansion |
| `lifespan_ask_p50_ms` | +49.96 bps | **0.125** | +0.41σ mean shift + 29% variance expansion |

These are *raw values* that depend on the current mix of market
participants. Participant mix is week-to-week non-stationary. Any
model that uses these as raw inputs with static normalization will
fail across capture windows.

The fix is **scale-invariant feature engineering**: replace raw values
with features that are robust to distribution shift by construction.
This addresses the root cause; rolling-window normalization would only
patch the symptom.

### 11.2 Candidate feature designs

For each problematic channel, three drift-robust alternatives to test:

**A. Percentile rank within rolling window.** Replace raw value with
its percentile rank against a recent window. E.g., `hidden_trade_rate`
becomes "what fraction of the last 6 hours of ticks had
hidden_trade_rate ≤ this tick's value?" Output ∈ [0, 1], scale-invariant
by construction.

- Rolling window: 6 hours, 24 hours, or "last N events" — tradeoff
  between responsiveness and statistical stability
- Cost: ~1 extra O(log N) operation per tick per channel (sorted
  deque); negligible
- Risk: information loss — percentile rank discards magnitude

**B. Ratios of co-moving channels.** Replace single channels with
ratios that should be more stable. Examples:
- `hidden_trade_rate / (hidden_trade_rate + visible_trade_rate)` →
  *fraction* of trades that are hidden, robust to overall trading
  intensity drift
- `lifespan_bid_p50 / lifespan_ask_p50` → bid/ask lifespan asymmetry,
  robust to overall lifespan regime shift
- `lifespan_p50 / event_interval_ms` → lifespan in "event-time
  units" rather than wall-time

**C. Z-score within rolling window.** Compute the channel's mean and
std over recent N ticks, output `(x - rolling_mean) / rolling_std`.
Continuous-valued (preserves magnitude info), but assumes near-Gaussian
within the window.

### 11.3 Recommended feature stack v2 design

For each of the 19 v1 channels, decide whether it stays raw, becomes
percentile-rank, becomes ratio, or gets dropped:

| v1 channel | v2 disposition | Rationale |
|---|---|---|
| event_density_per_s | **Drop or keep raw** | Already scale-relative (per-second normalization); but absolute event rate drifts. Keep raw initially. |
| arrival_rate_bid/ask_per_s | **Ratio: bid/(bid+ask)** | Imbalance is what matters, not absolute |
| cancel_rate_bid/ask_per_s | **Ratio: bid/(bid+ask)** | Same logic |
| aggressor_imbalance | **Keep raw** | Already a ratio ∈ [-1, 1], inherently scale-invariant |
| spread_bps | **Keep raw** | Already in bps; scale-invariant by definition |
| top_bid_size, top_ask_size | **Percentile rank within 6h window** | Sizes drift with market regime |
| depth_imbalance_top5 | **Keep raw** | Already a ratio ∈ [-1, 1] |
| hidden_trade_rate | **Percentile rank within 6h window** | The biggest drifter — needs hardest treatment |
| queue_depletion_bid/ask_per_s | **Percentile rank or ratio: bid/(bid+ask)** | Rate-based, drift-prone |
| lifespan_bid_p50_ms | **Percentile rank within 6h window** | Top-2 drifter — needs hardest treatment |
| lifespan_ask_p50_ms | **Percentile rank within 6h window** | Top-1 drifter — needs hardest treatment |
| lifespan_bid_p95_ms | **DROP** | Already flagged as net-noise in ablation; drifts and adds variance |
| lifespan_ask_p95_ms | **DROP** | Same |
| aggressor_autocorr_lag1 | **Keep raw** | Already an autocorrelation ∈ [-1, 1] |
| aggressor_autocorr_lag5 | **Keep raw** | Same |

Expected v2 input width: ~14-15 channels (down from 19), most either
inherently bounded or percentile-rank-transformed.

### 11.4 New L1 sensor work

Implementation tasks for the sensor layer:

1. **`RollingPercentileRanker`** class — maintains a rolling
   `sortedcontainers.SortedList` (or numpy heap) of recent values
   per channel; emits percentile rank per tick. Window configurable.
   Add to `layer1_l3_sensors.py`.

2. **Adapt `L3SensorArray.snapshot()`** to apply the per-channel
   transform table from §11.3. Outputs the v2 FEATURE_COLS list
   (different from v1).

3. **Update `FEATURE_COLS`** in `layer1_l3_sensors.py` to the v2 set.
   Update unit tests to verify ranges (most outputs should be in
   well-defined bounded intervals).

4. **Backward-compat path**: keep v1 outputs available behind a
   `feature_stack_version="v1"` flag so we can rebuild old CSVs if
   needed for cross-comparison.

### 11.5 Validation protocol

The key methodological change for v2 — **train+validate must include
multi-window evidence from day one**, not single-window with optimistic
"will it generalize" framing. Specifically:

1. **Train v2 model** on w1 (2026-05-12 → 23) — exactly the same
   training protocol as Phase 2, just with the v2 feature stack
2. **Mandatory in-sample sanity check**: same per-day fire-rate +
   threshold sweep diagnostics that revealed Phase 2's single-fire-day
   concentration. Run on w1 train + w1 val.
3. **Mandatory out-of-sample forward-test**: full w2 forward-test
   with per-day breakdown. The SAME analysis suite that revealed
   Phase 2's failure mechanism — `analysis_sol_w2_forward_test.py`
   and `analysis_sol_w2_feature_drift.py`.
4. **Pass criteria** (must clear all four):
   - In-sample: fires on ≥ 4 of w1 train days AND ≥ 1 of w1 val days
     with positive per-fire-day mean
   - Out-of-sample: ≥ 60% of w2 fire-days have positive mean bps
   - Bootstrap CI on pooled w2 fire-day trades > 0
   - Feature drift KS-D < 0.05 on all retained channels (the v2
     features should be drift-robust BY CONSTRUCTION; this is the
     check that the engineering worked)

### 11.6 Experiments to run, in order

1. **(1 day) Implement v2 sensor stack** — `RollingPercentileRanker`
   + transforms per §11.3. Unit tests for each new transform.
   Re-aggregate w1 raw events with v2 sensors → `l3_ticks_v2_{symbol}.csv`.
   Verify v2 KS-D between w1 train and w1 val on the recoded channels.
2. **(0.5 day) Re-aggregate w2** with v2 sensors. Verify KS-D between
   w1 and w2 on the v2 stack drops below 0.05 on the previously-drifted
   channels. If it doesn't, the feature engineering didn't work and
   we need to iterate before training.
3. **(0.5 day) Train per-symbol SOL models** on v2 w1 train slice
   at H ∈ {300, 1000} (the two that were least bad on v1). Sanity
   check in-sample fire pattern matches v1's rare-firer profile.
4. **(0.5 day) Forward-test v2 models on v2 w2**. Same analysis
   suite as before. Compare to v1 baseline (−18 to −30 bps mean).
5. **Decision gate**: did pooled w2 bps cross zero? If yes, signal
   exists in v2 — proceed to Layer 3 work (with the new feature
   stack). If no, deeper structural issue — consider symbol pivot
   or label-design changes.

### 11.7 What this DOESN'T attempt

- **Rolling normalization on raw features.** Considered but rejected
  by user preference for the more fundamental fix. Could be added
  later if v2 also fails on structural drift (e.g., the rolling
  window itself becomes the lever).
- **Pivoting symbols.** Stay on BTC/ETH/SOL initially. If v2 also
  fails on these three across w1+w2, then symbol pivot becomes
  the next experiment.
- **Adding regime-conditioning features** (recent vol, time-of-day,
  basis-to-HL). Useful future direction but adds complexity; first
  test whether drift-robust features alone are enough.
- **New label horizons.** v1 work already explored H=300/500/1000;
  no additional horizon experiments planned until v2 features clear
  the drift gate.

### 11.8 Honest expectation

Going in: I expect v2 to **improve** the w2 forward-test result
significantly (from −18-30 bps toward 0 or slightly positive) because
the drift mechanism is well-identified and the percentile-rank fix
is the textbook drift-robust transformation. But I would not predict
high confidence that v2 will be clearly profitable — there may be
additional structural issues (e.g., label horizon, the value of the
signal itself) that fixing drift alone won't solve.

If v2 lands in the −5 to +5 bps range on w2, that's progress worth
continuing into regime-conditioning + Layer 3 design. If v2 also
lands in the −20 bps range, the failure isn't just drift — it's
something deeper about the signal hypothesis, and a more aggressive
reset (different label, different instrument family) is warranted.

The point of Phase 3 isn't to ship; it's to localize where the L3
research hypothesis is right vs wrong with higher resolution than
Phase 2 provided.
