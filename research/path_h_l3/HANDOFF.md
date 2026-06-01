# Session Handoff — `research/path-h-l3`

> Created 2026-05-11 when the prior session was running low on context.
> This document is the entry point for the next Claude session resuming
> Path H (L3) work. Read this first, then [L3_RESEARCH_PLAN.md](L3_RESEARCH_PLAN.md)
> for the full strategic context.

## TL;DR of where we are

**Phase 2 complete. Per-symbol training works on all three captured
symbols.** Full L3 sensor stack (`layer1_l3_sensors.py`) trained on the
11-day Bitfinex corpus gives positive net per-trade edge on BTC, ETH,
and SOL simultaneously — first time in the project's history. The BTC
result was sanity-checked against the bearish val window and the random-
entry bootstrap; it's real signal, not trend capture.

Best per-symbol configs (q=1.0 fills, **11 bps round-trip fee assumption — see caveat below**, thr=0.90):
- BTC H=1000  →  **+11.96 bps/trade**  (215/216 long-precision in a -2.1% val window)
- ETH H=500   →  **+4.97 bps/trade**   (68.5% win on N=820)
- SOL H=300   →  **+26.34 bps/trade**  (66.0% win on N=853)

### Fee caveat — read before quoting these numbers

The bps figures above use the backtest's default fee model: 1 bp maker
+ 10 bps taker = **11 bps round-trip**. `backtest_l3_directional.py`
lines 16, 252-256 label this "Bitfinex retail" — **this label is wrong**.
Standard Bitfinex public retail crypto is 10 bps maker / 20 bps taker =
**30 bps round-trip** (19 bps higher per trade). Re-stated at standard
retail (linear adjustment, valid because fees apply per trade as a
fraction of notional):

| Symbol | @ 11 bps RT (headline) | @ 30 bps RT (standard retail) |
|---|---:|---:|
| BTC H=1000 thr=0.90 | +11.96 | **−7.04** ✗ |
| ETH H=500 thr=0.90  | +4.97  | **−14.03** ✗ |
| SOL H=300 thr=0.90  | +26.34 | **+7.34** ✓ |

Only SOL clearly survives standard retail. BTC/ETH need either higher
thresholds (the gross-edge-vs-threshold curve was steep; thr=0.95/0.97
untested), a fee-discount stack (LEO ~25% off taker, volume tiers,
affiliate code), or a cheaper venue with comparable signal. The
backtest's "Bitfinex retail" defaults should be relabeled or replaced
with realistic numbers — see outstanding item 0.

### Post-Phase-2 follow-up analysis (2026-05-24)

After the fee correction, ran a full in-sample test suite on SOL
(`analysis/analysis_sol_fulltest.py`, `_firerate_per_day.py`,
`_threshold_sweep.py`). Three findings — read these before quoting
any Phase 2 numbers:

**1. The "11-day val window" framing is misleading.** With train_frac=0.8
the val window is only 2.12 days (2026-05-21 14:07 → 2026-05-23 16:58).
The full capture is 11 days; the *test* slice is 2.

**2. All 423 SOL trades fire on a single calendar day (2026-05-23).** Val
spans 3 calendar dates. On May 21-22, the model produces predictions but
its confidence never crosses thr=0.95 (pred_max on May 22 = 0.945; on May
21 = 0.900). On May 23, fire rate jumps to 4.31% (423 trades, +11.97
bps/trade). All Phase 2 SOL statistics are therefore *single-day* evidence.

**3. The threshold is doing real noise-filtering** — not just narrowing
the model's view. Threshold sweep on val:

| thr  | May 21         | May 22         | May 23 (fire day) |
|------|---------------:|---------------:|------------------:|
| 0.90 | N=17, −12.34   | N=12, −22.47   | N=824, +9.05      |
| 0.93 | N=3, −5.60     | N=1, −7.06     | N=603, +9.82      |
| 0.95 | (no fire)      | (no fire)      | N=423, +11.97     |
| 0.97 | (no fire)      | (no fire)      | N=218, +19.45     |
| 0.98 | (no fire)      | (no fire)      | N=118, +23.66     |

Forcing May 21-22 to fire by lowering threshold reveals they were genuinely
non-signal days, not "almost-signal." Threshold ~0.95 is correctly tuned.

**Train-set fire-rate diagnostic.** Model fires on 9/10 training days
(varies 0.01%-7.30% fire rate, with one zero-fire day on May 18). So
the model is structurally a *rare-firer* with fluent firing across
regimes — not pathologically concentrated. The single-day val pattern
is consistent with "val happened to contain 2 non-fire days + 1 fire
day," not "model only knows one regime."

**Refined outlook.** SOL is a calibrated rare-firer-but-real *candidate*.
Per-fire-day economics are known for exactly one fire-day (May 23,
+11.97 bps at thr=0.95). Bootstrap CI on those 423 trades is
[+7.40, +16.51] (doesn't cross zero); 100th percentile vs random-entry
baseline (+41.68 bps over random). All true *within May 23*.

Generalization remains unverified: we need w2 to give us multiple
independent fire-days to estimate the per-fire-day distribution.
**Window-2 harvest is the gating experiment**, not "additional
confirmation." See [NEXT_PHASE_PLAN.md](NEXT_PHASE_PLAN.md).

### Window-2 forward-test outcome (2026-06-01) — Scenario A confirmed

Window 2 harvested 2026-05-24 → 2026-06-01 (8.43 days, 143k SOL ticks
+ 187k BTC ticks + 178k ETH ticks). Forward-tested the existing
per-symbol weights at standard retail fees on **all of w2 as out-of-sample**:

| Symbol | H | N | Win% | Mean bps | Fire days | Positive fire-days | CI 95% |
|---|---:|---:|---:|---:|---|---|---|
| SOL | 300 | 3,137 | 5.4% | **−30.04** | 8/9 | **0/8** | [−30.82, −29.16] |
| SOL | 500 | 4,590 | 15.1% | **−27.79** | 9/9 | **0/9** | [−28.83, −26.63] |
| SOL | 1000 | 1,379 | 20.7% | **−18.45** | 8/9 | 1/8 | [−20.21, −16.61] |
| BTC | 1000 | 1,089 | 3.5% | **−42.52** | 7/9 | 1/7 | [−43.84, −41.05] |

Every config: CI well below zero, ≤ 1 of 7-9 fire-days positive.
Per [NEXT_PHASE_PLAN §3 Scenario A](NEXT_PHASE_PLAN.md): regime overfit
confirmed. The Phase 2 result was a single-fire-day artifact specific
to the 2026-05-23 regime.

### Feature drift diagnostic — the failure mechanism

[analysis/analysis_sol_w2_feature_drift.py](analysis/analysis_sol_w2_feature_drift.py)
computed KS-D between w1 and w2 per channel. The three most load-bearing
channels (from the prior ablation) are the three biggest drifters:

| Channel | KS-D | Mean shift in training-σ | Std ratio w2/w1 | Ablation load (bps) |
|---|---:|---:|---:|---:|
| hidden_trade_rate | **0.239** | −0.52σ | 0.62 | +29.19 |
| lifespan_bid_p50_ms | **0.137** | +0.37σ | 1.16 | +34.49 |
| lifespan_ask_p50_ms | **0.125** | +0.41σ | 1.29 | +49.96 |

The model's prediction-output distribution shifted upward (median
0.524 → 0.564, p95 0.786 → 0.834). Long-fire rate at thr=0.95 went
0.45% → 1.34% (**3×**); short-fire rate barely moved. The model
became systematically over-confident on the long side because drifted
features push the sigmoid output rightward.

Mechanism: microstructure features like `hidden_trade_rate` and
`lifespan_p50` measure *who participates in the market and how they
behave*. Participant mix is week-to-week non-stationary. A model
trained with static normalization stats from w1 cannot generalize
across this kind of distribution shift.

**Horizon doesn't fix this.** SOL improves monotonically with H (−30
at H=300 → −18 at H=1000) but every horizon's CI is well below zero.
H affects what's predicted, not what's seen — the drift is on inputs.

### Status — L3 thesis falsified, feature engineering reset planned

**Phase 2 thesis falsified by w2 forward-test.** Current weights +
current static normalization + current 19-channel raw feature stack
do not generalize across capture windows. The "rare-firer-but-real"
candidate framing from §15.7 was wrong; it was rare-firer-but-overfit.

**L3 microstructure thesis as a research direction is NOT falsified.**
The sensors compute what they should; the harvester+aggregator pipeline
is sound; the channels themselves likely carry information. But the
deployment shape implied by Phase 2 ("train once, normalize once,
deploy with fixed weights") is wrong for non-stationary microstructure
features.

**Next research phase: Phase 3 — drift-robust feature engineering.**
Replace raw drifted channels (`hidden_trade_rate`, `lifespan_p50` bid
and ask) with scale-invariant alternatives (percentile ranks within
rolling windows, ratios of co-moving channels). Add explicit regime
features. Retest the train→forward-test pipeline with the new feature
stack.

L3 PPO / L4 / live-deployment work remains **paused** until a
generalizable signal exists.

See LAYER2_TRAINING.md §15 (chapter), §15.7 (Phase 2 follow-up
analysis), §15.8 (w2 forward-test + drift diagnostic). Phase 3
feature-engineering plan in [NEXT_PHASE_PLAN.md](NEXT_PHASE_PLAN.md) §11.

**Status: Phase 2 thesis falsified on w2 forward-test. Failure
mechanism identified as feature distribution drift on top load-bearing
channels. Phase 3 (drift-robust feature engineering) is the next
research direction. L3 execution policy / live deployment paused.**

## Status as of handoff

- **Branch**: `research/path-h-l3` at `5437c58` (Phase 1.5 committed);
  Phase 2 + per-symbol changes uncommitted as of this writing (about
  to commit).
- **L2 baseline**: `master` at `b72d1d9`; LAYER2_TRAINING.md §13-§14.
- **L3 chapter**: LAYER2_TRAINING.md §15 — full Phase 1.5 → Phase 2 →
  per-symbol → sanity check arc.
- **L3 architecture**: L3_RESEARCH_PLAN.md is the strategic doc;
  Phase 2 implementation matches §4.2's 7-sensor spec.
- **Code (Phase 2)**:
  - `harvest_bitfinex_l3.py` — live capture harvester.
  - `probe_bitfinex_ws.py` — diagnostic probe.
  - `aggregate_mbo_events.py` — thin event-counter that delegates
    feature math to `L3SensorArray`; `parse_bitfinex_l3` yields
    tagged tuples `("event"|"snapshot", symbol, payload)`.
  - `layer1_l3_sensors.py` — 7 sensors (OrderBook + OrderBookReconstructor
    + HiddenOrderDetector + QueueDepletion + OrderArrivalRate +
    CancellationVelocity + OrderLifespan with p50/p95 + AggressorSequence
    with lag-1/5 autocorr) + EventDensity + `L3SensorArray` orchestrator;
    19-channel `FEATURE_COLS`.
  - `train_l3_directional.py` — TCN trainer; auto-imports `FEATURE_COLS`
    from layer1_l3_sensors. Pushed config: 20 epochs, seq_len=200.
  - `backtest_l3_directional.py` — maker entry / taker exit PnL sim
    with Bitfinex retail fees + q-parameterized fill model.
  - `tests/test_layer1_l3_sensors.py` — 28 unit tests, all green.
  - `download_tardis_l3_free.py` — preserved for record.
  - `DEPLOY_HARVESTER.md` — AWS EC2 reference (still Coinbase-flavored).

### Critical implementation details (worth reading before any change)

Three fixes inside `layer1_l3_sensors.py` decided whether features were
sane or garbage:

1. **`L3SensorArray.seed_book` calls `self.book.reset()` before
   re-seeding.** Bitfinex emits a fresh snapshot on every reconnect;
   without reset, stale orders from prior sessions persist and cross
   the new best_bid/best_ask in 99%+ of ticks.
2. **`OrderBook.prune_far_from(last_trade, max_pct=0.01)` runs on
   every event.** Bitfinex's `book(R0, len=25)` doesn't emit deletes
   for orders that fall out of the visible top-25 window when price
   moves. Periodic pruning around the current market price evicts them.
   Without this, even Phase 2 sensors produced 99% negative spreads.
3. **`L3SensorArray.process()` enriches CANCEL events with recovered
   `(price, size)` from the OrderBook before dispatching to sensors.**
   Bitfinex CANCEL frames carry `price=0, amount=±1` (side marker only).
   Without enrichment, QueueDepletionTracker can't tell whether the
   cancel was at top-of-book.

If you refactor any of those three, re-run the smoke test and check
`neg_spread` rate drops back to <5%.

## Resolved by pivot to Bitfinex (2026-05-12)

Coinbase path closed and removed from this branch. Reason chain:

1. Probe confirmed Coinbase Exchange's `full` channel now requires HMAC
   auth: `{"type":"error","message":"Failed to subscribe","reason":
   "level2, level3, and full channels now require authentication"}`.
2. exchange.coinbase.com retail signup is gated — the only retail option
   is "Continue to Advanced Trading" (no L3) or "Start Business
   Application" (institutional Exchange API access priced at ~$5k).
3. Both retail Advanced Trade and Bitfinex are free; Advanced Trade is
   L2-deltas only, Bitfinex's `book` with `prec=R0` is true L3.

**Bitfinex public L3 raw book** is the working venue. No auth required.
`book` channel (`prec=R0`) emits per-`ORDER_ID` adds / modifies /
cancels; `trades` channel gives executed trades with aggressor side via
`AMOUNT` sign. Together they cover all L3 feature candidates in
[L3_RESEARCH_PLAN.md §2](L3_RESEARCH_PLAN.md) except taker-order-id
linkage (minor downgrade — partly recoverable by correlating trade
events with book-delete events at the same price/timestamp).

### Files (this branch)

| File | Purpose |
|---|---|
| `probe_bitfinex_ws.py` | Diagnostic probe for Bitfinex WS — prints raw frames + category counts |
| `harvest_bitfinex_l3.py` | Production harvester — subscribes to book(R0)+trades for all symbols, writes gzipped JSONL with UTC-date rotation, periodic flush every 5s, exponential-backoff reconnect |

### Smoke-test verified (2026-05-12)

- `probe_bitfinex_ws.py --duration 10 --symbol tBTCUSD` → 67 messages in
  10s, all categories present (snapshot, update, trade:te, trade:tu, hb)
- `harvest_bitfinex_l3.py --symbols tBTCUSD --status-interval-s 3` → 287
  recoverable lines in ~12s, gz file flushes every 5s (read mid-run with
  EOFError-tolerant reader; final clean close on graceful shutdown only)
- Rate: ~24 events/sec for tBTCUSD; expect ~50-100/sec across the
  default 3-symbol set (tBTCUSD, tETHUSD, tSOLUSD). 3-4 days accrual
  -> ~15-40M events, multi-GB compressed.

### Run the long harvest

```powershell
cd research\path_h_l3
& "..\..\.venv\Scripts\python.exe" -u harvest_bitfinex_l3.py `
    --out-dir .\l3_data `
    --log-file .\l3_data\bitfinex_harvester.log
```

Files land at `l3_data/bitfinex_l3_<YYYYMMDD>.jsonl.gz`. To stop:
**Ctrl+C in the PowerShell window where it's running** is the only way
to trigger Python's `finally` block on Windows (so the gz closes
cleanly). `Stop-Process` will kill it but lose the last in-flight
deflate block — periodic flush still preserves all but the last few
seconds of data.

### Known constraints

- **Windows signal handling**: `kill -INT` from Git Bash / Stop-Process
  with -Force does NOT trigger the harvester's SIGINT handler. Only
  Ctrl+C in the same PowerShell console works for a clean stop. This
  isn't an issue on EC2 (Linux signals work as expected).
- **No taker/maker order_id on trades**: Bitfinex's `trades` channel
  gives `[TRADE_ID, TS_MS, AMOUNT, PRICE]` but no `maker_order_id` /
  `taker_order_id`. The aggregator can still link trades to the book
  events that disappeared at the same price/timestamp (approximate but
  workable for the aggressor-sequence + hidden-order features).

## Outstanding (in priority order, updated 2026-05-24)

### Done in this session

- ✓ **Standard-retail fee correction sweep** (was item 0): Confirmed
  BTC −7.04 / ETH −14.03 / SOL +7.34 at thr=0.90 with 30 bps RT.
  Extended sweep thr=0.85-0.99: BTC crosses positive at thr=0.98
  (+0.06 bps); ETH never crosses; SOL peaks at thr=0.98 (+22.75 bps).
  ETH not viable at standard retail.
- ✓ **SOL inference-time channel ablation**: 19 single-channel zeroings
  ranked load-bearing channels (lifespan_p50 bid/ask, hidden_trade_rate,
  event_density dominate). Multi-channel ablation (10 "harmful" channels
  zeroed simultaneously) collapsed model — confirms inference-time
  ablation is rank-reliable but not recipe-reliable. See
  [calibration/ablation_sol_summary.json](calibration/ablation_sol_summary.json).
- ✓ **SOL full in-sample test suite** (see "Post-Phase-2 follow-up
  analysis" above): per-day breakdown, train fire-rate, threshold
  sweep, random-entry bootstrap, self-bootstrap CI, drawdown analysis.
  Surfaced the val-window concentration finding (all trades on May 23).

### Active

1. **Window-2 harvest** — IN PROGRESS. Launched with
   `--prefix bitfinex_l3_w2`. Need ~10-14 days of accrual. This is now
   the *gating experiment* for the whole strategy, not "additional
   confirmation." See [NEXT_PHASE_PLAN.md §3](NEXT_PHASE_PLAN.md)
   for the three scenarios and what each unlocks.

2. **Forward-test on w2 when ready** — re-aggregate, then re-run the
   per-symbol weights (SOL H=300 at minimum; BTC H=1000 if BTC-marginal
   case still relevant) on the w2 val slice. Use distinct CSV pattern
   `l3_ticks_w2_{symbol}.csv` so w1 and w2 results are separable.
   Re-run the analysis suite (`analysis/analysis_sol_fulltest.py` plus
   the threshold and fire-rate diagnostics) on the w2 result. Per-fire-day
   distribution across w2 is the answer to "is SOL real or single-day?"

### After w2 (gated on positive result — see NEXT_PHASE_PLAN.md scenarios)

3. **Layer 3 (execution policy) PPO training** — wire SOL L2 signal
   into `training/train_ppo.py` infrastructure (on master). Reward via
   net PnL; state augmentation = L2 signal + recent vol + time-of-day
   + recent flow. Goal: learn enter/skip + maker/taker + sizing +
   exit-timing jointly, replacing the hardcoded backtest policy.
4. **Layer 4 shadow validation** — KL-gated promotion of L3 policy
   via existing `training/validate_layer4.py`. Already structurally
   validated; needs a real L3 policy to gate.
5. **Live deployment on Hyperliquid** — Bitfinex L3 signal → L2 TCN →
   L3 PPO → HL SOL-PERP fills. Cross-venue verification in shadow first.
   Bitfinex is **not US-accessible for trading** (account creation
   blocked); HL is the natural execution venue (US-accessible via wallet,
   ~4 bps round-trip vs Bitfinex retail's 30, and engine.py already
   supports `EXCHANGE_ID=hyperliquid`).

### After w2 (gated on negative result)

6. **L2 reset** — different label horizon, different feature engineering,
   possibly different architecture or different instrument. The L3
   microstructure thesis itself isn't necessarily wrong; the current
   implementation may not be the right cut. See NEXT_PHASE_PLAN §3A.

## Key files (quick reference)

| File | What it is | Status |
|---|---|---|
| `L3_RESEARCH_PLAN.md` | Strategic plan | Venue references (Coinbase) stale; §4.2 sensor list now implemented |
| `LAYER2_TRAINING.md` §15 | L3 chapter | Phase 1.5 → Phase 2 → per-symbol → sanity check; full arc |
| `HANDOFF.md` | This file | Authoritative state |
| `harvest_bitfinex_l3.py` | Live WS harvester | Production-ready; ran 11 days stable |
| `probe_bitfinex_ws.py` | Diagnostic probe | Use to verify a fresh subscription before re-harvesting |
| `aggregate_mbo_events.py` | Thin event-counter | Delegates to L3SensorArray; yields tagged tuples |
| `layer1_l3_sensors.py` | 7-sensor stack + orchestrator | 19 channels, 28 unit tests green |
| `train_l3_directional.py` | TCN trainer | Pushed config: --epochs 20 --seq-len 200 |
| `backtest_l3_directional.py` | PnL backtester | Per-symbol or pooled via --symbols |
| `tests/test_layer1_l3_sensors.py` | Unit tests | `pytest tests/test_layer1_l3_sensors.py -v` |
| `calibration/tcn_weights_l3_phase2_persym_*.pt` | Per-symbol weights | gitignored; regenerate via training sweep |
| `calibration/l3_directional_metrics_phase2_persym_*.json` | Per-symbol metrics | Committed; documents every sweep result |
| `analysis/analysis_sol_fulltest.py` | Per-trade dump + bootstrap CI + drawdown + payoff distribution + per-day + time-of-day breakdown | Self-contained; reads weights + stats + CSV from `calibration/` |
| `analysis/analysis_sol_firerate_per_day.py` | Per-day fire-rate + prediction quantile distribution on train AND val | Discriminator between rare-firer-but-real and regime-overfit failure modes |
| `analysis/analysis_sol_threshold_sweep.py` | Threshold sweep with per-day breakdown | Confirms threshold is doing noise-filtering, not just narrowing model view |
| `deprecated/download_tardis_l3_free.py` | Tardis client | Preserved as historical record; Tardis paid-only |
| `NEXT_PHASE_PLAN.md` | Detailed progression plan post-Phase-2 | Three w2 scenarios + L3/L4 architecture path + live deployment plan |
| `DEPLOY_HARVESTER.md` | EC2 deploy guide | Coinbase-flavored; AWS-side instructions still valid |

### Reproducing the headline result

```powershell
cd research\path_h_l3
# 1. Aggregate (~5-10 min on full 12 daily files)
$files = (Get-ChildItem .\l3_data\bitfinex_l3_2026*.jsonl.gz |
          Where-Object Name -notlike '*partial*' | ForEach-Object FullName)
& "..\..\.venv\Scripts\python.exe" aggregate_mbo_events.py `
    --vendor bitfinex --in $files `
    --out "calibration\l3_ticks_{symbol}.csv" --tick-events 100

# 2. Per-symbol training (3 coins x 3 horizons; ~30-45 min on RTX 5070 Ti)
foreach ($sym in @('tBTCUSD', 'tETHUSD', 'tSOLUSD')) {
  foreach ($h in @(300, 500, 1000)) {
    & "..\..\.venv\Scripts\python.exe" train_l3_directional.py `
        --csv-dir calibration --symbols $sym `
        --horizon $h --seq-len 200 --epochs 20 `
        --weights-out "calibration\tcn_weights_l3_phase2_persym_${sym}_H${h}.pt" `
        --stats-out "calibration\l3_feature_stats_phase2_persym_${sym}.json" `
        --metrics-out "calibration\l3_directional_metrics_phase2_persym_${sym}_H${h}.json"
  }
}

# 3. Backtest the best per-symbol configs (~5 min)
& "..\..\.venv\Scripts\python.exe" backtest_l3_directional.py `
    --horizon 1000 --seq-len 200 --symbols tBTCUSD `
    --weights "calibration\tcn_weights_l3_phase2_persym_tBTCUSD_H1000.pt" `
    --stats-in "calibration\l3_feature_stats_phase2_persym_tBTCUSD.json" `
    --queue-fill-prob 1.0 --thresholds "0.85,0.90"
# Repeat for ETH at H=500 and SOL at H=300.
```

## L2 reference baselines to beat

Every L3 metric should be measured against these:

| metric | L2 value |
|---|---:|
| F2 (directional, H=100, val) | 0.106 |
| max prec / base rate | 1.5× |
| gross edge per trade | 0.06 bps |
| Net PnL after 5.5 bps round-trip | negative |

## Conversation memory bookmarks

For the next session, key context from the prior conversation:
- User is a student on a budget — paid vendors are out
- L2 investigation took ~3 days; user is engaged, fast at pivot
   decisions, and prefers honest assessments over hopium
- User's strategic instinct is sharp (caught the in-sample artifact
  issue at §14.1; correctly proposed event-clock over wall-clock;
  correctly killed Path A inventory framework as scope creep)
- Auto mode is the default working style — execute, don't ask
- Logs accumulate fast; gitignored in `logs/`
