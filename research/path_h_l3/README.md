# Path H — L3 (order-by-order) microstructure data

L2-aggregate features carried real directional signal in §14 of
[LAYER2_TRAINING.md](../../docs/LAYER2_TRAINING.md) (F2 = 0.106, +10pp
precision at H = 100 ticks) but at magnitudes too small to clear maker
+ taker round-trip cost (gross edge 0.06 bps per trade vs 5.5 bps cost).
The architectural premise — that *snapshots of book state* contain
enough information to predict tradeable directional moves — was
empirically falsified at the L2 venue/time-resolution combination.

Hypothesis: the queue dynamics that *precede* L2 state changes — per-
order arrivals, cancellations, modifications, hidden-order inference
from price-improvement events, order lifespan distributions — carry
information that L2 50 ms snapshots aggregate away. Tested by capturing
true order-by-order events ("Path H").

This directory is the Phase 1 data-acquisition pipeline for that test.
The strategic plan is in [L3_RESEARCH_PLAN.md](L3_RESEARCH_PLAN.md);
deployment notes in [DEPLOY_HARVESTER.md](DEPLOY_HARVESTER.md).

## Venue: Bitfinex public L3

Bitfinex's public WebSocket at `wss://api-pub.bitfinex.com/ws/2` emits
true L3 data via:

- **`book`** channel with `prec=R0` → per-`ORDER_ID` adds / modifies /
  cancels (raw book, no aggregation)
- **`trades`** channel → executed trades with aggressor side inferred
  from `AMOUNT` sign

No authentication required. Together they cover the L3 feature
candidates in [L3_RESEARCH_PLAN.md §2](L3_RESEARCH_PLAN.md) except taker
order-id linkage on trades — a minor downgrade, partially recoverable
by correlating `te` trade events with `book` delete events at the same
price/timestamp.

The Coinbase Exchange `full` channel went HMAC-auth-only in early 2026;
retail Exchange signup is closed (only retail options are "Continue to
Advanced Trading" — L2 deltas only, no L3 — or "Start Business
Application" priced ~$5k institutional API). Both Coinbase paths were
removed from this branch on 2026-05-12 once the institutional price was
confirmed. `download_tardis_l3_free.py` is preserved as a working
Tardis client for any later reactivation (Tardis 1st-of-month tier now
requires a free account API key; was anonymous before 2026-05-12).

## Files

| File | Purpose |
|---|---|
| `harvest_bitfinex_l3.py` | Live WS harvester. Subscribes to `book(R0)` + `trades` for all configured symbols, writes gzipped JSONL with UTC-date rotation, periodic 5-second flush, exponential-backoff reconnect. |
| `probe_bitfinex_ws.py` | Diagnostic probe. Connects briefly, prints raw frames + category counts. |
| `aggregate_mbo_events.py` | Event-clock aggregator (Phase 1.5). Canonical `OrderEvent` schema, vendor-parser dict; `parse_bitfinex_l3` is stubbed pending the long-capture wire-up. |
| `download_tardis_l3_free.py` | Tardis client preserved for historical record (paid-only as of 2026-05-12). |
| `DEPLOY_HARVESTER.md` | AWS EC2 deployment recipe (originally Coinbase-flavored; AWS-side instructions — instance type, systemd unit, gz rotation — carry over to the Bitfinex script unchanged). |
| `L3_RESEARCH_PLAN.md` | Strategic plan: feature candidates, success criteria, phase breakdown. |

## Verified by smoke test (2026-05-12)

- `probe_bitfinex_ws.py --duration 10 --symbol tBTCUSD` → 67 messages in
  10 s; all categories present (snapshot, update, trade:te, trade:tu, hb).
- `harvest_bitfinex_l3.py --symbols tBTCUSD --status-interval-s 3` → 287
  recoverable lines in ~12 s; gz file flushes every 5 s (readable
  mid-run with an EOFError-tolerant reader; final clean close on
  graceful shutdown only).
- Rate: ~24 events/s for `tBTCUSD` alone; ~50–100/s expected across the
  default 3-symbol set (`tBTCUSD` / `tETHUSD` / `tSOLUSD`). 3–4 day
  accrual targets ~15–40 M events, multi-GB compressed.

## Running the long capture

```powershell
cd research/path_h_l3
& "..\..\.venv\Scripts\python.exe" -u harvest_bitfinex_l3.py `
    --out-dir .\l3_data `
    --log-file .\l3_data\bitfinex_harvester.log
```

Files land at `l3_data/bitfinex_l3_<YYYYMMDD>.jsonl.gz`. Each UTC
midnight rotates to a fresh file with no gap.

Stop: **Ctrl+C in the same PowerShell window** is the only way on
Windows to trigger Python's `finally` block (clean gz close).
`Stop-Process -Force` will kill the process but lose the last in-flight
deflate block; the periodic 5-second flush still preserves all but the
last few seconds of data. On Linux/EC2, SIGINT works as expected.

For multi-day captures, EC2 is more reliable than a workstation that
might reboot or sleep. See `DEPLOY_HARVESTER.md`.

## Outstanding work

1. **Long capture (3–4 days).** Local workstation or EC2; targets
   ~15–40 M events, ~multi-GB compressed.
2. **Wire `parse_bitfinex_l3`** into `aggregate_mbo_events.py`'s
   `VENDORS` dict. Mapping (canonical `EventType` enum in
   `aggregate_mbo_events.py`):
   - `book` update with `PRICE != 0`, `ORDER_ID` not previously seen → `EventType.ADD`
   - `book` update with `PRICE != 0`, `ORDER_ID` known → `EventType.MODIFY`
   - `book` update with `PRICE == 0` → `EventType.CANCEL` (size from cached `ORDER_ID` state)
   - trade `te` with `AMOUNT > 0` → `EventType.TRADE`, `aggressor_side = "buy"`
   - trade `te` with `AMOUNT < 0` → `EventType.TRADE`, `aggressor_side = "sell"`
   - trade `tu` → skip (duplicates `te`)
   - heartbeat / info / subscribed / snapshot → skip in aggregator; the
     snapshot is consumed once at parser init to seed the live-orders
     dict.

   Bitfinex interleaves book + trades for all symbols in one file (each
   `chanId` identifies which channel + symbol); the parser must read
   `subscribed` events at the top of the stream to build the
   `chanId → (channel, symbol)` map, then partition events accordingly
   and emit one CSV per symbol.

   Test after wiring:

   ```powershell
   python aggregate_mbo_events.py --vendor bitfinex `
       --in l3_data\bitfinex_l3_<YYYYMMDD>.jsonl.gz `
       --out ..\..\calibration\l3_ticks_BTC.csv `
       --tick-events 100
   ```

## Subsequent phases

### Phase 2 — L3 sensors (~2–3 days)

Build `layer1_l3_sensors.py` at the project root per
[L3_RESEARCH_PLAN.md §4.2](L3_RESEARCH_PLAN.md):

- `OrderBookReconstructor` — full L3 book state from events (start here; the rest depend on it)
- `OrderArrivalRateTracker`
- `CancellationVelocityTracker`
- `OrderLifespanTracker`
- `HiddenOrderDetector`
- `QueueDepletionTracker`
- `AggressorSequenceTracker`

Each gets a synthetic-fixture unit test.

### Phase 3 — Distribution check (~1 day)

Apply the discipline from [research/path_g/](../path_g/) (denominator
floors, tanh saturation calibrated to empirical 95th percentile).
Confirm pre-shock KS-D values are non-trivially above the L2 baseline
of ~0.06.

### Phase 4 — Training (~2 days)

1. Add `TCN_L3_INPUT_CHANNELS` to `config.py` (target 10–15 channels)
2. Add `LAYER1_MODE = "l2_snapshots" | "l3_events"` env-overridable flag
3. Branch [training/train_stream.py](../../training/train_stream.py)
   feature stack on `LAYER1_MODE`
4. Train at directional H = 100 (matches L2 §14.1 protocol)
5. 80/20 temporal split per coin

### Phase 5 — Backtest + decision gate (~1 day)

[training/backtest_directional.py](../../training/backtest_directional.py)
works with any feature CSV.

Success criteria ([L3_RESEARCH_PLAN.md §6](L3_RESEARCH_PLAN.md)):

- F2 (directional, val) ≥ 0.15 (vs L2's 0.106)
- AND/OR gross edge ≥ 2 bps per trade (vs L2's 0.06 bps)

If yes → Phase 6 (engine integration). If no → document the negative
result and consider Pivot 4 (longer horizons, different model classes,
different venues).

## L2 reference baselines

Every L3 metric should be measured against these:

| Metric | L2 value |
|---|---:|
| F2 (directional, H = 100, val) | 0.106 |
| Max precision / base rate | 1.5× |
| Gross edge per trade | 0.06 bps |
| Net PnL after 5.5 bps round-trip | negative |

## Platform notes

- **Windows signal handling**: `kill -INT` from Git Bash and
  `Stop-Process -Force` do NOT trigger the harvester's SIGINT handler.
  Only Ctrl+C in the same PowerShell console does. Not an issue on
  Linux/EC2.
- **No taker/maker order_id on trades**: Bitfinex `trades` channel
  emits `[TRADE_ID, TS_MS, AMOUNT, PRICE]` with no `maker_order_id` or
  `taker_order_id`. Trade → book-event linkage is approximate (same
  price/timestamp match) but workable for the aggressor-sequence +
  hidden-order features.
