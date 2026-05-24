"""
aggregate_mbo_events.py — Phase 1.5 of L3_RESEARCH_PLAN.md.

Reads a raw MBO (Market-by-Order) event stream and aggregates it into
event-count ticks for TCN consumption. Uses an event clock (not wall-clock)
so the resulting tick stream has uniform information density: every K
events produces one row, regardless of how long that took. This is the
architectural fix to the §13-§14 quantization problem where 500ms could
contain zero info during quiet periods or thousands of state changes
during a shock.

Aggregated features per event-count window:
    arrival_rate_bid_per_s, arrival_rate_ask_per_s
    cancel_rate_bid_per_s,  cancel_rate_ask_per_s
    cancel_to_fill_bid,     cancel_to_fill_ask
    mean_lifespan_bid_ms,   mean_lifespan_ask_ms
    aggressor_imbalance     (buy_agg - sell_agg) / total_trades
    event_density_per_s     K / window_duration_s

Vendor-specific parsing is delegated to `parse_<vendor>_mbo(path)`
generators. Currently stubbed pending L3_RESEARCH_PLAN.md §5 Phase 1
data acquisition (Databento or Tardis 7-day MBO slice for BTC/ETH/SOL).
A `synthetic` parser reads our canonical OrderEvent CSV format — useful
for testing the aggregation logic before vendor data lands.

Usage:
    python aggregate_mbo_events.py --vendor synthetic \\
        --in calibration/l3_events_sample.csv \\
        --out calibration/l3_ticks_BTC.csv \\
        --tick-events 100

    python aggregate_mbo_events.py --vendor databento \\
        --in vendor_data/dbn_btcusd_20260301.dbn \\
        --out calibration/l3_ticks_BTC.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

import polars as pl


# ============================================================================
# Canonical event format — all vendor parsers convert to this
# ============================================================================

class EventType(str, Enum):
    ADD = "add"        # new order arrival
    CANCEL = "cancel"  # order canceled (no fill)
    MODIFY = "modify"  # order replaced (size or price update)
    TRADE = "trade"    # order executed (full or partial fill)


@dataclass(slots=True)
class OrderEvent:
    timestamp_ms: int
    event_type: EventType
    order_id: int                            # 0 = no resting order (hidden trade)
    side: str                                # "buy" or "sell"
    price: float
    size: float                              # 0 for full-cancel; remaining_size for partial
    aggressor_side: Optional[str] = None     # for TRADE events only


# ============================================================================
# Event-clock aggregator
# ============================================================================

class EventClockAggregator:
    """Buffers OrderEvents and emits a feature snapshot every `k_events`.

    Phase 2: feature math lives in `L3SensorArray` (layer1_l3_sensors.py).
    This class is now a thin event-counter that:
      1. Forwards each event to the sensor array (which holds all state).
      2. Tracks last_trade_price (the label source for directional training).
      3. On every k_events-th event, returns sensor_array.snapshot() plus
         timestamp_ms and last_trade_price as a single dict.
    """

    def __init__(self, k_events: int = 100,
                 sensor_array: Optional["L3SensorArray"] = None):
        # Delayed import to avoid a circular dep when layer1_l3_sensors.py
        # imports OrderEvent/EventType from this module.
        from layer1_l3_sensors import L3SensorArray as _L3SensorArray
        self.k_events = k_events
        self.sensor_array = sensor_array or _L3SensorArray()
        self._window_end_ms: int = 0
        self._event_count: int = 0
        # Last observed trade price — emitted on every tick for the
        # downstream trainer's directional label generation.
        self._last_trade_price: float = 0.0

    def seed_book(self, snapshot: list) -> None:
        """Route a vendor-emitted book snapshot to the sensor array's OrderBook."""
        self.sensor_array.seed_book(snapshot)

    def process(self, event: OrderEvent) -> Optional[dict]:
        """Forward event to sensors; emit aggregated tick every k_events."""
        self._window_end_ms = event.timestamp_ms
        if event.event_type == EventType.TRADE and event.price > 0:
            self._last_trade_price = event.price

        self.sensor_array.process(event)
        self._event_count += 1

        if self._event_count >= self.k_events:
            return self._emit_and_reset()
        return None

    def _emit_and_reset(self) -> dict:
        snap = self.sensor_array.snapshot()
        out = {
            "timestamp_ms": self._window_end_ms,
            "last_trade_price": self._last_trade_price,
            **snap,
        }
        self._event_count = 0
        return out


# ============================================================================
# Vendor-specific parsers
# ============================================================================
# Each parser is a generator that yields OrderEvent objects. Vendor schemas
# vary; the parser's job is to do the field mapping and event-type
# classification. See L3_RESEARCH_PLAN.md §3 for the venue decision.
# ============================================================================

def parse_databento_mbo(path: Path) -> Iterator[OrderEvent]:
    """Parse a Databento MBO file. Schema TBD — depends on whether we
    purchase the .dbn binary format or the CSV variant. STUB until data
    is acquired."""
    raise NotImplementedError(
        "Databento MBO parser not yet wired. Acquire a sample file first, "
        "then implement the field mapping. See L3_RESEARCH_PLAN.md §3."
    )


def parse_tardis_mbo(path: Path) -> Iterator[OrderEvent]:
    """Parse a Tardis.dev MBO CSV. Their incremental_book_L3 channel
    publishes one row per event. STUB until data is acquired."""
    raise NotImplementedError(
        "Tardis MBO parser not yet wired. Acquire a sample file first, "
        "then implement the field mapping. See L3_RESEARCH_PLAN.md §3."
    )


def parse_bitfinex_l3(paths: list[Path]) -> Iterator[tuple]:
    """Parse Bitfinex L3 raw-book + trades capture files.

    Files are gzipped JSONL produced by `harvest_bitfinex_l3.py`. Each
    file interleaves book(R0) and trades messages for multiple symbols;
    the chanId -> (channel, symbol) map is reconstructed from the
    `subscribed` events at the top of every connection's stream
    (reconnects produce fresh chanIds, also captured).

    Yields TAGGED tuples (Phase 2 contract):
        ("event",    symbol, OrderEvent)   -- normal ADD/MODIFY/CANCEL/TRADE
        ("snapshot", symbol, snapshot_list) -- vendor's initial book dump
    The caller routes events to per-symbol EventClockAggregators (so
    cancellation/lifespan/arrival features stay symbol-scoped) and seeds
    each aggregator's OrderBook from snapshots.

    Bitfinex wire-format -> canonical event mapping (HANDOFF.md):
      book update [ORDER_ID, PRICE, AMOUNT]:
        PRICE != 0, ORDER_ID new   -> ADD     (side: AMOUNT>0=buy, else=sell)
        PRICE != 0, ORDER_ID known -> MODIFY  (size: abs(AMOUNT))
        PRICE == 0                 -> CANCEL  (side from cached ADD; AMOUNT
                                                sign is also the side marker
                                                +-1 but we already know it)
      trade "te" [TRADE_ID, TS_MS, AMOUNT, PRICE] -> TRADE
        (aggressor_side: AMOUNT>0=buy, else=sell; order_id=0 since
         Bitfinex doesn't expose maker/taker order_ids on trades)
      trade "tu"  -> skip (duplicates "te" ~1s later)
      "hb", info, subscribed, snapshot -> skip in event stream
        (snapshot orders are NOT emitted as ADD because they were
         placed before our capture window — their later cancel will
         have no matching submit_t and the aggregator silently
         discards it. Acceptable: first ~minute per-file undercount
         is negligible across the 11-day corpus.)

    Files are read in lexicographic order (which is chronological for
    YYYYMMDD-suffixed names). The chan_map and per-symbol known_orders
    sets persist across files so reconnects mid-corpus don't lose
    state. EOFError on the last file is tolerated (the harvester may
    have been Stop-Process'd without a clean gzip close).

    Timestamps: book events on Bitfinex don't carry their own ts. We
    inherit the last-seen trade ts as the event's timestamp. With
    event-clock aggregation at k=100 events, this approximation is
    accurate to within milliseconds (since 100 events typically span
    <1s on a busy symbol).
    """
    chan_map: dict[int, tuple[str, str]] = {}
    # Per-symbol set of currently-resting ORDER_IDs we've observed an ADD for.
    # Used to distinguish first-time ADD vs MODIFY, and to recognize CANCELs
    # of pre-existing (snapshot) orders that should be silently dropped.
    known_orders: dict[str, set[int]] = {}
    last_ts_ms: int = 0

    for path in sorted(paths):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Control / housekeeping messages are dicts with "event".
                    if isinstance(msg, dict):
                        if msg.get("event") == "subscribed":
                            cid = msg.get("chanId")
                            channel = msg.get("channel")
                            symbol = msg.get("symbol")
                            if isinstance(cid, int) and channel and symbol:
                                chan_map[cid] = (channel, symbol)
                                known_orders.setdefault(symbol, set())
                        # info / error / pong / etc. -> ignore
                        continue

                    if not isinstance(msg, list) or len(msg) < 2:
                        continue

                    cid = msg[0]
                    if cid not in chan_map:
                        # Unknown channel (subscription ack lost or out of order).
                        continue
                    channel, symbol = chan_map[cid]

                    if channel == "book":
                        payload = msg[1]
                        if payload == "hb":
                            continue
                        if not isinstance(payload, list):
                            continue
                        # Snapshot: [[oid, price, amount], ...] vs single update [oid, price, amount].
                        if payload and isinstance(payload[0], list):
                            # Phase 2: route snapshot to caller so OrderBook
                            # can be seeded. ALSO track snapshot order_ids in
                            # our local known_orders set — otherwise CANCELs
                            # for them get dropped below ("order not in ko"),
                            # leaving stale orders in the book that crash
                            # spread/depth features (observed: spread went
                            # to -156 bps in one BTC test row).
                            ko_seed = known_orders.setdefault(symbol, set())
                            for row in payload:
                                if not isinstance(row, (list, tuple)) or len(row) < 3:
                                    continue
                                try:
                                    seed_oid = int(row[0])
                                    seed_price = float(row[1])
                                except (TypeError, ValueError):
                                    continue
                                if seed_price > 0:
                                    ko_seed.add(seed_oid)
                            yield ("snapshot", symbol, payload)
                            continue
                        if len(payload) != 3:
                            continue
                        try:
                            order_id = int(payload[0])
                            price = float(payload[1])
                            amount = float(payload[2])
                        except (TypeError, ValueError):
                            continue

                        # Hold all book events until the first trade
                        # establishes a real reference timestamp. Book
                        # frames don't carry ts; events tagged with ts=0
                        # would produce nonsensical window durations
                        # (1.7e12 ms = 1970-to-2026 gap) in the very
                        # first aggregated tick. Lost events: ~the first
                        # few seconds of book activity in the very first
                        # file; subsequent files inherit last_ts_ms.
                        if last_ts_ms == 0:
                            continue

                        ko = known_orders.setdefault(symbol, set())
                        if price == 0.0:
                            # Cancel (or filled-and-removed; book channel
                            # doesn't distinguish — that's fine because
                            # trades come on a different channel).
                            if order_id not in ko:
                                # Pre-existing or already-removed order; drop.
                                continue
                            ko.discard(order_id)
                            # AMOUNT sign on cancels is the side marker (+1=bid, -1=ask).
                            side = "buy" if amount > 0 else "sell"
                            yield ("event", symbol, OrderEvent(
                                timestamp_ms=last_ts_ms,
                                event_type=EventType.CANCEL,
                                order_id=order_id,
                                side=side,
                                price=0.0,
                                size=0.0,
                            ))
                        else:
                            side = "buy" if amount > 0 else "sell"
                            size = abs(amount)
                            if order_id in ko:
                                yield ("event", symbol, OrderEvent(
                                    timestamp_ms=last_ts_ms,
                                    event_type=EventType.MODIFY,
                                    order_id=order_id,
                                    side=side,
                                    price=price,
                                    size=size,
                                ))
                            else:
                                ko.add(order_id)
                                yield ("event", symbol, OrderEvent(
                                    timestamp_ms=last_ts_ms,
                                    event_type=EventType.ADD,
                                    order_id=order_id,
                                    side=side,
                                    price=price,
                                    size=size,
                                ))

                    elif channel == "trades":
                        payload = msg[1]
                        if payload == "hb":
                            continue
                        # Initial trades snapshot is [[trade...], ...]; skip.
                        if isinstance(payload, list):
                            continue
                        # Single trade: [chanid, "te"|"tu", [trade_id, ts_ms, amount, price]]
                        if payload not in ("te", "tu") or len(msg) < 3:
                            continue
                        if payload == "tu":
                            # "tu" is a refresh of an earlier "te"; skip to avoid double-count.
                            continue
                        trade = msg[2]
                        if not isinstance(trade, list) or len(trade) < 4:
                            continue
                        try:
                            ts_ms = int(trade[1])
                            amount = float(trade[2])
                            price = float(trade[3])
                        except (TypeError, ValueError):
                            continue
                        last_ts_ms = ts_ms
                        aggressor = "buy" if amount > 0 else "sell"
                        yield ("event", symbol, OrderEvent(
                            timestamp_ms=ts_ms,
                            event_type=EventType.TRADE,
                            order_id=0,  # Bitfinex doesn't expose maker/taker order_ids on trades
                            side="",
                            price=price,
                            size=abs(amount),
                            aggressor_side=aggressor,
                        ))
        except EOFError:
            # Last file may be truncated (harvester killed mid-flush). The
            # decompressor stops cleanly at the last SYNC_FLUSH boundary.
            print(f"[parse_bitfinex_l3] tolerated EOFError on {path.name}", file=sys.stderr)
            continue


def parse_synthetic_mbo(path: Path) -> Iterator[OrderEvent]:
    """Parse our own canonical OrderEvent CSV format. Useful for testing
    aggregation logic before vendor data lands. Schema (CSV columns):

        timestamp_ms, event_type, order_id, side, price, size, aggressor_side

    Where event_type ∈ {"add","cancel","modify","trade"}, side ∈ {"buy","sell"},
    and aggressor_side is "buy"/"sell"/"" (only set for trade rows).
    """
    df = pl.read_csv(path)
    for row in df.iter_rows(named=True):
        aggr = row.get("aggressor_side")
        if aggr is not None and not isinstance(aggr, str):
            aggr = None
        if aggr == "":
            aggr = None
        yield OrderEvent(
            timestamp_ms=int(row["timestamp_ms"]),
            event_type=EventType(row["event_type"]),
            order_id=int(row["order_id"]),
            side=row["side"],
            price=float(row["price"]),
            size=float(row["size"]),
            aggressor_side=aggr,
        )


SINGLE_FILE_VENDORS = {
    "databento": parse_databento_mbo,
    "tardis": parse_tardis_mbo,
    "synthetic": parse_synthetic_mbo,
}
MULTI_SYMBOL_VENDORS = {
    "bitfinex": parse_bitfinex_l3,  # yields (symbol, OrderEvent); fans out to per-symbol CSVs
}
ALL_VENDORS = list(SINGLE_FILE_VENDORS) + list(MULTI_SYMBOL_VENDORS)


# ============================================================================
# CLI
# ============================================================================

def _run_single_file(args: argparse.Namespace) -> None:
    parser_fn = SINGLE_FILE_VENDORS[args.vendor]
    if len(args.input_paths) != 1:
        raise SystemExit(
            f"--vendor {args.vendor} expects exactly one --in path; "
            f"got {len(args.input_paths)}"
        )
    aggregator = EventClockAggregator(k_events=args.tick_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ticks = 0
    with open(args.out, "w", newline="") as f_out:
        writer = None
        for event in parser_fn(args.input_paths[0]):
            tick = aggregator.process(event)
            if tick is None:
                continue
            if writer is None:
                writer = csv.DictWriter(f_out, fieldnames=list(tick.keys()))
                writer.writeheader()
            writer.writerow(tick)
            n_ticks += 1
    print(f"Wrote {n_ticks:,} ticks ({args.tick_events} events/tick) to {args.out}")


def _run_multi_symbol(args: argparse.Namespace) -> None:
    parser_fn = MULTI_SYMBOL_VENDORS[args.vendor]
    if "{symbol}" not in str(args.out):
        raise SystemExit(
            f"--vendor {args.vendor} produces per-symbol CSVs; --out must "
            f"contain a {{symbol}} placeholder (e.g. calibration/l3_ticks_{{symbol}}.csv)"
        )
    aggregators: dict[str, EventClockAggregator] = {}
    out_files: dict[str, "csv.DictWriter"] = {}
    file_handles: dict[str, object] = {}
    n_ticks: dict[str, int] = {}
    n_events: dict[str, int] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_agg(symbol: str) -> None:
        """Lazy per-symbol initialization: aggregator + output file."""
        if symbol in aggregators:
            return
        aggregators[symbol] = EventClockAggregator(k_events=args.tick_events)
        out_path = Path(str(args.out).format(symbol=symbol))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(out_path, "w", newline="")
        file_handles[symbol] = fh
        out_files[symbol] = None  # type: ignore[assignment]
        n_ticks[symbol] = 0
        n_events[symbol] = 0
        print(f"[{symbol}] opened {out_path}", file=sys.stderr)

    try:
        for item in parser_fn(args.input_paths):
            # Phase 2 contract: tagged tuples
            #   ("event", symbol, OrderEvent)
            #   ("snapshot", symbol, snapshot_list)
            kind = item[0]
            symbol = item[1]
            _ensure_agg(symbol)

            if kind == "snapshot":
                aggregators[symbol].seed_book(item[2])
                continue

            if kind != "event":
                # Unknown tag — ignore defensively.
                continue

            event = item[2]
            n_events[symbol] += 1
            tick = aggregators[symbol].process(event)
            if tick is None:
                continue
            if out_files[symbol] is None:
                writer = csv.DictWriter(file_handles[symbol], fieldnames=list(tick.keys()))
                writer.writeheader()
                out_files[symbol] = writer
            out_files[symbol].writerow(tick)
            n_ticks[symbol] += 1
    finally:
        for fh in file_handles.values():
            fh.close()

    print(f"\nDone. Events per symbol / ticks emitted at k={args.tick_events}:")
    for sym in sorted(n_ticks):
        print(f"  {sym:<10}  {n_events[sym]:>12,} events  ->  {n_ticks[sym]:>9,} ticks")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vendor", choices=ALL_VENDORS, required=True,
                   help="MBO source format. 'bitfinex' takes multiple gz files "
                        "and fans out per-symbol CSVs.")
    p.add_argument("--in", dest="input_paths", type=Path, nargs="+", required=True,
                   help="Path(s) to raw MBO file(s). Single path for "
                        "synthetic/databento/tardis; one or more for bitfinex.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output CSV path. For bitfinex, must contain {symbol} "
                        "placeholder (e.g. calibration/l3_ticks_{symbol}.csv).")
    p.add_argument("--tick-events", type=int, default=100,
                   help="Events per tick (event-clock K). Default 100; the "
                        "L3_RESEARCH_PLAN §7 initial choice.")
    args = p.parse_args()

    if args.vendor in MULTI_SYMBOL_VENDORS:
        _run_multi_symbol(args)
    else:
        _run_single_file(args)


if __name__ == "__main__":
    main()
