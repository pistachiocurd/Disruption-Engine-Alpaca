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
data acquisition (Bitfinex public L3 archive; Databento or Tardis for
historical replay). A `synthetic` parser reads the canonical OrderEvent
CSV format — useful for testing the aggregation logic before vendor
data lands.

Usage:
    python aggregate_mbo_events.py --vendor synthetic \\
        --in calibration/l3_events_sample.csv \\
        --out calibration/l3_ticks_BTC.csv \\
        --tick-events 100

    python aggregate_mbo_events.py --vendor bitfinex \\
        --in l3_data/bitfinex_l3_20260512.jsonl.gz \\
        --out calibration/l3_ticks_BTC.csv
"""
from __future__ import annotations

import argparse
import csv
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
    """Maintains rolling state over an MBO event stream and emits an
    aggregated feature row every `k_events` events.

    Tracks per-order submit times so cancellations can be tagged with
    their lifespan. Order IDs that fill are also removed from the live
    set (we don't compute lifespan on fills here — fills are a different
    signal class than cancels).
    """

    LIFESPAN_BUFFER_LEN = 2048  # rolling window for lifespan stats

    def __init__(self, k_events: int = 100):
        self.k_events = k_events
        self._order_submit_times: dict[int, int] = {}
        self._lifespans_bid: deque[int] = deque(maxlen=self.LIFESPAN_BUFFER_LEN)
        self._lifespans_ask: deque[int] = deque(maxlen=self.LIFESPAN_BUFFER_LEN)
        self._window_start_ms: Optional[int] = None
        self._window_end_ms: Optional[int] = None
        self._event_count: int = 0
        self._reset_window_counts()

    def _reset_window_counts(self) -> None:
        self.cancels_bid = 0
        self.cancels_ask = 0
        self.arrivals_bid = 0
        self.arrivals_ask = 0
        self.trades_buy_agg = 0
        self.trades_sell_agg = 0
        self.fills_bid = 0   # someone hit our resting bid (sold to us)
        self.fills_ask = 0   # someone lifted our resting ask (bought from us)

    def process(self, event: OrderEvent) -> Optional[dict]:
        """Process one event; return aggregated tick dict if window full,
        else None."""
        if self._window_start_ms is None:
            self._window_start_ms = event.timestamp_ms
        self._window_end_ms = event.timestamp_ms

        if event.event_type == EventType.ADD:
            self._order_submit_times[event.order_id] = event.timestamp_ms
            if event.side == "buy":
                self.arrivals_bid += 1
            else:
                self.arrivals_ask += 1

        elif event.event_type == EventType.CANCEL:
            submit_t = self._order_submit_times.pop(event.order_id, None)
            if submit_t is not None:
                lifespan_ms = event.timestamp_ms - submit_t
                if event.side == "buy":
                    self.cancels_bid += 1
                    self._lifespans_bid.append(lifespan_ms)
                else:
                    self.cancels_ask += 1
                    self._lifespans_ask.append(lifespan_ms)

        elif event.event_type == EventType.TRADE:
            if event.aggressor_side == "buy":
                self.trades_buy_agg += 1
                self.fills_ask += 1   # bought from an ask
            elif event.aggressor_side == "sell":
                self.trades_sell_agg += 1
                self.fills_bid += 1   # sold to a bid
            # Resting order is consumed (full or partial — we don't track
            # partial state at this level of aggregation).
            self._order_submit_times.pop(event.order_id, None)

        # MODIFY events are intentionally not counted here — a modify in
        # most vendor schemas appears as cancel + add. If a vendor emits
        # modify as a primitive, extend this branch to update submit time
        # for the order_id.

        self._event_count += 1
        if self._event_count >= self.k_events:
            return self._emit_and_reset()
        return None

    def _emit_and_reset(self) -> dict:
        duration_ms = max(self._window_end_ms - self._window_start_ms, 1)
        duration_s = duration_ms / 1000.0
        total_trades = self.trades_buy_agg + self.trades_sell_agg

        def _mean(buf: deque) -> float:
            return (sum(buf) / len(buf)) if buf else 0.0

        out = {
            "timestamp_ms": self._window_end_ms,
            "window_duration_s": duration_s,
            "event_density_per_s": self._event_count / duration_s,
            "arrival_rate_bid_per_s": self.arrivals_bid / duration_s,
            "arrival_rate_ask_per_s": self.arrivals_ask / duration_s,
            "cancel_rate_bid_per_s": self.cancels_bid / duration_s,
            "cancel_rate_ask_per_s": self.cancels_ask / duration_s,
            "cancel_to_fill_bid": self.cancels_bid / max(self.fills_bid, 1),
            "cancel_to_fill_ask": self.cancels_ask / max(self.fills_ask, 1),
            "mean_lifespan_bid_ms": _mean(self._lifespans_bid),
            "mean_lifespan_ask_ms": _mean(self._lifespans_ask),
            "aggressor_imbalance": (
                (self.trades_buy_agg - self.trades_sell_agg) / max(total_trades, 1)
            ),
        }

        self._reset_window_counts()
        self._window_start_ms = None
        self._window_end_ms = None
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
    """Parse a Databento MBO file. Schema TBD — depends on whether the
    .dbn binary format or the CSV variant is acquired. STUB until data
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


def parse_bitfinex_l3(path: Path) -> Iterator[OrderEvent]:
    """Parse a gzipped JSONL stream produced by harvest_bitfinex_l3.py.

    Bitfinex's raw-book frames at prec=R0 are lists keyed by chanId. The
    parser must first read all `subscribed` events at the top of the
    stream to build the chanId → (channel, symbol) map, then partition
    events accordingly. STUB pending Phase 1.5 wiring; see README.md."""
    raise NotImplementedError(
        "Bitfinex L3 parser not yet wired. Mapping:\n"
        "  book update PRICE!=0, ORDER_ID new      -> EventType.ADD\n"
        "  book update PRICE!=0, ORDER_ID known    -> EventType.MODIFY\n"
        "  book update PRICE==0                    -> EventType.CANCEL\n"
        "  trade 'te' AMOUNT>0                     -> EventType.TRADE buy-aggressor\n"
        "  trade 'te' AMOUNT<0                     -> EventType.TRADE sell-aggressor\n"
        "  trade 'tu'                              -> skip (duplicates 'te')\n"
        "  hb / info / subscribed / snapshot       -> skip (seed live-orders dict at init)\n"
    )


def parse_synthetic_mbo(path: Path) -> Iterator[OrderEvent]:
    """Parse the canonical OrderEvent CSV format. Useful for testing
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


VENDORS = {
    "databento": parse_databento_mbo,
    "tardis": parse_tardis_mbo,
    "bitfinex": parse_bitfinex_l3,
    "synthetic": parse_synthetic_mbo,
}


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vendor", choices=list(VENDORS.keys()), required=True,
                   help="MBO source format. 'synthetic' uses the canonical CSV; "
                        "'bitfinex' parses harvest_bitfinex_l3.py output; "
                        "'databento'/'tardis' require data acquisition first.")
    p.add_argument("--in", dest="input_path", type=Path, required=True,
                   help="Path to raw MBO file.")
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write aggregated event-clock tick CSV.")
    p.add_argument("--tick-events", type=int, default=100,
                   help="Events per tick (event-clock K). Default 100; the "
                        "L3_RESEARCH_PLAN §7 initial choice.")
    args = p.parse_args()

    parser_fn = VENDORS[args.vendor]
    aggregator = EventClockAggregator(k_events=args.tick_events)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ticks = 0
    fieldnames = None

    with open(args.out, "w", newline="") as f_out:
        writer = None
        for event in parser_fn(args.input_path):
            tick = aggregator.process(event)
            if tick is None:
                continue
            if writer is None:
                fieldnames = list(tick.keys())
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(tick)
            n_ticks += 1

    print(
        f"Wrote {n_ticks:,} event-clock ticks "
        f"({args.tick_events} events/tick) to {args.out}"
    )


if __name__ == "__main__":
    main()
