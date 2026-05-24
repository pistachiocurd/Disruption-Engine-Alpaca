"""
layer1_l3_sensors.py - L3 (order-by-order) microstructure sensors.

Phase 2 of L3_RESEARCH_PLAN.md §4.2. Each sensor consumes the canonical
`OrderEvent` stream produced by `aggregate_mbo_events.py` vendor parsers
and exposes feature values as properties. The `L3SensorArray` orchestrator
holds all sensors, dispatches every event to all of them, and produces a
single 19-channel feature dict on snapshot().

Mirrors the layer1_sensors.py pattern (per-tracker class with
constructor + process() + property getters + reset() for session
boundaries). Phase 2 stays OFFLINE — sensors run inside the aggregator
during CSV generation, not yet inside engine.py. Live wiring is Phase 6.

Sensors built:
  OrderBook                       (foundational; not a feature emitter)
  OrderBookReconstructorSensor    spread_bps, top_{bid,ask}_size, depth_imbalance_top5
  HiddenOrderDetector             hidden_trade_rate
  QueueDepletionTracker           queue_depletion_{bid,ask}_per_s
  OrderArrivalRateTracker         arrival_rate_{bid,ask}_per_s
  CancellationVelocityTracker     cancel_rate_{bid,ask}_per_s
  OrderLifespanTracker            lifespan_{bid,ask}_p{50,95}_ms
  AggressorSequenceTracker        aggressor_imbalance, _autocorr_lag{1,5}
  EventDensityTracker             event_density_per_s
  L3SensorArray                   orchestrator -> snapshot() dict
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from aggregate_mbo_events import EventType, OrderEvent


# ============================================================================
# OrderBook - foundational state
# ============================================================================

class OrderBook:
    """L3 book state reconstructed from an OrderEvent stream.

    Maintains a dict[order_id] -> (side, price, size). best_bid/best_ask
    and depth queries scan the dict O(n); for Bitfinex with ~500 typical
    live orders this is <100us per call. If profiling later shows it
    matters, swap to a sortedcontainers.SortedDict per side.

    The Bitfinex book(R0) channel emits an initial snapshot at subscribe
    time. Callers route those to `seed_from_snapshot()` so subsequent
    cancels of pre-existing orders are correctly accounted (and so the
    derived book features are reasonable from the first emitted tick).

    Capped at MAX_ORDERS_PER_SIDE (default 50) — Bitfinex book(R0) with
    len=25 sends ~25 levels x ~2 orders per level per side. When an
    order falls OUT of the visible top-25 window due to price moves,
    Bitfinex does NOT always emit a delete (verified empirically:
    without this cap, 99% of ticks ended up with negative spread from
    stale orders that fell out of the window and never came back).
    On overflow, evict the worst-priced order on the same side (the
    one furthest from mid) — that's the stale-est candidate.
    """

    MAX_ORDERS_PER_SIDE = 50

    def __init__(self):
        self._orders: dict[int, tuple[str, float, float]] = {}

    def seed_from_snapshot(self, snapshot: list) -> None:
        """Snapshot rows are [ORDER_ID, PRICE, AMOUNT] per Bitfinex docs.
        AMOUNT positive = bid, negative = ask. Size = abs(AMOUNT)."""
        for row in snapshot:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            try:
                oid = int(row[0])
                price = float(row[1])
                amount = float(row[2])
            except (TypeError, ValueError):
                continue
            if price == 0.0:
                continue
            side = "buy" if amount > 0 else "sell"
            self._orders[oid] = (side, price, abs(amount))

    def process(self, event: OrderEvent) -> None:
        """Apply a canonical event. ADDs insert, MODIFYs overwrite,
        CANCELs remove, TRADEs are no-ops (the book channel emits the
        cancel separately when a resting order fills)."""
        if event.event_type == EventType.ADD:
            self._orders[event.order_id] = (event.side, event.price, event.size)
            self._evict_worst_if_overcap(event.side)
        elif event.event_type == EventType.MODIFY:
            self._orders[event.order_id] = (event.side, event.price, event.size)
            self._evict_worst_if_overcap(event.side)
        elif event.event_type == EventType.CANCEL:
            self._orders.pop(event.order_id, None)
        # TRADE: no-op; the matching cancel arrives via the book channel

    def _evict_worst_if_overcap(self, side: str) -> None:
        """If `side` exceeds MAX_ORDERS_PER_SIDE, drop the worst-priced
        order on that side. Worst = lowest price for bids, highest for
        asks — the one furthest from mid and most likely stale."""
        on_side = [(oid, p) for oid, (s, p, _) in self._orders.items() if s == side]
        if len(on_side) <= self.MAX_ORDERS_PER_SIDE:
            return
        # Sort so the WORST is at index 0
        if side == "buy":
            on_side.sort(key=lambda x: x[1])  # ascending: lowest bid first
        else:
            on_side.sort(key=lambda x: -x[1])  # descending: highest ask first
        excess = len(on_side) - self.MAX_ORDERS_PER_SIDE
        for i in range(excess):
            self._orders.pop(on_side[i][0], None)

    def get(self, order_id: int) -> Optional[tuple[str, float, float]]:
        return self._orders.get(order_id)

    def best_bid(self) -> Optional[float]:
        best = None
        for side, price, _ in self._orders.values():
            if side == "buy" and (best is None or price > best):
                best = price
        return best

    def best_ask(self) -> Optional[float]:
        best = None
        for side, price, _ in self._orders.values():
            if side == "sell" and (best is None or price < best):
                best = price
        return best

    def top_size(self, side: str) -> float:
        target = self.best_bid() if side == "buy" else self.best_ask()
        if target is None:
            return 0.0
        total = 0.0
        for s, p, sz in self._orders.values():
            if s == side and p == target:
                total += sz
        return total

    def depth(self, side: str, n_levels: int = 5) -> float:
        """Total size across the n_levels best price levels on `side`."""
        levels: dict[float, float] = {}
        for s, p, sz in self._orders.values():
            if s == side:
                levels[p] = levels.get(p, 0.0) + sz
        if not levels:
            return 0.0
        # Bids: highest first; asks: lowest first.
        sorted_prices = sorted(levels.keys(), reverse=(side == "buy"))
        return sum(levels[p] for p in sorted_prices[:n_levels])

    def prune_far_from(self, ref_price: float, max_pct: float = 0.01) -> int:
        """Drop orders whose price is more than `max_pct` away from
        `ref_price` (typically last_trade_price). Returns count removed.

        Bitfinex's `book` channel with `prec=R0, len=25` only sends
        deletes for orders within the top-25 levels. When an order falls
        out of that window (price moves away), we never learn about its
        cancel or fill — it sits in our book forever, eventually
        producing crossed best_bid/best_ask. Periodic pruning around
        the current market price evicts these stragglers.

        Default max_pct=0.01 (1%) is wider than 25 levels for crypto
        majors (typical level spacing ~0.001-0.01% on BTC), so live
        in-window orders are never pruned; only orders that have
        wandered far from current market.
        """
        if ref_price <= 0:
            return 0
        lo = ref_price * (1.0 - max_pct)
        hi = ref_price * (1.0 + max_pct)
        to_remove = [oid for oid, (_, p, _) in self._orders.items()
                     if p < lo or p > hi]
        for oid in to_remove:
            self._orders.pop(oid, None)
        return len(to_remove)

    def reset(self) -> None:
        self._orders.clear()

    def __len__(self) -> int:
        return len(self._orders)


# ============================================================================
# Sensors
# ============================================================================

class OrderBookReconstructorSensor:
    """Thin wrapper over an OrderBook exposing derived features as properties.
    The book itself is updated by `L3SensorArray.process()`; this sensor only
    reads it."""

    def __init__(self, book: OrderBook):
        self._book = book

    @property
    def spread_bps(self) -> float:
        bid = self._book.best_bid()
        ask = self._book.best_ask()
        if bid is None or ask is None or bid <= 0:
            return 0.0
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            return 0.0
        return (ask - bid) / mid * 10000.0

    @property
    def top_bid_size(self) -> float:
        return self._book.top_size("buy")

    @property
    def top_ask_size(self) -> float:
        return self._book.top_size("sell")

    @property
    def depth_imbalance_top5(self) -> float:
        bid = self._book.depth("buy", 5)
        ask = self._book.depth("sell", 5)
        total = bid + ask
        if total <= 0:
            return 0.0
        return (bid - ask) / total

    def process(self, event: OrderEvent) -> None:
        pass  # stateless; book is updated upstream

    def reset(self) -> None:
        pass


class HiddenOrderDetector:
    """Count trades whose price has no matching resting order in the book.

    A buy aggressor that lifts price P should match a resting ASK at P.
    If the OrderBook has no ASK at P at the moment the trade is processed,
    the liquidity was probably hidden (iceberg, dark, etc.). Bitfinex
    sends the matching CANCEL on the book channel AFTER the trade message,
    so at trade-process time the resting order is normally still in book.

    Property: hidden_trade_rate over the last `trade_buffer_len` trades.
    """

    def __init__(self, book: "OrderBook", trade_buffer_len: int = 64,
                 price_tolerance: float = 1e-9):
        self._book = book
        self._price_tolerance = price_tolerance
        self._trade_outcomes: deque[bool] = deque(maxlen=trade_buffer_len)

    def process(self, event: OrderEvent) -> None:
        if event.event_type != EventType.TRADE or not event.aggressor_side:
            return
        resting_side = "sell" if event.aggressor_side == "buy" else "buy"
        for side, price, _ in self._book._orders.values():
            if side == resting_side and abs(price - event.price) < self._price_tolerance:
                self._trade_outcomes.append(False)  # matched visible
                return
        self._trade_outcomes.append(True)  # no match -> hidden

    @property
    def hidden_trade_rate(self) -> float:
        if not self._trade_outcomes:
            return 0.0
        return sum(self._trade_outcomes) / len(self._trade_outcomes)

    def reset(self) -> None:
        self._trade_outcomes.clear()


class QueueDepletionTracker:
    """Per-side rate of cancellations AT THE BEST PRICE LEVEL.

    Distinct from `CancellationVelocityTracker` (which counts ALL cancels):
    queue depletion measures specifically the rate at which the top-of-book
    queue shrinks via cancels, which is one of the classical pre-shock
    signals (informed liquidity pulling).

    Requires book state to know what "best" was at the time of the cancel.
    L3SensorArray dispatches sensors BEFORE updating the book on cancels,
    so this tracker can read pre-cancel best_bid/best_ask from the book.
    """

    def __init__(self, book: OrderBook, window_s: float = 2.0,
                 buffer_len: int = 2048):
        self._book = book
        self.window_s = window_s
        self._cancels_at_best: deque[tuple[int, str, float]] = deque(maxlen=buffer_len)
        self._last_ts_ms: int = 0

    def process(self, event: OrderEvent) -> None:
        if event.event_type == EventType.CANCEL:
            # Note: L3SensorArray has enriched the cancel event with the
            # order's recovered price+size before dispatching to us.
            if event.side == "buy":
                best = self._book.best_bid()
            else:
                best = self._book.best_ask()
            if best is not None and abs(event.price - best) < 1e-9:
                self._cancels_at_best.append((event.timestamp_ms, event.side, event.size))
        self._last_ts_ms = max(self._last_ts_ms, event.timestamp_ms)
        cutoff = self._last_ts_ms - int(self.window_s * 1000)
        while self._cancels_at_best and self._cancels_at_best[0][0] < cutoff:
            self._cancels_at_best.popleft()

    @property
    def queue_depletion_bid_per_s(self) -> float:
        if not self._cancels_at_best:
            return 0.0
        total_size = sum(sz for _, side, sz in self._cancels_at_best if side == "buy")
        return total_size / self.window_s

    @property
    def queue_depletion_ask_per_s(self) -> float:
        if not self._cancels_at_best:
            return 0.0
        total_size = sum(sz for _, side, sz in self._cancels_at_best if side == "sell")
        return total_size / self.window_s

    def reset(self) -> None:
        self._cancels_at_best.clear()
        self._last_ts_ms = 0


class OrderArrivalRateTracker:
    """Per-side ADD rate over a rolling wall-clock window."""

    def __init__(self, window_s: float = 2.0, buffer_len: int = 4096):
        self.window_s = window_s
        self._arrivals: deque[tuple[int, str]] = deque(maxlen=buffer_len)
        self._last_ts_ms: int = 0

    def process(self, event: OrderEvent) -> None:
        if event.event_type == EventType.ADD:
            self._arrivals.append((event.timestamp_ms, event.side))
        self._last_ts_ms = max(self._last_ts_ms, event.timestamp_ms)
        cutoff = self._last_ts_ms - int(self.window_s * 1000)
        while self._arrivals and self._arrivals[0][0] < cutoff:
            self._arrivals.popleft()

    @property
    def arrival_rate_bid_per_s(self) -> float:
        return sum(1 for _, s in self._arrivals if s == "buy") / self.window_s

    @property
    def arrival_rate_ask_per_s(self) -> float:
        return sum(1 for _, s in self._arrivals if s == "sell") / self.window_s

    def reset(self) -> None:
        self._arrivals.clear()
        self._last_ts_ms = 0


class CancellationVelocityTracker:
    """Per-side CANCEL rate over a rolling wall-clock window."""

    def __init__(self, window_s: float = 2.0, buffer_len: int = 4096):
        self.window_s = window_s
        self._cancels: deque[tuple[int, str]] = deque(maxlen=buffer_len)
        self._last_ts_ms: int = 0

    def process(self, event: OrderEvent) -> None:
        if event.event_type == EventType.CANCEL:
            self._cancels.append((event.timestamp_ms, event.side))
        self._last_ts_ms = max(self._last_ts_ms, event.timestamp_ms)
        cutoff = self._last_ts_ms - int(self.window_s * 1000)
        while self._cancels and self._cancels[0][0] < cutoff:
            self._cancels.popleft()

    @property
    def cancel_rate_bid_per_s(self) -> float:
        return sum(1 for _, s in self._cancels if s == "buy") / self.window_s

    @property
    def cancel_rate_ask_per_s(self) -> float:
        return sum(1 for _, s in self._cancels if s == "sell") / self.window_s

    def reset(self) -> None:
        self._cancels.clear()
        self._last_ts_ms = 0


class OrderLifespanTracker:
    """Per-side rolling-buffer distribution of order lifespans.

    On ADD, record the submit timestamp keyed by order_id. On CANCEL,
    pop the submit time and append (now - submit) to the per-side
    lifespan deque. TRADE also pops (resting order consumed; the
    fill-lifespan is a different signal class — we don't include it).

    Properties: p50 and p95 of bid and ask lifespans over the buffer.
    """

    def __init__(self, buffer_len: int = 2048):
        self.buffer_len = buffer_len
        self._submit_times: dict[int, int] = {}
        self._lifespans_bid: deque[int] = deque(maxlen=buffer_len)
        self._lifespans_ask: deque[int] = deque(maxlen=buffer_len)

    def process(self, event: OrderEvent) -> None:
        if event.event_type == EventType.ADD:
            self._submit_times[event.order_id] = event.timestamp_ms
        elif event.event_type == EventType.CANCEL:
            submit_t = self._submit_times.pop(event.order_id, None)
            if submit_t is not None:
                lifespan = event.timestamp_ms - submit_t
                if event.side == "buy":
                    self._lifespans_bid.append(lifespan)
                else:
                    self._lifespans_ask.append(lifespan)
        elif event.event_type == EventType.TRADE:
            # 0 == "no resting order linked" (hidden trade); pop is a safe no-op
            self._submit_times.pop(event.order_id, None)

    @staticmethod
    def _percentile(buf: deque, pct: float) -> float:
        if not buf:
            return 0.0
        return float(np.percentile(np.fromiter(buf, dtype=np.float32, count=len(buf)), pct))

    @property
    def lifespan_bid_p50_ms(self) -> float:
        return self._percentile(self._lifespans_bid, 50)

    @property
    def lifespan_bid_p95_ms(self) -> float:
        return self._percentile(self._lifespans_bid, 95)

    @property
    def lifespan_ask_p50_ms(self) -> float:
        return self._percentile(self._lifespans_ask, 50)

    @property
    def lifespan_ask_p95_ms(self) -> float:
        return self._percentile(self._lifespans_ask, 95)

    def reset(self) -> None:
        self._submit_times.clear()
        self._lifespans_bid.clear()
        self._lifespans_ask.clear()


class AggressorSequenceTracker:
    """Trade-aggressor sequence: mean imbalance + lag-1 / lag-5 autocorr.

    Circular buffer of recent aggressor signs (+1 buy, -1 sell). Lag-k
    autocorr is the Pearson correlation of the sequence with itself
    shifted by k positions. Positive lag-1 indicates runs (momentum);
    negative indicates flip-flopping (mean reversion in aggressor).
    """

    def __init__(self, window: int = 64):
        self.window = window
        self._signs: deque[int] = deque(maxlen=window)

    def process(self, event: OrderEvent) -> None:
        if event.event_type == EventType.TRADE and event.aggressor_side:
            self._signs.append(1 if event.aggressor_side == "buy" else -1)

    @property
    def aggressor_imbalance(self) -> float:
        if not self._signs:
            return 0.0
        return sum(self._signs) / len(self._signs)

    def _autocorr(self, lag: int) -> float:
        n = len(self._signs)
        if n <= lag + 1:
            return 0.0
        arr = np.fromiter(self._signs, dtype=np.float32, count=n)
        mu = arr.mean()
        denom = float(((arr - mu) ** 2).sum())
        if denom <= 1e-9:
            return 0.0
        num = float(((arr[:-lag] - mu) * (arr[lag:] - mu)).sum())
        return num / denom

    @property
    def aggressor_autocorr_lag1(self) -> float:
        return self._autocorr(1)

    @property
    def aggressor_autocorr_lag5(self) -> float:
        return self._autocorr(5)

    def reset(self) -> None:
        self._signs.clear()


class EventDensityTracker:
    """Events per second over a rolling window. Replaces Phase 1.5's
    `window_duration_s` feature with a cleaner, scale-invariant version."""

    def __init__(self, window_s: float = 2.0, buffer_len: int = 8192):
        self.window_s = window_s
        self._events: deque[int] = deque(maxlen=buffer_len)
        self._last_ts_ms: int = 0

    def process(self, event: OrderEvent) -> None:
        self._events.append(event.timestamp_ms)
        self._last_ts_ms = max(self._last_ts_ms, event.timestamp_ms)
        cutoff = self._last_ts_ms - int(self.window_s * 1000)
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    @property
    def event_density_per_s(self) -> float:
        return len(self._events) / self.window_s

    def reset(self) -> None:
        self._events.clear()
        self._last_ts_ms = 0


# ============================================================================
# L3SensorArray - orchestrator
# ============================================================================

# Canonical column order, matching `L3SensorArray.snapshot()` keys. The
# trainer's FEATURE_COLS is set from this so the two stay in sync.
FEATURE_COLS = [
    "event_density_per_s",
    "arrival_rate_bid_per_s",
    "arrival_rate_ask_per_s",
    "cancel_rate_bid_per_s",
    "cancel_rate_ask_per_s",
    "aggressor_imbalance",
    "spread_bps",
    "top_bid_size",
    "top_ask_size",
    "depth_imbalance_top5",
    "hidden_trade_rate",
    "queue_depletion_bid_per_s",
    "queue_depletion_ask_per_s",
    "lifespan_bid_p50_ms",
    "lifespan_bid_p95_ms",
    "lifespan_ask_p50_ms",
    "lifespan_ask_p95_ms",
    "aggressor_autocorr_lag1",
    "aggressor_autocorr_lag5",
]


class L3SensorArray:
    """Owns the OrderBook + all feature sensors; dispatches every
    OrderEvent to all of them; produces a 19-key feature dict on
    `snapshot()`.

    Dispatch order matters for CANCEL events:
      1. Recover the cancelled order's true price+size from the book
         (parser sets them to 0 since Bitfinex CANCEL frames don't
         carry that info).
      2. Dispatch the ENRICHED event to all sensors. QueueDepletion
         can now compare event.price to the (pre-cancel) book best
         to decide whether the cancel was at top-of-book.
      3. Apply the event to the book LAST so subsequent events see
         a consistent post-event state.
    """

    def __init__(
        self,
        density_window_s: float = 2.0,
        arrival_window_s: float = 2.0,
        cancel_window_s: float = 2.0,
        lifespan_buffer: int = 2048,
        hidden_trade_buffer: int = 64,
        queue_window_s: float = 2.0,
        aggressor_window: int = 64,
    ):
        self.book = OrderBook()
        self.book_sensor = OrderBookReconstructorSensor(self.book)
        self.density = EventDensityTracker(density_window_s)
        self.arrival = OrderArrivalRateTracker(arrival_window_s)
        self.cancel = CancellationVelocityTracker(cancel_window_s)
        self.lifespan = OrderLifespanTracker(lifespan_buffer)
        self.hidden = HiddenOrderDetector(self.book, hidden_trade_buffer)
        self.queue = QueueDepletionTracker(self.book, queue_window_s)
        self.aggressor = AggressorSequenceTracker(aggressor_window)
        self._stateful_sensors = [
            self.book_sensor, self.density, self.arrival, self.cancel,
            self.lifespan, self.hidden, self.queue, self.aggressor,
        ]
        # Reference for periodic stale-order pruning. Bitfinex's len=25
        # window means orders outside the top-25 levels never get explicit
        # deletes; without pruning the book accumulates stragglers that
        # cross best_bid/best_ask (observed: 99% of ticks crossed without
        # this safeguard).
        self._last_trade_price: float = 0.0

    def seed_book(self, snapshot: list) -> None:
        """Reset book and seed from this snapshot.

        Bitfinex emits a snapshot at every connection (initial + reconnect).
        Each one is the venue's authoritative current state, so anything
        we accumulated from the prior session is stale — orders that got
        cancelled or filled during a reconnect gap will never appear in
        our stream as CANCELs. Without reset(), those stale orders persist
        in our book forever, producing crossed best_bid/best_ask
        (observed: 99.5% of ticks had negative spread without this reset).
        """
        self.book.reset()
        self.book.seed_from_snapshot(snapshot)

    def process(self, event: OrderEvent) -> None:
        # Track last trade price for periodic book pruning (see below).
        if event.event_type == EventType.TRADE and event.price > 0:
            self._last_trade_price = event.price

        # Enrich CANCEL events with recovered price+size from the book BEFORE
        # dispatching to sensors, so QueueDepletion can compare to current best.
        if event.event_type == EventType.CANCEL:
            rec = self.book.get(event.order_id)
            if rec is not None:
                _side, price, size = rec
                event = OrderEvent(
                    timestamp_ms=event.timestamp_ms,
                    event_type=EventType.CANCEL,
                    order_id=event.order_id,
                    side=event.side,
                    price=price,
                    size=size,
                    aggressor_side=None,
                )
            # If the order isn't in book (pre-existing without snapshot seed,
            # or already cancelled), leave the event as-is; QueueDepletion
            # will skip it because event.price == 0 != best.

        # Dispatch BEFORE book mutation
        for s in self._stateful_sensors:
            s.process(event)

        # Apply to book
        self.book.process(event)

        # Periodically prune stale orders that have wandered out of
        # Bitfinex's len=25 visible window. Cheap (O(n) over <200 orders)
        # and only fires when we have a price reference.
        if self._last_trade_price > 0:
            self.book.prune_far_from(self._last_trade_price, max_pct=0.01)

    def snapshot(self) -> dict[str, float]:
        return {
            "event_density_per_s":       self.density.event_density_per_s,
            "arrival_rate_bid_per_s":    self.arrival.arrival_rate_bid_per_s,
            "arrival_rate_ask_per_s":    self.arrival.arrival_rate_ask_per_s,
            "cancel_rate_bid_per_s":     self.cancel.cancel_rate_bid_per_s,
            "cancel_rate_ask_per_s":     self.cancel.cancel_rate_ask_per_s,
            "aggressor_imbalance":       self.aggressor.aggressor_imbalance,
            "spread_bps":                self.book_sensor.spread_bps,
            "top_bid_size":              self.book_sensor.top_bid_size,
            "top_ask_size":              self.book_sensor.top_ask_size,
            "depth_imbalance_top5":      self.book_sensor.depth_imbalance_top5,
            "hidden_trade_rate":         self.hidden.hidden_trade_rate,
            "queue_depletion_bid_per_s": self.queue.queue_depletion_bid_per_s,
            "queue_depletion_ask_per_s": self.queue.queue_depletion_ask_per_s,
            "lifespan_bid_p50_ms":       self.lifespan.lifespan_bid_p50_ms,
            "lifespan_bid_p95_ms":       self.lifespan.lifespan_bid_p95_ms,
            "lifespan_ask_p50_ms":       self.lifespan.lifespan_ask_p50_ms,
            "lifespan_ask_p95_ms":       self.lifespan.lifespan_ask_p95_ms,
            "aggressor_autocorr_lag1":   self.aggressor.aggressor_autocorr_lag1,
            "aggressor_autocorr_lag5":   self.aggressor.aggressor_autocorr_lag5,
        }

    def reset(self) -> None:
        self.book.reset()
        for s in self._stateful_sensors:
            s.reset()
