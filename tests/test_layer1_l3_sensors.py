"""
test_layer1_l3_sensors.py - Unit tests for layer1_l3_sensors.

Each sensor: synthetic event sequence -> assert expected property value.
Edge cases covered: empty state, snapshot seeding, reset, single-side
streams, autocorrelation with too-few samples.

Run from path-h-l3:
    & ..\..\..\.venv\Scripts\python.exe -m pytest tests\test_layer1_l3_sensors.py -v
"""
from __future__ import annotations

import math
import os
import sys

import pytest

# Make the worktree root importable when pytest is run from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aggregate_mbo_events import EventType, OrderEvent
from layer1_l3_sensors import (
    FEATURE_COLS,
    AggressorSequenceTracker,
    CancellationVelocityTracker,
    EventDensityTracker,
    HiddenOrderDetector,
    L3SensorArray,
    OrderArrivalRateTracker,
    OrderBook,
    OrderBookReconstructorSensor,
    OrderLifespanTracker,
    QueueDepletionTracker,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ev(t_ms: int, typ: EventType, oid: int = 0, side: str = "buy",
        price: float = 100.0, size: float = 1.0,
        aggressor: str | None = None) -> OrderEvent:
    return OrderEvent(
        timestamp_ms=t_ms, event_type=typ, order_id=oid, side=side,
        price=price, size=size, aggressor_side=aggressor,
    )


# --------------------------------------------------------------------------
# OrderBook
# --------------------------------------------------------------------------

class TestOrderBook:
    def test_empty_book_has_no_best(self):
        b = OrderBook()
        assert b.best_bid() is None
        assert b.best_ask() is None
        assert b.top_size("buy") == 0.0
        assert b.depth("buy", 5) == 0.0

    def test_snapshot_seeding(self):
        b = OrderBook()
        # [ORDER_ID, PRICE, AMOUNT]; AMOUNT>0 bid, AMOUNT<0 ask
        b.seed_from_snapshot([
            [1, 99.5, 2.0],   # bid 2 @ 99.5
            [2, 99.0, 1.0],   # bid 1 @ 99.0
            [3, 100.5, -1.5], # ask 1.5 @ 100.5
            [4, 101.0, -0.5], # ask 0.5 @ 101.0
        ])
        assert b.best_bid() == 99.5
        assert b.best_ask() == 100.5
        assert b.top_size("buy") == 2.0
        assert b.top_size("sell") == 1.5
        # Top-2 depth on each side
        assert b.depth("buy", 2) == pytest.approx(3.0)
        assert b.depth("sell", 2) == pytest.approx(2.0)

    def test_add_cancel_roundtrip(self):
        b = OrderBook()
        b.process(_ev(1, EventType.ADD, oid=10, side="buy", price=100.0, size=1.0))
        assert b.best_bid() == 100.0
        b.process(_ev(2, EventType.CANCEL, oid=10, side="buy"))
        assert b.best_bid() is None

    def test_modify_overwrites(self):
        b = OrderBook()
        b.process(_ev(1, EventType.ADD, oid=5, side="sell", price=101.0, size=1.0))
        b.process(_ev(2, EventType.MODIFY, oid=5, side="sell", price=101.5, size=2.0))
        assert b.best_ask() == 101.5
        assert b.top_size("sell") == 2.0

    def test_multi_level_depth(self):
        b = OrderBook()
        # Two orders at the best, others deeper
        for i, (oid, side, p, s) in enumerate([
            (1, "buy", 100.0, 1.0),
            (2, "buy", 100.0, 0.5),  # also at best
            (3, "buy", 99.5, 2.0),
            (4, "buy", 99.0, 1.0),
        ]):
            b.process(_ev(i, EventType.ADD, oid=oid, side=side, price=p, size=s))
        assert b.best_bid() == 100.0
        assert b.top_size("buy") == pytest.approx(1.5)
        assert b.depth("buy", 1) == pytest.approx(1.5)
        assert b.depth("buy", 2) == pytest.approx(3.5)
        assert b.depth("buy", 3) == pytest.approx(4.5)

    def test_reset(self):
        b = OrderBook()
        b.process(_ev(1, EventType.ADD, oid=1, price=100.0))
        b.reset()
        assert b.best_bid() is None
        assert len(b) == 0


# --------------------------------------------------------------------------
# OrderBookReconstructorSensor
# --------------------------------------------------------------------------

class TestBookReconstructorSensor:
    def test_spread_bps(self):
        b = OrderBook()
        b.process(_ev(1, EventType.ADD, oid=1, side="buy", price=100.0, size=1.0))
        b.process(_ev(2, EventType.ADD, oid=2, side="sell", price=100.1, size=1.0))
        s = OrderBookReconstructorSensor(b)
        # spread = 0.1, mid = 100.05, bps = 0.1/100.05 * 10000 ~= 9.995
        assert s.spread_bps == pytest.approx(9.995, rel=1e-3)

    def test_depth_imbalance_top5(self):
        b = OrderBook()
        b.process(_ev(1, EventType.ADD, oid=1, side="buy", price=100.0, size=3.0))
        b.process(_ev(2, EventType.ADD, oid=2, side="sell", price=100.1, size=1.0))
        s = OrderBookReconstructorSensor(b)
        # (3 - 1) / (3 + 1) = 0.5
        assert s.depth_imbalance_top5 == pytest.approx(0.5)

    def test_one_sided_book_is_safe(self):
        b = OrderBook()
        b.process(_ev(1, EventType.ADD, oid=1, side="buy", price=100.0, size=1.0))
        s = OrderBookReconstructorSensor(b)
        assert s.spread_bps == 0.0  # no ask -> 0 fallback
        assert s.top_bid_size == 1.0
        assert s.top_ask_size == 0.0


# --------------------------------------------------------------------------
# HiddenOrderDetector
# --------------------------------------------------------------------------

class TestHiddenOrderDetector:
    def test_no_trades_means_zero_rate(self):
        book = OrderBook()
        d = HiddenOrderDetector(book)
        # Just adding events to the book without trades shouldn't affect rate
        book.process(_ev(0, EventType.ADD, oid=1, side="sell", price=100.0))
        assert d.hidden_trade_rate == 0.0

    def test_matched_trade_is_not_hidden(self):
        book = OrderBook()
        book.process(_ev(0, EventType.ADD, oid=1, side="sell", price=100.0))
        d = HiddenOrderDetector(book)
        d.process(_ev(50, EventType.TRADE, price=100.0, aggressor="buy"))
        # Resting ASK at 100 is in book -> trade matches it (not hidden)
        assert d.hidden_trade_rate == 0.0

    def test_unmatched_trade_is_hidden(self):
        book = OrderBook()
        # No ASK at 99.5 in book; trade at 99.5 counts as hidden
        book.process(_ev(0, EventType.ADD, oid=1, side="sell", price=100.0))
        d = HiddenOrderDetector(book)
        d.process(_ev(50, EventType.TRADE, price=99.5, aggressor="buy"))
        assert d.hidden_trade_rate == 1.0

    def test_long_lived_visible_order_is_not_hidden(self):
        """The window-based v1 of this detector falsely flagged trades
        against orders placed >window_ms ago. The book-based v2 only
        cares about current book state, so age doesn't matter."""
        book = OrderBook()
        book.process(_ev(0, EventType.ADD, oid=1, side="sell", price=100.0))
        d = HiddenOrderDetector(book)
        # Trade 5 seconds after the ADD: still in book -> not hidden
        d.process(_ev(5000, EventType.TRADE, price=100.0, aggressor="buy"))
        assert d.hidden_trade_rate == 0.0


# --------------------------------------------------------------------------
# OrderArrivalRateTracker / CancellationVelocityTracker
# --------------------------------------------------------------------------

class TestArrivalRate:
    def test_per_side_rates(self):
        tr = OrderArrivalRateTracker(window_s=1.0)
        # 3 bid adds + 1 ask add within 1s window
        for i, side in enumerate(["buy", "buy", "buy", "sell"]):
            tr.process(_ev(100 + i * 100, EventType.ADD, oid=i, side=side))
        assert tr.arrival_rate_bid_per_s == pytest.approx(3.0)
        assert tr.arrival_rate_ask_per_s == pytest.approx(1.0)

    def test_old_events_pruned(self):
        tr = OrderArrivalRateTracker(window_s=1.0)
        tr.process(_ev(0, EventType.ADD, oid=1, side="buy"))
        tr.process(_ev(2000, EventType.ADD, oid=2, side="buy"))  # >1s later
        # First event pruned; rate reflects only the latest
        assert tr.arrival_rate_bid_per_s == pytest.approx(1.0)


class TestCancelRate:
    def test_only_cancels_counted(self):
        tr = CancellationVelocityTracker(window_s=1.0)
        tr.process(_ev(0, EventType.ADD, oid=1, side="buy"))  # not a cancel
        tr.process(_ev(100, EventType.CANCEL, oid=1, side="buy"))
        tr.process(_ev(200, EventType.CANCEL, oid=2, side="sell"))
        assert tr.cancel_rate_bid_per_s == pytest.approx(1.0)
        assert tr.cancel_rate_ask_per_s == pytest.approx(1.0)


# --------------------------------------------------------------------------
# OrderLifespanTracker
# --------------------------------------------------------------------------

class TestOrderLifespan:
    def test_lifespan_percentiles(self):
        tr = OrderLifespanTracker(buffer_len=100)
        # Three bid lifespans: 10, 20, 100 ms (sorted)
        for oid, lifespan in [(1, 10), (2, 20), (3, 100)]:
            tr.process(_ev(0, EventType.ADD, oid=oid, side="buy"))
            tr.process(_ev(lifespan, EventType.CANCEL, oid=oid, side="buy"))
        # p50 = 20, p95 = 100 - 4 = 96 (np.percentile linear interp)
        assert tr.lifespan_bid_p50_ms == pytest.approx(20.0, abs=1e-3)
        assert tr.lifespan_bid_p95_ms > 80.0  # reasonable upper tail
        assert tr.lifespan_ask_p50_ms == 0.0  # no asks tracked

    def test_trade_consumes_without_lifespan(self):
        tr = OrderLifespanTracker()
        tr.process(_ev(0, EventType.ADD, oid=1, side="buy"))
        tr.process(_ev(50, EventType.TRADE, oid=1, side="buy", aggressor="sell"))
        assert tr.lifespan_bid_p50_ms == 0.0  # fill is not counted


# --------------------------------------------------------------------------
# AggressorSequenceTracker
# --------------------------------------------------------------------------

class TestAggressor:
    def test_imbalance(self):
        tr = AggressorSequenceTracker(window=10)
        for agg in ["buy", "buy", "buy", "sell"]:
            tr.process(_ev(0, EventType.TRADE, aggressor=agg))
        # mean of [+1, +1, +1, -1] = 0.5
        assert tr.aggressor_imbalance == pytest.approx(0.5)

    def test_autocorr_runs_positive(self):
        tr = AggressorSequenceTracker(window=20)
        # Strong runs: +1 +1 +1 +1 -1 -1 -1 -1 -> positive lag-1 autocorr
        for agg in ["buy"] * 4 + ["sell"] * 4 + ["buy"] * 4 + ["sell"] * 4:
            tr.process(_ev(0, EventType.TRADE, aggressor=agg))
        assert tr.aggressor_autocorr_lag1 > 0.3

    def test_autocorr_flipflop_negative(self):
        tr = AggressorSequenceTracker(window=20)
        # Alternating: -> strongly negative lag-1
        for i in range(20):
            tr.process(_ev(0, EventType.TRADE,
                           aggressor="buy" if i % 2 == 0 else "sell"))
        assert tr.aggressor_autocorr_lag1 < -0.5

    def test_too_few_samples_returns_zero(self):
        tr = AggressorSequenceTracker(window=20)
        tr.process(_ev(0, EventType.TRADE, aggressor="buy"))
        assert tr.aggressor_autocorr_lag1 == 0.0
        assert tr.aggressor_autocorr_lag5 == 0.0


# --------------------------------------------------------------------------
# QueueDepletionTracker (requires book context)
# --------------------------------------------------------------------------

class TestQueueDepletion:
    def test_cancel_at_best_counted(self):
        book = OrderBook()
        tr = QueueDepletionTracker(book, window_s=1.0)
        # Seed two bids: one at best (100), one deeper (99.5)
        book.process(_ev(0, EventType.ADD, oid=1, side="buy", price=100.0, size=2.0))
        book.process(_ev(0, EventType.ADD, oid=2, side="buy", price=99.5, size=1.0))
        # Cancel the order at best — simulate L3SensorArray enrichment (price+size populated)
        tr.process(_ev(100, EventType.CANCEL, oid=1, side="buy", price=100.0, size=2.0))
        # Size 2.0 over 1.0s window -> 2.0/s depletion on bid side
        assert tr.queue_depletion_bid_per_s == pytest.approx(2.0)
        assert tr.queue_depletion_ask_per_s == 0.0

    def test_cancel_off_best_not_counted(self):
        book = OrderBook()
        tr = QueueDepletionTracker(book, window_s=1.0)
        book.process(_ev(0, EventType.ADD, oid=1, side="buy", price=100.0, size=2.0))
        book.process(_ev(0, EventType.ADD, oid=2, side="buy", price=99.5, size=1.0))
        # Cancel the deeper order — should NOT count toward queue depletion at best
        tr.process(_ev(100, EventType.CANCEL, oid=2, side="buy", price=99.5, size=1.0))
        assert tr.queue_depletion_bid_per_s == 0.0


# --------------------------------------------------------------------------
# EventDensityTracker
# --------------------------------------------------------------------------

class TestEventDensity:
    def test_rate_over_window(self):
        tr = EventDensityTracker(window_s=1.0)
        # 5 events over a 500ms span — all within 1s window
        for i in range(5):
            tr.process(_ev(i * 100, EventType.ADD, oid=i))
        assert tr.event_density_per_s == pytest.approx(5.0)


# --------------------------------------------------------------------------
# L3SensorArray integration
# --------------------------------------------------------------------------

class TestL3SensorArray:
    def test_snapshot_has_all_19_keys(self):
        arr = L3SensorArray()
        # Feed a small synthetic stream
        arr.seed_book([[1, 99.5, 2.0], [2, 100.5, -1.5]])
        events = [
            _ev(100, EventType.ADD, oid=3, side="buy", price=99.6, size=0.5),
            _ev(150, EventType.ADD, oid=4, side="sell", price=100.4, size=0.5),
            _ev(200, EventType.TRADE, oid=2, price=100.5, aggressor="buy"),
            _ev(250, EventType.CANCEL, oid=2, side="sell"),  # parser sets price=0
        ]
        for e in events:
            arr.process(e)
        snap = arr.snapshot()
        assert set(snap.keys()) == set(FEATURE_COLS)
        # No NaN
        for k, v in snap.items():
            assert isinstance(v, float), f"{k} is not float: {type(v).__name__}"
            assert math.isfinite(v) or v == 0.0, f"{k} = {v}"

    def test_cancel_recovers_price_from_book(self):
        """L3SensorArray should fill in price+size on CANCEL events
        from the book so QueueDepletion can compare to best."""
        arr = L3SensorArray()
        # ADD a bid at the (initially-only) best price
        arr.process(_ev(0, EventType.ADD, oid=1, side="buy", price=100.0, size=3.0))
        # Parser-shape cancel: price=0, size=0
        arr.process(_ev(100, EventType.CANCEL, oid=1, side="buy", price=0.0, size=0.0))
        # Queue depletion should have counted size=3 (recovered from book)
        # but only because price matched best (100.0 was best at cancel time)
        # NOTE: arrival/cancel rate windows are 2s default; first ADD was 0ms,
        # cancel 100ms — both inside window. depletion = 3.0 / 2.0s = 1.5
        assert arr.queue.queue_depletion_bid_per_s == pytest.approx(1.5)

    def test_reset_clears_all(self):
        arr = L3SensorArray()
        arr.process(_ev(0, EventType.ADD, oid=1, side="buy", price=100.0, size=1.0))
        arr.process(_ev(50, EventType.TRADE, aggressor="buy"))
        arr.reset()
        snap = arr.snapshot()
        # All counts/rates back to 0; book empty
        assert snap["arrival_rate_bid_per_s"] == 0.0
        assert snap["aggressor_imbalance"] == 0.0
        assert snap["spread_bps"] == 0.0
        assert len(arr.book) == 0
