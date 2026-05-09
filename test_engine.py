"""Tests for engine.py risk gates: drawdown, position limit, session reset."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import config
from engine import Engine


class _Stub:
    """Duck-type for Engine risk-gate methods.

    Avoids spinning up a real Engine (Alpaca client, sensors, dashboard) just
    to exercise pure math. Each test sets only the fields it needs.

    Engine's gate methods call other gate methods via `self.<method>`, so we
    pin them as class attributes here — that way the lookup resolves on _Stub
    rather than failing because _Stub doesn't inherit from Engine.
    """

    _max_drawdown_breached = Engine._max_drawdown_breached
    _is_drawdown_breached = Engine._is_drawdown_breached
    _mtm_drawdown_breached = Engine._mtm_drawdown_breached
    _would_exceed_position_limit = Engine._would_exceed_position_limit
    _reset_session_risk_state = Engine._reset_session_risk_state

    def __init__(self) -> None:
        self.session_peak_pnl: float = 0.0
        self.session_pnl: float = 0.0
        self._position: float = 0.0
        self._cash: float = 0.0
        self._mtm_peak_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._total_filled_notional: float = 0.0
        # `sensors` and `drift_monitor` need to exist for the MTM gate / reset
        # to read off them. Tests override with stubs as needed.
        self.sensors = SimpleNamespace(state=SimpleNamespace(ob_snapshot=None))
        self.drift_monitor = None


def _book(bid: float, ask: float, size: float = 100.0) -> dict:
    return {
        "bids": [[bid, size]],
        "asks": [[ask, size]],
        "timestamp": 0,
    }


@pytest.fixture(autouse=True)
def reset_drawdown_config(monkeypatch):
    # Pin known values so a stale env doesn't poison the assertions.
    monkeypatch.setattr(config, "MAX_SESSION_DRAWDOWN_USD", 1000.0)
    monkeypatch.setattr(config, "MAX_SESSION_DRAWDOWN_PCT", 0.02)
    monkeypatch.setattr(config, "DRAWDOWN_PCT_MIN_PEAK_USD", 100.0)
    monkeypatch.setattr(config, "MAX_MTM_DRAWDOWN_USD", 20000.0)
    monkeypatch.setattr(config, "MAX_MTM_DRAWDOWN_PCT", 0.10)
    monkeypatch.setattr(config, "MAX_POSITION_LIMIT", 100.0)


# ============================================================================
# IS-based drawdown gate (existing behavior, unchanged)
# ============================================================================
def test_no_drawdown_when_pnl_above_peak():
    stub = _Stub()
    stub.session_peak_pnl = 50.0
    stub.session_pnl = 60.0
    assert Engine._max_drawdown_breached(stub) is False


def test_tiny_peak_with_modest_loss_does_not_trip_pct_gate():
    # Pre-fix bug: $0.91 peak, $9 loss = 1000% "drawdown" trips 2% gate.
    stub = _Stub()
    stub.session_peak_pnl = 0.91
    stub.session_pnl = -8.59
    assert Engine._max_drawdown_breached(stub) is False


def test_absolute_usd_cap_trips_at_large_loss_regardless_of_peak():
    stub = _Stub()
    stub.session_peak_pnl = 50_000.0
    stub.session_pnl = 48_500.0
    assert Engine._max_drawdown_breached(stub) is True


def test_pct_gate_trips_only_when_peak_above_floor():
    stub = _Stub()
    stub.session_peak_pnl = 50.0
    stub.session_pnl = 25.0
    assert Engine._max_drawdown_breached(stub) is False
    stub.session_peak_pnl = 200.0
    stub.session_pnl = 100.0
    assert Engine._max_drawdown_breached(stub) is True


def test_pct_gate_at_2pct_with_meaningful_peak():
    stub = _Stub()
    stub.session_peak_pnl = 10_000.0
    stub.session_pnl = 9_750.0
    assert Engine._max_drawdown_breached(stub) is True


def test_pct_gate_does_not_trip_below_pct_threshold():
    stub = _Stub()
    stub.session_peak_pnl = 10_000.0
    stub.session_pnl = 9_900.0
    assert Engine._max_drawdown_breached(stub) is False


# ============================================================================
# Position-limit gate
# ============================================================================
def test_position_limit_blocks_buy_at_long_cap():
    stub = _Stub()
    stub._position = 100.0  # at long limit
    mandate = SimpleNamespace(direction=1, target_size=25.0)
    assert Engine._would_exceed_position_limit(stub, mandate) is True


def test_position_limit_allows_sell_when_long():
    # Position at +100, sell flips it. Should be allowed even though we're at
    # the absolute long cap; selling reduces or flips, doesn't pile.
    stub = _Stub()
    stub._position = 100.0
    mandate = SimpleNamespace(direction=-1, target_size=25.0)
    assert Engine._would_exceed_position_limit(stub, mandate) is False


def test_position_limit_allows_buy_when_within_long_cap():
    stub = _Stub()
    stub._position = -50.0  # half-short, buy reduces
    mandate = SimpleNamespace(direction=1, target_size=25.0)
    assert Engine._would_exceed_position_limit(stub, mandate) is False


def test_position_limit_blocks_sell_at_short_cap():
    # Symmetric: position at -100, sell mandate would push deeper short.
    stub = _Stub()
    stub._position = -100.0
    mandate = SimpleNamespace(direction=-1, target_size=25.0)
    assert Engine._would_exceed_position_limit(stub, mandate) is True


# ============================================================================
# MTM drawdown gate
# ============================================================================
def test_mtm_no_trip_when_no_orderbook():
    # Without a mid available, MTM gate cannot evaluate; never trips.
    stub = _Stub()
    stub._cash = -1_000_000.0  # huge paper loss but no mid
    stub._position = 100.0
    assert Engine._mtm_drawdown_breached(stub) is False


def test_mtm_trips_on_absolute_dollar_loss():
    # 100 long shares with -$44,300 cash basis. Peak set to +$100.
    # Mid drops to $235 → MTM = -44,300 + 100*235 = -$20,800.
    # Drawdown = 100 - (-20,800) = $20,900 > $20K cap → trips.
    stub = _Stub()
    stub._cash = -44_300.0
    stub._position = 100.0
    stub._mtm_peak_pnl = 100.0  # set explicitly so we don't bump from current
    stub.sensors = SimpleNamespace(state=SimpleNamespace(ob_snapshot=_book(234.95, 235.05)))
    assert Engine._mtm_drawdown_breached(stub) is True


def test_mtm_no_trip_below_usd_cap_with_small_peak():
    # MTM peak of $50 (below MIN_PEAK floor). Pure-percentage gate wouldn't
    # apply. Only USD cap is relevant. -$200 MTM = $250 drawdown < $20K cap.
    stub = _Stub()
    stub._cash = 0.0
    stub._position = 0.0
    stub._mtm_peak_pnl = 50.0
    # Force a small-loss MTM via cash negative:
    stub._cash = -200.0
    stub.sensors = SimpleNamespace(state=SimpleNamespace(ob_snapshot=_book(99.95, 100.05)))
    assert Engine._mtm_drawdown_breached(stub) is False


def test_mtm_pct_gate_trips_at_meaningful_peak():
    # Peak $200K, current $179K = 10.5% drawdown > 10% gate → trip.
    # Position 0 here, the MTM is purely from cash gains/losses.
    stub = _Stub()
    stub._cash = 179_000.0
    stub._position = 0.0
    stub._mtm_peak_pnl = 200_000.0
    stub.sensors = SimpleNamespace(state=SimpleNamespace(ob_snapshot=_book(99.95, 100.05)))
    assert Engine._mtm_drawdown_breached(stub) is True


# ============================================================================
# Session-state reset on market open
# ============================================================================
def test_reset_zeroes_all_pnl_and_position_counters():
    stub = _Stub()
    stub.session_pnl = 123.0
    stub.session_peak_pnl = 456.0
    stub._position = 50.0
    stub._cash = -22_000.0
    stub._mtm_peak_pnl = 789.0
    stub._total_fees = 12.0
    stub._total_filled_notional = 30_000.0
    Engine._reset_session_risk_state(stub)
    assert stub.session_pnl == 0.0
    assert stub.session_peak_pnl == 0.0
    assert stub._position == 0.0
    assert stub._cash == 0.0
    assert stub._mtm_peak_pnl == 0.0
    assert stub._total_fees == 0.0
    assert stub._total_filled_notional == 0.0


def test_reset_clears_drawdown_state_so_next_loss_does_not_falsely_trip():
    # Stub had a $1500 peak and just dropped to $1000 — in the IS gate that's
    # a $500 drawdown, well within bounds. After reset, peak and pnl both 0,
    # so the gate is in a fresh-start state with no false trip.
    stub = _Stub()
    stub.session_peak_pnl = 1500.0
    stub.session_pnl = 1000.0
    Engine._reset_session_risk_state(stub)
    assert Engine._max_drawdown_breached(stub) is False
