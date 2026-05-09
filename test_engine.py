"""Tests for engine.py risk gates."""
from __future__ import annotations

import pytest

import config
from engine import Engine


class _Stub:
    """Minimal duck-type for Engine._max_drawdown_breached.

    Avoids spinning up a real Engine (Alpaca client, sensors, dashboard) just
    to exercise the drawdown math.
    """

    session_peak_pnl: float
    session_pnl: float


def _check(stub: _Stub) -> bool:
    return Engine._max_drawdown_breached(stub)


@pytest.fixture(autouse=True)
def reset_drawdown_config(monkeypatch):
    # Pin known values so a stale env doesn't poison the assertions.
    monkeypatch.setattr(config, "MAX_SESSION_DRAWDOWN_USD", 1000.0)
    monkeypatch.setattr(config, "MAX_SESSION_DRAWDOWN_PCT", 0.02)
    monkeypatch.setattr(config, "DRAWDOWN_PCT_MIN_PEAK_USD", 100.0)


def test_no_drawdown_when_pnl_above_peak():
    stub = _Stub()
    stub.session_peak_pnl = 50.0
    stub.session_pnl = 60.0
    assert _check(stub) is False


def test_tiny_peak_with_modest_loss_does_not_trip_pct_gate():
    # Pre-fix bug: $0.91 peak, $9 loss = 1000% "drawdown" trips 2% gate.
    stub = _Stub()
    stub.session_peak_pnl = 0.91
    stub.session_pnl = -8.59
    assert _check(stub) is False
    # Still under the absolute USD cap and below the percent-gate floor.


def test_absolute_usd_cap_trips_at_large_loss_regardless_of_peak():
    # Even with a large peak, a $1500 absolute drawdown trips the USD cap.
    stub = _Stub()
    stub.session_peak_pnl = 50_000.0
    stub.session_pnl = 48_500.0
    assert _check(stub) is True


def test_pct_gate_trips_only_when_peak_above_floor():
    # Peak just under floor: 50% drawdown ratio but no trip.
    stub = _Stub()
    stub.session_peak_pnl = 50.0
    stub.session_pnl = 25.0
    assert _check(stub) is False
    # Peak above floor with the same 50% ratio: trips.
    stub.session_peak_pnl = 200.0
    stub.session_pnl = 100.0
    assert _check(stub) is True


def test_pct_gate_at_2pct_with_meaningful_peak():
    # Production-like: $10K peak, $250 loss = 2.5% drawdown → trips.
    stub = _Stub()
    stub.session_peak_pnl = 10_000.0
    stub.session_pnl = 9_750.0
    assert _check(stub) is True


def test_pct_gate_does_not_trip_below_pct_threshold():
    # $10K peak, $100 loss = 1% drawdown < 2% → no trip.
    stub = _Stub()
    stub.session_peak_pnl = 10_000.0
    stub.session_pnl = 9_900.0
    assert _check(stub) is False
