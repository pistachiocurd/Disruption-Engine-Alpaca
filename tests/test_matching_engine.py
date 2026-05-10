"""Tests for matching_engine.py."""
from __future__ import annotations

import sys

import pytest

from matching_engine import LocalMatchingEngine


def _book(bid_levels: list, ask_levels: list, ts: int = 1_700_000_000_000) -> dict:
    return {"bids": bid_levels, "asks": ask_levels, "timestamp": ts}


# ============================================================================
# Taker path
# ============================================================================
class TestTakerFills:
    def test_market_buy_walks_asks(self):
        eng = LocalMatchingEngine()
        ob = _book([[99.5, 5.0]], [[100.0, 0.5], [100.5, 1.0], [101.0, 5.0]])
        result = eng.attempt_fill(action=-1.0, side="buy", remaining_volume=2.0,
                                  ob_snapshot=ob, prev_ob_snapshot=None)
        # Should fill all 2.0 walking through 0.5 + 1.0 + 0.5 = 2.0.
        assert result.filled_qty == pytest.approx(2.0)
        assert result.is_taker
        # Average price = (0.5*100 + 1.0*100.5 + 0.5*101) / 2.0 = 100.5
        assert result.fill_price == pytest.approx(100.5, rel=1e-3)

    def test_market_sell_walks_bids(self):
        eng = LocalMatchingEngine()
        ob = _book([[99.5, 1.0], [99.0, 1.0], [98.5, 1.0]], [[100.5, 5.0]])
        result = eng.attempt_fill(action=-1.0, side="sell", remaining_volume=2.5,
                                  ob_snapshot=ob, prev_ob_snapshot=None)
        assert result.filled_qty == pytest.approx(2.5)
        assert result.is_taker

    def test_market_partial_fill_when_depth_insufficient(self):
        eng = LocalMatchingEngine()
        ob = _book([[99.5, 5.0]], [[100.0, 0.3]])
        result = eng.attempt_fill(action=-1.0, side="buy", remaining_volume=10.0,
                                  ob_snapshot=ob, prev_ob_snapshot=None)
        assert result.filled_qty == pytest.approx(0.3)


# ============================================================================
# Maker path
# ============================================================================
class TestMakerFills:
    def test_first_tick_no_fill(self):
        eng = LocalMatchingEngine()
        ob = _book([[99.5, 5.0]], [[100.5, 5.0]])
        # First tick of a passive limit posts but cannot fill (no prev snapshot).
        result = eng.attempt_fill(action=1.0, side="buy", remaining_volume=1.0,
                                  ob_snapshot=ob, prev_ob_snapshot=None)
        assert result.filled_qty == 0.0
        assert not result.is_taker

    def test_limit_buy_fills_on_fresh_cross(self):
        eng = LocalMatchingEngine()
        # Post buy limit; book then crosses the limit price.
        ob_prev = _book([[99.5, 5.0]], [[100.5, 5.0]])
        # First call: post (no fill).
        eng.attempt_fill(action=0.5, side="buy", remaining_volume=1.0,
                         ob_snapshot=ob_prev, prev_ob_snapshot=None)
        # Now ask drops below where our limit would have been.
        ob_curr = _book([[99.5, 5.0]], [[99.95, 1.0]])
        result = eng.attempt_fill(action=0.5, side="buy", remaining_volume=1.0,
                                  ob_snapshot=ob_curr, prev_ob_snapshot=ob_prev)
        assert result.filled_qty > 0
        assert not result.is_taker

    def test_limit_no_fill_when_no_cross(self):
        eng = LocalMatchingEngine()
        ob = _book([[99.5, 5.0]], [[100.5, 5.0]])
        eng.attempt_fill(action=0.5, side="buy", remaining_volume=1.0,
                         ob_snapshot=ob, prev_ob_snapshot=None)
        # Book unchanged.
        result = eng.attempt_fill(action=0.5, side="buy", remaining_volume=1.0,
                                  ob_snapshot=ob, prev_ob_snapshot=ob)
        assert result.filled_qty == 0.0


# ============================================================================
# Cancel
# ============================================================================
class TestCancel:
    def test_cancel_clears_state(self):
        eng = LocalMatchingEngine()
        ob = _book([[99.5, 5.0]], [[100.5, 5.0]])
        eng.attempt_fill(action=0.5, side="buy", remaining_volume=1.0,
                         ob_snapshot=ob, prev_ob_snapshot=None)
        eng.cancel()
        assert eng._open_limit_price is None
        assert eng._open_limit_side is None


# ============================================================================
# Structural safety: no ccxt imports in the matching engine
# ============================================================================
class TestStructuralIsolation:
    def test_matching_engine_does_not_import_ccxt(self):
        # The module-level assertion in matching_engine.py runs at import time;
        # if it had failed, this test file wouldn't be importable. As a belt-
        # and-suspenders check, scan the source for actual import statements
        # naming ccxt.
        import re

        import matching_engine
        src_path = matching_engine.__file__
        with open(src_path) as f:
            src = f.read()
        # Forbid: `import ccxt`, `import ccxt.X`, `from ccxt import X`, etc.
        forbidden_patterns = [
            r"^\s*import\s+ccxt(\s|\.|$)",
            r"^\s*from\s+ccxt(\s|\.)",
        ]
        for line in src.splitlines():
            # Skip comments and string contents.
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pat in forbidden_patterns:
                assert not re.match(pat, line), (
                    f"matching_engine.py contains forbidden import: {line!r}"
                )
