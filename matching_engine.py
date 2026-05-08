"""
matching_engine.py — LocalMatchingEngine for shadow / paper trading.

CRITICAL CONSTRAINT: This module MUST NOT import ccxt or any exchange client.
The constraint is structural — if anyone adds `import ccxt` here, the module-level
assertion at the bottom will trip on first import.

The engine simulates fills against the LIVE order book snapshot held by the
SensorArray. It produces realistic partial-fill behavior (depth-limited market
orders, cross-sensitive limit orders) without ever touching the exchange API.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional

from config import (
    FEE_AGGRESSION_THRESHOLD,
    MAKER_FEE,
    MAX_PASSIVE_OFFSET_BPS,
    TAKER_FEE,
)

log = logging.getLogger(__name__)


@dataclass
class FillResult:
    """Result of a simulated fill attempt for one tick."""
    filled_qty: float          # base currency filled this step
    fill_price: float          # average fill price (0.0 if no fill)
    fee_rate: float            # rate applied (TAKER_FEE or MAKER_FEE)
    fee_paid: float            # absolute fee in quote currency
    order_type: str            # 'market' or 'limit'
    is_taker: bool


def _walk_book(levels: list, target_qty: float) -> tuple[float, float]:
    """
    Walk an order book side, consuming up to target_qty.

    levels: [[price, size], ...] sorted best-first.
    Returns: (avg_fill_price, qty_filled). qty_filled may be < target_qty if depth is exhausted.
    """
    remaining = target_qty
    cost = 0.0
    filled = 0.0
    for price, size in levels:
        if remaining <= 0:
            break
        take = min(size, remaining)
        cost += take * price
        filled += take
        remaining -= take
    if filled <= 0:
        return 0.0, 0.0
    return cost / filled, filled


class LocalMatchingEngine:
    """
    Paper-trading matching engine.

    Public API:
        attempt_fill(action, side, remaining_volume, ob_snapshot, prev_ob_snapshot=None)
            → FillResult

    Behavior:
        - action < FEE_AGGRESSION_THRESHOLD  → market order (taker), depth-limited.
        - action >= FEE_AGGRESSION_THRESHOLD → passive limit (maker). Fills only if
          the live book crosses the limit price between the previous and current
          snapshot. Otherwise zero fill that step (carries to next step).

    The action magnitude maps to a passive offset in basis points so that the
    PPO agent can learn the maker-vs-taker tradeoff and the depth of the
    passive post (best bid vs. one tick back vs. multiple ticks back).
    """

    def __init__(self) -> None:
        self._open_limit_price: Optional[float] = None
        self._open_limit_side: Optional[str] = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _mid(ob: dict) -> float:
        return 0.5 * (ob["bids"][0][0] + ob["asks"][0][0])

    @staticmethod
    def _action_to_limit_price(action: float, side: str, mid: float) -> float:
        """
        Map action ∈ [FEE_AGGRESSION_THRESHOLD, 1.0] to a limit price.
        action = 0.0 → at-best (top of book), action = 1.0 → MAX_PASSIVE_OFFSET_BPS deeper.
        For BUY: deeper = lower price. For SELL: deeper = higher price.
        """
        # Normalize action into [0, 1] over the passive band.
        passive_intensity = max(0.0, min(1.0, action))
        offset_bps = passive_intensity * MAX_PASSIVE_OFFSET_BPS
        offset = (offset_bps / 10_000.0) * mid
        if side == "buy":
            return mid - offset
        return mid + offset

    # --------------------------------------------------------------- fill API
    def attempt_fill(
        self,
        action: float,
        side: str,
        remaining_volume: float,
        ob_snapshot: dict,
        prev_ob_snapshot: Optional[dict] = None,
    ) -> FillResult:
        """
        Attempt to fill `remaining_volume` against `ob_snapshot`.

        Args:
            action: scalar in [-1.0, 1.0]. < threshold → market; ≥ threshold → limit.
            side: 'buy' or 'sell'.
            remaining_volume: base currency outstanding.
            ob_snapshot: current L2 snapshot (dict with 'bids', 'asks').
            prev_ob_snapshot: previous L2 snapshot, required for limit-cross detection.
        """
        if remaining_volume <= 0.0:
            return FillResult(0.0, 0.0, 0.0, 0.0, "noop", False)

        # ---------- TAKER PATH ---------------------------------------------
        if action < FEE_AGGRESSION_THRESHOLD:
            book_side = ob_snapshot["asks"] if side == "buy" else ob_snapshot["bids"]
            avg_price, filled = _walk_book(book_side, remaining_volume)
            if filled <= 0.0:
                return FillResult(0.0, 0.0, TAKER_FEE, 0.0, "market", True)
            fee_paid = filled * avg_price * TAKER_FEE
            self._open_limit_price = None
            self._open_limit_side = None
            return FillResult(filled, avg_price, TAKER_FEE, fee_paid, "market", True)

        # ---------- MAKER PATH ---------------------------------------------
        # Limit-order semantics: the price is locked when the order is FIRST
        # posted, not recomputed every tick. We re-compute only when no order
        # is currently open, OR when the side has changed (mandate re-attach).
        if (
            self._open_limit_price is None
            or self._open_limit_side != side
        ):
            mid = self._mid(ob_snapshot)
            limit_price = self._action_to_limit_price(action, side, mid)
            self._open_limit_price = limit_price
            self._open_limit_side = side
        else:
            limit_price = self._open_limit_price

        if prev_ob_snapshot is None:
            # First tick of episode — post but don't fill.
            return FillResult(0.0, 0.0, MAKER_FEE, 0.0, "limit", False)

        # Limit fills if the OPPOSING side of the current book crosses our price.
        # BUY limit at L fills if ask <= L. SELL limit at L fills if bid >= L.
        if side == "buy":
            best_ask_now = ob_snapshot["asks"][0][0]
            best_ask_prev = prev_ob_snapshot["asks"][0][0]
            crossed_now = best_ask_now <= limit_price
            crossed_prev = best_ask_prev <= limit_price
            if crossed_now and not crossed_prev:
                # Fresh cross — assume fill at limit price up to current best level depth.
                avail = ob_snapshot["asks"][0][1]
                filled = min(remaining_volume, avail)
                fee_paid = filled * limit_price * MAKER_FEE
                return FillResult(filled, limit_price, MAKER_FEE, fee_paid, "limit", False)
        else:  # sell
            best_bid_now = ob_snapshot["bids"][0][0]
            best_bid_prev = prev_ob_snapshot["bids"][0][0]
            crossed_now = best_bid_now >= limit_price
            crossed_prev = best_bid_prev >= limit_price
            if crossed_now and not crossed_prev:
                avail = ob_snapshot["bids"][0][1]
                filled = min(remaining_volume, avail)
                fee_paid = filled * limit_price * MAKER_FEE
                return FillResult(filled, limit_price, MAKER_FEE, fee_paid, "limit", False)

        return FillResult(0.0, 0.0, MAKER_FEE, 0.0, "limit", False)

    def cancel(self) -> None:
        """Cancel any open paper limit. Used at episode end / window expiry."""
        self._open_limit_price = None
        self._open_limit_side = None


# ============================================================================
# IMPORT-TIME SAFETY ASSERTION
# ============================================================================
# This is the structural guarantee that LocalMatchingEngine cannot accidentally
# place real orders. If any import in this module's transitive graph pulls in
# ccxt, the assertion fires immediately on first import.
_FORBIDDEN = ("ccxt", "ccxt.pro", "ccxtpro")
for _mod in _FORBIDDEN:
    assert _mod not in sys.modules or sys.modules[_mod] is None, (
        f"matching_engine.py must not transitively import {_mod}. "
        f"This is a hard safety constraint."
    )
