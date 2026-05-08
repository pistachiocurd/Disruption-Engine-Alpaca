"""
layer2_alpha.py — Alpha generation.

The ONLY place in the system where directional opinions are formed.

Components:
    TCNSpikePredictor   — temporal convnet → turbulence index
    HeatEquationSolver  — attrition-adjusted equilibrium price + PDE diffusion
    AlphaEngine         — orchestrator that produces TradeMandate objects
    TradeMandate        — typed payload consumed by Layer 3
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BASE_EXECUTION_WINDOW,
    CAPTURE_RATIO,
    EPSILON,
    FD_DOMAIN_PCT,
    FD_STABILITY_FACTOR,
    LAMBDA_ATTRITION,
    LAMBDA_SENSITIVITY_DELTA,
    MAX_POSITION_LIMIT,
    MIN_ORDER_SIZE,
    P_EQ_CONFIDENCE_BAND_BPS,
    SYMBOL,
    TCN_DILATIONS,
    TCN_HIDDEN_CHANNELS,
    TCN_INPUT_CHANNELS,
    TCN_INPUT_LENGTH,
    TICK_SIZE,
    TURBULENCE_THRESHOLD,
)

log = logging.getLogger(__name__)


# ============================================================================
# TradeMandate
# ============================================================================
@dataclass
class TradeMandate:
    direction: int                  # +1 long, -1 short
    target_size: float              # base currency
    p_decision: float               # mid at mandate generation (fixed IS reference)
    execution_window_seconds: float
    vpin: float
    alpha_calibrated: float
    viscosity: float
    symbol: str = SYMBOL


# ============================================================================
# TCN Spike Predictor
# ============================================================================
class CausalConv1d(nn.Module):
    """Causal 1D convolution. Pads only on the left so no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Two stacked dilated causal convs with a residual connection."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x + residual


class TCNSpikePredictor(nn.Module):
    """
    Input:  (batch, 3, 60) — [ce_ratio, obi, liquidation_rate] over 60 ticks.
    Output: (batch,) — turbulence_index in (0, 1).
    """

    def __init__(
        self,
        in_channels: int = TCN_INPUT_CHANNELS,
        hidden_channels: int = TCN_HIDDEN_CHANNELS,
        dilations: tuple = TCN_DILATIONS,
        seq_len: int = TCN_INPUT_LENGTH,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            TCNBlock(hidden_channels, dilation=d) for d in dilations
        )
        self.head = nn.Linear(hidden_channels, 1)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        # Take the LAST timestep (causal, so it summarizes the full window).
        x_last = x[:, :, -1]
        return torch.sigmoid(self.head(x_last)).squeeze(-1)


# ============================================================================
# Heat Equation Solver
# ============================================================================
class HeatEquationSolver:
    """
    Computes the post-shock equilibrium price by integrating attrition-adjusted
    book depth in the direction of the shock until the cumulative absorbed
    depth equals the imbalance volume Q.

    Also runs the attrition-sensitivity check: re-solving at λ ± δ. If the
    P_eq range exceeds the confidence band, the result is None and Layer 2
    suppresses the mandate.

    The PDE is used to bound the diffusion timescale (and to enforce the
    CFL stability condition on the explicit scheme), but the trade decision
    only needs P_eq itself.
    """

    def __init__(
        self,
        lambda_attrition: float = LAMBDA_ATTRITION,
        lambda_delta: float = LAMBDA_SENSITIVITY_DELTA,
        confidence_band_bps: float = P_EQ_CONFIDENCE_BAND_BPS,
        domain_pct: float = FD_DOMAIN_PCT,
        tick_size: float = TICK_SIZE,
    ) -> None:
        self.lambda_attrition = lambda_attrition
        self.lambda_delta = lambda_delta
        self.confidence_band_bps = confidence_band_bps
        self.domain_pct = domain_pct
        self.tick_size = tick_size

    @staticmethod
    def _shock_volume_and_direction(ob: dict) -> tuple[float, int]:
        """
        Q is the net order-book imbalance at the triggering tick, signed:
        positive Q with direction=+1 means buy pressure / shock will push price up.
        Returns (|Q|, direction).
        """
        bid_total = sum(sz for _, sz in ob["bids"])
        ask_total = sum(sz for _, sz in ob["asks"])
        net = bid_total - ask_total
        direction = 1 if net > 0 else -1
        return abs(net), direction

    def _solve_equilibrium_price(
        self,
        ob: dict,
        Q: float,
        direction: int,
        lambda_attr: float,
        vpin: float,
        p_current: float,
    ) -> Optional[float]:
        """
        Walk the side of the book that absorbs the shock, applying the attrition
        discount adjusted_depth_i = raw_i * exp(-λ * vpin * d_i_bps), and find
        the price at which the cumulative adjusted depth equals Q.
        """
        if Q <= 0:
            return None

        # If direction == +1 (buy shock pushing price up), the ASK side absorbs.
        side = ob["asks"] if direction == +1 else ob["bids"]
        if not side:
            return None

        cumulative = 0.0
        prev_price = p_current
        for price, size in side:
            d_bps = abs(price - p_current) / p_current * 10_000.0
            discount = np.exp(-lambda_attr * vpin * d_bps)
            adj = size * discount
            if cumulative + adj >= Q:
                # Linear interpolation within this level.
                needed = Q - cumulative
                frac = needed / max(adj, EPSILON)
                p_eq = prev_price + frac * (price - prev_price)
                return float(p_eq)
            cumulative += adj
            prev_price = price

        # Q exceeds available adjusted depth in domain — clamp to last level price
        # but also enforce the FD domain ceiling.
        domain_lim = (
            p_current * (1.0 + self.domain_pct) if direction == +1
            else p_current * (1.0 - self.domain_pct)
        )
        return float(min(prev_price, domain_lim) if direction == +1
                     else max(prev_price, domain_lim))

    def solve(
        self,
        ob: dict,
        physics_state,
        p_current: float,
    ) -> Optional[dict]:
        """
        Returns dict with {p_eq, direction, target_size, diffusion_timescale} or
        None if mandate should be suppressed (attrition-uncertainty).
        """
        Q, direction = self._shock_volume_and_direction(ob)
        if Q < MIN_ORDER_SIZE:
            return None

        vpin = physics_state.vpin

        # Sensitivity check: solve at λ-δ, λ, λ+δ.
        p_eq_low = self._solve_equilibrium_price(
            ob, Q, direction, self.lambda_attrition - self.lambda_delta, vpin, p_current
        )
        p_eq_base = self._solve_equilibrium_price(
            ob, Q, direction, self.lambda_attrition, vpin, p_current
        )
        p_eq_high = self._solve_equilibrium_price(
            ob, Q, direction, self.lambda_attrition + self.lambda_delta, vpin, p_current
        )
        if any(p is None for p in (p_eq_low, p_eq_base, p_eq_high)):
            log.info("MANDATE_SUPPRESSED: P_eq solution missing for one or more λ values")
            return None

        # Confidence band check (in bps from current price).
        p_lo = min(p_eq_low, p_eq_base, p_eq_high)
        p_hi = max(p_eq_low, p_eq_base, p_eq_high)
        range_bps = (p_hi - p_lo) / p_current * 10_000.0
        if range_bps > self.confidence_band_bps:
            log.info(
                f"MANDATE_SUPPRESSED_ATTRITION_UNCERTAINTY: "
                f"P_eq range = {range_bps:.2f} bps (limit {self.confidence_band_bps}); "
                f"λ={self.lambda_attrition} δ={self.lambda_delta}"
            )
            return None

        # Diffusion timescale from CFL: Δt = FD_STABILITY_FACTOR * Δx² / α_calibrated.
        alpha = max(EPSILON, physics_state.alpha_calibrated)
        dt = FD_STABILITY_FACTOR * (self.tick_size ** 2) / alpha

        # Target size derivation.
        target_size = min(Q * CAPTURE_RATIO, MAX_POSITION_LIMIT)
        # Round to lot size (assume MIN_ORDER_SIZE is the increment).
        target_size = max(0.0, round(target_size / MIN_ORDER_SIZE) * MIN_ORDER_SIZE)
        if target_size < MIN_ORDER_SIZE:
            return None

        return {
            "p_eq": p_eq_base,
            "direction": direction,
            "target_size": target_size,
            "diffusion_timescale_sec": dt,
            "Q": Q,
        }


# ============================================================================
# AlphaEngine — orchestrator
# ============================================================================
class AlphaEngine:
    """
    On each tick:
      1. Append latest [ce_ratio, obi, liquidation_rate] to rolling buffer.
      2. If buffer is full (60 ticks) and ood_flag is False, run the TCN.
      3. If turbulence_index > threshold, run the heat solver.
      4. Solver returns either P_eq (mandate) or None (suppressed).
      5. Build TradeMandate.
    """

    def __init__(
        self,
        tcn: Optional[TCNSpikePredictor] = None,
        solver: Optional[HeatEquationSolver] = None,
        device: str = "cpu",
        turbulence_threshold: float = TURBULENCE_THRESHOLD,
    ) -> None:
        self.device = torch.device(device)
        self.tcn = (tcn or TCNSpikePredictor()).to(self.device).eval()
        self.solver = solver or HeatEquationSolver()
        self.turbulence_threshold = turbulence_threshold
        self._buffer: deque = deque(maxlen=TCN_INPUT_LENGTH)
        # Rolling log of evaluation outcomes — read by the dashboard.
        self.decision_log: deque = deque(maxlen=200)
        self.last_turbulence: float = 0.0

    def _record(self, outcome: str, **fields) -> None:
        rec = {"ts": int(time.time() * 1000), "outcome": outcome, **fields}
        self.decision_log.append(rec)

    def _push_features(self, physics_state) -> None:
        feats = np.array([
            physics_state.ce_ratio,
            physics_state.obi,
            physics_state.liquidation_rate,
        ], dtype=np.float32)
        self._buffer.append(feats)

    def _tcn_predict(self) -> float:
        if len(self._buffer) < TCN_INPUT_LENGTH:
            return 0.0
        # (T, C) → (1, C, T)
        arr = np.stack(self._buffer, axis=0).T[None, :, :]
        x = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            y = self.tcn(x).item()
        return float(y)

    def evaluate(self, physics_state) -> Optional[TradeMandate]:
        """Returns a TradeMandate if a shock signature is confirmed, else None."""
        self._push_features(physics_state)

        if physics_state.ood_flag:
            log.info("MANDATE_SUPPRESSED_OOD: mahal_dist=%.3f", physics_state.mahal_dist)
            self._record("suppressed_ood", mahal_dist=float(physics_state.mahal_dist))
            return None

        if len(self._buffer) < TCN_INPUT_LENGTH:
            self._record("warming_up", buffer_size=len(self._buffer))
            return None

        turbulence = self._tcn_predict()
        self.last_turbulence = turbulence
        if turbulence < self.turbulence_threshold:
            self._record("below_turbulence", turbulence=float(turbulence))
            return None

        ob = physics_state.ob_snapshot
        if ob is None:
            self._record("no_orderbook")
            return None
        p_current = 0.5 * (ob["bids"][0][0] + ob["asks"][0][0])

        result = self.solver.solve(ob, physics_state, p_current)
        if result is None:
            self._record("suppressed_attrition", turbulence=float(turbulence))
            return None

        # Execution window shrinks with VPIN (high VPIN → tight window).
        exec_window = BASE_EXECUTION_WINDOW * (1.0 - physics_state.vpin)
        exec_window = max(1.0, exec_window)

        mandate = TradeMandate(
            direction=result["direction"],
            target_size=result["target_size"],
            p_decision=p_current,
            execution_window_seconds=exec_window,
            vpin=physics_state.vpin,
            alpha_calibrated=physics_state.alpha_calibrated,
            viscosity=physics_state.viscosity,
            symbol=SYMBOL,
        )
        self._record(
            "mandate",
            direction=int(mandate.direction),
            target_size=float(mandate.target_size),
            p_decision=float(mandate.p_decision),
            p_eq=float(result["p_eq"]),
            turbulence=float(turbulence),
            vpin=float(physics_state.vpin),
        )
        return mandate

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.tcn.load_state_dict(state)
        self.tcn.eval()
