"""Tests for layer2_alpha.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from layer1_sensors import PhysicsState
from layer2_alpha import (
    AlphaEngine,
    HeatEquationSolver,
    TCNSpikePredictor,
    TradeMandate,
)


# ============================================================================
# TCN
# ============================================================================
class TestTCN:
    def test_forward_shape(self):
        net = TCNSpikePredictor()
        x = torch.randn(4, 3, 60)
        y = net(x)
        assert y.shape == (4,)
        assert (y >= 0).all() and (y <= 1).all()

    def test_short_input_rejected_via_engine(self):
        # AlphaEngine should not invoke TCN until buffer is full.
        engine = AlphaEngine()
        state = _quiet_state()
        for _ in range(30):  # less than TCN_INPUT_LENGTH = 60
            mandate = engine.evaluate(state)
        assert mandate is None  # buffer not full → no mandate


# ============================================================================
# Heat equation solver
# ============================================================================
def _shocked_book(p_mid: float = 100.0) -> dict:
    """Heavy bid pressure → buy shock (direction = +1). Asks should absorb."""
    bids = [[p_mid - 0.5 * (i + 1), 5.0] for i in range(20)]
    asks = [[p_mid + 0.5 * (i + 1), 0.5] for i in range(20)]  # thin asks
    return {"bids": bids, "asks": asks, "timestamp": 1_700_000_000_000}


def _balanced_book(p_mid: float = 100.0) -> dict:
    bids = [[p_mid - 0.5 * (i + 1), 1.0] for i in range(20)]
    asks = [[p_mid + 0.5 * (i + 1), 1.0] for i in range(20)]
    return {"bids": bids, "asks": asks, "timestamp": 1_700_000_000_000}


def _quiet_state() -> PhysicsState:
    return PhysicsState(
        vpin=0.1, alpha_calibrated=0.018, viscosity=1.0,
        regime=0, liquidation_rate=0.0, ood_flag=False, mahal_dist=0.5,
        ce_ratio=1.0, obi=0.0, spread_velocity=0.0, timestamp=0,
        ob_snapshot=_balanced_book(),
    )


def _shocked_state(ood_flag: bool = False) -> PhysicsState:
    return PhysicsState(
        vpin=0.95, alpha_calibrated=0.171, viscosity=2.0,
        regime=2, liquidation_rate=0.4, ood_flag=ood_flag, mahal_dist=2.0,
        ce_ratio=8.0, obi=0.6, spread_velocity=2.0, timestamp=0,
        ob_snapshot=_shocked_book(),
    )


class TestHeatEquationSolver:
    def test_solver_finds_p_eq_for_buy_shock(self):
        solver = HeatEquationSolver(lambda_attrition=0.0,  # no attrition for clean depth-walk test
                                    lambda_delta=0.0,
                                    confidence_band_bps=10_000.0)
        state = _shocked_state()
        result = solver.solve(state.ob_snapshot, state, p_current=100.0)
        assert result is not None
        assert result["direction"] == +1
        assert result["p_eq"] > 100.0

    def test_attrition_sensitivity_suppresses_when_unstable(self):
        # Force a wide sensitivity band by setting δ very large and tight band threshold.
        solver = HeatEquationSolver(
            lambda_attrition=0.15, lambda_delta=2.0, confidence_band_bps=0.01,
        )
        state = _shocked_state()
        result = solver.solve(state.ob_snapshot, state, p_current=100.0)
        assert result is None

    def test_zero_imbalance_returns_none(self):
        solver = HeatEquationSolver()
        state = PhysicsState(
            vpin=0.95, alpha_calibrated=0.171, viscosity=1.0,
            regime=2, liquidation_rate=0.0, ood_flag=False, mahal_dist=0.0,
            ce_ratio=0.0, obi=0.0, spread_velocity=0.0, timestamp=0,
            ob_snapshot=_balanced_book(),
        )
        # Balanced book → near-zero net imbalance → below MIN_ORDER_SIZE.
        result = solver.solve(state.ob_snapshot, state, p_current=100.0)
        assert result is None


# ============================================================================
# AlphaEngine — OOD and turbulence gating
# ============================================================================
class TestAlphaEngineGating:
    def test_ood_flag_suppresses_mandate(self):
        engine = AlphaEngine(turbulence_threshold=0.0)  # accept any TCN output
        state = _shocked_state(ood_flag=True)
        # Fill buffer
        for _ in range(60):
            mandate = engine.evaluate(state)
        assert mandate is None  # OOD gate must trigger regardless of TCN

    def test_low_turbulence_suppresses_mandate(self):
        # Set threshold to 1.01 so the sigmoid-bounded TCN can never exceed it.
        engine = AlphaEngine(turbulence_threshold=1.01)
        state = _shocked_state()
        for _ in range(60):
            mandate = engine.evaluate(state)
        assert mandate is None

    def test_full_pipeline_can_emit_mandate(self):
        # With a permissive threshold and a proper shock state, we expect a mandate.
        engine = AlphaEngine(turbulence_threshold=0.0)
        state = _shocked_state(ood_flag=False)
        mandate = None
        for _ in range(60):
            mandate = engine.evaluate(state)
        assert mandate is not None
        assert mandate.direction in (+1, -1)
        assert mandate.target_size > 0
        assert mandate.execution_window_seconds > 0
