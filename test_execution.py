"""Tests for layer3_execution.py."""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from layer2_alpha import TradeMandate
import layer3_execution
from layer3_execution import ExecutionEnv, PPOAgent
from matching_engine import LocalMatchingEngine

# These tests pre-date the equities pivot — every fixture below uses BTC-shaped
# mandates (mid=$50K, target_size=0.05). With the equity default
# MIN_ORDER_SIZE=1.0 the env terminates after the first fill on a 0.05-unit
# mandate, which breaks tests that step the env multiple times. Pin a
# crypto-appropriate min order size at module load so these tests exercise
# the executor's contract on the size scale they were written for.
layer3_execution.MIN_ORDER_SIZE = 0.001


# ============================================================================
# Test fixtures
# ============================================================================
def _make_book(mid: float = 50_000.0) -> dict:
    bids = [[mid - 0.5 * (i + 1), 0.5] for i in range(10)]
    asks = [[mid + 0.5 * (i + 1), 0.5] for i in range(10)]
    return {"bids": bids, "asks": asks, "timestamp": int(time.time() * 1000)}


class _FakeStateProvider:
    """Stand-in for SensorArray during env tests."""

    class _State:
        def __init__(self, ob):
            self.ob_snapshot = ob

    def __init__(self, ob):
        self.state = self._State(ob)

    def update_book(self, ob):
        self.state.ob_snapshot = ob


def _make_mandate(direction: int = +1, size: float = 0.05) -> TradeMandate:
    return TradeMandate(
        direction=direction,
        target_size=size,
        p_decision=50_000.0,
        execution_window_seconds=10.0,
        vpin=0.95,
        alpha_calibrated=0.171,
        viscosity=1.5,
        symbol="BTC/USDT:USDT",
    )


# ============================================================================
# Env shape & step contract
# ============================================================================
class TestEnvShapes:
    def test_observation_shape(self):
        sp = _FakeStateProvider(_make_book())
        env = ExecutionEnv(live_state_provider=sp, shadow_mode=True)
        env.attach_mandate(_make_mandate())
        obs, _ = env.reset()
        assert obs.shape == (42,)
        assert obs.dtype == np.float32

    def test_action_space_bounds(self):
        sp = _FakeStateProvider(_make_book())
        env = ExecutionEnv(live_state_provider=sp, shadow_mode=True)
        env.attach_mandate(_make_mandate())
        env.reset()
        # Out-of-range actions should be clipped, not crash.
        env.step(np.array([5.0]))
        env.step(np.array([-5.0]))


# ============================================================================
# Reward signs
# ============================================================================
class TestRewardSigns:
    def test_long_taker_step_is_negative_reward(self):
        # Buying via market should produce IS_fee_adj > 0 (paid above decision price)
        # → reward = -IS - other_penalties is NEGATIVE.
        sp = _FakeStateProvider(_make_book())
        env = ExecutionEnv(live_state_provider=sp, shadow_mode=True)
        mandate = _make_mandate(direction=+1, size=0.1)
        env.attach_mandate(mandate)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0]))  # full taker
        assert info["filled"] > 0
        # We bought at ask > mid, so IS > 0 → step reward should be negative.
        assert reward < 0


# ============================================================================
# Terminal forced fill
# ============================================================================
class TestTerminalFill:
    def test_truncation_forces_market_fill(self):
        sp = _FakeStateProvider(_make_book())
        env = ExecutionEnv(live_state_provider=sp, shadow_mode=True)
        mandate = _make_mandate(direction=+1, size=0.05)
        # Force a near-zero time budget to trigger truncation immediately.
        mandate.execution_window_seconds = 0.001
        env.attach_mandate(mandate)
        env.reset()
        time.sleep(0.005)
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))  # passive — won't fill
        assert truncated
        # Terminal penalty should make reward strongly negative.
        assert reward < 0


# ============================================================================
# Agent — separate encoder weights and action bounds
# ============================================================================
class TestPPOAgent:
    def test_split_encoders_have_separate_parameters(self):
        agent = PPOAgent()
        bid_params = list(agent.bid_encoder.parameters())
        ask_params = list(agent.ask_encoder.parameters())
        # Different parameter objects (not weight-tied).
        for bp, ap in zip(bid_params, ask_params):
            assert bp is not ap

    def test_act_returns_bounded_action(self):
        agent = PPOAgent()
        obs = np.zeros(42, dtype=np.float32)
        for _ in range(20):
            action, log_prob, value = agent.act(obs)
            assert action.shape == (1,)
            assert -1.0 <= float(action[0]) <= 1.0

    def test_deterministic_act_is_repeatable(self):
        agent = PPOAgent()
        obs = np.random.randn(42).astype(np.float32)
        a1, _, _ = agent.act(obs, deterministic=True)
        a2, _, _ = agent.act(obs, deterministic=True)
        assert np.allclose(a1, a2)
