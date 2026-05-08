"""Tests for layer1_sensors.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from layer1_sensors import (
    CEInferenceEngine,
    OODDetector,
    QuoteCancellationProxy,
    StudentTHMM,
    StudentTKalmanFilter,
    VPINToxicityTracker,
    student_t_logpdf,
)


# ============================================================================
# VPIN
# ============================================================================
class TestVPIN:
    def test_balanced_flow_low_vpin(self):
        v = VPINToxicityTracker(n_buckets=5, volume_avg_window_sec=60)
        ts = 1_700_000_000_000
        for i in range(200):
            side = "buy" if i % 2 == 0 else "sell"
            v.add_trade({"timestamp": ts + i * 100, "amount": 1.0, "side": side})
        assert v.vpin < 0.2, f"balanced flow should produce low VPIN, got {v.vpin}"

    def test_one_sided_flow_high_vpin(self):
        v = VPINToxicityTracker(n_buckets=5, volume_avg_window_sec=60)
        ts = 1_700_000_000_000
        for i in range(200):
            v.add_trade({"timestamp": ts + i * 100, "amount": 1.0, "side": "buy"})
        assert v.vpin > 0.9, f"one-sided flow should produce high VPIN, got {v.vpin}"

    def test_average_volume_10s(self):
        v = VPINToxicityTracker()
        ts = 1_700_000_000_000
        for i in range(20):
            v.add_trade({"timestamp": ts + i * 500, "amount": 0.5, "side": "buy"})
        # last 10s = trades with ts >= ts_last - 10000. ts_last = ts + 19*500 = ts + 9500.
        # So cutoff = ts - 500. All 20 trades within window → 10.0.
        assert v.average_volume_10s == pytest.approx(10.0)


# ============================================================================
# Student-t Kalman vs spread spike
# ============================================================================
class TestStudentTKalman:
    def test_does_not_smooth_spread_spike(self):
        kf = StudentTKalmanFilter(nu=4.0, process_noise=1e-4, measurement_noise=1e-3)
        # Steady spread of 1.0
        for _ in range(50):
            kf.update(1.0)
        baseline = kf.x[0]
        # Sudden spike
        kf.update(10.0)
        spiked = kf.x[0]
        # The Student-t Kalman should respond to the spike; its post-spike state
        # should be somewhere meaningfully above baseline (not heavily smoothed).
        assert spiked > baseline + 0.5, "Student-t Kalman should not smooth spread spike"

    def test_initialization(self):
        kf = StudentTKalmanFilter()
        kf.update(2.5)
        assert kf.x[0] == pytest.approx(2.5)
        assert kf.x[1] == pytest.approx(0.0)


# ============================================================================
# C/E inference
# ============================================================================
class TestCEInference:
    def test_pure_cancellation(self):
        ce = CEInferenceEngine(window_sec=60)
        prev = {"bids": [[100.0, 5.0]], "asks": [[101.0, 5.0]]}
        curr = {"bids": [[100.0, 2.0]], "asks": [[101.0, 5.0]]}  # 3 units cancelled
        ce.update(prev, curr, trades_in_window=[], ts_ms=1_700_000_000_000)
        assert ce.ce_ratio > 0
        # No executions → ratio should hit clip ceiling (3 / EPSILON capped at MAX).
        assert ce.ce_ratio == pytest.approx(50.0)

    def test_pure_execution(self):
        ce = CEInferenceEngine(window_sec=60)
        prev = {"bids": [[100.0, 5.0]], "asks": [[101.0, 5.0]]}
        curr = {"bids": [[100.0, 2.0]], "asks": [[101.0, 5.0]]}
        trades = [{"price": 100.0, "amount": 3.0, "side": "sell", "timestamp": 0}]
        ce.update(prev, curr, trades, ts_ms=1_700_000_000_000)
        # All depth reduction was explained by trades → 0 cancellations.
        assert ce.ce_ratio == pytest.approx(0.0, abs=1e-6)

    def test_mixed(self):
        ce = CEInferenceEngine(window_sec=60)
        prev = {"bids": [[100.0, 10.0]], "asks": [[101.0, 5.0]]}
        curr = {"bids": [[100.0, 3.0]], "asks": [[101.0, 5.0]]}  # reduced by 7
        trades = [{"price": 100.0, "amount": 3.0, "side": "sell", "timestamp": 0}]
        ce.update(prev, curr, trades, ts_ms=1_700_000_000_000)
        # 4 cancelled, 3 executed → ratio ≈ 1.33
        assert ce.ce_ratio == pytest.approx(4.0 / 3.0, rel=1e-3)


# ============================================================================
# OOD gate
# ============================================================================
class TestOODDetector:
    def test_within_distribution_passes(self):
        rng = np.random.default_rng(0)
        train = rng.normal(0.0, 1.0, size=(1000, 3))
        det = OODDetector(mu=train.mean(axis=0), sigma=np.cov(train, rowvar=False), threshold=4.5)
        flag, dist = det.evaluate(np.array([0.0, 0.0, 0.0]))
        assert not flag

    def test_outlier_triggers(self):
        rng = np.random.default_rng(0)
        train = rng.normal(0.0, 1.0, size=(1000, 3))
        det = OODDetector(mu=train.mean(axis=0), sigma=np.cov(train, rowvar=False), threshold=4.5)
        flag, dist = det.evaluate(np.array([20.0, 20.0, 20.0]))
        assert flag
        assert dist > 4.5


# ============================================================================
# Student-t HMM
# ============================================================================
class TestStudentTHMM:
    def test_default_priors_starts_in_laminar(self):
        hmm = StudentTHMM.from_default_priors()
        # quiet observation
        s = hmm.update(np.array([1.0, 0.0, 0.0, 0.0]))
        assert s == 0  # laminar

    def test_shock_observation_moves_to_turbulent(self):
        hmm = StudentTHMM.from_default_priors()
        # Repeated extreme-turbulence observations.
        for _ in range(50):
            s = hmm.update(np.array([8.0, 0.5, 3.0, 1.0]))
        assert s == 2

    def test_logpdf_consistency(self):
        # Student-t with large ν should approach Gaussian.
        x = student_t_logpdf(0.0, 0.0, 1.0, 1000.0)
        gauss = -0.5 * math.log(2 * math.pi)
        assert abs(x - gauss) < 0.01


# ============================================================================
# Quote-cancellation proxy (equities path replacement for L2 C/E inference)
# ============================================================================
class TestQuoteCancellationProxy:
    def test_balanced_flow_low_ratio(self):
        # Equal quote-update rate and trade rate → no inferred cancellations.
        p = QuoteCancellationProxy(window_sec=5.0)
        ts = 1_700_000_000_000
        for i in range(20):
            p.on_quote(ts + i * 100)
            p.on_trade(ts + i * 100)
        assert p.ce_ratio == pytest.approx(0.0)

    def test_quote_heavy_flow_high_ratio(self):
        p = QuoteCancellationProxy(window_sec=5.0)
        ts = 1_700_000_000_000
        for i in range(40):
            p.on_quote(ts + i * 50)
        for i in range(2):
            p.on_trade(ts + i * 50)
        assert p.ce_ratio > 5.0  # 38 unmatched quotes / 2 trades

    def test_window_pruning(self):
        p = QuoteCancellationProxy(window_sec=1.0)
        ts = 1_700_000_000_000
        # Old quotes outside the window should not contribute.
        for i in range(10):
            p.on_quote(ts + i * 50)
        # Advance well past the 1s window.
        for i in range(5):
            p.on_quote(ts + 5_000 + i * 50)
            p.on_trade(ts + 5_000 + i * 50)
        # The 10 stale quotes must have been pruned; current window is balanced.
        assert p.ce_ratio == pytest.approx(0.0)

    def test_reset_session_clears_state(self):
        p = QuoteCancellationProxy(window_sec=5.0)
        ts = 1_700_000_000_000
        for i in range(20):
            p.on_quote(ts + i * 50)
        assert p.ce_ratio > 0
        p.reset_session()
        assert p.ce_ratio == 0.0


# ============================================================================
# Lee-Ready trade-side classification (equities VPIN)
# ============================================================================
class TestVPINLeeReady:
    def test_classifies_above_mid_as_buy(self):
        v = VPINToxicityTracker(n_buckets=5, volume_avg_window_sec=60)
        ts = 1_700_000_000_000
        for i in range(200):
            v.add_trade_classified(
                {"timestamp": ts + i * 100, "amount": 1.0, "price": 100.05},
                mid_at_trade=100.00,
                last_trade_price=None,
            )
        assert v.vpin > 0.9, "above-mid prints should be classified as buys → high VPIN"

    def test_classifies_below_mid_as_sell(self):
        v = VPINToxicityTracker(n_buckets=5, volume_avg_window_sec=60)
        ts = 1_700_000_000_000
        for i in range(200):
            v.add_trade_classified(
                {"timestamp": ts + i * 100, "amount": 1.0, "price": 99.95},
                mid_at_trade=100.00,
                last_trade_price=None,
            )
        assert v.vpin > 0.9

    def test_at_mid_falls_back_to_tick_test(self):
        v = VPINToxicityTracker(n_buckets=2, volume_avg_window_sec=60)
        ts = 1_700_000_000_000
        # Two prints exactly at mid; second is higher than first → buy.
        v.add_trade_classified(
            {"timestamp": ts, "amount": 1.0, "price": 100.00},
            mid_at_trade=100.00,
            last_trade_price=99.99,
        )
        v.add_trade_classified(
            {"timestamp": ts + 100, "amount": 1.0, "price": 100.00},
            mid_at_trade=100.00,
            last_trade_price=99.99,
        )
        # Both classified as buys via the tick test.
        assert v._cur_buy + sum(b[0] for b in v._buckets) > 0
        assert v._cur_sell + sum(b[1] for b in v._buckets) == 0


# ============================================================================
# Session reset (equities overnight gap handling)
# ============================================================================
class TestSessionReset:
    def test_kalman_reset_drops_state(self):
        kf = StudentTKalmanFilter()
        for _ in range(20):
            kf.update(1.5)
        assert kf._initialized
        kf.reset_session()
        assert not kf._initialized
        assert kf.x[0] == 0.0
        assert kf.x[1] == 0.0

    def test_hmm_reset_uniform_alpha(self):
        hmm = StudentTHMM.from_default_priors()
        for _ in range(20):
            hmm.update(np.array([8.0, 0.5, 3.0, 1.0]))
        assert hmm.alpha[2] > 0.5  # turbulent dominates after shocks
        hmm.reset_session()
        np.testing.assert_allclose(hmm.alpha, np.full(3, 1.0 / 3.0))

    def test_vpin_reset_clears_buckets(self):
        v = VPINToxicityTracker(n_buckets=5, volume_avg_window_sec=60)
        ts = 1_700_000_000_000
        for i in range(200):
            v.add_trade({"timestamp": ts + i * 100, "amount": 1.0, "side": "buy"})
        assert len(v._buckets) > 0
        v.reset_session()
        assert len(v._buckets) == 0
        assert v._cur_buy == 0.0 and v._cur_sell == 0.0
