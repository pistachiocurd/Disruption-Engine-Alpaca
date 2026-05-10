"""Tests for calibration.py."""
from __future__ import annotations

import numpy as np
import pytest

from calibration import (
    CalibrationDriftMonitor,
    fit_alpha_calibration_c,
    fit_ood_distribution,
    identify_shock_events,
)
from layer1_sensors import StudentTHMM


def _make_ob(price: float, ts_ms: int, spread: float = 0.5) -> dict:
    return {
        "timestamp": ts_ms,
        "bids": [[price - spread / 2, 5.0]],
        "asks": [[price + spread / 2, 5.0]],
    }


class TestShockEventIdentification:
    def test_no_shock_below_vpin_threshold(self):
        n = 100
        ob = [_make_ob(100.0 + 0.1 * i, 1_000_000 + 100 * i) for i in range(n)]
        vpin = [0.1] * n
        events = identify_shock_events(ob, vpin)
        assert events == []

    def test_no_shock_below_price_move_threshold(self):
        n = 100
        # Price drifts very slowly, never crossing 0.3% threshold.
        ob = [_make_ob(100.0 + 0.001 * i, 1_000_000 + 100 * i) for i in range(n)]
        vpin = [0.95] * n  # high VPIN but no price move
        events = identify_shock_events(ob, vpin)
        assert events == []

    def test_clean_shock_is_detected(self):
        # 50 ticks flat at 100, then a 1% jump to 101, then flat at 101.
        ob = []
        for i in range(50):
            ob.append(_make_ob(100.0, i * 1000))
        for i in range(50, 100):
            ob.append(_make_ob(101.0, i * 1000))
        vpin = [0.0] * 49 + [0.95] + [0.0] * 50
        events = identify_shock_events(ob, vpin)
        assert len(events) == 1
        ev = events[0]
        assert ev.t_index == 49
        assert ev.price_move_pct > 0.003

    def test_min_spacing_dedup(self):
        # Two close shocks within MIN_SHOCK_SPACING_SECONDS of each other.
        ob = []
        for i in range(50):
            ob.append(_make_ob(100.0, i * 100))   # 100ms apart, total 5s
        for i in range(50, 100):
            ob.append(_make_ob(101.0, i * 100))
        # Two high-VPIN spikes 1s apart — should dedup to one event.
        vpin = [0.0] * 100
        vpin[49] = 0.95
        vpin[59] = 0.95
        events = identify_shock_events(ob, vpin)
        assert len(events) == 1


class TestAlphaCalibrationFit:
    def test_returns_none_when_too_few_events(self):
        from calibration import ShockEvent
        events = [
            ShockEvent(0, 0, 0.95, 100.0, 101.0, 0.01, T_actual=10)
            for _ in range(5)
        ]
        c = fit_alpha_calibration_c(events)
        assert c is None

    def test_recovers_known_c(self):
        from calibration import ShockEvent
        # Generate 50 synthetic events with T = 1 / (c_true * vpin) + small noise.
        rng = np.random.default_rng(0)
        c_true = 0.18
        events = []
        for i in range(50):
            vpin = float(rng.uniform(0.91, 0.99))
            T = 1.0 / (c_true * vpin) + rng.normal(0, 0.5)
            events.append(ShockEvent(
                t_index=i, timestamp_ms=i * 60_000,
                vpin_at_trigger=vpin, p_trigger=100.0, p_post_shock=101.0,
                price_move_pct=0.01, T_actual=int(max(1, T)),
            ))
        c_hat = fit_alpha_calibration_c(events)
        assert c_hat is not None
        assert c_hat == pytest.approx(c_true, rel=0.15)


class TestDriftMonitor:
    def test_no_drift_when_predictions_accurate(self):
        c = 0.18
        m = CalibrationDriftMonitor(current_c=c, threshold=0.35, window=10)
        for _ in range(15):
            vpin = 0.95
            T_obs = int(round(1.0 / (c * vpin)))
            m.record(vpin, T_obs)
        assert not m.is_stale

    def test_drift_triggered_when_c_invalid(self):
        c_old = 0.18
        c_new = 0.40   # 2x change → predictions massively off
        m = CalibrationDriftMonitor(current_c=c_old, threshold=0.35, window=10)
        for _ in range(15):
            vpin = 0.95
            T_obs = int(round(1.0 / (c_new * vpin)))
            m.record(vpin, T_obs)
        assert m.is_stale

    def test_reset_clears_state(self):
        m = CalibrationDriftMonitor(current_c=0.18, threshold=0.35, window=10)
        for _ in range(15):
            m.record(0.95, 50)  # bad predictions
        m.reset(0.20)
        assert not m.is_stale
        assert m.latest_mape == 0.0


class TestOODFit:
    def test_basic(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, size=(1000, 3))
        mu, sigma = fit_ood_distribution(data)
        assert mu.shape == (3,)
        assert sigma.shape == (3, 3)
        assert np.allclose(mu, 0.0, atol=0.15)
