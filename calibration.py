"""
calibration.py — Offline calibration jobs.

Three responsibilities:
    1. identify_shock_events()      — strict algorithmic definition of a shock.
    2. fit_alpha_calibration_c()    — exponentially weighted OLS over shock events.
    3. CalibrationDriftMonitor      — live MAPE-based drift detection.

Plus:
    4. fit_ood_distribution()       — μ, Σ for the OODDetector.
    5. fit_hmm_emissions()          — per-state Student-t emission parameters.

These are NOT real-time. Run as a scheduled job every CALIBRATION_INTERVAL_HOURS
(default 48). Outputs are written to disk and reloaded by the live engine.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    DRIFT_THRESHOLD,
    EPSILON,
    EQUILIBRIUM_BAND_PCT,
    EQUILIBRIUM_STABILITY_TICKS,
    EWLS_DECAY,
    HMM_DOF_LAMINAR,
    HMM_DOF_TRANSITION,
    HMM_DOF_TURBULENT,
    MAX_DIFFUSION_TICKS,
    MIN_CALIBRATION_EVENTS,
    MIN_SHOCK_SPACING_SECONDS,
    N_DRIFT_WINDOW,
    SHOCK_PRICE_MOVE_PCT,
    TURBULENCE_THRESHOLD,
)

log = logging.getLogger(__name__)


# ============================================================================
# Shock event identification
# ============================================================================
@dataclass
class ShockEvent:
    t_index: int                 # tick index in the input series
    timestamp_ms: int
    vpin_at_trigger: float
    p_trigger: float
    p_post_shock: float
    price_move_pct: float
    T_actual: int                # ticks from trigger to equilibrium


def _mid_price(ob: dict) -> float:
    return 0.5 * (ob["bids"][0][0] + ob["asks"][0][0])


def _find_equilibrium_tick(
    ob_snapshots: list[dict],
    start_idx: int,
    band_pct: float = EQUILIBRIUM_BAND_PCT,
    stability_ticks: int = EQUILIBRIUM_STABILITY_TICKS,
) -> Optional[int]:
    """
    First tick after `start_idx` where price stays within `band_pct` of the
    starting price for `stability_ticks` consecutive observations.
    """
    n = len(ob_snapshots)
    if start_idx >= n:
        return None
    p_ref = _mid_price(ob_snapshots[start_idx])
    consecutive = 0
    for i in range(start_idx, min(n, start_idx + MAX_DIFFUSION_TICKS)):
        p_i = _mid_price(ob_snapshots[i])
        if abs(p_i - p_ref) / p_ref <= band_pct:
            consecutive += 1
            if consecutive >= stability_ticks:
                return i - stability_ticks + 1
        else:
            consecutive = 0
            p_ref = p_i  # re-anchor
    return None


def identify_shock_events(
    ob_snapshots: list[dict],
    vpin_series: list[float],
) -> list[ShockEvent]:
    """
    Returns the list of shock events satisfying ALL three conditions from the spec.

    Args:
        ob_snapshots: aligned list of L2 snapshots (each with 'timestamp', 'bids', 'asks').
        vpin_series: VPIN value at each snapshot (same length as ob_snapshots).

    Conditions for a valid shock:
        1. vpin_at_t > TURBULENCE_THRESHOLD
        2. |price_move| / price > SHOCK_PRICE_MOVE_PCT within MAX_DIFFUSION_TICKS
        3. > MIN_SHOCK_SPACING_SECONDS since the previous accepted event
    """
    if len(ob_snapshots) != len(vpin_series):
        raise ValueError("ob_snapshots and vpin_series must be the same length.")

    events: list[ShockEvent] = []
    last_event_ts_ms = -1e18
    n = len(ob_snapshots)

    for t in range(n):
        vpin = vpin_series[t]
        if vpin < TURBULENCE_THRESHOLD:
            continue
        ts = int(ob_snapshots[t]["timestamp"])
        if (ts - last_event_ts_ms) < MIN_SHOCK_SPACING_SECONDS * 1000:
            continue

        p_trigger = _mid_price(ob_snapshots[t])
        moved = False
        for dt in range(1, min(MAX_DIFFUSION_TICKS, n - t)):
            p_later = _mid_price(ob_snapshots[t + dt])
            move_pct = abs(p_later - p_trigger) / p_trigger
            if move_pct > SHOCK_PRICE_MOVE_PCT:
                eq_tick = _find_equilibrium_tick(ob_snapshots, t + dt)
                if eq_tick is None:
                    break
                T_actual = eq_tick - t
                events.append(ShockEvent(
                    t_index=t,
                    timestamp_ms=ts,
                    vpin_at_trigger=vpin,
                    p_trigger=p_trigger,
                    p_post_shock=_mid_price(ob_snapshots[eq_tick]),
                    price_move_pct=move_pct,
                    T_actual=T_actual,
                ))
                last_event_ts_ms = ts
                moved = True
                break
        if not moved:
            continue

    return events


# ============================================================================
# Exponentially Weighted OLS for ALPHA_CALIBRATION_C
# ============================================================================
def fit_alpha_calibration_c(
    events: list[ShockEvent],
    decay: float = EWLS_DECAY,
) -> Optional[float]:
    """
    Fit  T_actual = 1 / (c * vpin)  via exponentially weighted OLS.

    Equivalent to fitting  1 / T_actual = c * vpin  with WLS where weights
    decay geometrically toward older events.

    Returns None if fewer than MIN_CALIBRATION_EVENTS events are available.
    """
    if len(events) < MIN_CALIBRATION_EVENTS:
        log.warning(
            f"insufficient calibration events: {len(events)} < {MIN_CALIBRATION_EVENTS}"
        )
        return None

    # Sort oldest → newest for EWLS weighting.
    events = sorted(events, key=lambda e: e.timestamp_ms)
    n = len(events)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])

    x = np.array([e.vpin_at_trigger for e in events])
    y = np.array([1.0 / max(EPSILON, e.T_actual) for e in events])

    # WLS estimator for slope-only model y = c*x:
    # c = Σ w_i x_i y_i / Σ w_i x_i^2
    num = np.sum(weights * x * y)
    den = np.sum(weights * x * x)
    if den < EPSILON:
        return None
    c = float(num / den)
    return c


# ============================================================================
# Drift monitor
# ============================================================================
class CalibrationDriftMonitor:
    """
    Maintains a rolling window of completed live shocks. After each completed
    shock, computes MAPE of (predicted T_actual = 1/(c*vpin)) vs (observed
    T_actual). When MAPE crosses DRIFT_THRESHOLD, sets `is_stale = True`,
    signaling the engine to suspend mandate generation until recalibration.
    """

    def __init__(
        self,
        current_c: float,
        threshold: float = DRIFT_THRESHOLD,
        window: int = N_DRIFT_WINDOW,
    ) -> None:
        self.current_c = current_c
        self.threshold = threshold
        self.window = window
        self._buf: deque = deque(maxlen=window)
        self.is_stale = False
        self.latest_mape: float = 0.0

    def record(self, vpin_at_trigger: float, observed_T_actual: int) -> None:
        if vpin_at_trigger <= 0 or observed_T_actual <= 0:
            return
        self._buf.append((vpin_at_trigger, observed_T_actual))
        if len(self._buf) >= self.window:
            errs = []
            for vpin, T_obs in self._buf:
                T_pred = 1.0 / max(EPSILON, self.current_c * vpin)
                errs.append(abs(T_obs - T_pred) / max(EPSILON, T_obs))
            self.latest_mape = float(np.mean(errs))
            self.is_stale = self.latest_mape > self.threshold
            if self.is_stale:
                log.warning(
                    f"calibration drift detected: MAPE={self.latest_mape:.3f} "
                    f"> threshold={self.threshold}"
                )

    def reset(self, new_c: float) -> None:
        self.current_c = new_c
        self._buf.clear()
        self.is_stale = False
        self.latest_mape = 0.0


# ============================================================================
# OOD distribution fit
# ============================================================================
def fit_ood_distribution(
    feature_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute μ and Σ for [ce_ratio, obi, liquidation_rate] over the calibration window.

    Args:
        feature_history: shape (T, 3).

    Returns:
        (mu, sigma).
    """
    if feature_history.ndim != 2 or feature_history.shape[1] != 3:
        raise ValueError("feature_history must be shape (T, 3).")
    mu = feature_history.mean(axis=0)
    sigma = np.cov(feature_history, rowvar=False)
    return mu, sigma


# ============================================================================
# HMM emission parameter fit (per-state, given Viterbi-labeled training data)
# ============================================================================
def fit_hmm_emissions(
    feature_history: np.ndarray,
    state_labels: np.ndarray,
) -> list:
    """
    Fits per-state Student-t emission parameters.

    Per the spec: ν is FIT OFFLINE per state and frozen. The per-state ν values
    are taken from config (HMM_DOF_LAMINAR/TRANSITION/TURBULENT), which were
    themselves established by an offline ν-fit procedure (e.g., method-of-moments
    on tail kurtosis). We fit μ and σ here via robust estimators (median, MAD).

    Args:
        feature_history: shape (T, n_features).
        state_labels:    shape (T,) integer in {0,1,2}.
    """
    from layer1_sensors import HMMStateParams

    nu_per_state = [HMM_DOF_LAMINAR, HMM_DOF_TRANSITION, HMM_DOF_TURBULENT]
    out = []
    for s in range(3):
        mask = state_labels == s
        if mask.sum() < 10:
            log.warning(f"HMM state {s} has too few samples ({mask.sum()}); using priors.")
            mu = np.zeros(feature_history.shape[1])
            sigma = np.ones(feature_history.shape[1])
        else:
            block = feature_history[mask]
            mu = np.median(block, axis=0)
            mad = np.median(np.abs(block - mu), axis=0)
            sigma = np.maximum(1.4826 * mad, 1e-4)  # robust σ from MAD
        out.append(HMMStateParams(mu=mu, sigma=sigma, nu=nu_per_state[s]))
    return out


# ============================================================================
# Persistence
# ============================================================================
def save_calibration(
    path: Path,
    alpha_c: float,
    ood_mu: np.ndarray,
    ood_sigma: np.ndarray,
) -> None:
    payload = {
        "alpha_calibration_c": alpha_c,
        "ood_mu": ood_mu.tolist(),
        "ood_sigma": ood_sigma.tolist(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> dict:
    return json.loads(Path(path).read_text())
