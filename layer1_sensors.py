"""
layer1_sensors.py — Sensor Array.

Translates raw L2 / trade / liquidation streams into physical market parameters.
No trading logic lives here. The output is a single physics_state dict consumed
by Layer 2.

Components:
    VPINToxicityTracker      — volume-bucketed VPIN (informed flow)
    StudentTKalmanFilter     — robust Kalman over [spread, spread_velocity]
    StudentTHMM              — 3-state HMM, Student-t emissions, online inference
    LiquidationCascadeTracker — Binance forceOrder stream → normalized rate
    OODDetector              — Mahalanobis gate over the TCN input vector
    CEInferenceEngine        — cancellation-from-snapshot-deltas
    SensorArray              — orchestrator
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import (
    CE_RATIO_MAX,
    CE_ROLLING_WINDOW_SEC,
    EPSILON,
    HMM_DOF_LAMINAR,
    HMM_DOF_TRANSITION,
    HMM_DOF_TURBULENT,
    IS_EQUITY,
    KALMAN_DOF,
    KYLES_LAMBDA_LOOKBACK_TICKS,
    LIQUIDATION_WINDOW_SEC,
    MIN_ORDER_SIZE,
    MLOFI_DEPTH_LEVELS,
    OOD_THRESHOLD,
    SENSOR_LOOP_INTERVAL_SEC,
    SYMBOL,
    TICK_SIZE,
    VPIN_BUCKETS,
    VPIN_VOLUME_AVG_WINDOW_SEC,
)

log = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================
def student_t_logpdf(x: float, mu: float, sigma: float, nu: float) -> float:
    """Log-pdf of univariate Student-t. Numerically stable."""
    if sigma <= 0.0 or not np.isfinite(sigma):
        return -np.inf
    z = (x - mu) / sigma
    coef = math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
    coef -= 0.5 * math.log(nu * math.pi) + math.log(sigma)
    return coef - 0.5 * (nu + 1.0) * math.log1p((z * z) / nu)


# ============================================================================
# 1.1 VPIN
# ============================================================================
class VPINToxicityTracker:
    """
    Volume-bucketed VPIN adapted for 24/7 crypto markets.

    Bucket size = V_avg_1h / N_buckets, recomputed continuously. Each trade is
    classified by its `side` field (no Lee-Ready). VPIN = |V_buy - V_sell| / V_total
    over the trailing N_buckets buckets.
    """

    def __init__(
        self,
        n_buckets: int = VPIN_BUCKETS,
        volume_avg_window_sec: float = VPIN_VOLUME_AVG_WINDOW_SEC,
    ) -> None:
        self.n_buckets = n_buckets
        self.volume_avg_window_sec = volume_avg_window_sec

        self._volume_history: deque = deque()  # (ts_ms, qty)
        self._buckets: deque = deque(maxlen=n_buckets)  # each: (V_buy, V_sell)

        self._cur_buy = 0.0
        self._cur_sell = 0.0
        self._cur_total = 0.0
        self._bucket_size = 1.0  # initial; updated on first trade

        self._latest_trade_ts_ms: int = 0

    def add_trade_classified(
        self,
        trade: dict,
        mid_at_trade: float,
        last_trade_price: Optional[float],
    ) -> None:
        """Lee-Ready classification for venues that don't report trade side.

        Equity prints (NASDAQ/NYSE/IEX) carry no buyer/seller initiator field,
        so we infer it from the trade price relative to the prevailing
        midpoint, with the tick test as tiebreaker.
        """
        price = float(trade["price"])
        if mid_at_trade is None or not math.isfinite(mid_at_trade) or mid_at_trade <= 0.0:
            side = "buy"  # degenerate; will be averaged out by VPIN bucketing
        elif price > mid_at_trade:
            side = "buy"
        elif price < mid_at_trade:
            side = "sell"
        elif last_trade_price is not None and price != last_trade_price:
            side = "buy" if price > last_trade_price else "sell"
        else:
            side = "buy"
        self.add_trade({**trade, "side": side})

    def reset_session(self) -> None:
        """Drop all bucketed state. Called at each market open for equities."""
        self._volume_history.clear()
        self._buckets.clear()
        self._cur_buy = 0.0
        self._cur_sell = 0.0
        self._cur_total = 0.0
        self._bucket_size = 1.0
        self._latest_trade_ts_ms = 0

    def add_trade(self, trade: dict) -> None:
        ts = int(trade["timestamp"])
        qty = float(trade["amount"])
        side = trade["side"]
        self._latest_trade_ts_ms = ts
        self._volume_history.append((ts, qty))

        # Drop trades older than the rolling window.
        cutoff = ts - int(self.volume_avg_window_sec * 1000)
        while self._volume_history and self._volume_history[0][0] < cutoff:
            self._volume_history.popleft()

        v_total_window = sum(q for _, q in self._volume_history)
        if v_total_window > 0:
            self._bucket_size = max(EPSILON, v_total_window / self.n_buckets)

        if side == "buy":
            self._cur_buy += qty
        else:
            self._cur_sell += qty
        self._cur_total += qty

        while self._cur_total >= self._bucket_size:
            # Close out a bucket. Allocate proportionally if the trade overshoots.
            overflow = self._cur_total - self._bucket_size
            ratio = (self._bucket_size / self._cur_total) if self._cur_total > 0 else 0.0
            buy_close = self._cur_buy * ratio
            sell_close = self._cur_sell * ratio
            self._buckets.append((buy_close, sell_close))
            # Carry remainder.
            self._cur_buy -= buy_close
            self._cur_sell -= sell_close
            self._cur_total = overflow

    @property
    def vpin(self) -> float:
        if len(self._buckets) < 2:
            return 0.0
        v_buy = sum(b[0] for b in self._buckets)
        v_sell = sum(b[1] for b in self._buckets)
        v_total = v_buy + v_sell
        if v_total < EPSILON:
            return 0.0
        return abs(v_buy - v_sell) / v_total

    @property
    def average_volume_10s(self) -> float:
        """Used by LiquidationCascadeTracker for normalization."""
        if not self._volume_history:
            return 0.0
        now = self._latest_trade_ts_ms
        cutoff = now - 10_000
        recent = [q for ts, q in self._volume_history if ts >= cutoff]
        return sum(recent)


# ============================================================================
# 1.2 Student-t Kalman
# ============================================================================
class StudentTKalmanFilter:
    """
    Robust Kalman filter with Student-t observation noise.

    State: x = [spread, spread_velocity]^T
    Transition (constant velocity, dt=1 step): F = [[1,1],[0,1]]
    Observation (spread only):                  H = [1, 0]

    Update step uses the IRLS-style re-weighting that arises from a Student-t
    noise prior: weight w = (ν + 1) / (ν + d²) where d² is the squared
    Mahalanobis distance of the innovation. This downweights outlier spreads
    rather than smoothing them.

    `KALMAN_DOF` is fitted offline and FROZEN at runtime — see config.py.
    """

    def __init__(
        self,
        nu: float = KALMAN_DOF,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
    ) -> None:
        self.nu = nu
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.eye(2, dtype=np.float64) * process_noise
        self.R = np.array([[measurement_noise]], dtype=np.float64)
        self._initialized = False

    def update(self, observed_spread: float) -> None:
        if not self._initialized:
            self.x[0] = observed_spread
            self.x[1] = 0.0
            self._initialized = True
            return

        # Predict (without Q first, so we can compute baseline innovation)
        x_pred = self.F @ self.x
        P_no_Q = self.F @ self.P @ self.F.T

        # Baseline innovation
        y = observed_spread - (self.H @ x_pred)[0]
        S_baseline = (self.H @ (P_no_Q + self.Q) @ self.H.T + self.R)[0, 0]
        if S_baseline <= 0.0 or not np.isfinite(S_baseline):
            return

        # Heavy-tailed Student-t prior on STATE EVOLUTION (not observation noise).
        # Under a Student-t state prior, large innovations are more consistent
        # with genuine state jumps than with observation noise — so we INFLATE
        # process noise for this step rather than downweight the observation.
        # This is the spec's intent: respond to spread spikes, do not smooth them.
        d2 = (y * y) / S_baseline
        q_factor = (self.nu + d2) / (self.nu + 1.0)   # >= 1, large for outliers
        Q_eff = self.Q * q_factor

        P_pred = P_no_Q + Q_eff
        S = (self.H @ P_pred @ self.H.T + self.R)[0, 0]
        if S <= 0.0 or not np.isfinite(S):
            return
        K = (P_pred @ self.H.T).flatten() / S
        self.x = x_pred + K * y
        self.P = (np.eye(2) - np.outer(K, self.H[0])) @ P_pred

    def reset_session(self) -> None:
        """Re-initialize the filter. Called at each market open for equities so
        the overnight gap doesn't get fed in as a 1-tick spread innovation.
        """
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self._initialized = False

    @property
    def viscosity(self) -> float:
        """
        Viscosity is computed from the filtered spread state. Higher spread →
        higher friction. Spread velocity contributes via a small bonus that
        captures rapid widening as additional cost.
        """
        spread = max(0.0, float(self.x[0]))
        vel = float(self.x[1])
        return spread * (1.0 + 0.5 * abs(vel))


# ============================================================================
# 1.3 Student-t HMM (online forward inference only)
# ============================================================================
@dataclass
class HMMStateParams:
    """Per-state Student-t emission parameters for each of the 4 features."""
    mu: np.ndarray      # shape (n_features,)
    sigma: np.ndarray   # shape (n_features,)
    nu: float           # scalar; one ν per state per spec


class StudentTHMM:
    """
    3-state HMM with Student-t emissions. ONLINE INFERENCE ONLY.

    The transition matrix `A`, emission means/sigmas, and per-state ν are all
    fitted offline (see calibration.py). This class consumes those parameters
    and runs the forward algorithm online, producing the most-likely current
    regime at each step.

    States: 0 = Laminar, 1 = Transition, 2 = Turbulent.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        state_params: list[HMMStateParams],
        initial_distribution: Optional[np.ndarray] = None,
    ) -> None:
        assert transition_matrix.shape == (3, 3)
        assert len(state_params) == 3
        self.A = transition_matrix
        self.params = state_params
        if initial_distribution is None:
            initial_distribution = np.array([0.7, 0.2, 0.1])
        self.alpha = initial_distribution.copy()  # forward variable, normalized

    @classmethod
    def from_default_priors(cls) -> "StudentTHMM":
        """
        Build an HMM with sensible priors for cold-start before calibration.
        These should be REPLACED by a fitted set on first calibration run.
        """
        A = np.array([
            [0.97, 0.025, 0.005],
            [0.10, 0.85, 0.05],
            [0.05, 0.15, 0.80],
        ])
        # 4 emission features: [ce_ratio, obi, spread_d1, liquidation_rate]
        params = [
            HMMStateParams(
                mu=np.array([1.0, 0.0, 0.0, 0.0]),
                sigma=np.array([0.5, 0.2, 0.5, 0.05]),
                nu=HMM_DOF_LAMINAR,
            ),
            HMMStateParams(
                mu=np.array([3.0, 0.2, 1.0, 0.2]),
                sigma=np.array([1.5, 0.3, 1.0, 0.3]),
                nu=HMM_DOF_TRANSITION,
            ),
            HMMStateParams(
                mu=np.array([8.0, 0.5, 3.0, 1.0]),
                sigma=np.array([3.0, 0.4, 2.0, 1.0]),
                nu=HMM_DOF_TURBULENT,
            ),
        ]
        return cls(A, params)

    def reset_session(self) -> None:
        """Reset the forward variable to a uniform prior. Called at each
        market open for equities so yesterday's regime distribution doesn't
        bias the first morning observations.
        """
        n = self.A.shape[0]
        self.alpha = np.full(n, 1.0 / n, dtype=np.float64)

    def update(self, observation: np.ndarray) -> int:
        """
        Forward step. Returns the most-likely current state.
        observation shape: (n_features,)
        """
        # Predict: alpha_pred = A^T @ alpha
        alpha_pred = self.A.T @ self.alpha

        # Emission likelihood per state (in log-space, then exponentiate carefully)
        log_lik = np.empty(3)
        for s, p in enumerate(self.params):
            ll = 0.0
            for i in range(len(observation)):
                ll += student_t_logpdf(observation[i], p.mu[i], p.sigma[i], p.nu)
            log_lik[s] = ll

        # Normalize in log-space for stability, then convert.
        m = log_lik.max()
        lik = np.exp(log_lik - m)

        new_alpha = alpha_pred * lik
        s = new_alpha.sum()
        if s < EPSILON or not np.isfinite(s):
            # Pathological observation — keep the predicted distribution.
            self.alpha = alpha_pred / (alpha_pred.sum() + EPSILON)
        else:
            self.alpha = new_alpha / s
        return int(np.argmax(self.alpha))


# ============================================================================
# 1.4 Liquidation Cascade Tracker
# ============================================================================
class LiquidationCascadeTracker:
    """
    Subscribes to Binance's public !forceOrder stream:
      wss://fstream.binance.com/ws/!forceOrder@arr

    On non-Binance exchanges, instantiate with `enabled=False` and the tracker
    permanently emits liquidation_rate = 0.0.
    """

    BINANCE_LIQUIDATION_WS = "wss://fstream.binance.com/ws/!forceOrder@arr"

    def __init__(
        self,
        symbol: str = SYMBOL,
        enabled: bool = True,
        window_sec: float = LIQUIDATION_WINDOW_SEC,
    ) -> None:
        self.enabled = enabled
        self.window_sec = window_sec
        # Symbol on Binance perp is e.g. "BTCUSDT" (no slash, no settle suffix).
        # Equity symbols have no slash — derive a harmless filter that the
        # liquidation feed will never match (the tracker is disabled in that
        # case anyway, but instantiation must not crash).
        if "/" in symbol:
            self.symbol_filter = symbol.split("/")[0] + symbol.split("/")[1].split(":")[0]
        else:
            self.symbol_filter = symbol
        self._events: deque = deque()  # (ts_ms, qty)
        self.running = False

    def _record(self, ts_ms: int, qty: float) -> None:
        self._events.append((ts_ms, qty))
        cutoff = ts_ms - int(self.window_sec * 1000)
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def record(self, ts_ms: int, qty: float) -> None:
        """Public injection hook for adapters that source liquidations from
        a non-Binance feed (e.g. parsed out of HL trade events during
        harvest replay). The Binance WS path uses _record directly; this
        wraps it so external callers don't reach into a private method."""
        if not self.enabled:
            return
        self._record(ts_ms, qty)

    def liquidation_rate(self, normalizer_volume: float) -> float:
        """
        Total liquidated qty in the window divided by `normalizer_volume`
        (typically VPIN's average_volume_10s). Returns 0.0 if the tracker is
        disabled or normalizer is too small.
        """
        if not self.enabled or normalizer_volume < EPSILON:
            return 0.0
        total = sum(q for _, q in self._events)
        return total / normalizer_volume

    async def run(self) -> None:
        """Outer reconnection loop. Subscribes to the raw Binance WS feed.

        For non-Binance venues (e.g. Hyperliquid), the tracker is enabled
        but events come via `record()` from a venue-specific adapter
        rather than from this WS loop. We early-return so we don't try
        to connect to the Binance URL on a non-Binance run.
        """
        if not self.enabled:
            log.info("LiquidationCascadeTracker disabled.")
            return
        from config import EXCHANGE_ID
        if EXCHANGE_ID != "binance":
            log.info(
                "LiquidationCascadeTracker enabled but no live WS source for "
                "EXCHANGE_ID=%s; events arrive via record() from a venue adapter "
                "(e.g. HL harvester).", EXCHANGE_ID,
            )
            return
        try:
            import websockets  # local import — only needed for Binance live
        except ImportError:
            log.error("websockets package not installed — liquidation tracker disabled.")
            self.enabled = False
            return

        self.running = True
        while self.running:
            try:
                async with websockets.connect(self.BINANCE_LIQUIDATION_WS) as ws:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            order = msg.get("o", {})
                            if order.get("s") != self.symbol_filter:
                                continue
                            ts = int(order.get("T", time.time() * 1000))
                            qty = float(order.get("q", 0.0))
                            self._record(ts, qty)
                        except Exception as parse_err:
                            log.warning(f"liquidation parse error: {parse_err}")
            except Exception as e:
                log.warning(f"Liquidation stream interrupted: {e}. Reconnecting in 2s.")
                await asyncio.sleep(2)


# ============================================================================
# 1.5 OOD Detector
# ============================================================================
class OODDetector:
    """
    Mahalanobis-distance gate over the 3-feature TCN input vector
    [ce_ratio, obi, liquidation_rate].

    μ and Σ are computed offline from the calibration dataset and FROZEN at
    runtime. Updating during a live shock would disable the guard precisely
    when it is needed most.
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        threshold: float = OOD_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        if mu is None:
            mu = np.zeros(3)
        if sigma is None:
            sigma = np.eye(3)
        self.mu = mu.astype(np.float64)
        self._set_inverse(sigma)

    def _set_inverse(self, sigma: np.ndarray) -> None:
        sigma = sigma + np.eye(sigma.shape[0]) * 1e-6  # numerical jitter
        self.sigma = sigma.astype(np.float64)
        self.sigma_inv = np.linalg.inv(self.sigma)

    def load(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.mu = mu.astype(np.float64)
        self._set_inverse(sigma)

    def evaluate(self, observation: np.ndarray) -> tuple[bool, float]:
        delta = observation - self.mu
        d2 = float(delta @ self.sigma_inv @ delta)
        d = math.sqrt(max(0.0, d2))
        return d > self.threshold, d


# ============================================================================
# C/E Ratio inference
# ============================================================================
class CEInferenceEngine:
    """
    Infers cancellations from consecutive book snapshots.

    For each price level present in the previous snapshot but reduced in the
    current snapshot, any depth reduction NOT explained by trades at that
    price is counted as a cancellation. Symmetric across bid and ask.
    """

    def __init__(self, window_sec: float = CE_ROLLING_WINDOW_SEC) -> None:
        self.window_sec = window_sec
        self._cancellations: deque = deque()  # (ts_ms, qty)
        self._executions: deque = deque()     # (ts_ms, qty)

    @staticmethod
    def _book_to_dict(side_levels: list) -> dict[float, float]:
        # Round price to TICK_SIZE to avoid floating-point key drift.
        out: dict[float, float] = {}
        for p, sz in side_levels:
            key = round(round(p / TICK_SIZE) * TICK_SIZE, 8)
            out[key] = out.get(key, 0.0) + float(sz)
        return out

    def _level_match_tolerance(self, price: float) -> float:
        return TICK_SIZE * 0.5

    def update(
        self,
        prev_ob: dict,
        curr_ob: dict,
        trades_in_window: list[dict],
        ts_ms: int,
    ) -> None:
        prev_bids = self._book_to_dict(prev_ob.get("bids", []))
        curr_bids = self._book_to_dict(curr_ob.get("bids", []))
        prev_asks = self._book_to_dict(prev_ob.get("asks", []))
        curr_asks = self._book_to_dict(curr_ob.get("asks", []))

        cancels = 0.0
        execs = 0.0

        # Bid side: 'sell'-aggressor trades consume bids.
        for price, prev_depth in prev_bids.items():
            curr_depth = curr_bids.get(price, 0.0)
            depth_red = prev_depth - curr_depth
            if depth_red <= 0:
                continue
            fills = sum(
                t["amount"] for t in trades_in_window
                if t["side"] == "sell"
                and abs(t["price"] - price) < self._level_match_tolerance(price)
            )
            execs += min(depth_red, fills)
            cancels += max(0.0, depth_red - fills)

        # Ask side: 'buy'-aggressor trades consume asks.
        for price, prev_depth in prev_asks.items():
            curr_depth = curr_asks.get(price, 0.0)
            depth_red = prev_depth - curr_depth
            if depth_red <= 0:
                continue
            fills = sum(
                t["amount"] for t in trades_in_window
                if t["side"] == "buy"
                and abs(t["price"] - price) < self._level_match_tolerance(price)
            )
            execs += min(depth_red, fills)
            cancels += max(0.0, depth_red - fills)

        if cancels > 0:
            self._cancellations.append((ts_ms, cancels))
        if execs > 0:
            self._executions.append((ts_ms, execs))

        cutoff = ts_ms - int(self.window_sec * 1000)
        while self._cancellations and self._cancellations[0][0] < cutoff:
            self._cancellations.popleft()
        while self._executions and self._executions[0][0] < cutoff:
            self._executions.popleft()

    @property
    def ce_ratio(self) -> float:
        c = sum(q for _, q in self._cancellations)
        e = sum(q for _, q in self._executions)
        ratio = c / (e + EPSILON)
        return min(ratio, CE_RATIO_MAX)

    def reset_session(self) -> None:
        self._cancellations.clear()
        self._executions.clear()


# ============================================================================
# C/E Ratio — L1-only proxy for equities (no L2 depth available)
# ============================================================================
class QuoteCancellationProxy:
    """L1-compatible replacement for CEInferenceEngine on equities.

    Without full-depth L2 we cannot diff levels. Instead we count NBBO
    quote updates and trades over a rolling window. Quote updates beyond
    those matched 1:1 with trades are treated as inferred cancellations,
    matching the semantic of the L2 C/E ratio (cancellation pressure
    relative to execution flow).

    Exposes the same `.ce_ratio` property as `CEInferenceEngine` so the
    rest of the pipeline (HMM, OOD, FeatureDumper) is unchanged.
    """

    def __init__(self, window_sec: float = CE_ROLLING_WINDOW_SEC) -> None:
        self.window_sec = window_sec
        self._quote_updates: deque = deque()  # ts_ms
        self._trades: deque = deque()         # ts_ms
        self._ce_ratio = 0.0
        self._latest_ts_ms: int = 0

    def _prune(self, now_ms: int) -> None:
        cutoff = now_ms - int(self.window_sec * 1000)
        while self._quote_updates and self._quote_updates[0] < cutoff:
            self._quote_updates.popleft()
        while self._trades and self._trades[0] < cutoff:
            self._trades.popleft()

    def _recompute(self) -> None:
        n_q = len(self._quote_updates)
        n_t = len(self._trades)
        # Match each trade with a co-incident quote update (executions); the
        # remainder are inferred cancellations. Bound by CE_RATIO_MAX to match
        # the historical clamp from the L2 engine.
        execs = min(n_q, n_t)
        cancels = max(0, n_q - execs)
        self._ce_ratio = min(cancels / (execs + EPSILON), CE_RATIO_MAX)

    def on_quote(self, ts_ms: int) -> None:
        self._latest_ts_ms = ts_ms
        self._quote_updates.append(ts_ms)
        self._prune(ts_ms)
        self._recompute()

    def on_trade(self, ts_ms: int) -> None:
        self._latest_ts_ms = ts_ms
        self._trades.append(ts_ms)
        self._prune(ts_ms)
        self._recompute()

    @property
    def ce_ratio(self) -> float:
        return self._ce_ratio

    def reset_session(self) -> None:
        self._quote_updates.clear()
        self._trades.clear()
        self._ce_ratio = 0.0
        self._latest_ts_ms = 0


# ============================================================================
# 1.6 Crypto-native microstructure features (Path D / P3.6, 2026-05-10)
# ============================================================================
# Added to replace the dead liquidation_rate channel on Hyperliquid and
# expand the TCN input from 3 to 5 features. See LAYER2_TRAINING.md
# §13.4 and TODO.md P3.6 for motivation.
#
# Equity path doesn't use these (no L2 depth from Alpaca, no per-trade
# initiator field) — they stay at default 0.0 in PhysicsState and the
# equity TCN_INPUT_CHANNELS=3 model doesn't read them.


class MLOFITracker:
    """Multi-Level Order Flow Imbalance (Xu et al. 2018) across top-K levels.

    For each book snapshot, computes per-level (Δbid_size - Δask_size)
    summed over the top-K depth levels and normalized by total top-K depth.
    Captures bid/ask flow pressure beyond top-of-book OBI; less susceptible
    to top-of-book spoofing because deeper levels are harder to fake.

    Pure function of the current and previous snapshot — no internal
    rolling window; smoothing is handled (if needed) by downstream
    consumers via deque/EMA in the TCN's own receptive field.

    Returns a scalar in roughly [-1, 1] for typical book depth.
    """

    def __init__(self, depth_levels: int = MLOFI_DEPTH_LEVELS) -> None:
        self.depth_levels = depth_levels

    def compute(self, prev_ob: Optional[dict], curr_ob: dict) -> float:
        if prev_ob is None or not curr_ob.get("bids") or not curr_ob.get("asks"):
            return 0.0
        K = self.depth_levels
        prev_bids = prev_ob.get("bids", [])[:K]
        curr_bids = curr_ob.get("bids", [])[:K]
        prev_asks = prev_ob.get("asks", [])[:K]
        curr_asks = curr_ob.get("asks", [])[:K]

        ofi = 0.0
        for i in range(K):
            pb = float(prev_bids[i][1]) if i < len(prev_bids) else 0.0
            cb = float(curr_bids[i][1]) if i < len(curr_bids) else 0.0
            pa = float(prev_asks[i][1]) if i < len(prev_asks) else 0.0
            ca = float(curr_asks[i][1]) if i < len(curr_asks) else 0.0
            ofi += (cb - pb) - (ca - pa)

        total_depth = (
            sum(float(b[1]) for b in curr_bids)
            + sum(float(a[1]) for a in curr_asks)
        )
        if total_depth < EPSILON:
            return 0.0
        return ofi / total_depth

    def reset_session(self) -> None:
        pass  # stateless across the prev/curr pair


class VAMPDeviationComputer:
    """Volume-Adjusted Mid Price — queue-position-weighted mid (Stoikov 2018).

    VAMP = (ask_sz * bid_px + bid_sz * ask_px) / (bid_sz + ask_sz).
    Pulls the "true" mid toward whichever side has more queue weight: a
    heavy ask queue → VAMP closer to bid (selling pressure expected).

    Returns the *deviation* of VAMP from the arithmetic mid in basis
    points (signed). Stateless; computed per snapshot.
    """

    @staticmethod
    def compute(ob: dict, mid: float) -> float:
        if not ob.get("bids") or not ob.get("asks") or mid <= 0:
            return 0.0
        bid_px = float(ob["bids"][0][0])
        bid_sz = float(ob["bids"][0][1])
        ask_px = float(ob["asks"][0][0])
        ask_sz = float(ob["asks"][0][1])
        denom = bid_sz + ask_sz
        if denom < EPSILON:
            return 0.0
        vamp = (ask_sz * bid_px + bid_sz * ask_px) / denom
        return (vamp - mid) / mid * 10_000.0  # signed basis points


class KylesLambdaTracker:
    """Kyle's λ — rolling regression of mid-return on signed trade volume.

    Maintains deques of (per-tick) signed volume and mid prices. Trade
    flow is accumulated between book snapshots via on_trade(); each
    book snapshot commits the accumulated flow and recomputes λ from
    the trailing lookback window using OLS:

        λ ≈ cov(signed_vol_norm, returns) / var(signed_vol_norm)

    Signed volume is normalized by recent average |flow| so λ is
    dimensionless and roughly scale-invariant across coins. Typical
    values: [0, 0.01] in calm markets, spiking during liquidity crises.

    Higher λ = more price-impact-per-unit-flow = thinner book / less
    resilient liquidity.
    """

    def __init__(self, lookback_ticks: int = KYLES_LAMBDA_LOOKBACK_TICKS) -> None:
        self.lookback_ticks = lookback_ticks
        self._signed_vol_buffer: deque = deque(maxlen=lookback_ticks)
        self._mid_buffer: deque = deque(maxlen=lookback_ticks)
        self._signed_vol_accum: float = 0.0
        self._vol_window: deque = deque(maxlen=lookback_ticks)

    def on_trade(self, side: str, qty: float) -> None:
        """Accumulate signed volume between book snapshots.
        side='buy' (taker bought ask) → +1; side='sell' → -1.
        """
        sign = 1.0 if side == "buy" else -1.0
        self._signed_vol_accum += sign * qty

    def on_book(self, mid: float) -> float:
        """Commit accumulated flow as the current tick's signed volume
        and recompute λ from the buffer. Returns scalar λ.
        """
        signed_vol = self._signed_vol_accum
        self._signed_vol_accum = 0.0
        self._signed_vol_buffer.append(signed_vol)
        self._mid_buffer.append(mid)
        self._vol_window.append(abs(signed_vol))

        n = len(self._mid_buffer)
        if n < 5:
            return 0.0

        # Normalize signed volume by recent average |flow| → dimensionless.
        vol_avg = sum(self._vol_window) / max(len(self._vol_window), 1)
        if vol_avg < EPSILON:
            return 0.0

        mids = np.asarray(self._mid_buffer, dtype=np.float64)
        rets = np.diff(mids) / np.maximum(mids[:-1], EPSILON)  # length n-1
        sv = np.asarray(self._signed_vol_buffer, dtype=np.float64)[1:] / vol_avg

        sv_var = float(np.var(sv))
        if sv_var < EPSILON:
            return 0.0
        cov = float(np.cov(sv, rets, bias=True)[0, 1])
        return cov / sv_var

    def reset_session(self) -> None:
        self._signed_vol_buffer.clear()
        self._mid_buffer.clear()
        self._vol_window.clear()
        self._signed_vol_accum = 0.0


# ============================================================================
# 1.11 Path G — Characteristic scales and dimensionless π-groups
# ============================================================================
# Phase 2+3 of `mini_projects/path_G_dimensionless/RESEARCH_PLAN.md` on
# branch `research/path-g-dimensionless`. THEORY.md derives the π-groups;
# BOUNDARY_CONDITIONS.md mandates the floor/saturation discipline below.
# Not yet consumed by the TCN — TCN_INPUT_CHANNELS stays at 5 until a
# harvest with these columns and a successful Phase 4 retraining lands.

# Per-π tanh saturation scales — calibrated from the raw-ratio
# 95th-percentile divided by 2, so typical values land in tanh's linear
# range and saturation kicks in around p95. Pooled from 8.67M rows of
# multi-coin harvest (BTC/ETH/SOL/HYPE × 14 days, HL 2026-03-24→04-06);
# see LAYER2_TRAINING.md §13.13. `pi_kappa`'s 208× shift from the
# BTC-only smoke calibration came from ETH/SOL/HYPE's thinner books
# (larger κ) — depth heterogeneity is what made the pooled calibration
# essential. Re-run calibrate_path_g_scales.py if the coin set changes.
PATH_G_TANH_SCALES = {
    "fo_market": 3.448,
    "sr": 0.5566,
    "pi_kappa": 0.0002875,
    "pi_vamp_dim": 0.2315,
}


class CharacteristicScales:
    """Path G — online rolling estimates of microstructure characteristic scales.

    Maintains τ_c, L_c, D_c, V_c, κ_c online for use as denominators in the
    dimensionless π-groups (Fo_market, Sr, π_κ, π_vamp_dim). All denominators
    returned by `current()` are floored to their quantization unit
    (`TICK_SIZE` for L_c, `min_tau_ms` for τ_c, `min_depth` for V_c) so the
    π-groups don't blow up at the discrete boundaries of market data — see
    `BOUNDARY_CONDITIONS.md`.

        τ_c (tau_c_s) : rolling mean(Δt)        units: s
        L_c           : rolling mean(spread)    units: P
        D_c           : rolling mean(Δp²/Δt)    units: P²/s
        V_c           : rolling Σ(qty)/window   units: Q/s
        κ_c (kappa_c) : rolling mean(Kyle's λ)  units: P/Q

    Feed via `on_trade(qty)` between book snapshots and `on_book(...)` on
    each snapshot. The on_book call closes the per-tick volume bucket.
    """

    def __init__(
        self,
        lookback_ticks: int = 200,
        tick_size: float = TICK_SIZE,
        min_tau_ms: float = 1.0,
        min_depth: float = MIN_ORDER_SIZE,
    ) -> None:
        self.lookback_ticks = lookback_ticks
        self.tick_size = float(tick_size)
        self.min_tau_ms = float(min_tau_ms)
        self.min_depth = float(min_depth)

        self._dt_ms_buffer: deque = deque(maxlen=lookback_ticks)
        self._spread_buffer: deque = deque(maxlen=lookback_ticks)
        self._dp_sq_per_s_buffer: deque = deque(maxlen=lookback_ticks)
        self._volume_window: deque = deque(maxlen=lookback_ticks)
        self._lambda_buffer: deque = deque(maxlen=lookback_ticks)

        self._last_ts_ms: Optional[int] = None
        self._last_mid: Optional[float] = None
        self._trade_vol_accum: float = 0.0

    def on_trade(self, qty: float) -> None:
        """Accumulate per-trade volume between book snapshots. Sign-blind:
        V_c tracks total throughput, not signed flow (Kyle's λ already
        owns signed flow)."""
        try:
            self._trade_vol_accum += abs(float(qty))
        except (TypeError, ValueError):
            pass

    def on_book(
        self,
        ts_ms: int,
        mid: float,
        spread: float,
        kyles_lambda: float,
    ) -> None:
        """Update scales on each book snapshot. Closes the per-tick volume
        bucket. Δt and Δp² samples require a previous snapshot — first
        call is warmup."""
        if self._last_ts_ms is not None:
            dt_ms = max(self.min_tau_ms, float(ts_ms - self._last_ts_ms))
            self._dt_ms_buffer.append(dt_ms)
            if self._last_mid is not None:
                dp = float(mid) - float(self._last_mid)
                dt_s = max(dt_ms / 1000.0, EPSILON)
                self._dp_sq_per_s_buffer.append((dp * dp) / dt_s)

        self._spread_buffer.append(max(float(spread), self.tick_size))
        self._volume_window.append(self._trade_vol_accum)
        self._trade_vol_accum = 0.0
        # Only push real samples; Kyle's λ is exactly 0 during warmup.
        if kyles_lambda != 0.0:
            self._lambda_buffer.append(float(kyles_lambda))

        self._last_ts_ms = int(ts_ms)
        self._last_mid = float(mid)

    def current(self) -> dict:
        """Return floored scale estimates. Safe to call at any time
        (warm-up returns floor values for unsamplable quantities)."""
        n_dt = len(self._dt_ms_buffer)
        tau_c_s = (
            (sum(self._dt_ms_buffer) / n_dt / 1000.0)
            if n_dt > 0
            else self.min_tau_ms / 1000.0
        )
        tau_c_s = max(tau_c_s, self.min_tau_ms / 1000.0)

        n_s = len(self._spread_buffer)
        L_c = (sum(self._spread_buffer) / n_s) if n_s > 0 else self.tick_size
        L_c = max(L_c, self.tick_size)

        n_d = len(self._dp_sq_per_s_buffer)
        D_c = (sum(self._dp_sq_per_s_buffer) / n_d) if n_d > 0 else 0.0

        n_v = len(self._volume_window)
        total_vol = sum(self._volume_window) if n_v > 0 else 0.0
        window_s = max(tau_c_s * max(n_v, 1), EPSILON)
        V_c = max(self.min_depth, total_vol / window_s)

        n_l = len(self._lambda_buffer)
        kappa_c = (sum(self._lambda_buffer) / n_l) if n_l > 0 else 0.0

        return {
            "tau_c_s": tau_c_s,
            "L_c": L_c,
            "D_c": D_c,
            "V_c": V_c,
            "kappa_c": kappa_c,
        }

    def reset_session(self) -> None:
        self._dt_ms_buffer.clear()
        self._spread_buffer.clear()
        self._dp_sq_per_s_buffer.clear()
        self._volume_window.clear()
        self._lambda_buffer.clear()
        self._last_ts_ms = None
        self._last_mid = None
        self._trade_vol_accum = 0.0


def compute_dimensionless_features(
    dt_ms: float,
    vamp_minus_mid_abs: float,
    scales: dict,
    tanh_scales: Optional[dict] = None,
) -> dict:
    """Compute Path G's dimensionless π-groups for one tick.

    Wraps each raw ratio in tanh(raw / scale_factor) so the model sees a
    bounded, differentiable saturation at quantization boundaries
    instead of an unbounded blow-up (BOUNDARY_CONDITIONS.md §"Mitigation 2").
    Denominators in `scales` are already floored by `CharacteristicScales`.

    Returns four keys, each in (-1, +1):
        fo_market   = tanh(D·Δt/L² / scale)       diffusion vs tick interval
        sr          = tanh(Δt/τ_c   / scale)      local vs recent cadence
        pi_kappa    = tanh(κ·V·τ/L  / scale)      dimensionless price impact
        pi_vamp_dim = tanh((VAMP-mid)/L / scale)  queue deviation in spread units
    """
    if tanh_scales is None:
        tanh_scales = PATH_G_TANH_SCALES

    L_c = scales["L_c"]
    L_c_sq = L_c * L_c
    tau_c_s = scales["tau_c_s"]
    D_c = scales["D_c"]
    V_c = scales["V_c"]
    kappa_c = scales["kappa_c"]

    dt_s = max(float(dt_ms) / 1000.0, EPSILON)

    fo_market_raw = D_c * dt_s / L_c_sq
    fo_market = math.tanh(fo_market_raw / tanh_scales["fo_market"])

    sr_raw = dt_s / tau_c_s
    sr = math.tanh(sr_raw / tanh_scales["sr"])

    pi_kappa_raw = kappa_c * V_c * tau_c_s / L_c
    pi_kappa = math.tanh(pi_kappa_raw / tanh_scales["pi_kappa"])

    pi_vamp_raw = float(vamp_minus_mid_abs) / L_c
    pi_vamp_dim = math.tanh(pi_vamp_raw / tanh_scales["pi_vamp_dim"])

    return {
        "fo_market": float(fo_market),
        "sr": float(sr),
        "pi_kappa": float(pi_kappa),
        "pi_vamp_dim": float(pi_vamp_dim),
    }


# ============================================================================
# Sensor Array — orchestrator
# ============================================================================
@dataclass
class PhysicsState:
    """Snapshot of all sensor outputs. Layer 2 reads this dict every tick."""
    vpin: float = 0.0
    alpha_calibrated: float = 0.0
    viscosity: float = 0.0
    regime: int = 0
    liquidation_rate: float = 0.0
    ood_flag: bool = False
    mahal_dist: float = 0.0
    ce_ratio: float = 0.0
    obi: float = 0.0                 # order book imbalance
    spread_velocity: float = 0.0
    # Path D crypto-native features (P3.6 / §13.4 of LAYER2_TRAINING.md).
    # Populated for HL crypto; zero on equities (no L2 depth from Alpaca,
    # equity TCN_INPUT_CHANNELS=3 doesn't consume them).
    mlofi: float = 0.0               # multi-level OFI, top-K levels
    vamp: float = 0.0                # VAMP - mid, in basis points
    kyles_lambda: float = 0.0        # price-impact-per-unit-flow
    # Path G dimensionless features (research/path-g-dimensionless;
    # §13.13). Tanh-bounded (-1, +1). NOT yet consumed by the TCN —
    # TCN_INPUT_CHANNELS stays at 5 until a harvest with these columns
    # and a successful Phase 4 retraining lands.
    fo_market: float = 0.0           # tanh(D·Δt/L²)
    sr: float = 0.0                  # tanh(Δt/τ_c)
    pi_kappa: float = 0.0            # tanh(κ·V·τ/L)
    pi_vamp_dim: float = 0.0         # tanh((VAMP−mid)/L)
    # Raw characteristic scales — preserved alongside the π-groups so a
    # future harvest can re-derive the dimensionless features with a
    # different tanh saturation calibration without re-running the engine.
    tau_c_s: float = 0.0
    L_c: float = 0.0
    D_c: float = 0.0
    V_c: float = 0.0
    kappa_c: float = 0.0
    timestamp: int = 0
    ob_snapshot: Optional[dict] = field(default=None, repr=False)
    prev_ob_snapshot: Optional[dict] = field(default=None, repr=False)


class FeatureDumper:
    """
    Append-only CSV writer for sensor features. Used to build a calibration
    dataset for `fit_ood_distribution()` — see fit_ood_from_csv.py.

    Writes one row per order-book tick: the full feature triple consumed by
    the OOD detector plus a few diagnostics. Line-buffered so a Ctrl+C
    won't lose the recent rows.
    """

    # Path D columns: ce_ratio, obi, mlofi, vamp, kyles_lambda (§13.4).
    # Path G columns: fo_market, sr, pi_kappa, pi_vamp_dim (the π-groups)
    # plus the raw characteristic scales (tau_c_s, L_c, D_c, V_c, kappa_c)
    # and the raw top-of-book sizes (bid_sz_top, ask_sz_top) — the latter
    # two are book-level quantities a future harvest can use to re-derive
    # the π-groups under a different tanh calibration. Old CSVs that lack
    # the Path G columns remain readable by polars (column-name lookup);
    # readers that touch the new columns must handle absence.
    HEADER = (
        "timestamp_ms,ce_ratio,obi,liquidation_rate,vpin,regime,mahal_dist,"
        "best_bid,best_ask,mlofi,vamp,kyles_lambda,"
        "fo_market,sr,pi_kappa,pi_vamp_dim,"
        "tau_c_s,L_c,D_c,V_c,kappa_c,"
        "bid_sz_top,ask_sz_top\n"
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        # buffering=1 → line-buffered; safe under Ctrl+C
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        if new_file:
            self._fh.write(self.HEADER)
        self.rows_written = 0

    def record(self, state: "PhysicsState") -> None:
        try:
            best_bid = best_ask = 0.0
            bid_sz_top = ask_sz_top = 0.0
            ob = state.ob_snapshot
            if ob:
                if ob.get("bids"):
                    best_bid = float(ob["bids"][0][0])
                    bid_sz_top = float(ob["bids"][0][1])
                if ob.get("asks"):
                    best_ask = float(ob["asks"][0][0])
                    ask_sz_top = float(ob["asks"][0][1])
            self._fh.write(
                f"{state.timestamp},{state.ce_ratio:.6f},{state.obi:.6f},"
                f"{state.liquidation_rate:.6f},{state.vpin:.6f},{state.regime},"
                f"{state.mahal_dist:.6f},{best_bid:.6f},{best_ask:.6f},"
                f"{state.mlofi:.6f},{state.vamp:.6f},{state.kyles_lambda:.8f},"
                f"{state.fo_market:.6f},{state.sr:.6f},{state.pi_kappa:.6f},"
                f"{state.pi_vamp_dim:.6f},{state.tau_c_s:.6f},{state.L_c:.6f},"
                f"{state.D_c:.8f},{state.V_c:.6f},{state.kappa_c:.8f},"
                f"{bid_sz_top:.6f},{ask_sz_top:.6f}\n"
            )
            self.rows_written += 1
        except Exception as e:
            log.warning("FeatureDumper write failed: %s", e)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class SensorArray:
    """
    Orchestrates all Layer 1 components. Runs as background asyncio coroutines
    consuming ccxt.pro WebSocket streams. Exposes a single PhysicsState object
    that is read each tick by Layer 2.
    """

    def __init__(
        self,
        exchange: Any,
        symbol: str = SYMBOL,
        liquidation_tracker: Optional[LiquidationCascadeTracker] = None,
        ood_detector: Optional[OODDetector] = None,
        hmm: Optional[StudentTHMM] = None,
        alpha_calibration_c: float = 0.18,
        feature_dumper: Optional[FeatureDumper] = None,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.alpha_calibration_c = alpha_calibration_c

        self.vpin = VPINToxicityTracker()
        self.kalman = StudentTKalmanFilter()
        self.hmm = hmm or StudentTHMM.from_default_priors()

        from config import EXCHANGE_ID
        is_binance = (EXCHANGE_ID == "binance")
        is_hyperliquid = (EXCHANGE_ID == "hyperliquid")
        # Equities have no liquidation feed; non-Binance crypto venues feed
        # the tracker via record() from venue-specific adapters (e.g. HL
        # harvester parses liquidations out of trade events). Tracker stays
        # enabled in those cases so the rate isn't pinned to 0.
        liq_enabled = (is_binance or is_hyperliquid) and not IS_EQUITY
        self.liquidation = liquidation_tracker or LiquidationCascadeTracker(
            symbol=symbol, enabled=liq_enabled
        )

        self.ood = ood_detector or OODDetector()
        # On equities we have no L2 depth, so the L2-diff C/E inference can't
        # work. Use the L1 quote-update proxy instead. Both classes expose
        # `.ce_ratio` and `.reset_session()` so the rest of the pipeline is
        # agnostic.
        self.ce = QuoteCancellationProxy() if IS_EQUITY else CEInferenceEngine()
        self.feature_dumper = feature_dumper

        # Path D crypto-native trackers. Instantiated only for crypto (no
        # L2 depth or per-trade initiator field on equities). MLOFI is
        # stateless across snapshots; VAMP is a static helper, no instance.
        # Path G adds CharacteristicScales for the dimensionless π-groups
        # (research/path-g-dimensionless; §13.13). Same crypto-only gating.
        if not IS_EQUITY:
            self.mlofi = MLOFITracker(depth_levels=MLOFI_DEPTH_LEVELS)
            self.kyles = KylesLambdaTracker(lookback_ticks=KYLES_LAMBDA_LOOKBACK_TICKS)
            self.scales = CharacteristicScales(lookback_ticks=KYLES_LAMBDA_LOOKBACK_TICKS)
        else:
            self.mlofi = None
            self.kyles = None
            self.scales = None

        self.state = PhysicsState()
        self._prev_ob: Optional[dict] = None
        self._recent_trades: deque = deque()  # for C/E inference window
        self._spread_prev: Optional[float] = None
        self._spread_prev_ts_ms: Optional[int] = None
        # Cached for Lee-Ready trade-side classification when prints don't
        # carry an initiator field (equities). Updated by every order-book tick.
        self._last_mid: Optional[float] = None
        self._last_trade_price: Optional[float] = None
        # OOD warmup window after a session reset — see reset_session().
        self._ood_quarantine_until_ms: int = 0
        self.running = False

    # -------------------------------------------------------- WebSocket loops
    async def _stream_order_book(self) -> None:
        while self.running:
            try:
                ob = await self.exchange.watch_order_book(self.symbol)
                self._process_order_book(ob)
            except Exception as e:
                log.warning(f"order book stream interrupted: {e}. Reconnect in 2s.")
                await asyncio.sleep(2)

    async def _stream_trades(self) -> None:
        while self.running:
            try:
                trades = await self.exchange.watch_trades(self.symbol)
                for t in trades:
                    self._process_trade(t)
            except Exception as e:
                log.warning(f"trade stream interrupted: {e}. Reconnect in 2s.")
                await asyncio.sleep(2)

    # ------------------------------------------------------- per-event hooks
    def _process_trade(self, trade: dict) -> None:
        ts = int(trade["timestamp"])
        if IS_EQUITY:
            # Equity prints carry no initiator field — derive via Lee-Ready
            # against the most recent NBBO midpoint we cached from the last
            # quote tick. If no quote has been seen yet, the classifier
            # falls back gracefully to the tick test.
            self.vpin.add_trade_classified(
                trade,
                mid_at_trade=self._last_mid,
                last_trade_price=self._last_trade_price,
            )
            self._last_trade_price = float(trade["price"])
            # Feed the C/E proxy so it can match this trade against quote
            # updates in the same window.
            if isinstance(self.ce, QuoteCancellationProxy):
                self.ce.on_trade(ts)
        else:
            self.vpin.add_trade(trade)
            qty = float(trade["amount"])
            # Path D — accumulate signed flow for Kyle's λ between book ticks.
            # ccxt + HL archive both expose trade["side"] ∈ {"buy", "sell"}.
            if self.kyles is not None:
                self.kyles.on_trade(trade["side"], qty)
            # Path G — accumulate per-trade volume for V_c. Sign-blind: V_c
            # tracks throughput (Q/s rate); signed flow is Kyle's λ's job.
            if self.scales is not None:
                self.scales.on_trade(qty)

        self._recent_trades.append(trade)
        cutoff = ts - int(CE_ROLLING_WINDOW_SEC * 1000)
        while self._recent_trades and self._recent_trades[0]["timestamp"] < cutoff:
            self._recent_trades.popleft()

    def _process_order_book(self, ob: dict) -> None:
        if not ob.get("bids") or not ob.get("asks"):
            return
        ts_ms = int(ob.get("timestamp") or time.time() * 1000)
        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        spread = best_ask - best_bid
        mid = 0.5 * (best_bid + best_ask)

        # Kalman update
        self.kalman.update(spread)

        # Spread first derivative
        if self._spread_prev is not None and self._spread_prev_ts_ms is not None:
            dt_ms_since_prev = float(ts_ms - self._spread_prev_ts_ms)
            dt = max(1, ts_ms - self._spread_prev_ts_ms) / 1000.0
            spread_d1 = (spread - self._spread_prev) / dt
        else:
            dt_ms_since_prev = 0.0
            spread_d1 = 0.0
        self._spread_prev = spread
        self._spread_prev_ts_ms = ts_ms

        # OBI — top-1 imbalance, signed: + means bid-heavy.
        bid_sz = ob["bids"][0][1]
        ask_sz = ob["asks"][0][1]
        obi = (bid_sz - ask_sz) / (bid_sz + ask_sz + EPSILON)

        # Cache midpoint for Lee-Ready trade classification on the next prints.
        self._last_mid = mid

        # C/E ratio. Two backends, same `.ce_ratio` contract:
        # - Equities (QuoteCancellationProxy): driven by quote-event timestamps,
        #   no L2 diff, naturally robust to WS reconnects.
        # - Crypto  (CEInferenceEngine): L2-level diff; skip across reconnect
        #   gaps so cumulative changes aren't attributed to phantom cancels.
        if IS_EQUITY:
            self.ce.on_quote(ts_ms)
        elif self._prev_ob is not None:
            prev_ts = int(self._prev_ob.get("timestamp") or 0)
            gap_ms = ts_ms - prev_ts
            if 0 < gap_ms < 1000:
                self.ce.update(
                    prev_ob=self._prev_ob,
                    curr_ob=ob,
                    trades_in_window=list(self._recent_trades),
                    ts_ms=ts_ms,
                )
            elif gap_ms >= 1000:
                log.debug(
                    "CE update skipped: %dms gap between snapshots (likely WS reconnect)",
                    gap_ms,
                )

        # Liquidation rate (normalized by recent volume)
        liq_rate = self.liquidation.liquidation_rate(self.vpin.average_volume_10s)

        # Path D crypto-native features. Computed only for crypto venues;
        # equities don't have multi-level depth or trade-side fields, and
        # the equity TCN_INPUT_CHANNELS=3 model doesn't consume them.
        if self.mlofi is not None:
            mlofi = self.mlofi.compute(self._prev_ob, ob)
            vamp = VAMPDeviationComputer.compute(ob, mid)
            kyles_lambda = self.kyles.on_book(mid)
        else:
            mlofi = 0.0
            vamp = 0.0
            kyles_lambda = 0.0

        # Path G — characteristic scales + dimensionless π-groups
        # (research/path-g-dimensionless; LAYER2_TRAINING.md §13.13).
        # Same crypto-only gating as Path D. The π-groups are tanh-bounded
        # in compute_dimensionless_features per BOUNDARY_CONDITIONS.md.
        # NOT yet consumed by the TCN — these columns are emitted only so
        # a future re-harvest produces training data with the new schema.
        if self.scales is not None:
            self.scales.on_book(ts_ms, mid, spread, kyles_lambda)
            scales_now = self.scales.current()
            # VAMP is in basis points relative to mid; convert back to
            # absolute P for the dimensionless π_vamp ratio.
            vamp_minus_mid_abs = (vamp / 1e4) * mid
            pi_groups = compute_dimensionless_features(
                dt_ms=dt_ms_since_prev,
                vamp_minus_mid_abs=vamp_minus_mid_abs,
                scales=scales_now,
            )
            fo_market = pi_groups["fo_market"]
            sr_val = pi_groups["sr"]
            pi_kappa = pi_groups["pi_kappa"]
            pi_vamp_dim = pi_groups["pi_vamp_dim"]
            tau_c_s = scales_now["tau_c_s"]
            L_c = scales_now["L_c"]
            D_c = scales_now["D_c"]
            V_c = scales_now["V_c"]
            kappa_c = scales_now["kappa_c"]
        else:
            fo_market = 0.0
            sr_val = 0.0
            pi_kappa = 0.0
            pi_vamp_dim = 0.0
            tau_c_s = 0.0
            L_c = 0.0
            D_c = 0.0
            V_c = 0.0
            kappa_c = 0.0

        # HMM update — 4 emission features (unchanged; HMM stays on the
        # original [ce_ratio, obi, spread_d1, liq_rate] vector; refitting
        # the HMM for HL is deferred — see TODO.md P5).
        hmm_obs = np.array([self.ce.ce_ratio, obi, spread_d1, liq_rate])
        regime = self.hmm.update(hmm_obs)

        # OOD check — 3-feature vector. Stays on the original feature set
        # for back-compat with existing per-symbol OOD calibrations. HL's
        # OOD is currently uncalibrated (TODO.md P1) so the dead
        # liquidation channel here is a known degenerate axis until
        # either P1 lands a real source or OOD is refit on the new
        # crypto-native vector.
        ood_obs = np.array([self.ce.ce_ratio, obi, liq_rate])
        ood_flag, mahal = self.ood.evaluate(ood_obs)
        # During the post-open warmup window, suppress OOD trips so the first
        # quotes don't fire a false-positive shock signal while the filters
        # re-warm. The mahalanobis distance is still recorded for telemetry.
        if ts_ms < self._ood_quarantine_until_ms:
            ood_flag = False

        # Atomic state update
        self.state = PhysicsState(
            vpin=self.vpin.vpin,
            alpha_calibrated=self.alpha_calibration_c * self.vpin.vpin,
            viscosity=self.kalman.viscosity,
            regime=regime,
            liquidation_rate=liq_rate,
            ood_flag=ood_flag,
            mahal_dist=mahal,
            ce_ratio=self.ce.ce_ratio,
            obi=obi,
            spread_velocity=spread_d1,
            mlofi=mlofi,
            vamp=vamp,
            kyles_lambda=kyles_lambda,
            fo_market=fo_market,
            sr=sr_val,
            pi_kappa=pi_kappa,
            pi_vamp_dim=pi_vamp_dim,
            tau_c_s=tau_c_s,
            L_c=L_c,
            D_c=D_c,
            V_c=V_c,
            kappa_c=kappa_c,
            timestamp=ts_ms,
            ob_snapshot=ob,
            prev_ob_snapshot=self._prev_ob,
        )
        self._prev_ob = ob

        if self.feature_dumper is not None:
            self.feature_dumper.record(self.state)

    # -------------------------------------------------------- session control
    def reset_session(self, warmup_seconds: float = 0.0) -> None:
        """Reset all path-dependent sensor state.

        Called by the engine at each market open. Without this the overnight
        gap is fed to the Kalman filter as a 1-tick spread innovation, the
        HMM scores the first morning observation against yesterday's regime
        distribution, and the VPIN buckets carry partial yesterday-evening
        volume into today — every one of those produces an OOD storm.
        """
        from config import (
            SESSION_RESET_HMM,
            SESSION_RESET_KALMAN,
            SESSION_RESET_VPIN,
        )
        if SESSION_RESET_VPIN:
            self.vpin.reset_session()
        if SESSION_RESET_KALMAN:
            self.kalman.reset_session()
        if SESSION_RESET_HMM:
            self.hmm.reset_session()
        # Both CE backends expose reset_session().
        self.ce.reset_session()
        # Path D trackers (None on equities).
        if self.kyles is not None:
            self.kyles.reset_session()
        if self.mlofi is not None:
            self.mlofi.reset_session()
        # Path G — characteristic scales (None on equities).
        if self.scales is not None:
            self.scales.reset_session()

        self._prev_ob = None
        self._spread_prev = None
        self._spread_prev_ts_ms = None
        self._recent_trades.clear()
        self._last_mid = None
        self._last_trade_price = None

        if warmup_seconds > 0.0:
            self._ood_quarantine_until_ms = (
                int(time.time() * 1000) + int(warmup_seconds * 1000)
            )
        else:
            self._ood_quarantine_until_ms = 0
        log.info(
            "SensorArray.reset_session: state cleared, OOD quarantined for %.1fs",
            warmup_seconds,
        )

    # ------------------------------------------------ Alpaca stream callbacks
    async def _on_alpaca_quote(self, quote: Any) -> None:
        """Handler registered with `StockDataStream.subscribe_quotes`.

        Synthesises an order-book-shaped dict from the NBBO update so that
        `_process_order_book` is identical to the crypto path.
        """
        try:
            ts = quote.timestamp
            ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else int(ts)
            ob = {
                "bids": [[float(quote.bid_price), float(quote.bid_size)]],
                "asks": [[float(quote.ask_price), float(quote.ask_size)]],
                "timestamp": ts_ms,
            }
            self._process_order_book(ob)
        except Exception as e:
            log.warning(f"alpaca quote handler failed: {e}")

    async def _on_alpaca_trade(self, trade_evt: Any) -> None:
        """Handler registered with `StockDataStream.subscribe_trades`."""
        try:
            ts = trade_evt.timestamp
            ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else int(ts)
            trade = {
                "timestamp": ts_ms,
                "price": float(trade_evt.price),
                "amount": float(trade_evt.size),
            }
            self._process_trade(trade)
        except Exception as e:
            log.warning(f"alpaca trade handler failed: {e}")

    # ----------------------------------------------------------- entry point
    async def run(self) -> None:
        self.running = True
        if IS_EQUITY:
            # Alpaca path: register handlers and let the SDK drive its own loop.
            # `self.exchange` is an alpaca.data.live.StockDataStream.
            try:
                self.exchange.subscribe_quotes(self._on_alpaca_quote, self.symbol)
                self.exchange.subscribe_trades(self._on_alpaca_trade, self.symbol)
                # `_run_forever` is the public coroutine in alpaca-py>=0.13.
                await self.exchange._run_forever()
            except Exception as e:
                log.error(f"Alpaca data stream terminated: {e}")
            return

        # Crypto path: ccxt.pro polling loops.
        await asyncio.gather(
            self._stream_order_book(),
            self._stream_trades(),
            self.liquidation.run() if self.liquidation.enabled else asyncio.sleep(0),
            return_exceptions=True,
        )

    def stop(self) -> None:
        self.running = False
        self.liquidation.running = False
        if self.feature_dumper is not None:
            self.feature_dumper.close()
