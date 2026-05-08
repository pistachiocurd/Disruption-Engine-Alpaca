"""
engine.py — Main entry point. Asyncio orchestration of all four layers.

Topology:
    Stream 1: SensorArray loops (order book + trades + liquidations)
    Stream 2: ShadowSimulator loop (continuous training on GPU)
    Stream 3: Main strategy loop (Layer 2 alpha + Layer 3 execution)

Kill switches:
    - Unhandled exception in any task → TRADING_ENABLED=False, cancel all orders.
    - Session drawdown > MAX_SESSION_DRAWDOWN_PCT → trading halted.
    - Calibration drift detected → mandate generation suspended.
    - SIGINT / SIGTERM → graceful shutdown.

Run:
    python engine.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config

# ----------------------------------------------------------------------------
# Windows DNS workaround.
# aiohttp prefers `aiodns` (c-ares) when installed. On Windows, c-ares often
# fails with "Could not contact DNS servers" because it can't read the system
# DNS config from the registry, even when `Resolve-DnsName` succeeds. Force
# aiohttp back to its ThreadedResolver, which uses `socket.getaddrinfo`.
# Then restrict getaddrinfo to IPv4 to skip AAAA records on networks without
# working IPv6 egress.
# ----------------------------------------------------------------------------
if sys.platform == "win32" and os.environ.get("DAE_FORCE_IPV4", "1") == "1":
    try:
        import aiohttp.connector as _aiohttp_connector
        import aiohttp.resolver as _aiohttp_resolver
        _aiohttp_resolver.aiodns_default = False
        _aiohttp_resolver.DefaultResolver = _aiohttp_resolver.ThreadedResolver
        # connector imported DefaultResolver by name at module load — rebind it.
        _aiohttp_connector.DefaultResolver = _aiohttp_resolver.ThreadedResolver
    except Exception:
        pass

    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_only_getaddrinfo(host, port, family=0, *args, **kwargs):
        return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)

    socket.getaddrinfo = _ipv4_only_getaddrinfo
from calibration import CalibrationDriftMonitor, load_calibration
from dashboard import DashboardServer
from layer1_sensors import (
    FeatureDumper,
    LiquidationCascadeTracker,
    OODDetector,
    SensorArray,
    StudentTHMM,
)
from layer2_alpha import AlphaEngine, TradeMandate
from layer3_execution import ExecutionEnv, PPOAgent
from layer4_shadow import ReplayEntry, ShadowSimulator
from matching_engine import LocalMatchingEngine

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("engine")

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# ============================================================================
# SessionManager — equity market-hours gate
# ============================================================================
class SessionManager:
    """Owns the trading calendar and answers `is_market_open()`.

    Pulls the schedule from Alpaca's `/v2/calendar` so half-days and holidays
    are handled without a separate dependency. Cache is refreshed once per
    day; the engine treats stale cache as closed (fail-safe).
    """

    def __init__(self, trading_client) -> None:
        self.trading = trading_client
        self._sessions: list = []           # list of (open_dt_utc, close_dt_utc)
        self._cache_loaded_for: Optional[str] = None  # YYYY-MM-DD (UTC)

    def _ensure_cache(self) -> None:
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._cache_loaded_for == today and self._sessions:
            return
        try:
            from alpaca.trading.requests import GetCalendarRequest
            req = GetCalendarRequest(
                start=(datetime.now(timezone.utc) - timedelta(days=1)).date(),
                end=(datetime.now(timezone.utc) + timedelta(days=14)).date(),
            )
            cal = self.trading.get_calendar(req)
        except Exception as e:
            log.warning("SessionManager: failed to refresh calendar (%s).", e)
            return

        sessions = []
        for c in cal:
            # Each entry has .date, .open, .close — all naive local NY times.
            # Combine and convert to UTC.
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(config.MARKET_TZ)
                d = c.date if hasattr(c, "date") else c["date"]
                op = c.open if hasattr(c, "open") else c["open"]
                cl = c.close if hasattr(c, "close") else c["close"]

                # alpaca-py returns datetime.date / datetime.time; some
                # response shapes return strings. Handle both, plus datetimes.

                if isinstance(d, str):
                    from datetime import date as _date
                    d = _date.fromisoformat(d)
                elif isinstance(d, datetime):
                    d = d.date()
                    
                if isinstance(op, str):
                    from datetime import time as _time
                    op = _time.fromisoformat(op)
                elif isinstance(op, datetime):
                    op = op.time()
                    
                if isinstance(cl, str):
                    from datetime import time as _time
                    cl = _time.fromisoformat(cl)
                elif isinstance(cl, datetime):
                    cl = cl.time()
                    
                open_dt = datetime.combine(d, op, tzinfo=tz).astimezone(timezone.utc)
                close_dt = datetime.combine(d, cl, tzinfo=tz).astimezone(timezone.utc)
                sessions.append((open_dt, close_dt))
            except Exception as e:
                log.debug("SessionManager: skipping malformed calendar entry (%s)", e)
        self._sessions = sessions
        self._cache_loaded_for = today
        log.info("SessionManager: cached %d sessions through 14 days", len(sessions))

    def is_market_open(self) -> bool:
        from datetime import datetime, timezone
        self._ensure_cache()
        if not self._sessions:
            return False
        now = datetime.now(timezone.utc)
        for op, cl in self._sessions:
            if op <= now <= cl:
                return True
        return False

    def seconds_until_open(self) -> float:
        from datetime import datetime, timezone
        self._ensure_cache()
        if not self._sessions:
            return 60.0  # retry-the-cache cadence
        now = datetime.now(timezone.utc)
        for op, _cl in self._sessions:
            if op > now:
                return max(0.0, (op - now).total_seconds())
        return 60.0  # all cached sessions in the past — refresh and retry

    async def wait_until_open(self) -> None:
        # Coarse wait-then-poll: long sleeps when far from open, fine-grained
        # near the bell. Avoids tight loops without missing the open.
        while True:
            if self.is_market_open():
                return
            wait = self.seconds_until_open()
            sleep_for = min(60.0, max(1.0, wait))
            await asyncio.sleep(sleep_for)


# ============================================================================
# Engine
# ============================================================================
class Engine:
    def __init__(self) -> None:
        # On equities, _init_exchange returns the live data stream (used by
        # SensorArray) and also populates `self.trading` / `self.hist` for the
        # broker REST API. On crypto, `self.exchange` is the ccxt.pro instance
        # (which serves both data and trading).
        self.trading = None  # alpaca.trading.client.TradingClient (equity only)
        self.hist = None     # alpaca.data.historical.StockHistoricalDataClient
        self.exchange = self._init_exchange()
        self.shadow_mode = config.SHADOW_MODE
        # SessionManager only meaningful for equities; crypto runs 24/7.
        self.session: Optional[SessionManager] = (
            SessionManager(self.trading) if config.IS_EQUITY else None
        )

        # Optional preloaded calibration (μ, Σ for OOD; α_calibration_c for solver).
        # Per-symbol path so SOL data doesn't load BTC's prior, etc.
        ood = OODDetector()
        alpha_c = config.ALPHA_CALIBRATION_C
        cal_path = Path(config.OOD_CALIBRATION_PATH)
        if cal_path.exists():
            try:
                payload = load_calibration(cal_path)
                ood.load(np.array(payload["ood_mu"]), np.array(payload["ood_sigma"]))
                alpha_c = float(payload.get("alpha_calibration_c", alpha_c))
                log.info("loaded calibration from %s (α=%.4f)", cal_path, alpha_c)
            except Exception as e:
                log.warning("failed to load calibration; using defaults: %s", e)
        else:
            log.info("no calibration at %s; OOD using cold-start prior", cal_path)

        liquidation = LiquidationCascadeTracker(
            symbol=config.SYMBOL,
            enabled=(config.EXCHANGE_ID == "binance"),
        )

        # Optional feature dumper — set DUMP_FEATURES=1 to record per-tick
        # [ce_ratio, obi, liquidation_rate] history for offline OOD fitting.
        # Default path is per-symbol: feature_history_{SLUG}.csv.
        feature_dumper = None
        if os.environ.get("DUMP_FEATURES", "0") == "1":
            dump_path = os.environ.get("FEATURE_DUMP_PATH", config.FEATURE_DUMP_PATH)
            feature_dumper = FeatureDumper(dump_path)
            log.info("FeatureDumper writing to %s", dump_path)

        self.sensors = SensorArray(
            exchange=self.exchange,
            symbol=config.SYMBOL,
            liquidation_tracker=liquidation,
            ood_detector=ood,
            hmm=StudentTHMM.from_default_priors(),
            alpha_calibration_c=alpha_c,
            feature_dumper=feature_dumper,
        )
        # Initial turbulence threshold from config; train_tcn.py's sidecar
        # JSON, if present, overrides this with a value derived from the
        # validation PR curve.
        turb_thresh = float(config.TURBULENCE_THRESHOLD)
        self._turbulence_threshold_source = "config"
        self._tcn_threshold_payload: dict | None = None

        thresh_path = Path(config.TCN_THRESHOLD_PATH)
        if thresh_path.exists():
            try:
                payload = json.loads(thresh_path.read_text())
                turb_thresh = float(payload["threshold"])
                self._tcn_threshold_payload = payload
                self._turbulence_threshold_source = "trained"
                log.info(
                    "loaded TCN threshold %.4f from %s "
                    "(target_prec=%.2f, achieved_prec=%.3f, recall=%.3f, target_satisfied=%s)",
                    turb_thresh, thresh_path,
                    payload.get("target_precision", 0),
                    payload.get("achieved_precision", 0),
                    payload.get("achieved_recall", 0),
                    payload.get("satisfied_target", False),
                )
            except Exception as e:
                log.warning(
                    "TCN threshold at %s failed to load (%s); using config default %.3f",
                    thresh_path, e, turb_thresh,
                )

        self.alpha = AlphaEngine(turbulence_threshold=turb_thresh)

        # Auto-load TCN weights if present. Produced by train_tcn.py.
        # Per-symbol path so a SOL-trained TCN doesn't load when running BTC.
        tcn_path = Path(config.TCN_WEIGHTS_PATH)
        if tcn_path.exists():
            try:
                self.alpha.load_weights(str(tcn_path))
                log.info("loaded TCN weights from %s", tcn_path)
            except Exception as e:
                log.warning(
                    "TCN weights at %s failed to load (%s); using random init",
                    tcn_path, e,
                )
        else:
            log.info("no TCN weights at %s; TCN remains random-init", tcn_path)

        # Live and shadow agents.
        self.live_agent = PPOAgent()
        self.shadow_agent = PPOAgent()
        self.shadow_agent.load_state_dict(self.live_agent.state_dict())

        self.shadow_sim = ShadowSimulator(
            live_agent=self.live_agent,
            shadow_agent=self.shadow_agent,
            live_regime_provider=lambda: self.sensors.state.regime,
        )

        # Drift monitor — current_c sourced from calibration registry.
        self.drift_monitor = CalibrationDriftMonitor(current_c=alpha_c)

        # Risk tracking — execution-shortfall PnL for drawdown.
        self.session_pnl: float = 0.0
        self.session_peak_pnl: float = 0.0
        # Theoretical mark-to-market PnL — what a real trader would see if these
        # simulated fills had executed. Updated per fill, MTM'd against live mid.
        self._cash: float = 0.0
        self._position: float = 0.0
        self._total_fees: float = 0.0
        self._total_filled_notional: float = 0.0
        self._recent_mandates: deque = deque(maxlen=256)
        self._recent_episodes: deque = deque(maxlen=50)
        # Per-fill order log — every step that resulted in a placement attempt.
        # Read by the dashboard.
        self._recent_orders: deque = deque(maxlen=500)
        self._next_episode_id: int = 0
        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

        # Dashboard server — read-only view onto engine state.
        self.dashboard = DashboardServer(self)

    # ---------------------------------------------------- exchange wiring
    def _init_exchange(self):
        if config.IS_EQUITY:
            return self._init_alpaca()
        return self._init_ccxt()

    def _init_alpaca(self):
        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise ImportError(
                "alpaca-py is required for equities. install via `pip install alpaca-py`"
            ) from e

        key = config.EXCHANGE_API_KEY
        secret = config.EXCHANGE_SECRET
        if not key or not secret:
            log.warning(
                "EXCHANGE_API_KEY / EXCHANGE_SECRET are empty. Alpaca will reject "
                "auth on first request — set them before going live."
            )

        feed = config.ALPACA_DATA_FEED
        data_stream = StockDataStream(key, secret, feed=feed)
        self.hist = StockHistoricalDataClient(key, secret)
        # `paper=True` when EXCHANGE_LIVE is not literally "true". Mirrors the
        # ccxt sandbox fallback below.
        self.trading = TradingClient(key, secret, paper=not config.EXCHANGE_LIVE)
        log.info(
            "alpaca initialized: feed=%s paper=%s symbol=%s",
            feed, not config.EXCHANGE_LIVE, config.SYMBOL,
        )
        if config.EXCHANGE_LIVE:
            log.warning("EXCHANGE_LIVE=true → REAL CAPITAL AT RISK.")
        return data_stream

    def _init_ccxt(self):
        try:
            import ccxt.pro as ccxtpro
        except ImportError as e:
            raise ImportError(
                "ccxt.pro is required. install via `pip install ccxt` (pro is bundled in recent versions)"
            ) from e

        cls = getattr(ccxtpro, config.EXCHANGE_ID)

        kwargs = dict(
            apiKey=config.EXCHANGE_API_KEY,
            secret=config.EXCHANGE_SECRET,
            options={
                "defaultType": "spot",       # Coinbase Advanced is spot-only.
                "fetchCurrencies": False,    # Skip the v2 metadata call (404s on sandbox).
            },
            enableRateLimit=True,
        )

        ex = cls(kwargs)

        # Pre-seed the configured market so the engine doesn't depend on
        # exchange.load_markets() — that call hits the v2 metadata endpoint
        # which is unreliable on Coinbase Advanced sandbox / NYC-restricted IPs.
        # Symbol is parsed from BASE/QUOTE form; ccxt's coinbaseadvanced uses
        # BASE-QUOTE as the market id.
        try:
            base, quote = config.SYMBOL.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                f"SYMBOL must be in BASE/QUOTE form, got {config.SYMBOL!r}"
            ) from exc
        market_id = f"{base}-{quote}"
        market = {
            "id": market_id,
            "symbol": config.SYMBOL,
            "base": base,
            "quote": quote,
            "baseId": base,
            "quoteId": quote,
            "active": True,
            "spot": True,
            "type": "spot",
            "precision": {"amount": 8, "price": 2},
            "limits": {
                "amount": {"min": float(config.MIN_ORDER_SIZE)},
                "price": {"min": float(config.TICK_SIZE)},
            },
        }
        ex.markets = {config.SYMBOL: market}
        # ccxt's safe_market() expects markets_by_id values to be a list (so it
        # can disambiguate spot vs futures sharing one market id).
        ex.markets_by_id = {market_id: [market]}
        ex.symbols = [config.SYMBOL]

        if not config.EXCHANGE_LIVE:
            try:
                ex.set_sandbox_mode(True)
                log.info("EXCHANGE_LIVE=false → sandbox enabled (%s).", config.EXCHANGE_ID)
            except Exception as e:
                # Coinbase Advanced does not expose a public sandbox in all ccxt
                # builds. Fall back to live data; SHADOW_MODE keeps orders simulated.
                log.warning(
                    "sandbox unavailable for %s (%s); reading live market data. "
                    "SHADOW_MODE=%s controls whether orders are simulated.",
                    config.EXCHANGE_ID, e, config.SHADOW_MODE,
                )
        else:
            log.warning("EXCHANGE_LIVE=true → REAL CAPITAL AT RISK.")
        return ex

    # ---------------------------------------------- main strategy loop
    async def _strategy_loop(self) -> None:
        log.info("strategy loop started")
        # Tracks whether we've already done the post-open sensor reset for
        # this calendar session; toggled when the market closes again.
        was_open = True
        while config.TRADING_ENABLED and not self._stop_event.is_set():
            try:
                # Equity session gate. Without this the HMM, Kalman, and VPIN
                # buckets would carry overnight state into the next morning's
                # first ticks and trip a false-positive OOD storm.
                if self.session is not None and not self.session.is_market_open():
                    if was_open:
                        log.info("market closed — pausing strategy and cancelling open orders")
                        await self._cancel_all_open_orders()
                        was_open = False
                    await self.session.wait_until_open()
                    log.info("market open — resetting sensor state")
                    self.sensors.reset_session(
                        warmup_seconds=config.SESSION_WARMUP_SECONDS
                    )
                    # Clear any replay entries spanning the overnight gap so
                    # the shadow simulator doesn't mix sessions.
                    if hasattr(self.shadow_sim, "buffer") and \
                            hasattr(self.shadow_sim.buffer, "clear"):
                        try:
                            self.shadow_sim.buffer.clear()
                        except Exception as e:
                            log.warning("shadow buffer clear failed: %s", e)
                    was_open = True

                await asyncio.sleep(0.05)
                state = self.sensors.state
                if state.ob_snapshot is None:
                    continue

                # Drift / drawdown gates
                if self.drift_monitor.is_stale:
                    continue
                if self._max_drawdown_breached():
                    log.error("MAX DRAWDOWN BREACHED — halting trading")
                    config.TRADING_ENABLED = False
                    await self._cancel_all_open_orders()
                    break

                mandate = self.alpha.evaluate(state)
                if mandate is None:
                    continue
                self._recent_mandates.append(mandate)
                await self._execute_mandate(mandate)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("unhandled in strategy loop: %s — kill switch tripped", e)
                config.TRADING_ENABLED = False
                await self._cancel_all_open_orders()
                break

    # -------------------------------------------- per-mandate execution
    async def _execute_mandate(self, mandate: TradeMandate) -> None:
        env = ExecutionEnv(
            live_state_provider=self.sensors,
            matching_engine=LocalMatchingEngine(),
            exchange=self.exchange,
            shadow_mode=self.shadow_mode,
        )
        env.attach_mandate(mandate)
        obs, _ = env.reset()
        cumulative_reward = 0.0
        episode_start = time.monotonic()
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        side = "buy" if mandate.direction > 0 else "sell"
        while True:
            action, log_prob, value = self.live_agent.act(obs)
            obs_next, reward, terminated, truncated, info = env.step(action)

            # Record this step as an order placement attempt — visible on the dashboard.
            self._record_order(
                mandate=mandate,
                episode_id=episode_id,
                action=float(action[0]) if hasattr(action, "__len__") else float(action),
                side=side,
                info=info,
            )

            self.shadow_sim.push_transition(ReplayEntry(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=bool(terminated or truncated),
                regime=self.sensors.state.regime,
                timestamp=time.monotonic(),
            ))
            obs = obs_next
            cumulative_reward += reward
            if terminated or truncated:
                break
            # Hard stop: if we've blown through window * multiplier, force-cancel any limit.
            if time.monotonic() - episode_start > mandate.execution_window_seconds * \
                    config.EXECUTION_HARD_STOP_MULTIPLIER:
                log.warning("execution hard stop triggered — cancelling open orders")
                await self._cancel_all_open_orders()
                break

        result = env.episode_result()
        self._update_session_pnl(result, mandate)
        self._recent_episodes.append({
            "ts": int(time.time() * 1000),
            "direction": int(mandate.direction),
            "target_size": float(mandate.target_size),
            "total_filled": float(result.total_filled),
            "mean_is": float(result.mean_is),
            "fill_price_std": float(result.fill_price_std),
            "fee_paid_total": float(result.fee_paid_total),
            "n_fills": int(result.n_fills),
            "forced_terminal": bool(result.forced_terminal),
        })
        log.info(
            "episode done: dir=%+d size=%.4f mean_IS=%.5f fee=%.4f forced=%s",
            mandate.direction, mandate.target_size, result.mean_is,
            result.fee_paid_total, result.forced_terminal,
        )

    # -------------------------------------------------- order log helper
    def _record_order(
        self,
        mandate: TradeMandate,
        episode_id: int,
        action: float,
        side: str,
        info: dict,
    ) -> None:
        """One row per ExecutionEnv step. Visible in the dashboard's Orders panel."""
        is_taker = bool(info.get("is_taker", False))
        filled = float(info.get("filled", 0.0) or 0.0)
        fill_price = float(info.get("fill_price", 0.0) or 0.0)
        # Order intent based on action vs threshold (matches LocalMatchingEngine logic).
        if action < config.FEE_AGGRESSION_THRESHOLD:
            order_type = "market"
        else:
            order_type = "limit"
        # Forced terminal market fill is signalled by terminal_reward != 0.
        terminal_reward = float(info.get("terminal_reward", 0.0) or 0.0)
        if terminal_reward != 0.0:
            order_type = "forced_market"
        fee_rate = config.TAKER_FEE if is_taker else config.MAKER_FEE
        fee_paid_step = filled * fill_price * fee_rate if filled > 0 else 0.0
        # Implementation shortfall in bps (fee-adjusted) for this fill.
        step_is = float(info.get("step_is", 0.0) or 0.0)
        is_bps = step_is * 10_000.0

        # Theoretical PnL accounting — cash + MTM model. Each fill moves cash
        # one way and position the other; net PnL is unrealized at any time.
        if filled > 0 and fill_price > 0:
            notional = filled * fill_price
            if side == "buy":
                self._cash -= notional + fee_paid_step
                self._position += filled
            else:
                self._cash += notional - fee_paid_step
                self._position -= filled
            self._total_fees += fee_paid_step
            self._total_filled_notional += notional

        self._recent_orders.append({
            "ts": int(time.time() * 1000),
            "episode_id": episode_id,
            "side": side,
            "direction": int(mandate.direction),
            "action": action,
            "order_type": order_type,
            "is_taker": is_taker,
            "filled_qty": filled,
            "fill_price": fill_price,
            "fee_paid": fee_paid_step,
            "fee_rate": fee_rate,
            "p_decision": float(mandate.p_decision),
            "is_bps": is_bps,
            "remaining_after": float(info.get("remaining_volume", 0.0) or 0.0),
            "target_size": float(mandate.target_size),
        })

    # ------------------------------------------------------- risk helpers
    def _update_session_pnl(self, result, mandate: TradeMandate) -> None:
        # Approximate PnL contribution from this episode (negative IS = good for us).
        pnl_contrib = -result.mean_is * mandate.target_size * mandate.p_decision
        self.session_pnl += pnl_contrib
        self.session_peak_pnl = max(self.session_peak_pnl, self.session_pnl)

    def _max_drawdown_breached(self) -> bool:
        if self.session_peak_pnl <= 0:
            return False
        drawdown = (self.session_peak_pnl - self.session_pnl) / self.session_peak_pnl
        return drawdown > config.MAX_SESSION_DRAWDOWN_PCT

    async def _cancel_all_open_orders(self) -> None:
        try:
            if config.IS_EQUITY and self.trading is not None:
                # Alpaca's cancel-all is sync. Run it off the event loop so
                # the strategy loop isn't blocked by network latency.
                await asyncio.to_thread(self.trading.cancel_orders)
            elif hasattr(self.exchange, "cancel_all_orders"):
                await self.exchange.cancel_all_orders(config.SYMBOL)
        except Exception as e:
            log.warning("cancel_all_orders failed: %s", e)

    # ---------------------------------------------------- lifecycle
    async def run(self) -> None:
        log.info("engine starting — symbol=%s exchange=%s shadow=%s",
                 config.SYMBOL, config.EXCHANGE_ID, self.shadow_mode)
        self._tasks = [
            asyncio.create_task(self.sensors.run(), name="sensors"),
            asyncio.create_task(self.shadow_sim.run(), name="shadow"),
            asyncio.create_task(self._strategy_loop(), name="strategy"),
            asyncio.create_task(self.dashboard.run(), name="dashboard"),
        ]
        await self._stop_event.wait()
        await self._shutdown()

    async def _shutdown(self) -> None:
        log.info("engine shutdown initiated")
        config.TRADING_ENABLED = False
        self.sensors.stop()
        self.shadow_sim.stop()
        self.dashboard.stop()
        await self._cancel_all_open_orders()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            if config.IS_EQUITY:
                # alpaca-py's StockDataStream uses .stop_ws() / .close() depending
                # on version. Try both, swallow the rest.
                close = getattr(self.exchange, "stop_ws", None) or \
                        getattr(self.exchange, "close", None)
                if close is not None:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
            else:
                await self.exchange.close()
        except Exception:
            pass
        log.info("engine shutdown complete")

    def request_stop(self) -> None:
        self._stop_event.set()


# ============================================================================
# Entry point
# ============================================================================
async def amain() -> None:
    engine = Engine()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.request_stop)
        except NotImplementedError:
            # Windows: signal.signal() is the only path.
            pass
    try:
        await engine.run()
    finally:
        # Ensures the session is ALWAYS closed, even on DNS errors or Ctrl+C.
        log.info("Closing exchange resources...")
        await engine._shutdown()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        log.info("interrupted")
    except Exception as e:
        log.exception("fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
