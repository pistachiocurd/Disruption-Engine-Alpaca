"""
dashboard.py — In-process aiohttp server exposing engine state for the live UI.

Mounted as a 4th asyncio task by engine.run(). Serves:
    GET  /                   → dashboard.html
    GET  /state              → JSON snapshot of current engine state
    POST /api/threshold      → set the live turbulence threshold (no restart)

Localhost-only by default. Override with $env:DASHBOARD_HOST / $env:DASHBOARD_PORT.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from aiohttp import web

import config

log = logging.getLogger(__name__)

DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))


def _bid(ob: Any) -> float | None:
    if ob and ob.get("bids"):
        return float(ob["bids"][0][0])
    return None


def _ask(ob: Any) -> float | None:
    if ob and ob.get("asks"):
        return float(ob["asks"][0][0])
    return None


def _drawdown_usd(engine) -> float:
    """Absolute dollar drawdown from peak. Always meaningful, never pathological."""
    return max(0.0, engine.session_peak_pnl - engine.session_pnl)


def _drawdown_pct(engine) -> float:
    """Percentage drawdown — only meaningful once peak exceeds the floor below
    which the ratio explodes (a $0.91 peak with -$9 PnL is not "1000% drawdown",
    it's noise on a tiny base). Mirrors the gate logic in
    engine._max_drawdown_breached so the display agrees with the kill-switch."""
    if engine.session_peak_pnl < config.DRAWDOWN_PCT_MIN_PEAK_USD:
        return 0.0
    if engine.session_peak_pnl <= 0:
        return 0.0
    return (engine.session_peak_pnl - engine.session_pnl) / engine.session_peak_pnl


def _mtm_drawdown_usd(engine, mid: float | None) -> float:
    """Absolute MTM drawdown from peak. Returns 0 when no mid is available
    (warmup, between sessions, etc.) — matches the gate behavior in
    engine._mtm_drawdown_breached."""
    if mid is None:
        return 0.0
    mtm_pnl = float(getattr(engine, "_cash", 0.0)) + float(getattr(engine, "_position", 0.0)) * mid
    peak = float(getattr(engine, "_mtm_peak_pnl", 0.0))
    return max(0.0, peak - mtm_pnl)


def _mtm_drawdown_pct(engine, mid: float | None) -> float:
    """Percentage MTM drawdown — same floor logic as the IS gate."""
    if mid is None:
        return 0.0
    peak = float(getattr(engine, "_mtm_peak_pnl", 0.0))
    if peak < config.DRAWDOWN_PCT_MIN_PEAK_USD or peak <= 0:
        return 0.0
    mtm_pnl = float(getattr(engine, "_cash", 0.0)) + float(getattr(engine, "_position", 0.0)) * mid
    return max(0.0, (peak - mtm_pnl) / peak)


def _episode_to_dict(rec: dict) -> dict:
    # Episode records are already dicts of primitives.
    return rec


class DashboardServer:
    """Lightweight in-process aiohttp server exposing engine state."""

    def __init__(
        self,
        engine,
        host: str = DASHBOARD_HOST,
        port: int = DASHBOARD_PORT,
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self._runner: web.AppRunner | None = None
        self._stop_event = asyncio.Event()

    # ---------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        e = self.engine
        s = e.sensors.state
        sh = e.shadow_sim
        alpha = e.alpha

        # Theoretical mark-to-market PnL.
        ob = s.ob_snapshot
        mid = None
        if ob and ob.get("bids") and ob.get("asks"):
            mid = 0.5 * (float(ob["bids"][0][0]) + float(ob["asks"][0][0]))
        cash = float(getattr(e, "_cash", 0.0))
        position = float(getattr(e, "_position", 0.0))
        total_fees = float(getattr(e, "_total_fees", 0.0))
        total_notional = float(getattr(e, "_total_filled_notional", 0.0))
        mtm_value = position * mid if mid is not None else 0.0
        theoretical_pnl = cash + mtm_value

        return {
            "ts_ms": int(time.time() * 1000),
            "engine": {
                "symbol": config.SYMBOL,
                "exchange_id": config.EXCHANGE_ID,
                "shadow_mode": e.shadow_mode,
                "trading_enabled": config.TRADING_ENABLED,
                "exchange_live": config.EXCHANGE_LIVE,
                "calibration_stale": e.drift_monitor.is_stale,
                "drift_mape": float(e.drift_monitor.latest_mape),
                "session_pnl": float(e.session_pnl),
                "session_peak_pnl": float(e.session_peak_pnl),
                "drawdown_pct": float(_drawdown_pct(e)),
                "drawdown_usd": float(_drawdown_usd(e)),
                "max_drawdown_pct": float(config.MAX_SESSION_DRAWDOWN_PCT),
                "max_drawdown_usd": float(config.MAX_SESSION_DRAWDOWN_USD),
                "drawdown_pct_min_peak_usd": float(config.DRAWDOWN_PCT_MIN_PEAK_USD),
                # MTM drawdown — see engine._mtm_drawdown_breached. Distinct
                # from the IS drawdown above because IS only sees execution
                # cost while MTM sees directional exposure losses.
                "mtm_peak_pnl": float(getattr(e, "_mtm_peak_pnl", 0.0)),
                "mtm_drawdown_usd": float(_mtm_drawdown_usd(e, mid)),
                "mtm_drawdown_pct": float(_mtm_drawdown_pct(e, mid)),
                "max_mtm_drawdown_usd": float(config.MAX_MTM_DRAWDOWN_USD),
                "max_mtm_drawdown_pct": float(config.MAX_MTM_DRAWDOWN_PCT),
                "is_equity": bool(config.IS_EQUITY),
                "max_position_limit": float(config.MAX_POSITION_LIMIT),
                # Theoretical PnL (would-be fills MTM'd at live mid).
                "pnl_theoretical": float(theoretical_pnl),
                "pnl_cash": float(cash),
                "pnl_position": float(position),
                "pnl_position_value": float(mtm_value),
                "pnl_total_fees": float(total_fees),
                "pnl_total_notional": float(total_notional),
                "pnl_mid": float(mid) if mid is not None else None,
            },
            "physics_state": {
                "vpin": float(s.vpin),
                "alpha_calibrated": float(s.alpha_calibrated),
                "viscosity": float(s.viscosity),
                "regime": int(s.regime),
                "regime_label": ["laminar", "transition", "turbulent"][s.regime],
                "ce_ratio": float(s.ce_ratio),
                "obi": float(s.obi),
                "spread_velocity": float(s.spread_velocity),
                "liquidation_rate": float(s.liquidation_rate),
                "ood_flag": bool(s.ood_flag),
                "mahal_dist": float(s.mahal_dist),
                "ood_threshold": float(config.OOD_THRESHOLD),
                "timestamp_ms": int(s.timestamp),
                "best_bid": _bid(s.ob_snapshot),
                "best_ask": _ask(s.ob_snapshot),
            },
            "alpha": {
                "last_turbulence": float(getattr(alpha, "last_turbulence", 0.0)),
                # Live threshold (mutable via POST /api/threshold).
                "turbulence_threshold": float(getattr(alpha, "turbulence_threshold", config.TURBULENCE_THRESHOLD)),
                "turbulence_threshold_source": getattr(e, "_turbulence_threshold_source", "config"),
                # Trained-on-data PR curve, if a sidecar exists.
                "tcn_threshold_payload": getattr(e, "_tcn_threshold_payload", None),
                "decisions": list(getattr(alpha, "decision_log", []))[-50:],
                "tcn_buffer_size": len(alpha._buffer),
                "tcn_buffer_target": int(config.TCN_INPUT_LENGTH),
            },
            "execution": {
                "recent_episodes": list(getattr(e, "_recent_episodes", []))[-10:],
                "recent_orders": list(getattr(e, "_recent_orders", []))[-100:],
                "total_orders": len(getattr(e, "_recent_orders", [])),
                "total_episodes_run": int(getattr(e, "_next_episode_id", 0)),
            },
            "shadow": {
                "replay_size": len(sh.replay),
                "steps_since_last_push": int(sh.steps_since_last_push),
                "min_shadow_steps": int(config.MIN_SHADOW_STEPS),
                "kl_threshold": float(config.KL_THRESHOLD),
                "device": str(sh.device),
            },
        }

    # ----------------------------------------------------------- handlers
    async def state_handler(self, request: web.Request) -> web.Response:
        try:
            data = self.snapshot()
        except Exception as exc:
            log.exception("snapshot failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response(data)

    async def index_handler(self, request: web.Request) -> web.Response:
        html_path = Path(__file__).parent / "dashboard.html"
        if not html_path.exists():
            return web.Response(text="dashboard.html not found", status=404)
        return web.FileResponse(html_path)

    async def vendor_handler(self, request: web.Request) -> web.Response:
        """Serve files from ./vendor/ — used to host Chart.js locally so the
        dashboard works without external CDN requests (some browsers' tracking
        prevention blocks jsdelivr/unpkg)."""
        name = request.match_info["name"]
        # Reject path traversal.
        if "/" in name or "\\" in name or ".." in name:
            return web.Response(status=400, text="invalid filename")
        path = Path(__file__).parent / "vendor" / name
        if not path.exists() or not path.is_file():
            return web.Response(status=404, text=f"vendor/{name} not found")
        return web.FileResponse(path)

    async def threshold_handler(self, request: web.Request) -> web.Response:
        """POST /api/threshold {threshold: float, persist?: bool}"""
        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"invalid JSON: {exc}"}, status=400)
        thresh = body.get("threshold")
        persist = bool(body.get("persist", False))
        try:
            thresh = float(thresh)
        except (TypeError, ValueError):
            return web.json_response({"error": "threshold must be a number"}, status=400)
        if not (0.0 <= thresh <= 1.0):
            return web.json_response({"error": "threshold must be in [0, 1]"}, status=400)

        # Live update — takes effect on the next strategy tick. No restart.
        self.engine.alpha.turbulence_threshold = thresh
        self.engine._turbulence_threshold_source = "manual"

        persisted_to: str | None = None
        if persist:
            try:
                payload = self.engine._tcn_threshold_payload or {}
                payload = dict(payload)  # shallow copy
                payload["threshold"] = thresh
                payload["manually_set_at_ms"] = int(time.time() * 1000)
                path = Path(config.TCN_THRESHOLD_PATH)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2))
                self.engine._tcn_threshold_payload = payload
                persisted_to = str(path)
                log.info("threshold %.4f persisted to %s (manual)", thresh, path)
            except Exception as exc:
                log.exception("failed to persist threshold: %s", exc)
                return web.json_response({
                    "ok": True,
                    "threshold": thresh,
                    "persisted": False,
                    "persist_error": str(exc),
                })

        log.info("threshold updated to %.4f via dashboard", thresh)
        return web.json_response({
            "ok": True,
            "threshold": thresh,
            "persisted": persist,
            "persisted_to": persisted_to,
        })

    # ---------------------------------------------------------- lifecycle
    async def run(self) -> None:
        app = web.Application()
        app.router.add_get("/", self.index_handler)
        app.router.add_get("/state", self.state_handler)
        app.router.add_get("/vendor/{name}", self.vendor_handler)
        app.router.add_post("/api/threshold", self.threshold_handler)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host=self.host, port=self.port)
        try:
            await site.start()
            self._runner = runner
            log.info("dashboard listening on http://%s:%d", self.host, self.port)
            await self._stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("dashboard server failed: %s", exc)
        finally:
            await runner.cleanup()
            log.info("dashboard server stopped")

    def stop(self) -> None:
        self._stop_event.set()
