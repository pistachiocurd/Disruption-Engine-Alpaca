"""
fetch_history_alpaca.py — Historical harvester for the equities pivot.

Replays the last N trading days of Alpaca quote and trade data through the
live SensorArray code path (driving _process_order_book / _process_trade
directly), so the resulting feature_history_*.csv is byte-identical to what
the live FeatureDumper produces. This means check_shocks.py, fit_ood_from_csv.py
and train_tcn.py can run today on real-market data without waiting for live
capture.

Usage:
    python fetch_history_alpaca.py --symbol NVDA --days 14

Environment:
    EXCHANGE_API_KEY, EXCHANGE_SECRET — Alpaca paper or live keys
    ALPACA_DATA_FEED                  — "iex" (free) or "sip" (paid)
"""
from __future__ import annotations

import argparse
import heapq
import logging
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import config
from layer1_sensors import FeatureDumper, SensorArray, StudentTHMM

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch_history")

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

def _resolve_session_window(cal_entry, market_tz):
    """Return (open_utc, close_utc) for a single Alpaca calendar entry."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(market_tz)
    d = cal_entry.date if hasattr(cal_entry, "date") else cal_entry["date"]
    op = cal_entry.open if hasattr(cal_entry, "open") else cal_entry["open"]
    cl = cal_entry.close if hasattr(cal_entry, "close") else cal_entry["close"]

    if isinstance(d, str):
        d = date.fromisoformat(d)
    elif isinstance(d, datetime):
        d = d.date()

    if isinstance(op, str):
        op = time.fromisoformat(op)
    elif isinstance(op, datetime):
        op = op.time()

    if isinstance(cl, str):
        cl = time.fromisoformat(cl)
    elif isinstance(cl, datetime):
        cl = cl.time()

    open_dt = datetime.combine(d, op, tzinfo=tz).astimezone(timezone.utc)
    close_dt = datetime.combine(d, cl, tzinfo=tz).astimezone(timezone.utc)
    return open_dt, close_dt

def _fetch_calendar(trading_client, days: int):
    from alpaca.trading.requests import GetCalendarRequest
    today = datetime.now(timezone.utc).date()
    req = GetCalendarRequest(start=today - timedelta(days=days + 5), end=today)
    cal = trading_client.get_calendar(req)
    # Keep only the last `days` sessions ending strictly before "now" (don't
    # try to harvest a still-in-progress session).
    sessions = []
    now_utc = datetime.now(timezone.utc)
    for entry in cal:
        op, cl = _resolve_session_window(entry, config.MARKET_TZ)
        if cl < now_utc:
            sessions.append((op, cl))
    return sessions[-days:]


def _replay_session(
    sensors: SensorArray,
    hist_client,
    symbol: str,
    open_utc: datetime,
    close_utc: datetime,
    feed: str,
) -> tuple[int, int]:
    """Replay one trading session. Returns (n_quotes, n_trades) consumed."""
    from alpaca.data.requests import StockQuotesRequest, StockTradesRequest

    quote_req = StockQuotesRequest(
        symbol_or_symbols=symbol, start=open_utc, end=close_utc, feed=feed,
    )
    trade_req = StockTradesRequest(
        symbol_or_symbols=symbol, start=open_utc, end=close_utc, feed=feed,
    )

    quotes_resp = hist_client.get_stock_quotes(quote_req)
    trades_resp = hist_client.get_stock_trades(trade_req)

    # alpaca-py returns either a dict-like keyed by symbol or a Quote/TradeSet
    # depending on the version. Normalize.
    def _list_for(resp, sym):
        if hasattr(resp, "data"):
            return resp.data.get(sym, [])
        if isinstance(resp, dict):
            return resp.get(sym, [])
        return resp[sym] if sym in resp else []

    quotes = _list_for(quotes_resp, symbol)
    trades = _list_for(trades_resp, symbol)

    n_q = n_t = 0

    def _q_events():
        nonlocal n_q
        for q in quotes:
            n_q += 1
            yield (q.timestamp, "q", q)

    def _t_events():
        nonlocal n_t
        for t in trades:
            n_t += 1
            yield (t.timestamp, "t", t)

    events_processed = 0
    for ts, kind, payload in heapq.merge(
        _q_events(), _t_events(), key=lambda e: e[0]
    ):
        events_processed += 1
        if events_processed % 100_000 == 0:
            log.info("  ... churning: %s events processed so far in this session", f"{events_processed:,}")

        ts_ms = int(ts.timestamp() * 1000)
        if kind == "q":
            ob = {
                "bids": [[float(payload.bid_price), float(payload.bid_size)]],
                "asks": [[float(payload.ask_price), float(payload.ask_size)]],
                "timestamp": ts_ms,
            }
            sensors._process_order_book(ob)
        else:
            trade = {
                "timestamp": ts_ms,
                "price": float(payload.price),
                "amount": float(payload.size),
            }
            sensors._process_trade(trade)
    return n_q, n_t


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.SYMBOL,
                        help="Equity symbol to harvest (default from config.SYMBOL)")
    parser.add_argument("--days", type=int, default=14,
                        help="Number of past trading sessions to replay")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: per-symbol calibration file)")
    parser.add_argument("--feed", default=config.ALPACA_DATA_FEED,
                        help='Alpaca data feed: "iex" or "sip"')
    args = parser.parse_args(argv)

    if "/" in args.symbol:
        log.error("--symbol must be an equity ticker (no slash). got %r", args.symbol)
        return 2

    out_path = args.out or f"./calibration/feature_history_{args.symbol}.csv"
    out_path = Path(out_path)
    if out_path.exists():
        log.warning("output %s already exists — appending; "
                    "delete the file first for a clean rebuild.", out_path)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
    except ImportError:
        log.error("alpaca-py is required. install via `pip install alpaca-py`")
        return 1

    key = config.EXCHANGE_API_KEY
    secret = config.EXCHANGE_SECRET
    if not key or not secret:
        log.error("EXCHANGE_API_KEY / EXCHANGE_SECRET must be set in the environment")
        return 1

    hist = StockHistoricalDataClient(key, secret)
    trading = TradingClient(key, secret, paper=True)

    sessions = _fetch_calendar(trading, args.days)
    if not sessions:
        log.error("no completed sessions returned for the requested window")
        return 1
    log.info("harvesting %d sessions for %s (feed=%s) → %s",
             len(sessions), args.symbol, args.feed, out_path)

    dumper = FeatureDumper(out_path)
    sensors = SensorArray(
        exchange=None,                     # we drive _process_* directly
        symbol=args.symbol,
        hmm=StudentTHMM.from_default_priors(),
        feature_dumper=dumper,
    )

    total_q = total_t = 0
    for i, (op, cl) in enumerate(sessions, start=1):
        # Mirror the live behavior: every market open begins with a clean
        # sensor state. This is what the trained model will see live.
        sensors.reset_session()
        try:
            n_q, n_t = _replay_session(
                sensors, hist, args.symbol, op, cl, args.feed,
            )
            log.info("session %d/%d %s: %d quotes, %d trades",
                     i, len(sessions), op.date().isoformat(), n_q, n_t)
            total_q += n_q
            total_t += n_t
        except Exception as e:
            log.exception("session %s failed (%s) — continuing", op.date(), e)

    dumper.close()
    log.info("done: %d sessions, %d quotes, %d trades, %d rows in %s",
             len(sessions), total_q, total_t, dumper.rows_written, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
