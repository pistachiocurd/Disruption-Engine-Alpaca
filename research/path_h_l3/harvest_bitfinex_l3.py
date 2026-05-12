"""
harvest_bitfinex_l3.py - Bitfinex public WS L3 (raw book) harvester.

Pivot target after Coinbase Exchange went institutional-only in 2026
and the retail Coinbase Advanced Trade API stopped exposing the `full`
channel. Bitfinex's public book channel at `prec=R0` is the only known
free venue still emitting true order-by-order L3 (individual ORDER_IDs
on every add / modify / cancel). See L3_RESEARCH_PLAN.md for the
strategic justification.

This harvester subscribes to TWO channels per symbol:
  - book(prec=R0)  - adds / modifies / cancels keyed by ORDER_ID
  - trades         - executed trades (with aggressor side via AMOUNT sign)

Both channels' raw messages are interleaved into a single gzipped JSONL
file per UTC date. Each line is the raw WS frame; the chanId -> channel
mapping is reconstructed offline by reading the subscription-ack
messages (also written verbatim). Reconnect-induced chanId changes are
handled because each reconnect emits fresh ack messages into the stream.

Output: <out-dir>/bitfinex_l3_<YYYYMMDD>.jsonl.gz (UTC date in filename).

Tested loops:
  - WS connect -> subscribe (book + trades per symbol) -> stream
  - UTC midnight -> rotate output file (no gap)
  - WS drop -> exponential backoff reconnect, re-subscribe
  - SIGTERM -> drain pending message, close file cleanly
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import websockets


WS_URI = "wss://api-pub.bitfinex.com/ws/2"
DEFAULT_SYMBOLS = ["tBTCUSD", "tETHUSD", "tSOLUSD"]

# Reconnect backoff: 1s -> 2 -> 4 -> 8 -> 16 -> 32 -> 60 (cap)
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0


def utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def setup_logging(log_path: Optional[Path]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    logging.Formatter.converter = time.gmtime


class RotatingGzipWriter:
    """Append-only gzipped JSONL writer.

    Rotates output on UTC date change. Flushes gzip buffer every
    FLUSH_INTERVAL_S seconds so the file is readable mid-run and
    crash recovery loses at most the last few seconds of events
    (Python's gzip.flush defaults to Z_SYNC_FLUSH, which is
    recoverable on partial-file reads).
    """

    FLUSH_INTERVAL_S = 5.0

    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._current_date = utc_date_str()
        self._fh: Optional[gzip.GzipFile] = None
        self._last_flush_t = 0.0
        self._open_current()

    def _path_for_date(self, date_str: str) -> Path:
        return self.out_dir / f"{self.prefix}_{date_str}.jsonl.gz"

    def _open_current(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self._path_for_date(self._current_date)
        self._fh = gzip.open(path, "at", encoding="utf-8")
        self._last_flush_t = time.monotonic()
        logging.info("writing to %s", path)

    def write(self, line: str) -> None:
        today = utc_date_str()
        if today != self._current_date:
            logging.info(
                "UTC date rolled %s -> %s; rotating output file",
                self._current_date, today,
            )
            self._current_date = today
            self._open_current()
        self._fh.write(line + "\n")
        now = time.monotonic()
        if now - self._last_flush_t >= self.FLUSH_INTERVAL_S:
            self._fh.flush()
            self._last_flush_t = now

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def build_subscribe_messages(symbols: list[str], skip_trades: bool) -> list[dict]:
    """Two subscribe messages per symbol: book (raw L3) and trades."""
    msgs: list[dict] = []
    for sym in symbols:
        msgs.append({
            "event": "subscribe",
            "channel": "book",
            "symbol": sym,
            "prec": "R0",
            "len": "25",
        })
        if not skip_trades:
            msgs.append({
                "event": "subscribe",
                "channel": "trades",
                "symbol": sym,
            })
    return msgs


async def harvest(args: argparse.Namespace) -> None:
    writer = RotatingGzipWriter(args.out_dir, args.prefix)
    subscribes = build_subscribe_messages(args.symbols, args.skip_trades)
    backoff = INITIAL_BACKOFF_S
    event_count = 0
    byte_count = 0
    last_status_t = time.monotonic()

    shutdown = asyncio.Event()

    def _on_signal(_sig: int, _frame: object) -> None:
        logging.info("signal received; draining and exiting")
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, AttributeError):
            pass  # Windows / nested loop - best-effort

    try:
        while not shutdown.is_set():
            try:
                logging.info("connecting to %s", WS_URI)
                async with websockets.connect(
                    WS_URI,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=None,
                ) as ws:
                    for sub in subscribes:
                        await ws.send(json.dumps(sub))
                    logging.info(
                        "subscribed: %d channels across symbols=%s",
                        len(subscribes), args.symbols,
                    )
                    backoff = INITIAL_BACKOFF_S  # reset on success

                    pending_subs = {
                        (s["channel"], s["symbol"]) for s in subscribes
                    }
                    sub_errors: list[str] = []

                    while not shutdown.is_set():
                        message = await ws.recv()
                        writer.write(message)
                        event_count += 1
                        byte_count += len(message)

                        # Track subscription confirmations and errors so we
                        # don't silently run with a half-broken subscription.
                        try:
                            parsed = json.loads(message)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            ev = parsed.get("event")
                            if ev == "subscribed":
                                key = (parsed.get("channel"), parsed.get("symbol"))
                                pending_subs.discard(key)
                                logging.info(
                                    "ack: %s %s chanId=%s",
                                    parsed.get("channel"),
                                    parsed.get("symbol"),
                                    parsed.get("chanId"),
                                )
                                if not pending_subs and not sub_errors:
                                    logging.info("all subscriptions confirmed")
                            elif ev == "error":
                                err = f"{parsed.get('msg')} (code={parsed.get('code')})"
                                sub_errors.append(err)
                                logging.error("bitfinex error: %s", err)
                            elif ev == "info":
                                code = parsed.get("code")
                                if code:
                                    # 20051 reconnect, 20060 maintenance start,
                                    # 20061 maintenance end
                                    logging.warning(
                                        "info code=%s msg=%s",
                                        code, parsed.get("msg"),
                                    )

                        now = time.monotonic()
                        if now - last_status_t >= args.status_interval_s:
                            elapsed = now - last_status_t
                            rate = event_count / max(elapsed, 0.001)
                            logging.info(
                                "status: %d events (%.1f/s), %d KB written in last %.0fs",
                                event_count, rate, byte_count // 1024, elapsed,
                            )
                            event_count = 0
                            byte_count = 0
                            last_status_t = now

            except (websockets.ConnectionClosed,
                    websockets.WebSocketException,
                    OSError) as e:
                logging.warning(
                    "WS lost (%s); reconnect in %.1fs", e, backoff,
                )
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=backoff)
                    break  # shutdown signaled during backoff
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, MAX_BACKOFF_S)
            except Exception as e:
                logging.exception("unexpected: %s", e)
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=backoff)
                    break
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, MAX_BACKOFF_S)
    finally:
        writer.close()
        logging.info("harvester exiting cleanly")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, default=Path("./l3_data"),
                   help="Directory for gz output files. Default ./l3_data")
    p.add_argument("--prefix", default="bitfinex_l3",
                   help="Filename prefix. Default 'bitfinex_l3'.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   help="Bitfinex symbols with 't' prefix "
                        "(default tBTCUSD tETHUSD tSOLUSD).")
    p.add_argument("--skip-trades", action="store_true",
                   help="Skip the trades-channel subscription "
                        "(book-only mode; ~half the message volume).")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional log file (in addition to stderr).")
    p.add_argument("--status-interval-s", type=float, default=300.0,
                   help="Log harvester rate/bytes every N seconds. Default 300 (5 min).")
    args = p.parse_args()
    setup_logging(args.log_file)
    asyncio.run(harvest(args))


if __name__ == "__main__":
    main()
