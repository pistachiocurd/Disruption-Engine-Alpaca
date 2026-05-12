"""
download_tardis_l3_free.py — Tardis.dev free-tier L3 puller (preserved for record).

Initially intended to pull true Level-3 (`level3`) order-by-order data
for Coinbase BTC-USD, ETH-USD, SOL-USD using Tardis.dev's free tier
(first day of every month). Anonymous access empirically returns 404
on `/data-feeds` (2026-05-12); a free Tardis account + API key is
required even for the 1st-of-month tier. Coinbase's `level3` channel
also went auth-only on the live WS feed around the same time. The
project pivoted to Bitfinex public L3 (see harvest_bitfinex_l3.py).
This script is preserved as a working Tardis client for any later
reactivation.

Output: one gzipped JSONL file per (coin, date) at
    calibration/l3_raw/coinbase_l3_<COIN>_<YYYYMMDD>.jsonl.gz

Each line is a JSON object with keys:
    local_timestamp   — Tardis-side capture time (microsecond)
    message           — raw Coinbase WS payload (received/open/done/match/change)

Usage:
    python download_tardis_l3_free.py                       # 5 most recent month-firsts
    python download_tardis_l3_free.py --dates 2026-05-01 2026-04-01
    python download_tardis_l3_free.py --coins BTC-USD       # single coin
    python download_tardis_l3_free.py --out-dir ./l3_data   # custom output

Notes:
    - Requires TARDIS_API_KEY (env var or --api-key). Free account suffices.
    - Tardis caches the raw data to a local cache dir before streaming;
      first run for a (date, coin) will be slower than subsequent runs.
    - Each (coin, day) is a separate task; failures don't abort the run.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from tardis_dev import replay, Channel

DEFAULT_COINS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_OUT_DIR = Path("./calibration/l3_raw")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )


def latest_month_firsts(n: int, reference: date | None = None) -> list[date]:
    """Return the most recent `n` first-of-month dates strictly before today."""
    today = reference or date.today()
    out = []
    cur = date(today.year, today.month, 1)
    if cur >= today:
        # If today IS the 1st, Tardis may not have published it yet —
        # back off to the previous month-first to be safe.
        cur = (cur - timedelta(days=1)).replace(day=1)
    while len(out) < n:
        out.append(cur)
        cur = (cur - timedelta(days=1)).replace(day=1)
    return out


async def download_one(
    coin: str,
    day: date,
    out_dir: Path,
    overwrite: bool = False,
    api_key: str = "",
) -> tuple[bool, int, float]:
    """Download a single (coin, day) slice. Returns (success, n_messages,
    elapsed_s)."""
    out_path = out_dir / f"coinbase_l3_{coin}_{day.strftime('%Y%m%d')}.jsonl.gz"
    if out_path.exists() and not overwrite:
        logging.info("%s exists; skipping (use --overwrite to re-download)", out_path.name)
        return True, 0, 0.0

    from_date = day.isoformat()
    to_date = (day + timedelta(days=1)).isoformat()

    logging.info("downloading %s for %s -> %s", coin, from_date, out_path.name)
    start_t = time.monotonic()
    # Coinbase L3 on Tardis is split into 5 normalized channels (the
    # `level3` aggregate channel does not exist as a Tardis filter).
    # Subscribe to all 5 to reconstruct the full order-by-order stream.
    L3_CHANNELS = ["received", "open", "done", "change", "match"]
    messages = replay(
        exchange="coinbase",
        from_date=from_date,
        to_date=to_date,
        filters=[Channel(name=ch, symbols=[coin]) for ch in L3_CHANNELS],
        api_key=api_key,
    )

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    n = 0
    last_log_t = start_t
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
            async for response in messages:
                if response is None:
                    continue
                msg = response.message
                # message may be dict (after decode) or str
                if not isinstance(msg, dict):
                    try:
                        msg = json.loads(msg)
                    except (TypeError, ValueError):
                        msg = {"raw": str(msg)}
                msg["_tardis_local_ts_us"] = response.local_timestamp
                fh.write(json.dumps(msg, separators=(",", ":")) + "\n")
                n += 1
                now = time.monotonic()
                if now - last_log_t >= 30.0:
                    elapsed = now - start_t
                    rate = n / max(elapsed, 0.001)
                    size_mb = out_path.with_suffix(out_path.suffix + ".part").stat().st_size / 1e6
                    logging.info(
                        "  %s/%s: %d messages (%.0f/s, %.1f MB on disk)",
                        coin, day, n, rate, size_mb,
                    )
                    last_log_t = now
        tmp_path.replace(out_path)
        elapsed = time.monotonic() - start_t
        size_mb = out_path.stat().st_size / 1e6
        logging.info(
            "  %s/%s DONE: %d messages, %.1f MB, %.0fs",
            coin, day, n, size_mb, elapsed,
        )
        return True, n, elapsed
    except Exception as e:
        logging.error("  %s/%s FAILED: %s: %s", coin, day, type(e).__name__, e)
        if tmp_path.exists():
            tmp_path.unlink()
        return False, n, time.monotonic() - start_t


async def main_async(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dates:
        days = [datetime.strptime(d, "%Y-%m-%d").date() for d in args.dates]
    else:
        days = latest_month_firsts(args.n_months)

    coins = args.coins
    total_tasks = len(days) * len(coins)
    logging.info(
        "plan: %d days x %d coins = %d slices -> %s",
        len(days), len(coins), total_tasks, out_dir,
    )
    for d in days:
        logging.info("  day: %s", d.isoformat())
    logging.info("  coins: %s", coins)

    successes = failures = 0
    total_messages = 0
    overall_start = time.monotonic()
    for day in days:
        for coin in coins:
            ok, n, _ = await download_one(coin, day, out_dir, args.overwrite, args.api_key)
            if ok:
                successes += 1
                total_messages += n
            else:
                failures += 1

    elapsed = time.monotonic() - overall_start
    logging.info(
        "complete: %d ok / %d failed, %d total messages, %.0fs",
        successes, failures, total_messages, elapsed,
    )


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-months", type=int, default=5,
                   help="Number of most-recent month-firsts to fetch. Default 5.")
    p.add_argument("--dates", nargs="+", default=None,
                   help="Override: explicit list of YYYY-MM-DD dates "
                        "(typically the first of each month for Tardis free tier).")
    p.add_argument("--coins", nargs="+", default=DEFAULT_COINS,
                   help="Coinbase product IDs to fetch.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output directory for gzipped JSONL files.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download even if the output file already exists.")
    p.add_argument("--api-key", default="",
                   help="Tardis API key. Free account suffices for "
                        "1st-of-month tier. Can also be set via TARDIS_API_KEY "
                        "env var.")
    args = p.parse_args()
    import os
    if not args.api_key:
        args.api_key = os.environ.get("TARDIS_API_KEY", "")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
