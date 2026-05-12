"""
probe_bitfinex_ws.py — diagnostic probe for Bitfinex public WS feed.

Connects, subscribes to the `book` (raw L3, prec=R0) and `trades`
channels for a configurable symbol, prints all incoming messages for
a bounded duration, and summarizes counts. Used to verify Bitfinex's
public WS is reachable from this network and to learn the exact wire
format (snapshot, single-update, delete-by-zero-price) before writing
the production harvester.

Bitfinex public WS endpoint: wss://api-pub.bitfinex.com/ws/2
No authentication required for the `book`/`trades` channels.

Usage:
    python probe_bitfinex_ws.py --symbol tBTCUSD --duration 10
    python probe_bitfinex_ws.py --symbol tBTCUSD --duration 30 --no-trades
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter

import websockets


WS_URI = "wss://api-pub.bitfinex.com/ws/2"


async def probe(args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    total = 0
    start = time.monotonic()
    deadline = start + args.duration

    print(f"[probe] connecting to {WS_URI}", flush=True)
    try:
        async with websockets.connect(
            WS_URI,
            ping_interval=20,
            ping_timeout=10,
            max_size=None,
        ) as ws:
            subs = [{
                "event": "subscribe",
                "channel": "book",
                "symbol": args.symbol,
                "prec": "R0",
                "len": "25",
            }]
            if not args.no_trades:
                subs.append({
                    "event": "subscribe",
                    "channel": "trades",
                    "symbol": args.symbol,
                })
            for s in subs:
                await ws.send(json.dumps(s))
                print(f"[probe] sent subscribe: {s}", flush=True)

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                total += 1
                category = "<unknown>"
                try:
                    msg = json.loads(raw)
                    if isinstance(msg, dict):
                        category = f"event:{msg.get('event', '<noevent>')}"
                    elif isinstance(msg, list):
                        # [CHANID, payload] or [CHANID, "hb"] heartbeat
                        if len(msg) >= 2 and msg[1] == "hb":
                            category = "hb"
                        elif len(msg) >= 2 and isinstance(msg[1], list):
                            # Snapshot if payload is list of lists; update if list of scalars
                            if msg[1] and isinstance(msg[1][0], list):
                                category = "snapshot"
                            else:
                                category = "update"
                        elif len(msg) >= 3 and isinstance(msg[1], str):
                            # Trades: [CHANID, "te"|"tu", [...]]
                            category = f"trade:{msg[1]}"
                        else:
                            category = "list-other"
                except json.JSONDecodeError:
                    category = "<non-json>"

                counts[category] += 1
                if category not in samples:
                    samples[category] = raw[: args.per_msg_chars]
                truncated = (
                    raw if len(raw) <= args.per_msg_chars
                    else raw[: args.per_msg_chars] + "..."
                )
                print(f"[{total:04d}] {category:<22} | {truncated}", flush=True)
    except (websockets.WebSocketException, OSError) as e:
        print(f"[probe] WS error: {type(e).__name__}: {e}", flush=True)
    except KeyboardInterrupt:
        print("[probe] interrupted; printing summary", flush=True)
    finally:
        elapsed = time.monotonic() - start
        print("\n" + "=" * 60, flush=True)
        rate = total / max(elapsed, 0.001)
        print(
            f"[summary] total={total} elapsed={elapsed:.1f}s rate={rate:.2f}/s",
            flush=True,
        )
        print("[summary] counts (desc):", flush=True)
        for k, v in counts.most_common():
            print(f"  {k:<22} {v:>6}", flush=True)
        print("[samples] one example per category:", flush=True)
        for cat, sample in samples.items():
            print(f"  {cat}: {sample}", flush=True)
        print("=" * 60, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="tBTCUSD",
                   help="Bitfinex symbol with 't' prefix. Default tBTCUSD.")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Seconds to probe. Default 10.")
    p.add_argument("--per-msg-chars", type=int, default=200,
                   help="Truncate each printed message to N chars. Default 200.")
    p.add_argument("--no-trades", action="store_true",
                   help="Skip the trades-channel subscription.")
    args = p.parse_args()
    asyncio.run(probe(args))


if __name__ == "__main__":
    main()
