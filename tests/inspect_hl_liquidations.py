"""
inspect_hl_liquidations.py - Identify the liquidation marker in HL trade events.

Hyperliquid's `node_fills_by_block` archive contains all fills with a
`dir` field that describes the user's directional intent. Liquidations
are surfaced via this field with a distinct value (commonly "Liquidate
Long" / "Liquidate Short" or similar). This scanner downloads a few
hours of trade data, flattens block events, and reports:
  - unique `dir` values + counts
  - representative samples of rare dirs (likely liquidation markers)
  - a heuristic check on `closedPnl` for events with rare dirs

Run before wiring up the HL liquidation adapter so we know which `dir`
values to flag. ~30 seconds per hour scanned.

Usage:
    python tests/inspect_hl_liquidations.py
    python tests/inspect_hl_liquidations.py --date 20260405 --hours 0,1,2,3
    python tests/inspect_hl_liquidations.py --coin BTC --hours 0,1
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fetch_history_hyperliquid import (
    HLArchiveClient,
    NODE_BUCKET,
    _flatten_block_events,
)


def scan_hour(client: HLArchiveClient, yyyymmdd: str, hour: int,
              coin_filter: str = None) -> dict:
    """Return aggregated stats for one hour: dir counts, trade_dir_override
    counts, top counterparty addresses, samples by dir."""
    key = f"node_fills_by_block/hourly/{yyyymmdd}/{hour}.lz4"
    print(f"  scanning {key} ...", flush=True)
    dir_counts: Counter = Counter()
    override_counts: Counter = Counter()
    user_addr_counts: Counter = Counter()
    samples_by_dir: dict[str, list] = defaultdict(list)
    samples_by_override: dict[str, list] = defaultdict(list)
    n_events = 0
    try:
        for block in client.get_lz4_jsonl(NODE_BUCKET, key):
            for user_addr, fill in _flatten_block_events(block):
                if coin_filter and fill.get("coin") != coin_filter:
                    continue
                n_events += 1
                d = fill.get("dir", "<missing>")
                dir_counts[d] += 1
                tdo = str(fill.get("trade_dir_override", "<missing>"))
                override_counts[tdo] += 1
                # Track user (the address whose perspective this fill is from)
                # plus side_info participants
                if user_addr:
                    user_addr_counts[user_addr] += 1
                side_info = fill.get("side_info") or []
                for entry in side_info:
                    if isinstance(entry, dict):
                        u = entry.get("user")
                        if u:
                            user_addr_counts[u] += 1
                if len(samples_by_dir[d]) < 3:
                    samples_by_dir[d].append({
                        "coin": fill.get("coin"),
                        "side": fill.get("side"),
                        "px": fill.get("px"),
                        "sz": fill.get("sz"),
                        "startPosition": fill.get("startPosition"),
                        "closedPnl": fill.get("closedPnl"),
                        "crossed": fill.get("crossed"),
                        "dir": fill.get("dir"),
                        "trade_dir_override": fill.get("trade_dir_override"),
                        "side_info_users": [
                            (e.get("user") or "")[:10] + "..." for e in side_info
                            if isinstance(e, dict)
                        ],
                    })
                if tdo != "Na" and tdo != "<missing>" and len(samples_by_override[tdo]) < 5:
                    samples_by_override[tdo].append({
                        "coin": fill.get("coin"),
                        "dir": fill.get("dir"),
                        "trade_dir_override": tdo,
                        "side": fill.get("side"),
                        "px": fill.get("px"),
                        "sz": fill.get("sz"),
                        "startPosition": fill.get("startPosition"),
                        "closedPnl": fill.get("closedPnl"),
                        "side_info_users": [
                            (e.get("user") or "")[:10] + "..." for e in side_info
                            if isinstance(e, dict)
                        ],
                    })
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            print(f"    (missing — skipped)")
        else:
            print(f"    ERROR: {e}")
    print(f"    {n_events:,} events scanned, {len(dir_counts)} unique dirs, "
          f"{len(override_counts)} unique trade_dir_overrides")
    return {
        "dir_counts": dir_counts,
        "override_counts": override_counts,
        "user_addr_counts": user_addr_counts,
        "samples_by_dir": samples_by_dir,
        "samples_by_override": samples_by_override,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD (default: 35 days ago)")
    parser.add_argument("--hours", default="0,6,12,18",
                        help="Comma-separated hours to scan (default 4 spread "
                             "across the day to capture different volume regimes)")
    parser.add_argument("--coin", default=None,
                        help="Filter to one coin (e.g. BTC). Default: all coins, "
                             "which is more representative for HL since "
                             "liquidations are alts-heavy.")
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now(timezone.utc).date() - timedelta(days=35)).strftime("%Y%m%d")

    hours = [int(h) for h in args.hours.split(",")]
    print(f"\n=== HL trade dir-value scan: date={target_date}, hours={hours}, "
          f"coin={'all' if args.coin is None else args.coin} ===\n")

    client = HLArchiveClient.make()

    total_dir: Counter = Counter()
    total_override: Counter = Counter()
    total_users: Counter = Counter()
    all_dir_samples: dict[str, list] = defaultdict(list)
    all_override_samples: dict[str, list] = defaultdict(list)
    for h in hours:
        result = scan_hour(client, target_date, h, args.coin)
        total_dir.update(result["dir_counts"])
        total_override.update(result["override_counts"])
        total_users.update(result["user_addr_counts"])
        for d, samples in result["samples_by_dir"].items():
            for sample in samples:
                if len(all_dir_samples[d]) < 3:
                    all_dir_samples[d].append(sample)
        for o, samples in result["samples_by_override"].items():
            for sample in samples:
                if len(all_override_samples[o]) < 5:
                    all_override_samples[o].append(sample)

    print()
    print("=== Aggregated dir value counts ===")
    total = sum(total_dir.values())
    for d, count in total_dir.most_common():
        pct = 100.0 * count / max(total, 1)
        print(f"  {count:>10,}  ({pct:>5.2f}%)  dir={d!r}")

    print()
    print("=== Aggregated trade_dir_override counts ===")
    for o, count in total_override.most_common():
        pct = 100.0 * count / max(total, 1)
        print(f"  {count:>10,}  ({pct:>5.2f}%)  trade_dir_override={o!r}")

    print()
    print("=== Top 15 counterparty addresses (by participation count) ===")
    print("(High-frequency addresses are likely market makers OR HL's liquidator vault)")
    for u, count in total_users.most_common(15):
        pct = 100.0 * count / max(total, 1)
        print(f"  {count:>10,}  ({pct:>5.2f}%)  user={u}")

    print()
    print("=== Samples for each non-Na trade_dir_override ===")
    if not all_override_samples:
        print("  (none — all trade_dir_override values are 'Na')")
    for o in sorted(all_override_samples.keys()):
        print(f"\n  trade_dir_override={o!r}  (count={total_override[o]:,})")
        for s in all_override_samples[o]:
            print(f"    coin={s['coin']!r:<10}  dir={s['dir']!r}  side={s['side']!r}  "
                  f"px={s['px']}  sz={s['sz']}  "
                  f"startPos={s['startPosition']}  closedPnl={s['closedPnl']}")
            print(f"      side_info_users: {s['side_info_users']}")

    print()
    print("=== Liquidation-marker hypothesis check ===")
    likely_liq_dirs = [d for d in total_dir if "liquid" in d.lower()]
    likely_liq_overrides = [o for o in total_override if o not in ("Na", "<missing>")]
    if likely_liq_dirs:
        print(f"  Found dir values with 'liquid' in the name: {likely_liq_dirs}")
    elif likely_liq_overrides:
        print(f"  Non-Na trade_dir_override values present: {likely_liq_overrides}")
        print(f"  These are likely the liquidation markers. Update the harvester's")
        print(f"  is_hl_liquidation() to check trade_dir_override instead of dir.")
    else:
        print(f"  No explicit liquidation marker found in `dir` or `trade_dir_override`.")
        print(f"  Liquidations on HL are likely identifiable only via:")
        print(f"    1. side_info[*].user matching HL's known liquidator vault address")
        print(f"    2. The separate hyperliquid-dex/historical_data/liquidations.csv")
        print(f"  See top counterparty addresses above for candidate liquidator addresses")
        print(f"  (look for ones with disproportionate volume given they're not MMs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
