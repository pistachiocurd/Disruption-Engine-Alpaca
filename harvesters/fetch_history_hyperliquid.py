"""
fetch_history_hyperliquid.py — Historical harvester for the Hyperliquid perps path.

Replays L2 book snapshots and trades from Hyperliquid's public S3 archives
through the live SensorArray code path so the resulting feature_history_*.csv
is byte-identical to live FeatureDumper output. Same train/serve parity
guarantee as fetch_history_alpaca.py — this module does NOT reimplement
feature computation; it drives _process_order_book / _process_trade directly.

Two buckets, both Requester Pays:
    s3://hyperliquid-archive/market_data/[YYYYMMDD]/[hour]/l2Book/[COIN].lz4
    s3://hl-mainnet-node-data/node_fills_by_block/...        (path TBD; --probe)

Update cadence is ~monthly with no SLA — backtests are inherently month-stale.

Usage:
    # 1. Verify the schema before committing to a multi-day download.
    python fetch_history_hyperliquid.py --coin BTC --probe

    # 2. Harvest one day to validate the pipeline end-to-end.
    python fetch_history_hyperliquid.py --coin BTC --start-date 20260415 --end-date 20260415

    # 3. Full multi-day harvest.
    python fetch_history_hyperliquid.py --coin BTC --days 14

Environment:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY — boto3 picks these up
    AWS_REGION (optional)                   — default us-east-1
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import heapq
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.request import urlretrieve

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from layer1_sensors import FeatureDumper, SensorArray, StudentTHMM  # noqa: E402

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch_hl")
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

ARCHIVE_BUCKET = "hyperliquid-archive"
NODE_BUCKET = "hl-mainnet-node-data"
NODE_FILLS_PREFIX = "node_fills_by_block"

# HL surfaces liquidations in a separate official repo, not in node_fills_by_block.
# Verified empirically (see tests/inspect_hl_liquidations.py): no `dir` value
# starting with "Liquidate", `trade_dir_override` always 'Na'/missing, no
# dominant counterparty address indicating a liquidator vault. The labeled
# source is this CSV. Schema:
#     time(ISO8601 with Z),user(0x...),liquidated_ntl_pos(USD),
#     liquidated_account_value(USD),leverage_type(Cross|Isolated)
# Note: no `coin` field — events are aggregate cross-market notional.
# Treated as a market-wide stress signal injected uniformly across per-coin
# harvests. Per-coin VPIN normalization in LiquidationCascadeTracker.
HL_LIQUIDATIONS_CSV_URL = (
    "https://raw.githubusercontent.com/hyperliquid-dex/historical_data/master/liquidations.csv"
)
HL_LIQUIDATIONS_CACHE = "./calibration/hl_liquidations.csv"


# ============================================================================
# Schema converters
# ============================================================================

def _to_ms(t) -> int:
    """Coerce HL's timestamp encodings to epoch ms.

    Two encodings observed in HL archives:
    - int / float epoch ms (the inner `data.time` field) — passes through.
    - ISO-8601 string with up to 9 fractional digits (the wrapper `time`
      field is nanosecond-precision, e.g. "2026-04-05T00:00:10.661552759").
      Python's `datetime.fromisoformat` only accepts up to 6 fractional
      digits (microsecond), so truncate.
    """
    if isinstance(t, (int, float)):
        return int(t)
    # ISO-8601 string. HL stores naive UTC; treat trailing Z as UTC.
    if t.endswith("Z"):
        t = t[:-1]
    # Truncate fractional seconds to <=6 digits, preserving any timezone
    # suffix that came AFTER the fractional part.
    if "." in t:
        head, frac = t.split(".", 1)
        tz = ""
        for sep in ("+", "-"):
            if sep in frac:
                idx = frac.index(sep)
                tz, frac = frac[idx:], frac[:idx]
                break
        t = f"{head}.{frac[:6]}{tz}"
    has_tz = ("+" in t) or t.endswith("00:00") or (t.count("-") >= 3)
    if not has_tz:
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(t)
    return int(dt.timestamp() * 1000)


def _unwrap(rec: dict) -> dict:
    """HL archive records are wrapped: {time, ver_num, raw: {channel, data}}.

    Returns the inner `raw.data` payload. If the record is already
    unwrapped (legacy or third-party tooling), returns it as-is.
    """
    if "raw" in rec and isinstance(rec["raw"], dict) and "data" in rec["raw"]:
        return rec["raw"]["data"]
    return rec


def _l2_record_to_ob(rec: dict) -> dict:
    """Convert an HL l2Book record to the dict shape SensorArray expects.

    Inner shape: {coin, levels: [bids[], asks[]], time}. Each level is
    {px:str, sz:str, n:int}. We pass the FULL depth (not just top-of-book)
    because the crypto path uses CEInferenceEngine which diffs all levels.
    Inner `data.time` is epoch ms — preferred over the wrapper's
    nanosecond ISO time (simpler, matches what the live engine sees).
    """
    data = _unwrap(rec)
    levels = data.get("levels") or [[], []]
    bids = [[float(L["px"]), float(L["sz"])] for L in (levels[0] or [])]
    asks = [[float(L["px"]), float(L["sz"])] for L in (levels[1] or [])]
    ts = data.get("time", rec.get("time"))
    return {
        "bids": bids,
        "asks": asks,
        "timestamp": _to_ms(ts),
    }


def _trade_record_to_dict(fill: dict) -> dict:
    """Convert an HL fill dict (after flattening) to the SensorArray trade dict.

    Inner schema (verified via --probe on 2025-07-27 BTC fills):
        {coin, px:str, sz:str, side: 'B'|'A', time: int_ms,
         startPosition:str, dir: 'Open Long'|'Close Short'|...,
         closedPnl:str, hash:str (always zero), oid:int,
         crossed: bool, fee:str, tid:int, cloid:str, feeToken}

    NOTE: This function expects the FILL dict (i.e. ev[1] from the
    [user_addr, fill_dict] event tuple), not the raw event. `_trade_iter`
    handles the unpacking and aggressor-side filter.

    Side mapping: 'B' = aggressor bought (taker hit ask), 'A' = aggressor
    sold (taker hit bid). Maps to ccxt 'buy'/'sell'. The aggressor side
    is whichever fill has `crossed: True`.
    """
    side_str = fill.get("side", "B")
    return {
        "timestamp": _to_ms(fill["time"]),
        "price": float(fill["px"]),
        "amount": float(fill["sz"]),
        "side": "buy" if side_str == "B" else "sell",
        "info": {
            "tid": fill.get("tid"),
            "dir": fill.get("dir"),
            "crossed": fill.get("crossed"),
        },
    }


def _flatten_block_events(block: dict) -> Iterator[tuple]:
    """Yield (user_addr, fill_dict) pairs from one block record.

    Block shape: {local_time, block_time, block_number, events: [...]}
    Each event is a 2-tuple [user_addr_str, fill_dict].
    """
    events = block.get("events") or []
    # Defensive: handle the wrapped form in case future archives wrap blocks
    if not events and "raw" in block:
        raw = block["raw"]
        if isinstance(raw, dict):
            events = (raw.get("data") or {}).get("events") or []
    for ev in events:
        if isinstance(ev, (list, tuple)) and len(ev) == 2 and isinstance(ev[1], dict):
            yield ev[0], ev[1]
        elif isinstance(ev, dict):
            # Some legacy archive variants may emit fill dicts directly
            yield None, ev


def load_hl_liquidations(
    cache_path: str = HL_LIQUIDATIONS_CACHE,
    force_refresh: bool = False,
) -> list[tuple[int, float]]:
    """Download (if not cached) the official HL liquidations CSV and return
    a list of (ts_ms, notional_usd) tuples sorted by time.

    The CSV is small (single-digit MB; ~3 years of HL liquidations) and
    static between repo updates, so a local cache is reasonable. Pass
    force_refresh=True to re-download.

    Note: the upstream CSV has no `coin` field. The notional values are
    cross-market and represent aggregate liquidation pressure. Inject
    uniformly across per-coin harvests; per-coin normalization happens
    in LiquidationCascadeTracker.liquidation_rate().
    """
    p = Path(cache_path)
    if not p.exists() or force_refresh:
        log.info("downloading HL liquidations CSV from %s ...", HL_LIQUIDATIONS_CSV_URL)
        p.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(HL_LIQUIDATIONS_CSV_URL, str(p))
        log.info("cached at %s", p)

    events: list[tuple[int, float]] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            try:
                ts = _to_ms(row["time"])
                ntl = float(row["liquidated_ntl_pos"])
                events.append((ts, ntl))
            except (KeyError, ValueError, TypeError):
                continue
    events.sort(key=lambda e: e[0])
    log.info("loaded %d HL liquidation events from %s", len(events), p)
    return events


def _liquidation_iter_csv(
    events: list[tuple[int, float]],
    start_ms: int,
    end_ms: int,
) -> Iterator[dict]:
    """Yield {timestamp, qty} for liquidations in [start_ms, end_ms]."""
    for ts, ntl in events:
        if ts < start_ms:
            continue
        if ts > end_ms:
            break
        yield {"timestamp": ts, "qty": float(ntl)}


# ============================================================================
# S3 archive client
# ============================================================================

@dataclass
class HLArchiveClient:
    """Thin wrapper over boto3 that injects RequestPayer="requester" on every
    call to either HL bucket. Forgetting this on any single call gets a 403."""
    s3: object  # boto3.client("s3")

    @classmethod
    def make(cls) -> "HLArchiveClient":
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise SystemExit(
                "boto3 is required for the Hyperliquid harvester. "
                "Install via `pip install boto3 lz4`."
            ) from e
        s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5}))
        return cls(s3=s3)

    def list_keys(
        self,
        bucket: str,
        prefix: str,
        max_keys: int = 1000,
        max_total: Optional[int] = None,
    ) -> list[str]:
        """List keys under prefix. `max_total` caps total returned across
        pages — useful for buckets like `node_fills_by_block` that may
        contain millions of keys from years of mainnet history."""
        keys: list[str] = []
        kwargs = {"Bucket": bucket, "Prefix": prefix, "RequestPayer": "requester"}
        while True:
            resp = self.s3.list_objects_v2(**kwargs, MaxKeys=max_keys)
            for o in resp.get("Contents", []):
                keys.append(o["Key"])
                if max_total is not None and len(keys) >= max_total:
                    return keys
            if resp.get("IsTruncated"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        return keys

    def list_common_prefixes(
        self,
        bucket: str,
        prefix: str,
        delimiter: str = "/",
        max_keys: int = 100,
    ) -> list[str]:
        """List the immediate sub-prefixes under `prefix` (single page).
        Use this to discover bucket layout without paginating millions of
        keys. Returns prefixes like 'node_fills_by_block/2025/'."""
        resp = self.s3.list_objects_v2(
            Bucket=bucket, Prefix=prefix, Delimiter=delimiter,
            RequestPayer="requester", MaxKeys=max_keys,
        )
        return [p["Prefix"] for p in resp.get("CommonPrefixes", [])]

    def get_lz4_jsonl(self, bucket: str, key: str) -> Iterator[dict]:
        """Download an LZ4-compressed JSON-lines file and yield records."""
        try:
            import lz4.frame as lz4f
        except ImportError as e:
            raise SystemExit(
                "lz4 is required. Install via `pip install lz4`."
            ) from e
        obj = self.s3.get_object(Bucket=bucket, Key=key, RequestPayer="requester")
        blob = obj["Body"].read()
        try:
            text = lz4f.decompress(blob).decode("utf-8")
        except RuntimeError as e:
            log.error("LZ4 decompress failed for %s: %s", key, e)
            return
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("JSON decode failed in %s: %s", key, e)
                continue


# ============================================================================
# Probe mode — verify the schema before committing to a full harvest
# ============================================================================

def cmd_probe(client: HLArchiveClient, coin: str, probe_date: str) -> int:
    """Run all the schema-verification checks from §6.5 of the plan.

    1. List one hour of L2 keys to confirm path scheme + zero-padding.
    2. Download one L2 file, dump the first record's schema.
    3. Probe the node_fills bucket for path layout.
    4. Download one trade file, dump the first record's schema.
    """
    print(f"\n=== Hyperliquid S3 schema probe — coin={coin}, date={probe_date} ===\n")

    # ---- L2 archive bucket ----
    print(f"[1/4] Listing {ARCHIVE_BUCKET}/market_data/{probe_date}/0/ ...")
    try:
        l2_hour0 = client.list_keys(
            ARCHIVE_BUCKET, f"market_data/{probe_date}/0/", max_keys=10,
        )
    except Exception as e:
        log.error("L2 list failed: %s", e)
        l2_hour0 = []
    print(f"    found {len(l2_hour0)} keys in hour 0; first 5: {l2_hour0[:5]}")

    # Try zero-padded variant in case the doc example was wrong
    if not l2_hour0:
        print(f"    no keys with hour=0; trying hour=00 ...")
        l2_hour0 = client.list_keys(
            ARCHIVE_BUCKET, f"market_data/{probe_date}/00/", max_keys=10,
        )
        print(f"    zero-padded variant returned {len(l2_hour0)} keys")

    # ---- One L2 file decompress ----
    target_l2 = next(
        (k for k in l2_hour0 if k.endswith(f"/l2Book/{coin}.lz4")),
        None,
    )
    if target_l2 is None:
        # Try listing l2Book/ subprefix specifically
        sub = client.list_keys(
            ARCHIVE_BUCKET, f"market_data/{probe_date}/0/l2Book/", max_keys=20,
        )
        print(f"    market_data/{probe_date}/0/l2Book/ contents: {sub[:10]}")
        target_l2 = next((k for k in sub if k.endswith(f"/{coin}.lz4")), None)

    if target_l2:
        print(f"\n[2/4] Downloading {target_l2} ...")
        records = list(client.get_lz4_jsonl(ARCHIVE_BUCKET, target_l2))
        print(f"    decompressed to {len(records)} JSONL records")
        if records:
            first = records[0]
            print(f"    wrapper keys: {list(first.keys())}")
            data = _unwrap(first)
            wrapped = data is not first
            print(f"    wrapped form: {wrapped}")
            if wrapped:
                raw = first.get("raw", {})
                print(f"    raw.channel: {raw.get('channel')!r}")
                print(f"    wrapper.time (type={type(first.get('time')).__name__}): "
                      f"{first.get('time')!r}")
            print(f"    inner data keys: {list(data.keys())}")
            if "time" in data:
                t = data["time"]
                print(f"    inner data.time (type={type(t).__name__}): {t!r}")
            if "coin" in data:
                print(f"    inner data.coin: {data['coin']!r}")
            if "levels" in data:
                lv = data["levels"]
                print(f"    inner data.levels: outer length={len(lv)} (expect 2: bids,asks)")
                if len(lv) >= 1 and lv[0]:
                    print(f"      levels[0][0] (best bid?): {lv[0][0]}")
                if len(lv) >= 2 and lv[1]:
                    print(f"      levels[1][0] (best ask?): {lv[1][0]}")
                # Verify the parser produces a sane ob dict
                try:
                    ob = _l2_record_to_ob(first)
                    print(f"    parser→ob: {len(ob['bids'])} bids, {len(ob['asks'])} asks, "
                          f"timestamp={ob['timestamp']} ms")
                except Exception as e:
                    print(f"    PARSER ERROR: {e}")
    else:
        print(f"    SKIP: no l2Book file for {coin} on {probe_date}")

    # ---- node_fills_by_block bucket ----
    # The full prefix can contain millions of keys (every block since HL
    # mainnet launch). Use Delimiter="/" to discover the layout (sub-folders)
    # rather than enumerating every fill file.
    print(f"\n[3/4] Discovering layout under {NODE_BUCKET}/{NODE_FILLS_PREFIX}/ ...")
    try:
        sub_prefixes = client.list_common_prefixes(
            NODE_BUCKET, f"{NODE_FILLS_PREFIX}/", delimiter="/", max_keys=50,
        )
    except Exception as e:
        log.error("node_fills sub-prefix listing failed: %s", e)
        sub_prefixes = []
    print(f"    found {len(sub_prefixes)} sub-prefixes (sub-folders); first 10:")
    for p in sub_prefixes[:10]:
        print(f"      {p}")

    # Also list the first 10 actual keys directly under the prefix (single
    # page, no pagination). Some layouts put files at the top level.
    try:
        first_keys = client.list_keys(
            NODE_BUCKET, f"{NODE_FILLS_PREFIX}/", max_keys=10, max_total=10,
        )
    except Exception as e:
        log.error("node_fills key listing failed: %s", e)
        first_keys = []
    print(f"    first 10 keys directly under prefix:")
    for k in first_keys:
        print(f"      {k}")

    # If sub-prefixes found, drill into the first one to show its contents
    drill_target = sub_prefixes[0] if sub_prefixes else None
    if drill_target:
        try:
            drill_keys = client.list_keys(
                NODE_BUCKET, drill_target, max_keys=10, max_total=10,
            )
            print(f"    first 10 keys under {drill_target}:")
            for k in drill_keys:
                print(f"      {k}")
        except Exception as e:
            log.error("drill listing failed: %s", e)

    # ---- One trade file decompress ----
    sample_trade_key = (first_keys + (
        client.list_keys(NODE_BUCKET, drill_target, max_total=1)
        if drill_target else []
    ))
    sample_trade_key = sample_trade_key[0] if sample_trade_key else None
    if sample_trade_key:
        print(f"\n[4/4] Downloading {sample_trade_key} ...")
        try:
            trade_records = list(client.get_lz4_jsonl(NODE_BUCKET, sample_trade_key))
            print(f"    decompressed to {len(trade_records)} JSONL records")
            if trade_records:
                first = trade_records[0]
                print(f"    block-record keys: {list(first.keys())}")
                # Find the first block with non-empty events to expose the
                # event schema. Most blocks are empty (no fills in that
                # ~76ms slot); scan up to 5000 records.
                scan_budget = min(5000, len(trade_records))
                event_record = None
                for r in trade_records[:scan_budget]:
                    evs = r.get("events") or []
                    if evs:
                        event_record = r
                        break
                if event_record is None:
                    print(f"    NO non-empty events in first {scan_budget} blocks "
                          f"(quiet hour?); try a different hour")
                else:
                    print(f"    first non-empty block: block_number="
                          f"{event_record.get('block_number')}, "
                          f"events count={len(event_record['events'])}")
                    ev = event_record["events"][0]
                    if isinstance(ev, (list, tuple)) and len(ev) == 2:
                        user_addr, fill = ev
                        print(f"    event format: [user_addr, fill_dict] tuple")
                        print(f"    user_addr: {user_addr!r}")
                        print(f"    fill keys: {list(fill.keys())}")
                        print(f"    fill (full):")
                        print(f"      {json.dumps(fill, indent=8)[:800]}")
                        # Look for one with crossed=True specifically
                        crossed_fill = None
                        for ev2 in event_record["events"]:
                            if isinstance(ev2, (list, tuple)) and len(ev2) == 2:
                                f2 = ev2[1]
                                if isinstance(f2, dict) and f2.get("crossed"):
                                    crossed_fill = f2
                                    break
                        if crossed_fill is None:
                            print(f"    NOTE: no crossed=True in this block; "
                                  f"_trade_iter would skip all events here.")
                            crossed_fill = fill  # fall back for parser test
                        try:
                            td = _trade_record_to_dict(crossed_fill)
                            print(f"    parser→trade: ts={td['timestamp']} ms, "
                                  f"price={td['price']}, amount={td['amount']}, "
                                  f"side={td['side']!r}, "
                                  f"crossed={crossed_fill.get('crossed')}")
                        except Exception as e:
                            print(f"    PARSER ERROR: {e}")
                    elif isinstance(ev, dict):
                        print(f"    event format: bare dict (legacy variant)")
                        print(f"    fill (full):")
                        print(f"      {json.dumps(ev, indent=8)[:800]}")
        except Exception as e:
            log.error("trade decompress failed: %s", e)
    else:
        print(f"\n[4/4] SKIP: no trade keys discovered.")

    print("\n=== Probe complete. Update fetch_history_hyperliquid.py if any "
          "field above doesn't match the assumed schema. ===\n")
    return 0


# ============================================================================
# Harvest mode — chronological replay through SensorArray
# ============================================================================

def _date_range(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _l2_keys_for_date(coin: str, d: date) -> list[str]:
    yyyymmdd = d.strftime("%Y%m%d")
    # 24 hours, NOT zero-padded (per HL docs example "9/" not "09/")
    return [
        f"market_data/{yyyymmdd}/{h}/l2Book/{coin}.lz4"
        for h in range(24)
    ]


def _l2_iter(client: HLArchiveClient, coin: str, days: list[date]) -> Iterator[dict]:
    """Iterate L2 records chronologically across all date+hour files."""
    for d in days:
        for key in _l2_keys_for_date(coin, d):
            try:
                for rec in client.get_lz4_jsonl(ARCHIVE_BUCKET, key):
                    yield rec
            except Exception as e:
                # Missing hours are common (HL docs warn data may be missing)
                if "NoSuchKey" in str(e) or "404" in str(e):
                    log.debug("missing key (skipped): %s", key)
                else:
                    log.warning("error fetching %s: %s", key, e)


def _trade_keys_for_date(d: date) -> list[str]:
    """Per-hour trade key paths under hl-mainnet-node-data.

    Layout discovered via --probe: node_fills_by_block/hourly/YYYYMMDD/H.lz4
    Hour is NOT zero-padded (matches the L2 archive convention).
    Earliest data is 2025-07-27 in this format; older data lives in the
    legacy `node_fills` / `node_trades` paths and isn't covered here.
    """
    yyyymmdd = d.strftime("%Y%m%d")
    return [
        f"node_fills_by_block/hourly/{yyyymmdd}/{h}.lz4"
        for h in range(24)
    ]


def _trade_iter(
    client: HLArchiveClient,
    coin: str,
    days: list[date],
    trade_keys: Optional[list[str]] = None,
) -> Iterator[dict]:
    """Iterate aggressor-side fill dicts chronologically.

    HL trade archives are batched per-block: each line is a block record
    {local_time, block_time, block_number, events: [[addr, fill], ...]}.
    We flatten one level via `_flatten_block_events`, then:
      - filter by `coin` (the archive multiplexes all coins per file),
      - keep only `crossed: True` records (the aggressor side — each
        trade has 2 records, one per user; the crossed-True record
        carries the trade's aggressor-side directionality),
      - dedupe by `tid` defensively.

    `hash` is always 0x000...000 in this archive — do NOT use it for
    dedup (we did originally; --probe revealed it).

    If `trade_keys` is None, keys are constructed per-date from the
    `node_fills_by_block/hourly/<date>/<hour>.lz4` layout. Missing hours
    (404s) are silently skipped.

    Liquidations are NOT identifiable from this archive (verified by
    tests/inspect_hl_liquidations.py — no marker in `dir` or
    `trade_dir_override`). Use load_hl_liquidations() + the external
    CSV for liquidation events.
    """
    if trade_keys is None:
        trade_keys = []
        for d in days:
            trade_keys.extend(_trade_keys_for_date(d))

    seen_tids: set[int] = set()
    for key in trade_keys:
        try:
            for block in client.get_lz4_jsonl(NODE_BUCKET, key):
                for _user_addr, fill in _flatten_block_events(block):
                    if fill.get("coin") and fill["coin"] != coin:
                        continue
                    if not fill.get("crossed", False):
                        continue
                    tid = fill.get("tid")
                    if tid is not None:
                        if tid in seen_tids:
                            continue
                        seen_tids.add(tid)
                    yield fill
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                log.debug("missing trade key (skipped): %s", key)
            else:
                log.warning("error fetching %s: %s", key, e)


def cmd_harvest(
    client: HLArchiveClient,
    coin: str,
    days: list[date],
    out_path: Path,
    trade_keys: Optional[list[str]] = None,
    limit_records: Optional[int] = None,
    liquidation_events: Optional[list[tuple[int, float]]] = None,
) -> int:
    log.info(
        "harvesting %d day(s) for coin=%s → %s%s",
        len(days), coin, out_path,
        f" (limit {limit_records:,} records)" if limit_records else "",
    )
    log.info("dates: %s", [d.isoformat() for d in days])

    if trade_keys is None:
        # Construct keys automatically using the known per-date layout
        # discovered via --probe. Missing hours are skipped at fetch time.
        trade_keys = []
        for d in days:
            trade_keys.extend(_trade_keys_for_date(d))
        log.info(
            "constructed %d trade keys (hourly per date) for "
            "node_fills_by_block/hourly/", len(trade_keys),
        )

    # Date-range bounds for slicing the cross-market liquidation CSV.
    start_ms = int(datetime.combine(
        days[0], datetime.min.time(), tzinfo=timezone.utc,
    ).timestamp() * 1000)
    end_ms = int(datetime.combine(
        days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc,
    ).timestamp() * 1000)

    dumper = FeatureDumper(out_path)
    sensors = SensorArray(
        exchange=None,
        symbol=coin,
        hmm=StudentTHMM.from_default_priors(),
        feature_dumper=dumper,
    )

    # No reset_session() between days — HL is 24/7.

    n_l2 = n_trade = n_liq = n_processed = 0

    def _l2_events() -> Iterator[tuple[int, str, dict]]:
        nonlocal n_l2
        for rec in _l2_iter(client, coin, days):
            n_l2 += 1
            ob = _l2_record_to_ob(rec)
            yield (ob["timestamp"], "ob", ob)

    def _trade_events() -> Iterator[tuple[int, str, dict]]:
        nonlocal n_trade
        for rec in _trade_iter(client, coin, days, trade_keys):
            n_trade += 1
            t = _trade_record_to_dict(rec)
            yield (t["timestamp"], "t", t)

    def _liq_events() -> Iterator[tuple[int, str, dict]]:
        nonlocal n_liq
        if not liquidation_events:
            return
        for liq in _liquidation_iter_csv(liquidation_events, start_ms, end_ms):
            n_liq += 1
            yield (liq["timestamp"], "liq", liq)

    try:
        merged = heapq.merge(
            _l2_events(), _trade_events(), _liq_events(), key=lambda e: e[0],
        )
        for ts, kind, payload in merged:
            n_processed += 1
            if n_processed % 100_000 == 0:
                log.info(
                    "  ... %s events processed (l2=%s, trade=%s, liq=%s, rows=%s)",
                    f"{n_processed:,}", f"{n_l2:,}", f"{n_trade:,}",
                    f"{n_liq:,}", f"{dumper.rows_written:,}",
                )
            if kind == "ob":
                sensors._process_order_book(payload)
            elif kind == "t":
                sensors._process_trade(payload)
            else:  # "liq"
                sensors.liquidation.record(payload["timestamp"], payload["qty"])
            if limit_records and n_processed >= limit_records:
                log.info("reached --limit-records %d, stopping early", limit_records)
                break
    finally:
        dumper.close()

    log.info(
        "done: %d L2 records, %d trades, %d liquidation events, %d rows in %s",
        n_l2, n_trade, n_liq, dumper.rows_written, out_path,
    )
    return 0


# ============================================================================
# CLI
# ============================================================================

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--coin", default="BTC",
                        help="Hyperliquid coin symbol (HL bucket name, e.g. BTC, ETH, SOL). "
                             "Ignored when --coins is set.")
    parser.add_argument("--coins", default=None,
                        help="Comma-separated list of coins for multi-coin harvest "
                             "(e.g. 'BTC,ETH,SOL'). Each coin gets its own CSV at "
                             "calibration/feature_history_<COIN>.csv. Each coin runs "
                             "in its own SensorArray (correct VPIN/Kalman state per "
                             "coin). Cost scales linearly with coin count.")
    parser.add_argument("--days", type=int, default=14,
                        help="Number of past days to harvest (ignored if --start-date given)")
    parser.add_argument("--start-date", type=_parse_date, default=None,
                        help="Start date YYYYMMDD (inclusive)")
    parser.add_argument("--end-date", type=_parse_date, default=None,
                        help="End date YYYYMMDD (inclusive)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: ./calibration/feature_history_<COIN>.csv)")
    parser.add_argument("--probe", action="store_true",
                        help="Schema-verify mode: list buckets and dump one record of each kind. "
                             "Run this once before the first real harvest.")
    parser.add_argument("--probe-date", type=_parse_date, default=None,
                        help="Date to probe in --probe mode (defaults to ~30 days ago)")
    parser.add_argument("--trade-key-prefix", default=None,
                        help="Override prefix for trade key discovery in node_fills_by_block. "
                             "Use this once --probe reveals the actual layout.")
    parser.add_argument("--limit-records", type=int, default=None,
                        help="Stop harvest after N events (testing).")
    parser.add_argument("--no-liquidations", action="store_true",
                        help="Skip the cross-market HL liquidations CSV. "
                             "Result: liquidation_rate stays at 0 (Sigma_diag[2] "
                             "degenerate; see research review).")
    parser.add_argument("--refresh-liquidations", action="store_true",
                        help="Force re-download of the HL liquidations CSV "
                             "from GitHub (default: use local cache if present).")
    args = parser.parse_args(argv)

    client = HLArchiveClient.make()

    # ---------- probe mode ----------
    if args.probe:
        probe_date = (
            args.probe_date
            or (datetime.now(timezone.utc).date() - timedelta(days=35))
        )
        return cmd_probe(client, args.coin, probe_date.strftime("%Y%m%d"))

    # ---------- harvest mode ----------
    if args.start_date and args.end_date:
        days = list(_date_range(args.start_date, args.end_date))
    elif args.start_date:
        days = [args.start_date]
    else:
        # HL archive update cadence is ~monthly. Default to a window known
        # to be uploaded — end ≈ 35 days ago, start = end - days.
        end = datetime.now(timezone.utc).date() - timedelta(days=35)
        start = end - timedelta(days=args.days - 1)
        days = list(_date_range(start, end))

    if not days:
        log.error("empty date range")
        return 2

    # Resolve coin list: --coins (multi) takes precedence over --coin (single).
    if args.coins:
        coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    else:
        coins = [args.coin]
    if not coins:
        log.error("no coins specified")
        return 2
    if args.coins and args.out:
        log.error("--out cannot be combined with --coins (path is per-coin)")
        return 2

    # Trade key discovery is gated on --probe verification. Until the
    # node_fills_by_block layout is confirmed, the user must pass
    # --trade-key-prefix explicitly OR run L2-only first.
    trade_keys: Optional[list[str]] = None
    if args.trade_key_prefix:
        log.info("listing trade keys under %s ...", args.trade_key_prefix)
        try:
            trade_keys = client.list_keys(NODE_BUCKET, args.trade_key_prefix)
            log.info("found %d trade keys", len(trade_keys))
        except Exception as e:
            log.error("trade key listing failed: %s", e)
            return 1

    # Load liquidations CSV once (shared across per-coin harvests since
    # the upstream feed is cross-market — same events injected uniformly
    # into each coin's harvest).
    liquidation_events: Optional[list[tuple[int, float]]] = None
    if not args.no_liquidations:
        try:
            liquidation_events = load_hl_liquidations(
                force_refresh=args.refresh_liquidations,
            )
        except Exception as e:
            log.error(
                "failed to load HL liquidations CSV (%s); continuing without. "
                "liquidation_rate will be 0. Pass --no-liquidations to silence.",
                e,
            )
            liquidation_events = None

    overall_rc = 0
    for coin in coins:
        coin_slug = coin.replace("/", "_").replace(":", "_")
        out_path = Path(
            args.out or f"./calibration/feature_history_{coin_slug}.csv"
        )
        if out_path.exists():
            log.warning(
                "output %s already exists — appending; delete the file first "
                "for a clean rebuild.", out_path,
            )
        if len(coins) > 1:
            log.info("=== coin %s of %s: %s ===",
                     coins.index(coin) + 1, len(coins), coin)
        rc = cmd_harvest(
            client=client,
            coin=coin,
            days=days,
            out_path=out_path,
            trade_keys=trade_keys,
            limit_records=args.limit_records,
            liquidation_events=liquidation_events,
        )
        if rc != 0 and overall_rc == 0:
            overall_rc = rc
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
