"""
Replay shim + tests. ReplaySensorArray drives the engine off a pre-computed
feature_history CSV instead of a live Alpaca feed, so Layers 2/3/4 + dashboard
can be smoke-tested when markets are closed. engine.py imports
ReplaySensorArray from here when REPLAY_CSV is set; otherwise this file is
just a pytest module and is ignored by the live path.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import config
from layer1_sensors import PhysicsState

REPLAY_TICK_SECONDS = float(os.environ.get("REPLAY_TICK_SECONDS", "0.05"))
# Total synthesized book size (bid_total + ask_total). Split bid vs ask
# proportional to obi so layer2's shock-volume calc Q = |bid - ask| reflects
# the harvest's OBI signal — without this, equal sides => Q=0 => every mandate
# suppressed at layer2_alpha.py:228 (Q < MIN_ORDER_SIZE).
REPLAY_FAKE_DEPTH_SIZE = float(os.environ.get("REPLAY_FAKE_DEPTH_SIZE", "1000.0"))


class ReplaySensorArray:
    """
    Drop-in for SensorArray that yields PhysicsState rows from a feature_history
    CSV. Same four-method interface engine.py uses: .run() (async task),
    .stop(), .reset_session(warmup_seconds), and .state attribute.
    """

    def __init__(self, csv_path: str, symbol: str, alpha_calibration_c: float,
                 ood_detector=None, **_):
        # Accept (optionally) the engine's pre-loaded OODDetector so we can
        # recompute mahal_dist against the calibrated μ/Σ instead of using the
        # CSV's stored value (which was computed with the identity-prior at
        # harvest time and is meaningless once latest_*.json exists).
        # The remaining SensorArray kwargs (exchange, hmm, liquidation_tracker,
        # feature_dumper) are absorbed silently so engine.py doesn't have to
        # branch the construction site by argument shape.
        self._csv_path = Path(csv_path)
        self._symbol = symbol
        self._alpha_c = float(alpha_calibration_c)
        self._ood = ood_detector
        # ob_snapshot=None on the default — strategy loop holds at engine.py:473
        # until the first row arrives.
        self.state = PhysicsState()
        self._stopped = False
        self._prev_ob: dict | None = None
        self._prev_spread: float = 0.0

    async def run(self):
        # Match train_stream.py's batched-streaming pattern so we don't load
        # the full multi-GB CSV into memory at once.
        reader = pl.read_csv_batched(str(self._csv_path))
        batches = reader.next_batches(1)
        while batches and not self._stopped:
            for batch in batches:
                for row in batch.iter_rows(named=True):
                    if self._stopped:
                        return
                    self.state = self._row_to_state(row)
                    await asyncio.sleep(REPLAY_TICK_SECONDS)
            batches = reader.next_batches(1)

    def _row_to_state(self, row) -> PhysicsState:
        bid = float(row["best_bid"])
        ask = float(row["best_ask"])
        spread = ask - bid
        # Skew synthesized depth by OBI so Q = |bid_total - ask_total| in the
        # layer2 solver = REPLAY_FAKE_DEPTH_SIZE * |obi|. With size=1000 and
        # obi=0.1, Q=100, well clear of MIN_ORDER_SIZE=1.0. Direction matches
        # the harvest's actual buy/sell pressure.
        obi = float(row["obi"])
        bid_size = REPLAY_FAKE_DEPTH_SIZE * (1.0 + obi) / 2.0
        ask_size = REPLAY_FAKE_DEPTH_SIZE * (1.0 - obi) / 2.0
        ob = {
            "bids": [[bid, bid_size]],
            "asks": [[ask, ask_size]],
            "timestamp": int(row["timestamp_ms"]),
        }
        prev_ob = self._prev_ob
        spread_velocity = spread - self._prev_spread
        self._prev_ob = ob
        self._prev_spread = spread
        # Compute mahal_dist from the loaded OODDetector if available; this
        # mirrors what live SensorArray does at layer1_sensors.py:896. The CSV
        # column's value was produced with whatever μ/Σ existed at harvest
        # (likely identity-prior, hence inflated 50+ values), so prefer the
        # in-memory calibrated detector when present.
        if self._ood is not None:
            tcn_obs = np.array([
                float(row["ce_ratio"]),
                float(row["obi"]),
                float(row["liquidation_rate"]),
            ])
            ood_flag, mahal = self._ood.evaluate(tcn_obs)
        else:
            mahal = float(row["mahal_dist"])
            ood_flag = mahal > config.OOD_THRESHOLD
        return PhysicsState(
            vpin=float(row["vpin"]),
            alpha_calibrated=self._alpha_c,
            # Heat solver in layer2 divides by viscosity; never zero.
            viscosity=max(spread, 1e-4),
            regime=int(row["regime"]),
            liquidation_rate=float(row["liquidation_rate"]),
            ood_flag=bool(ood_flag),
            mahal_dist=float(mahal),
            ce_ratio=float(row["ce_ratio"]),
            obi=float(row["obi"]),
            spread_velocity=spread_velocity,
            timestamp=int(row["timestamp_ms"]),
            ob_snapshot=ob,
            prev_ob_snapshot=prev_ob,
        )

    def reset_session(self, warmup_seconds: float = 0):
        pass  # no session boundaries in replay

    def stop(self):
        self._stopped = True


# ---- pytest tests ----------------------------------------------------------

@pytest.fixture
def sample_row():
    return {
        "timestamp_ms": 1_700_000_000_000,
        "ce_ratio": 0.42,
        "obi": 0.1,
        "liquidation_rate": 0.0,
        "vpin": 0.3,
        "regime": 0,
        "mahal_dist": 1.5,
        "best_bid": 100.0,
        "best_ask": 100.05,
    }


def test_row_to_state_populates_required_fields(sample_row):
    arr = ReplaySensorArray(csv_path="<unused>", symbol="TSLA", alpha_calibration_c=0.25)
    s = arr._row_to_state(sample_row)
    assert s.ob_snapshot is not None
    assert s.ob_snapshot["bids"][0][0] == 100.0
    assert s.ob_snapshot["asks"][0][0] == 100.05
    assert s.ce_ratio == pytest.approx(0.42)
    assert s.regime == 0
    assert s.viscosity > 0  # solver-safe
    assert s.alpha_calibrated == pytest.approx(0.25)


def test_ood_flag_derived_from_threshold(sample_row):
    arr = ReplaySensorArray(csv_path="<unused>", symbol="TSLA", alpha_calibration_c=0.0)
    # mahal_dist=1.5 (< OOD_THRESHOLD=4.5) → in-distribution
    s_in = arr._row_to_state({**sample_row, "mahal_dist": 1.5})
    assert s_in.ood_flag is False
    # mahal_dist=50 (>> OOD_THRESHOLD) → flagged so AlphaEngine suppresses
    s_out = arr._row_to_state({**sample_row, "mahal_dist": 50.0})
    assert s_out.ood_flag is True


def test_obi_skews_synthesized_depth(sample_row):
    # Positive OBI ⇒ heavier bid side; layer2 solver reads net imbalance > 0
    # and direction = +1 (buy pressure). Sizes must sum to REPLAY_FAKE_DEPTH_SIZE.
    arr = ReplaySensorArray(csv_path="<unused>", symbol="TSLA", alpha_calibration_c=0.0)
    s = arr._row_to_state({**sample_row, "obi": 0.4})
    bid_size = s.ob_snapshot["bids"][0][1]
    ask_size = s.ob_snapshot["asks"][0][1]
    assert bid_size > ask_size
    assert bid_size + ask_size == pytest.approx(REPLAY_FAKE_DEPTH_SIZE)
    # Net imbalance = total * obi.
    assert (bid_size - ask_size) == pytest.approx(REPLAY_FAKE_DEPTH_SIZE * 0.4)


def test_prev_ob_snapshot_threads_across_rows(sample_row):
    arr = ReplaySensorArray(csv_path="<unused>", symbol="TSLA", alpha_calibration_c=0.25)
    s1 = arr._row_to_state(sample_row)
    assert s1.prev_ob_snapshot is None
    s2 = arr._row_to_state({**sample_row, "best_ask": 100.06})
    assert s2.prev_ob_snapshot is not None
    assert s2.prev_ob_snapshot["asks"][0][0] == pytest.approx(100.05)


def test_stop_breaks_run_loop(tmp_path, sample_row):
    csv = tmp_path / "tiny.csv"
    pl.DataFrame([sample_row, sample_row, sample_row]).write_csv(csv)
    arr = ReplaySensorArray(csv_path=str(csv), symbol="TSLA", alpha_calibration_c=0.0)

    async def driver():
        task = asyncio.create_task(arr.run())
        await asyncio.sleep(0.01)
        arr.stop()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(driver())
