"""
eval_ppo.py — Compare a trained PPO checkpoint against naive baselines.

Replays a held-out slice of the feature_history CSV through the same
(ReplaySensorArray → AlphaEngine → ExecutionEnv) stack used by train_ppo.py,
running the SAME mandate stream through several execution policies:

    trained_ppo   — loaded from `--weights`
    random_init   — PPOAgent with no training (sanity floor)
    always_taker  — deterministic action = -1.0
    always_maker  — deterministic action = +1.0
    naive_random  — uniform action in [-0.5, +0.5]

Metric: mean fee-adjusted implementation shortfall (IS) per episode, plus
forced-terminal rate and fill rate.

Phase 2 decision gate (per plan):
    PRIMARY:   trained_ppo mean_IS  ≤ 0.6 × always_taker mean_IS  (≥ 40% reduction)
    SECONDARY: trained_ppo term_rate ≤ 1.5 × always_maker term_rate
    SANITY:    trained_ppo mean_ep_reward  >  random_init mean_ep_reward

If PRIMARY fails the script exits with non-zero status to surface the failure.

Usage (PowerShell):
    $env:SYMBOL = "TSLA"
    $env:MAX_POSITION_LIMIT = "10000"
    python eval_ppo.py --symbol TSLA `
        --csv calibration/feature_history_TSLA_balanced.csv `
        --weights calibration/ppo_weights_TSLA.pt `
        --random-init-weights calibration/ppo_weights_TSLA_random_init.pt `
        --eval-frac 0.8 1.0 --n-episodes 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _real_time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Parse --symbol BEFORE config import (so config.SYMBOL is correct).
# ---------------------------------------------------------------------------
def _pre_parse_symbol() -> None:
    if "--symbol" in sys.argv:
        idx = sys.argv.index("--symbol")
        if idx + 1 < len(sys.argv):
            os.environ["SYMBOL"] = sys.argv[idx + 1]


_pre_parse_symbol()

# Repo-root resolution for the new training/ subdir layout.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Synthetic clock — same monkey-patch pattern as train_ppo.py
# ---------------------------------------------------------------------------
class _TickClock:
    def __init__(self) -> None:
        self.t = 0.0

    def reset(self, t: float = 0.0) -> None:
        self.t = t

    def advance(self, dt: float) -> None:
        self.t += dt


_clock = _TickClock()
import layer3_execution as _l3  # noqa: E402

_l3.time = SimpleNamespace(monotonic=lambda: _clock.t)


# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from layer1_sensors import OODDetector  # noqa: E402
from layer2_alpha import AlphaEngine, TCNSpikePredictor  # noqa: E402
from layer3_execution import ExecutionEnv, PPOAgent  # noqa: E402
from matching_engine import LocalMatchingEngine  # noqa: E402
from tests.test_replay import ReplaySensorArray  # noqa: E402

log = logging.getLogger("eval_ppo")

CLOCK_TICK_SECONDS = 0.05
TRAINING_TIME_BUDGET_SECONDS = 5.0
MAX_TICKS_PER_EPISODE = 120


# ---------------------------------------------------------------------------
# Calibration loader — mirrors train_ppo.py
# ---------------------------------------------------------------------------
def load_calibration_for(symbol: str) -> tuple[OODDetector, float]:
    ood = OODDetector()
    alpha_c = float(config.CALIBRATION_REGISTRY.get(symbol, 0.25))
    cal_path = Path(f"calibration/latest_{symbol}.json")
    if cal_path.exists():
        try:
            payload = json.loads(cal_path.read_text())
            ood.load(np.array(payload["ood_mu"]), np.array(payload["ood_sigma"]))
            alpha_c = float(payload.get("alpha_calibration_c", alpha_c))
            log.info("Loaded OOD calibration from %s", cal_path)
        except Exception as e:
            log.warning("OOD load failed: %s — cold-start prior", e)
    return ood, alpha_c


def load_alpha_engine(symbol: str, turbulence_threshold: float) -> AlphaEngine:
    tcn = TCNSpikePredictor()
    alpha = AlphaEngine(tcn=tcn, turbulence_threshold=turbulence_threshold)
    weights_path = Path(f"calibration/tcn_weights_{symbol}.pt")
    if weights_path.exists():
        try:
            alpha.load_weights(str(weights_path))
            log.info("Loaded TCN weights from %s", weights_path)
        except Exception as e:
            log.warning("TCN load failed: %s — random-init", e)
    return alpha


# ---------------------------------------------------------------------------
# Policy interface — every baseline is a callable obs → action
# ---------------------------------------------------------------------------
class _PolicyResult:
    """Aggregated metrics across episodes for one policy."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.episodes_mean_is: list[float] = []
        self.episodes_reward: list[float] = []
        self.episodes_fill_frac: list[float] = []
        self.episodes_terminal: list[int] = []
        self.episodes_taker_count: list[int] = []
        self.episodes_maker_count: list[int] = []

    def summary(self) -> dict:
        def _m(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0
        def _med(xs: list[float]) -> float:
            return float(np.median(xs)) if xs else 0.0
        return {
            "name": self.name,
            "n_episodes": len(self.episodes_mean_is),
            "mean_is": _m(self.episodes_mean_is),
            "median_is": _med(self.episodes_mean_is),
            "mean_reward": _m(self.episodes_reward),
            "mean_fill_frac": _m(self.episodes_fill_frac),
            "terminal_rate": _m([float(t) for t in self.episodes_terminal]),
            "mean_taker_count": _m([float(c) for c in self.episodes_taker_count]),
            "mean_maker_count": _m([float(c) for c in self.episodes_maker_count]),
        }


def _stochastic_ppo(agent: PPOAgent):
    def fn(obs: np.ndarray) -> float:
        action, _, _ = agent.act(obs, deterministic=False)
        return float(action[0])
    return fn


def _deterministic_ppo(agent: PPOAgent):
    def fn(obs: np.ndarray) -> float:
        action, _, _ = agent.act(obs, deterministic=True)
        return float(action[0])
    return fn


def _always_taker(_obs: np.ndarray) -> float:
    return -1.0


def _always_maker(_obs: np.ndarray) -> float:
    return 1.0


def _naive_random(rng: np.random.Generator):
    def fn(_obs: np.ndarray) -> float:
        return float(rng.uniform(-0.5, 0.5))
    return fn


# ---------------------------------------------------------------------------
# CSV slice streaming
# ---------------------------------------------------------------------------
def stream_csv_rows_in_range(csv_path: Path, start_frac: float, end_frac: float) -> Iterator[dict]:
    """Stream rows from a fractional slice of the CSV.

    Uses a two-pass approach: first pass counts rows (cheap with polars), second
    pass yields rows in the [start, end) range. Slow for huge files, but tolerable
    for eval-window sizes (typically ~10MB).
    """
    total_rows = pl.scan_csv(str(csv_path)).select(pl.len()).collect().item()
    start_idx = int(total_rows * start_frac)
    end_idx = int(total_rows * end_frac)
    log.info(
        "Eval slice: rows %d..%d of %d (frac %.2f..%.2f)",
        start_idx, end_idx, total_rows, start_frac, end_frac,
    )
    reader = pl.read_csv_batched(str(csv_path))
    batches = reader.next_batches(1)
    cursor = 0
    while batches:
        for batch in batches:
            batch_len = batch.height
            if cursor + batch_len <= start_idx:
                cursor += batch_len
                continue
            if cursor >= end_idx:
                return
            for row in batch.iter_rows(named=True):
                if start_idx <= cursor < end_idx:
                    yield row
                cursor += 1
                if cursor >= end_idx:
                    return
        batches = reader.next_batches(1)


# ---------------------------------------------------------------------------
# Run a single policy over a fresh AlphaEngine + sensor stream
# ---------------------------------------------------------------------------
def evaluate_policy(
    name: str,
    policy_fn,
    symbol: str,
    csv_path: Path,
    start_frac: float,
    end_frac: float,
    n_episodes: int,
    turbulence_threshold: float,
    target_multiplier: float,
) -> _PolicyResult:
    ood, alpha_c = load_calibration_for(symbol)
    sensor = ReplaySensorArray(
        csv_path=str(csv_path),
        symbol=symbol,
        alpha_calibration_c=alpha_c,
        ood_detector=ood,
    )
    alpha = load_alpha_engine(symbol, turbulence_threshold)
    env = ExecutionEnv(
        live_state_provider=sensor,
        matching_engine=LocalMatchingEngine(),
        shadow_mode=True,
    )
    result = _PolicyResult(name)

    row_iter = stream_csv_rows_in_range(csv_path, start_frac, end_frac)

    def advance_row() -> Optional[dict]:
        try:
            row = next(row_iter)
        except StopIteration:
            return None
        sensor.state = sensor._row_to_state(row)
        return row

    while len(result.episodes_mean_is) < n_episodes:
        row = advance_row()
        if row is None:
            log.warning(
                "[%s] CSV slice exhausted after %d episodes (target %d)",
                name, len(result.episodes_mean_is), n_episodes,
            )
            break
        mandate = alpha.evaluate(sensor.state)
        if mandate is None:
            continue
        if target_multiplier != 1.0:
            mandate.target_size = float(mandate.target_size * target_multiplier)

        _clock.reset(0.0)
        env.attach_mandate(mandate)
        obs, _ = env.reset()
        env.time_budget = TRAINING_TIME_BUDGET_SECONDS

        ep_reward = 0.0
        ep_terminal = False
        ep_taker = 0
        ep_maker = 0

        done = False
        for tick in range(MAX_TICKS_PER_EPISODE):
            action_scalar = policy_fn(obs)
            action = np.array([float(np.clip(action_scalar, -1.0, 1.0))], dtype=np.float32)
            if action[0] < config.FEE_AGGRESSION_THRESHOLD:
                ep_taker += 1
            else:
                ep_maker += 1
            next_row = advance_row()
            if next_row is None:
                done = True
                break
            _clock.advance(CLOCK_TICK_SECONDS)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            if truncated:
                ep_terminal = True
            if terminated or truncated:
                done = True
                break
        if not done:
            side = "buy" if mandate.direction > 0 else "sell"
            terminal_r = float(env._force_terminal_fill(side))
            ep_reward += terminal_r
            ep_terminal = True

        ep_result = env.episode_result()
        fill_frac = 1.0 - env.remaining_volume / max(mandate.target_size, 1e-9)
        result.episodes_mean_is.append(float(ep_result.mean_is))
        result.episodes_reward.append(ep_reward)
        result.episodes_fill_frac.append(float(fill_frac))
        result.episodes_terminal.append(1 if ep_terminal else 0)
        result.episodes_taker_count.append(ep_taker)
        result.episodes_maker_count.append(ep_maker)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="TSLA")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path,
                        help="Trained PPO weights (calibration/ppo_weights_<SYMBOL>.pt)")
    parser.add_argument("--random-init-weights", type=Path, default=None,
                        help="Random-init PPO weights for sanity floor (optional).")
    parser.add_argument("--eval-frac", nargs=2, type=float, default=[0.8, 1.0],
                        metavar=("START", "END"),
                        help="Fractional slice of the CSV to evaluate on.")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--turbulence-threshold", type=float, default=0.05)
    parser.add_argument("--target-multiplier", type=float, default=20.0,
                        help="MUST match train_ppo.py for apples-to-apples comparison.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("calibration/eval_ppo_results.json"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.csv.exists():
        log.error("CSV %s not found.", args.csv)
        sys.exit(1)
    if not args.weights.exists():
        log.error("Trained weights %s not found.", args.weights)
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start_frac, end_frac = args.eval_frac

    # Load policies
    trained_agent = PPOAgent()
    trained_agent.load_state_dict(torch.load(args.weights, map_location="cpu"))
    trained_agent.eval()

    random_agent: Optional[PPOAgent] = None
    if args.random_init_weights and args.random_init_weights.exists():
        random_agent = PPOAgent()
        random_agent.load_state_dict(torch.load(args.random_init_weights, map_location="cpu"))
        random_agent.eval()
        log.info("Loaded random-init PPO from %s", args.random_init_weights)

    common = dict(
        symbol=args.symbol,
        csv_path=args.csv,
        start_frac=start_frac,
        end_frac=end_frac,
        n_episodes=args.n_episodes,
        turbulence_threshold=args.turbulence_threshold,
        target_multiplier=args.target_multiplier,
    )

    results: dict[str, dict] = {}
    summaries: list[dict] = []
    policies: list[tuple[str, callable]] = [
        ("trained_ppo_det", _deterministic_ppo(trained_agent)),
        ("trained_ppo_stoch", _stochastic_ppo(trained_agent)),
        ("always_taker", _always_taker),
        ("always_maker", _always_maker),
        ("naive_random", _naive_random(rng)),
    ]
    if random_agent is not None:
        policies.append(("random_init_ppo", _deterministic_ppo(random_agent)))

    wall = _real_time.monotonic()
    for name, fn in policies:
        log.info("=== Evaluating: %s ===", name)
        t0 = _real_time.monotonic()
        res = evaluate_policy(name, fn, **common)
        summary = res.summary()
        summary["wall_time_s"] = _real_time.monotonic() - t0
        summaries.append(summary)
        results[name] = summary
        log.info(
            "%-20s | n=%d | IS=%.5f (med=%.5f) | r=%.4f | fill=%.3f | term=%.2f | taker/maker=%.1f/%.1f | t=%.1fs",
            name, summary["n_episodes"], summary["mean_is"], summary["median_is"],
            summary["mean_reward"], summary["mean_fill_frac"], summary["terminal_rate"],
            summary["mean_taker_count"], summary["mean_maker_count"], summary["wall_time_s"],
        )

    # ---- Decision gate ---------------------------------------------------
    taker_is = results.get("always_taker", {}).get("mean_is", 0.0)
    maker_term = results.get("always_maker", {}).get("terminal_rate", 0.0)
    ppo_is = results.get("trained_ppo_det", {}).get("mean_is", 0.0)
    ppo_term = results.get("trained_ppo_det", {}).get("terminal_rate", 0.0)
    ppo_reward = results.get("trained_ppo_det", {}).get("mean_reward", 0.0)
    random_reward = results.get("random_init_ppo", {}).get("mean_reward", 0.0)

    # IS sign convention: lower is better (cost). For shorts the framing differs;
    # for now compare absolute reduction vs taker baseline.
    primary_ratio = (ppo_is / taker_is) if abs(taker_is) > 1e-12 else float("inf")
    primary_pass = primary_ratio <= 0.6
    secondary_ratio = (ppo_term / maker_term) if maker_term > 1e-12 else float("inf")
    secondary_pass = secondary_ratio <= 1.5
    sanity_pass = (random_agent is None) or (ppo_reward > random_reward)

    gate = {
        "primary": {
            "criterion": "trained_ppo_det.mean_is <= 0.6 * always_taker.mean_is",
            "ppo_is": ppo_is,
            "taker_is": taker_is,
            "ratio": primary_ratio,
            "passed": bool(primary_pass),
        },
        "secondary": {
            "criterion": "trained_ppo_det.terminal_rate <= 1.5 * always_maker.terminal_rate",
            "ppo_term": ppo_term,
            "maker_term": maker_term,
            "ratio": secondary_ratio,
            "passed": bool(secondary_pass),
        },
        "sanity": {
            "criterion": "trained_ppo_det.mean_reward > random_init_ppo.mean_reward",
            "ppo_reward": ppo_reward,
            "random_reward": random_reward,
            "passed": bool(sanity_pass),
        },
    }

    log.info("=" * 70)
    log.info(
        "DECISION GATE: primary=%s (ratio=%.3f) secondary=%s (ratio=%.3f) sanity=%s",
        "PASS" if primary_pass else "FAIL", primary_ratio,
        "PASS" if secondary_pass else "FAIL", secondary_ratio,
        "PASS" if sanity_pass else "FAIL",
    )

    payload = {
        "summaries": summaries,
        "gate": gate,
        "config": {
            "symbol": args.symbol,
            "csv": str(args.csv),
            "weights": str(args.weights),
            "eval_frac": [start_frac, end_frac],
            "n_episodes": args.n_episodes,
            "turbulence_threshold": args.turbulence_threshold,
            "target_multiplier": args.target_multiplier,
            "seed": args.seed,
        },
        "wall_total_s": _real_time.monotonic() - wall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", args.output)

    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
