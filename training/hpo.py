"""
hpo.py — Optuna hyperparameter optimization for the execution reward weights.

Searches over (ETA, GAMMA_INV, TERMINAL_PENALTY_MULTIPLIER). Trains a fresh
PPO agent per trial against the shadow simulator's environment, evaluates on a
held-out window, and validates the winning trial on the SUBSEQUENT week to
guard against overfitting.

Data providers come from `make_csv_data_provider(csv_path, start_frac, end_frac)`
which streams a fractional slice of a feature_history CSV through the same
ReplaySensorArray + AlphaEngine stack used by train_ppo.py.

Run:
    python hpo.py --n-trials 3 --csv calibration/feature_history_TSLA_balanced.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _real_time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, Optional

# Repo-root resolution for the new training/ subdir layout — must come BEFORE
# any layer3_execution / config / layer*_* import below.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Tick clock — replaces time.monotonic in layer3_execution so HPO can run
# its short training/eval loops without burning wall-clock seconds on the
# inventory-time penalty.
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

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import config  # noqa: E402
from config import (  # noqa: E402
    HPO_DEGRADATION_THRESHOLD,
    HPO_N_TRIALS,
    HPO_TRAIN_STEPS,
)
from layer1_sensors import OODDetector  # noqa: E402
from layer2_alpha import AlphaEngine, TCNSpikePredictor  # noqa: E402
from layer3_execution import ExecutionEnv, PPOAgent, PPOTrainer, RolloutBuffer  # noqa: E402
from matching_engine import LocalMatchingEngine  # noqa: E402
from tests.test_replay import ReplaySensorArray  # noqa: E402

log = logging.getLogger(__name__)

_HPO_CLOCK_TICK = 0.05
_HPO_TIME_BUDGET = 5.0
_HPO_MAX_TICKS = 120


@dataclass
class HPOTrialResult:
    eta: float
    gamma_inv: float
    terminal_mult: float
    objective: float
    mean_is: float
    fill_price_std: float
    terminal_penalty_rate: float


def _run_episode(
    env: ExecutionEnv,
    agent: PPOAgent,
    state_advancer: Callable[[], bool],
    rollout: Optional[RolloutBuffer] = None,
    deterministic: bool = False,
) -> tuple[int, bool]:
    """Run one ExecutionEnv episode against a state-advancing callable.

    `state_advancer()` advances the live sensor's state to the next CSV row and
    returns True on success, False when the stream is exhausted.
    Returns: (transitions_appended, episode_completed_normally).
    """
    _clock.reset(0.0)
    obs, _ = env.reset()
    env.time_budget = _HPO_TIME_BUDGET
    transitions = 0
    completed = False
    for _ in range(_HPO_MAX_TICKS):
        action, log_prob, value = agent.act(obs, deterministic=deterministic)
        if rollout is not None:
            rollout.obs.append(obs.copy())
            rollout.actions.append(action.copy())
            rollout.log_probs.append(log_prob)
            rollout.values.append(value)
        if not state_advancer():
            if rollout is not None:
                rollout.rewards.append(0.0)
                rollout.dones.append(True)
            break
        _clock.advance(_HPO_CLOCK_TICK)
        obs, reward, terminated, truncated, _ = env.step(action)
        if rollout is not None:
            rollout.rewards.append(float(reward))
            rollout.dones.append(bool(terminated or truncated))
        transitions += 1
        if terminated or truncated:
            completed = True
            break
    return transitions, completed


def evaluate_objective(
    eta: float,
    gamma_inv: float,
    terminal_mult: float,
    train_data_provider: Callable,
    eval_data_provider: Callable,
    train_steps: int = HPO_TRAIN_STEPS,
) -> HPOTrialResult:
    """
    Train an agent with given reward weights on `train_data_provider` then
    evaluate on `eval_data_provider`. Each provider yields tuples of
    `(mandate, env, state_advancer)` — env shares the sensor with the
    advancer so env.step sees fresh ob_snapshots between ticks.

    Returns the composite objective and the three component metrics.
    """
    agent = PPOAgent()
    trainer = PPOTrainer(agent)
    rollout = RolloutBuffer()

    # Training loop ----------------------------------------------------------
    steps = 0
    for mandate, env_train, state_advancer in train_data_provider(
        eta=eta, gamma_inv=gamma_inv, terminal_mult=terminal_mult,
    ):
        if steps >= train_steps:
            break
        env_train.attach_mandate(mandate)
        appended, _ = _run_episode(env_train, agent, state_advancer, rollout=rollout)
        steps += appended
        if len(rollout) >= 2048:
            trainer.update(rollout, last_value=0.0)
            rollout.clear()

    # Evaluation -------------------------------------------------------------
    is_list: list[float] = []
    std_list: list[float] = []
    terminal_count = 0
    n_episodes = 0
    for mandate, env_eval, state_advancer in eval_data_provider(
        eta=eta, gamma_inv=gamma_inv, terminal_mult=terminal_mult,
    ):
        env_eval.attach_mandate(mandate)
        _, _ = _run_episode(env_eval, agent, state_advancer, rollout=None, deterministic=True)
        result = env_eval.episode_result()
        is_list.append(result.mean_is)
        std_list.append(result.fill_price_std)
        if result.forced_terminal:
            terminal_count += 1
        n_episodes += 1
    if n_episodes == 0:
        return HPOTrialResult(eta, gamma_inv, terminal_mult, float("inf"), 0, 0, 0)

    mean_is = float(np.mean(is_list))
    fill_std = float(np.mean(std_list))
    term_rate = terminal_count / n_episodes
    objective = mean_is + 0.5 * fill_std + 2.0 * term_rate
    return HPOTrialResult(eta, gamma_inv, terminal_mult, objective, mean_is, fill_std, term_rate)


# ---------------------------------------------------------------------------
# CSV-backed data provider — yields (mandate, env, state_advancer) tuples.
# ---------------------------------------------------------------------------
def _load_calibration_for(symbol: str) -> tuple[OODDetector, float]:
    ood = OODDetector()
    alpha_c = float(config.CALIBRATION_REGISTRY.get(symbol, 0.25))
    cal = Path(f"calibration/latest_{symbol}.json")
    if cal.exists():
        try:
            payload = json.loads(cal.read_text())
            ood.load(np.array(payload["ood_mu"]), np.array(payload["ood_sigma"]))
            alpha_c = float(payload.get("alpha_calibration_c", alpha_c))
        except Exception as e:
            log.warning("OOD load failed: %s", e)
    return ood, alpha_c


def _load_alpha_engine(symbol: str, turbulence_threshold: float) -> AlphaEngine:
    tcn = TCNSpikePredictor()
    alpha = AlphaEngine(tcn=tcn, turbulence_threshold=turbulence_threshold)
    weights = Path(f"calibration/tcn_weights_{symbol}.pt")
    if weights.exists():
        try:
            alpha.load_weights(str(weights))
        except Exception as e:
            log.warning("TCN load failed: %s", e)
    return alpha


def _stream_rows_in_range(csv_path: Path, start_frac: float, end_frac: float) -> Iterator[dict]:
    total = pl.scan_csv(str(csv_path)).select(pl.len()).collect().item()
    start_idx = int(total * start_frac)
    end_idx = int(total * end_frac)
    reader = pl.read_csv_batched(str(csv_path))
    batches = reader.next_batches(1)
    cursor = 0
    while batches:
        for batch in batches:
            if cursor + batch.height <= start_idx:
                cursor += batch.height
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


def make_csv_data_provider(
    csv_path: Path,
    symbol: str,
    start_frac: float,
    end_frac: float,
    turbulence_threshold: float = 0.05,
    target_multiplier: float = 20.0,
) -> Callable:
    """Return a provider callable usable as train/eval/holdout in `run_hpo`.

    The provider builds a fresh sensor, alpha engine, and ExecutionEnv per
    call (one per HPO trial) so the trial's hyperparams (eta, gamma_inv,
    terminal_mult) flow into the env on construction.
    """
    def provider(eta: float, gamma_inv: float, terminal_mult: float):
        ood, alpha_c = _load_calibration_for(symbol)
        sensor = ReplaySensorArray(
            csv_path=str(csv_path),
            symbol=symbol,
            alpha_calibration_c=alpha_c,
            ood_detector=ood,
        )
        alpha = _load_alpha_engine(symbol, turbulence_threshold)
        env = ExecutionEnv(
            live_state_provider=sensor,
            matching_engine=LocalMatchingEngine(),
            shadow_mode=True,
            eta=eta,
            gamma_inv=gamma_inv,
            terminal_mult=terminal_mult,
        )
        row_iter = _stream_rows_in_range(csv_path, start_frac, end_frac)

        def state_advancer() -> bool:
            try:
                row = next(row_iter)
            except StopIteration:
                return False
            sensor.state = sensor._row_to_state(row)
            return True

        # Warm up the AlphaEngine TCN buffer before yielding mandates.
        for row in row_iter:
            sensor.state = sensor._row_to_state(row)
            mandate = alpha.evaluate(sensor.state)
            if mandate is None:
                continue
            if target_multiplier != 1.0:
                mandate.target_size = float(mandate.target_size * target_multiplier)
            yield (mandate, env, state_advancer)

    return provider


def run_hpo(
    train_data_provider: Callable,
    eval_data_provider: Callable,
    holdout_data_provider: Callable,
    n_trials: int = HPO_N_TRIALS,
    output_path: Path = Path("./calibration/hpo_winning_trial.json"),
) -> dict:
    """
    Optuna search over the three reward weights. Validates the winning trial
    on `holdout_data_provider` (week N+1). If the holdout objective degrades
    by more than HPO_DEGRADATION_THRESHOLD relative to training, the trial is
    REJECTED and current defaults are retained.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("optuna is required for HPO; install via `pip install optuna`") from e

    def objective(trial):
        eta = trial.suggest_float("eta", 0.05, 1.0, log=True)
        gamma_inv = trial.suggest_float("gamma_inv", 0.1, 2.0, log=True)
        terminal_mult = trial.suggest_float("terminal_mult", 1.5, 5.0)
        result = evaluate_objective(
            eta, gamma_inv, terminal_mult,
            train_data_provider, eval_data_provider,
        )
        trial.set_user_attr("mean_is", result.mean_is)
        trial.set_user_attr("fill_std", result.fill_price_std)
        trial.set_user_attr("term_rate", result.terminal_penalty_rate)
        return result.objective

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    log.info("best trial: %s (objective=%.4f)", best.params, best.value)

    # Holdout validation -----------------------------------------------------
    holdout = evaluate_objective(
        best.params["eta"], best.params["gamma_inv"], best.params["terminal_mult"],
        train_data_provider, holdout_data_provider,
    )
    train_obj = best.value
    degradation = (holdout.objective - train_obj) / max(abs(train_obj), 1e-8)

    accepted = degradation <= HPO_DEGRADATION_THRESHOLD
    payload = {
        "params": best.params,
        "training_objective": train_obj,
        "holdout_objective": holdout.objective,
        "degradation": degradation,
        "accepted": accepted,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    output_path.write_text(json.dumps(payload, indent=2))
    if not accepted:
        log.warning(
            "REJECTED: holdout degradation %.3f > %.3f. Keeping defaults.",
            degradation, HPO_DEGRADATION_THRESHOLD,
        )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=HPO_N_TRIALS)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--symbol", default=os.environ.get("SYMBOL", "TSLA"))
    parser.add_argument("--turbulence-threshold", type=float, default=0.05)
    parser.add_argument("--target-multiplier", type=float, default=20.0)
    parser.add_argument(
        "--splits", nargs=3, type=float,
        default=[0.6, 0.8, 1.0],
        metavar=("TRAIN_END", "EVAL_END", "HOLDOUT_END"),
        help="Cumulative fractions: train=[0, TRAIN_END), eval=[TRAIN_END, EVAL_END), "
             "holdout=[EVAL_END, HOLDOUT_END]. Default 0.6/0.8/1.0.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("./calibration/hpo_winning_trial.json"),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.csv.exists():
        log.error("CSV %s not found.", args.csv)
        sys.exit(1)
    if args.symbol:
        os.environ["SYMBOL"] = args.symbol

    train_end, eval_end, holdout_end = args.splits
    train_provider = make_csv_data_provider(
        args.csv, args.symbol, 0.0, train_end,
        turbulence_threshold=args.turbulence_threshold,
        target_multiplier=args.target_multiplier,
    )
    eval_provider = make_csv_data_provider(
        args.csv, args.symbol, train_end, eval_end,
        turbulence_threshold=args.turbulence_threshold,
        target_multiplier=args.target_multiplier,
    )
    holdout_provider = make_csv_data_provider(
        args.csv, args.symbol, eval_end, holdout_end,
        turbulence_threshold=args.turbulence_threshold,
        target_multiplier=args.target_multiplier,
    )

    log.info(
        "HPO start: csv=%s n_trials=%d splits=train[0,%.2f) eval[%.2f,%.2f) holdout[%.2f,%.2f]",
        args.csv, args.n_trials, train_end, train_end, eval_end, eval_end, holdout_end,
    )
    t0 = _real_time.monotonic()
    payload = run_hpo(
        train_data_provider=train_provider,
        eval_data_provider=eval_provider,
        holdout_data_provider=holdout_provider,
        n_trials=args.n_trials,
        output_path=args.output,
    )
    log.info(
        "HPO done in %.1fs. accepted=%s degradation=%.3f. params=%s",
        _real_time.monotonic() - t0, payload["accepted"],
        payload["degradation"], payload["params"],
    )
