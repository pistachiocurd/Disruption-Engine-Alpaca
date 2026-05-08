"""
hpo.py — Optuna hyperparameter optimization for the execution reward weights.

Searches over (ETA, GAMMA_INV, TERMINAL_PENALTY_MULTIPLIER). Trains a fresh
PPO agent per trial against the shadow simulator's environment, evaluates on a
held-out window, and validates the winning trial on the SUBSEQUENT week to
guard against overfitting.

Run:
    python hpo.py
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from config import (
    HPO_DEGRADATION_THRESHOLD,
    HPO_N_TRIALS,
    HPO_TRAIN_STEPS,
)
from layer3_execution import ExecutionEnv, PPOAgent, PPOTrainer, RolloutBuffer

log = logging.getLogger(__name__)


@dataclass
class HPOTrialResult:
    eta: float
    gamma_inv: float
    terminal_mult: float
    objective: float
    mean_is: float
    fill_price_std: float
    terminal_penalty_rate: float


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
    evaluate on `eval_data_provider`. Both providers should yield mandate +
    state stream tuples. The actual provider implementation is environment-
    specific — see `engine.py` for a real one.

    Returns the composite objective and the three component metrics.
    """
    agent = PPOAgent()
    trainer = PPOTrainer(agent)
    env_train = ExecutionEnv(eta=eta, gamma_inv=gamma_inv, terminal_mult=terminal_mult)
    rollout = RolloutBuffer()

    # Training loop ----------------------------------------------------------
    steps = 0
    for mandate, state_stream in train_data_provider():
        if steps >= train_steps:
            break
        env_train.attach_mandate(mandate)
        obs, _ = env_train.reset()
        for _ in range(1024):
            action, log_prob, value = agent.act(obs)
            rollout.obs.append(obs)
            rollout.actions.append(action)
            rollout.log_probs.append(log_prob)
            rollout.values.append(value)
            obs, reward, terminated, truncated, _ = env_train.step(action)
            rollout.rewards.append(reward)
            rollout.dones.append(terminated or truncated)
            steps += 1
            if terminated or truncated:
                break
        if len(rollout) >= 2048:
            trainer.update(rollout, last_value=0.0)
            rollout.clear()

    # Evaluation -------------------------------------------------------------
    env_eval = ExecutionEnv(eta=eta, gamma_inv=gamma_inv, terminal_mult=terminal_mult)
    is_list, std_list, terminal_count, n_episodes = [], [], 0, 0
    for mandate, _ in eval_data_provider():
        env_eval.attach_mandate(mandate)
        obs, _ = env_eval.reset()
        done = False
        while not done:
            action, _, _ = agent.act(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env_eval.step(action)
            done = terminated or truncated
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
    args = parser.parse_args()
    log.warning(
        "Running HPO requires real train/eval/holdout data providers. "
        "Stub providers will not produce meaningful results. "
        "Plumb in your archived L2 + mandate stream from engine.py."
    )
