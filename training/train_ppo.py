"""
train_ppo.py — Offline PPO training for the Layer 3 execution agent.

Replays a `feature_history_<SYMBOL>.csv` through (ReplaySensorArray → AlphaEngine
→ ExecutionEnv) to generate mandates and train the PPOAgent against simulated
fills from LocalMatchingEngine. Output:

    calibration/ppo_weights_<SYMBOL>.pt        — final policy weights
    calibration/ppo_train_log_<SYMBOL>.csv     — per-update metrics

Time is controlled by a tick-driven simulated clock that replaces
`time.monotonic` in `layer3_execution` so episodes step deterministically
with CSV rows rather than wall clock. `env.time_budget` is clamped to
TRAINING_TIME_BUDGET so episodes have a bounded simulated duration.

Usage (PowerShell):
    $env:SYMBOL = "TSLA"
    python train_ppo.py --symbol TSLA `
        --csv calibration/feature_history_TSLA_balanced.csv `
        --total-updates 200 --turbulence-threshold 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import os
import sys
import time as _real_time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Parse --symbol BEFORE importing config so config.SYMBOL is correct.
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
# Synthetic clock — replaces time.monotonic in layer3_execution.
# Set before any other import that pulls in layer3_execution transitively.
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
# Standard imports (after clock patch).
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from layer1_sensors import OODDetector  # noqa: E402
from layer2_alpha import AlphaEngine, TCNSpikePredictor  # noqa: E402
from layer3_execution import ExecutionEnv, PPOAgent, PPOTrainer, RolloutBuffer  # noqa: E402
from matching_engine import LocalMatchingEngine  # noqa: E402
from tests.test_replay import ReplaySensorArray  # noqa: E402

log = logging.getLogger("train_ppo")

CLOCK_TICK_SECONDS = 0.05            # simulated time per env step
TRAINING_TIME_BUDGET_SECONDS = 5.0   # clamp env.time_budget for compute-bounded episodes
MAX_TICKS_PER_EPISODE = 120          # hard cap (> TIME_BUDGET / TICK so time_up fires first)


# ---------------------------------------------------------------------------
# Calibration loaders — mirror engine.py's path
# ---------------------------------------------------------------------------
def load_calibration_for(symbol: str) -> tuple[OODDetector, float]:
    """Return (OODDetector with loaded μ/Σ if available, alpha_c)."""
    ood = OODDetector()
    alpha_c = float(config.CALIBRATION_REGISTRY.get(symbol, 0.25))
    cal_path = Path(f"calibration/latest_{symbol}.json")
    if not cal_path.exists():
        log.warning("OOD calibration %s missing; cold-start prior", cal_path)
        return ood, alpha_c
    try:
        payload = json.loads(cal_path.read_text())
        mu = np.array(payload["ood_mu"], dtype=np.float64)
        sigma = np.array(payload["ood_sigma"], dtype=np.float64)
        ood.load(mu, sigma)
        alpha_c = float(payload.get("alpha_calibration_c", alpha_c))
        log.info("Loaded OOD calibration from %s (α=%.4f)", cal_path, alpha_c)
    except Exception as e:
        log.warning("Failed to load %s: %s — using cold-start prior", cal_path, e)
    return ood, alpha_c


def load_alpha_engine(symbol: str, turbulence_threshold: float, device: str) -> AlphaEngine:
    tcn = TCNSpikePredictor()
    alpha = AlphaEngine(tcn=tcn, device=device, turbulence_threshold=turbulence_threshold)
    weights_path = Path(f"calibration/tcn_weights_{symbol}.pt")
    if weights_path.exists():
        try:
            alpha.load_weights(str(weights_path))
            log.info("Loaded TCN weights from %s", weights_path)
        except Exception as e:
            log.warning("Failed to load %s: %s — random-init TCN", weights_path, e)
    else:
        log.warning("TCN weights %s missing — random-init TCN (mandate stream will be noise-driven)", weights_path)
    return alpha


# ---------------------------------------------------------------------------
# CSV streaming — re-creates reader when the file is exhausted.
# ---------------------------------------------------------------------------
def stream_csv_rows(csv_path: Path, restart: bool = True) -> Iterator[dict]:
    while True:
        reader = pl.read_csv_batched(str(csv_path))
        batches = reader.next_batches(1)
        while batches:
            for batch in batches:
                for row in batch.iter_rows(named=True):
                    yield row
            batches = reader.next_batches(1)
        if not restart:
            return


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(
    symbol: str,
    csv_path: Path,
    output_dir: Path,
    total_updates: int,
    turbulence_threshold: float,
    rollout_size: int,
    save_every: int,
    seed: int,
    device: str,
    log_every_updates: int,
    target_multiplier: float,
    lr: float,
    entropy_coef: float,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"ppo_train_log_{symbol}.csv"
    final_weights_path = output_dir / f"ppo_weights_{symbol}.pt"

    ood, alpha_c = load_calibration_for(symbol)
    sensor = ReplaySensorArray(
        csv_path=str(csv_path),
        symbol=symbol,
        alpha_calibration_c=alpha_c,
        ood_detector=ood,
    )
    alpha = load_alpha_engine(symbol, turbulence_threshold, device=device)

    agent = PPOAgent()
    trainer = PPOTrainer(agent, device=device, lr=lr, entropy_coef=entropy_coef)
    log.info("PPOTrainer: lr=%.1e entropy_coef=%.3f", lr, entropy_coef)
    env = ExecutionEnv(
        live_state_provider=sensor,
        matching_engine=LocalMatchingEngine(),
        shadow_mode=True,
    )

    # Save random-init weights as a sanity baseline for Phase 2 comparison.
    random_init_path = output_dir / f"ppo_weights_{symbol}_random_init.pt"
    torch.save(agent.state_dict(), random_init_path)
    log.info("Saved random-init checkpoint → %s", random_init_path)

    metric_handle = log_path.open("w", newline="")
    metric_writer = _csv.writer(metric_handle)
    metric_writer.writerow([
        "update", "episodes", "transitions",
        "mean_ep_reward", "mean_ep_is", "mean_terminal_rate", "mean_filled_frac",
        "policy_loss", "value_loss", "entropy", "kl",
        "wall_time_s",
    ])

    rollout = RolloutBuffer()
    update_idx = 0
    n_episodes_total = 0
    ep_rewards: list[float] = []
    ep_mean_is: list[float] = []
    ep_terminal: list[int] = []
    ep_filled_frac: list[float] = []
    wall_start = _real_time.monotonic()

    row_iter = stream_csv_rows(csv_path, restart=True)

    def advance_row() -> Optional[dict]:
        try:
            row = next(row_iter)
        except StopIteration:
            return None
        sensor.state = sensor._row_to_state(row)
        return row

    log.info(
        "Starting training: symbol=%s csv=%s total_updates=%d turb_thresh=%.3f "
        "rollout=%d device=%s",
        symbol, csv_path, total_updates, turbulence_threshold, rollout_size, device,
    )

    try:
        while update_idx < total_updates:
            row = advance_row()
            if row is None:
                # Should not happen since stream_csv_rows restarts; defensive.
                continue
            mandate = alpha.evaluate(sensor.state)
            if mandate is None:
                continue

            # CAPTURE_RATIO=0.05 + 1-level synthesized book leaves target_size
            # an order of magnitude below opposing depth; episodes collapse to
            # 1-2 ticks and PPO never sees the maker/taker decision boundary.
            # Multiplier inflates target_size during training so episodes
            # exercise the full passive-vs-aggressive dynamic.
            if target_multiplier != 1.0:
                mandate.target_size = float(mandate.target_size * target_multiplier)

            # Episode setup
            _clock.reset(0.0)
            env.attach_mandate(mandate)
            obs, _ = env.reset()
            # Override mandate's natural window to compute-bounded training budget
            env.time_budget = TRAINING_TIME_BUDGET_SECONDS

            ep_reward_sum = 0.0
            episode_filled = 0.0
            episode_term = False

            csv_exhausted = False
            done_flag = False
            for tick in range(MAX_TICKS_PER_EPISODE):
                action, log_prob, value = agent.act(obs)
                rollout.obs.append(obs.copy())
                rollout.actions.append(action.copy())
                rollout.log_probs.append(log_prob)
                rollout.values.append(value)

                next_row = advance_row()
                if next_row is None:
                    csv_exhausted = True
                    rollout.rewards.append(0.0)
                    rollout.dones.append(True)
                    done_flag = True
                    break
                _clock.advance(CLOCK_TICK_SECONDS)

                obs, reward, terminated, truncated, info = env.step(action)
                rollout.rewards.append(float(reward))
                done_step = bool(terminated or truncated)
                rollout.dones.append(done_step)
                ep_reward_sum += float(reward)
                episode_filled = float(info.get("remaining_volume", 0.0))
                if truncated:
                    episode_term = True
                if done_step:
                    done_flag = True
                    break

            if not done_flag:
                # Loop hit MAX_TICKS without env termination — force terminal fill.
                side = "buy" if mandate.direction > 0 else "sell"
                terminal_r = float(env._force_terminal_fill(side))
                if rollout.rewards:
                    rollout.rewards[-1] += terminal_r
                    rollout.dones[-1] = True
                ep_reward_sum += terminal_r
                episode_term = True

            result = env.episode_result()
            filled_frac = (
                1.0 - (env.remaining_volume / max(mandate.target_size, 1e-9))
            )
            ep_rewards.append(ep_reward_sum)
            ep_mean_is.append(float(result.mean_is))
            ep_terminal.append(1 if episode_term else 0)
            ep_filled_frac.append(float(filled_frac))
            n_episodes_total += 1

            # PPO update gate
            if len(rollout) >= rollout_size:
                # Bootstrap last_value if the final transition isn't terminal.
                last_value = 0.0
                if rollout.dones and not rollout.dones[-1]:
                    with torch.no_grad():
                        x = torch.from_numpy(obs[None, :]).float().to(trainer.device)
                        _, v = agent(x)
                        last_value = float(v.item())
                metrics = trainer.update(rollout, last_value=last_value)
                wall = _real_time.monotonic() - wall_start
                mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                mean_is = float(np.mean(ep_mean_is)) if ep_mean_is else 0.0
                term_rate = float(np.mean(ep_terminal)) if ep_terminal else 0.0
                fill_rate = float(np.mean(ep_filled_frac)) if ep_filled_frac else 0.0

                metric_writer.writerow([
                    update_idx, n_episodes_total, len(rollout),
                    f"{mean_r:.6f}", f"{mean_is:.6f}", f"{term_rate:.4f}", f"{fill_rate:.4f}",
                    f"{metrics['policy_loss']:.6f}", f"{metrics['value_loss']:.6f}",
                    f"{metrics['entropy']:.6f}", f"{metrics['kl']:.6f}",
                    f"{wall:.1f}",
                ])
                metric_handle.flush()
                if update_idx % log_every_updates == 0 or update_idx == total_updates - 1:
                    log.info(
                        "upd %3d | eps %5d trans %5d | r=%.4f IS=%.5f term=%.2f fill=%.2f | "
                        "ploss=%.4f vloss=%.4f ent=%.3f kl=%.4f | t=%.1fs",
                        update_idx, n_episodes_total, len(rollout),
                        mean_r, mean_is, term_rate, fill_rate,
                        metrics["policy_loss"], metrics["value_loss"],
                        metrics["entropy"], metrics["kl"], wall,
                    )
                rollout.clear()
                ep_rewards.clear()
                ep_mean_is.clear()
                ep_terminal.clear()
                ep_filled_frac.clear()
                update_idx += 1
                if save_every > 0 and update_idx % save_every == 0:
                    ckpt = output_dir / f"ppo_weights_{symbol}_step{update_idx:05d}.pt"
                    torch.save(agent.state_dict(), ckpt)
                    log.info("Checkpoint → %s", ckpt)
            if csv_exhausted:
                # advance_row signaled exhaustion mid-episode; reset iterator.
                row_iter = stream_csv_rows(csv_path, restart=True)
    finally:
        metric_handle.close()
        torch.save(agent.state_dict(), final_weights_path)
        log.info(
            "Done. Final weights → %s. Episodes=%d, updates=%d, wall=%.1fs",
            final_weights_path, n_episodes_total, update_idx,
            _real_time.monotonic() - wall_start,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="TSLA")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("calibration"), type=Path)
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument(
        "--turbulence-threshold", type=float, default=0.05,
        help="Override config.TURBULENCE_THRESHOLD during training (lower → denser mandates).",
    )
    parser.add_argument("--rollout-size", type=int, default=2048)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-every-updates", type=int, default=1)
    parser.add_argument(
        "--target-multiplier", type=float, default=20.0,
        help="Scale mandate.target_size to force multi-tick episodes. "
             "Production CAPTURE_RATIO=0.05 sizes single-tick fill on the "
             "1-level synthesized replay book.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override PPO_LR (default 3e-4). Lower (e.g. 1e-4) for tiny-reward stability.",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=None,
        help="Override PPO_ENTROPY_COEF (default 0.01). Set 0 if entropy term dominates the tiny IS-reward.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.csv.exists():
        log.error("CSV %s not found.", args.csv)
        sys.exit(1)
    train(
        symbol=args.symbol,
        csv_path=args.csv,
        output_dir=args.output_dir,
        total_updates=args.total_updates,
        turbulence_threshold=args.turbulence_threshold,
        rollout_size=args.rollout_size,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        log_every_updates=args.log_every_updates,
        target_multiplier=args.target_multiplier,
        lr=(args.lr if args.lr is not None else config.PPO_LR),
        entropy_coef=(args.entropy_coef if args.entropy_coef is not None else config.PPO_ENTROPY_COEF),
    )


if __name__ == "__main__":
    main()
