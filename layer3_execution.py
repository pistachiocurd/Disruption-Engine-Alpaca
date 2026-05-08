"""
layer3_execution.py — Optimal execution.

ExecutionEnv:
    Gymnasium environment. One mandate = one episode. 42-dim observation,
    continuous scalar action. Reward = mean-variance fee-adjusted IS with
    inventory urgency penalty. CRITICAL: no directional information in the
    state space; the agent has no opinion about price direction, only about
    how to fill the requested volume cheaply.

PPOAgent:
    Two SEPARATE 1D CNN encoders (bid / ask), then fusion with context, then
    actor (Normal head) and critic (V) heads. From-scratch PPO trainer below.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import (
    ACTION_DIM,
    EPSILON,
    ETA,
    EXECUTION_HARD_STOP_MULTIPLIER,
    FEE_AGGRESSION_THRESHOLD,
    GAMMA_INV,
    MAKER_FEE,
    MIN_ORDER_SIZE,
    OBS_DIM,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_VALUE_COEF,
    SHADOW_MODE,
    TAKER_FEE,
    TERMINAL_PENALTY_MULTIPLIER,
)
from matching_engine import LocalMatchingEngine

log = logging.getLogger(__name__)


# ============================================================================
# ExecutionEnv
# ============================================================================
@dataclass
class EpisodeResult:
    mean_is: float
    fill_price_std: float
    forced_terminal: bool
    total_filled: float
    target_size: float
    fee_paid_total: float
    n_fills: int


class ExecutionEnv(gym.Env):
    """
    One mandate → one episode.

    State (42 dims, ALL normalized):
        bid_prices_bps        : 10  (top 10 bids as bps from mid)
        bid_cum_vol_norm      : 10  (cumulative bid depth / target_size)
        ask_prices_bps        : 10  (top 10 asks as bps from mid)
        ask_cum_vol_norm      : 10  (cumulative ask depth / target_size)
        remaining_vol_frac    :  1
        time_remaining_frac   :  1

    Action: scalar in [-1.0, 1.0].
        action <  threshold → market (taker)
        action >= threshold → passive limit (maker), depth = action * MAX_PASSIVE_OFFSET_BPS

    Reward (per step, fee-adjusted):
        r = -IS_fee_adj  -  ETA * fill_var  -  GAMMA_INV * inventory_time_penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        live_state_provider: Any = None,    # something exposing .state.ob_snapshot
        matching_engine: Optional[LocalMatchingEngine] = None,
        exchange: Any = None,
        shadow_mode: bool = SHADOW_MODE,
        eta: float = ETA,
        gamma_inv: float = GAMMA_INV,
        terminal_mult: float = TERMINAL_PENALTY_MULTIPLIER,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.live_state_provider = live_state_provider
        self.matching_engine = matching_engine or LocalMatchingEngine()
        self.exchange = exchange
        self.shadow_mode = shadow_mode
        self.eta = eta
        self.gamma_inv = gamma_inv
        self.terminal_mult = terminal_mult

        # Episode state
        self.mandate = None
        self.remaining_volume = 0.0
        self.time_start: float = 0.0
        self.time_budget: float = 0.0
        self.fill_history: list[float] = []          # fee-adjusted fill prices
        self.fee_total: float = 0.0
        self.prev_ob: Optional[dict] = None
        self.curr_ob: Optional[dict] = None
        self._episode_done = False
        self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)

    # ------------------------------------------------------------ life-cycle
    def attach_mandate(self, mandate) -> None:
        """Bind a TradeMandate. Must be called before reset()."""
        self.mandate = mandate

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.mandate is None:
            raise RuntimeError("attach_mandate() must be called before reset().")
        self.remaining_volume = float(self.mandate.target_size)
        self.time_start = time.monotonic()
        self.time_budget = float(self.mandate.execution_window_seconds)
        self.fill_history = []
        self.fee_total = 0.0
        self.prev_ob = None
        self.curr_ob = self._get_live_ob()
        self._episode_done = False
        self.matching_engine = LocalMatchingEngine()  # fresh per-episode
        obs = self._build_observation()
        self._last_obs = obs
        return obs, {}

    # --------------------------------------------------------- live ob hook
    def _get_live_ob(self) -> Optional[dict]:
        if self.live_state_provider is None:
            return None
        state = getattr(self.live_state_provider, "state", None)
        return getattr(state, "ob_snapshot", None) if state else None

    # ------------------------------------------------------------- step API
    def step(self, action):
        if self._episode_done:
            raise RuntimeError("step() called after episode completion.")
        a = float(np.clip(action, -1.0, 1.0)[0] if hasattr(action, "__len__") else action)
        self.prev_ob = self.curr_ob
        self.curr_ob = self._get_live_ob() or self.curr_ob

        side = "buy" if self.mandate.direction > 0 else "sell"

        if self.shadow_mode or self.exchange is None:
            fill = self.matching_engine.attempt_fill(
                action=a,
                side=side,
                remaining_volume=self.remaining_volume,
                ob_snapshot=self.curr_ob,
                prev_ob_snapshot=self.prev_ob,
            )
            filled = fill.filled_qty
            fill_price = fill.fill_price
            fee_paid = fill.fee_paid
            is_taker = fill.is_taker
        else:
            # Live path — submit a real order. Wrap in try; on failure treat as no-fill.
            filled, fill_price, fee_paid, is_taker = self._live_submit(a, side)

        if filled > 0:
            fee_rate = TAKER_FEE if is_taker else MAKER_FEE
            if self.mandate.direction > 0:
                fee_adj = fill_price * (1.0 + fee_rate)
            else:
                fee_adj = fill_price * (1.0 - fee_rate)
            self.fill_history.append(fee_adj)
            self.fee_total += fee_paid
            self.remaining_volume = max(0.0, self.remaining_volume - filled)

        # Reward components ----------------------------------------------------
        step_is = self._step_is_fee_adj(filled, fill_price, is_taker)
        fill_var = self._fill_variance()
        inv_pen = self._inventory_time_penalty()
        reward = -step_is - self.eta * fill_var - self.gamma_inv * inv_pen

        # Termination ---------------------------------------------------------
        time_elapsed = time.monotonic() - self.time_start
        time_up = time_elapsed >= self.time_budget
        volume_done = self.remaining_volume <= MIN_ORDER_SIZE * 0.5

        terminated = volume_done
        truncated = time_up and not volume_done

        terminal_reward = 0.0
        if truncated and self.remaining_volume > MIN_ORDER_SIZE * 0.5:
            terminal_reward = self._force_terminal_fill(side)

        reward += terminal_reward
        self._episode_done = terminated or truncated

        obs = self._build_observation()
        self._last_obs = obs

        info = {
            "filled": filled,
            "fill_price": fill_price,
            "is_taker": is_taker,
            "remaining_volume": self.remaining_volume,
            "step_is": step_is,
            "fill_var": fill_var,
            "inv_pen": inv_pen,
            "terminal_reward": terminal_reward,
            "fee_total": self.fee_total,
        }
        return obs, float(reward), terminated, truncated, info

    # --------------------------------------------------- reward sub-routines
    def _step_is_fee_adj(self, filled: float, fill_price: float, is_taker: bool) -> float:
        if filled <= 0:
            return 0.0
        fee_rate = TAKER_FEE if is_taker else MAKER_FEE
        if self.mandate.direction > 0:
            fee_adj = fill_price * (1.0 + fee_rate)
            return (fee_adj - self.mandate.p_decision) / self.mandate.p_decision
        fee_adj = fill_price * (1.0 - fee_rate)
        return (self.mandate.p_decision - fee_adj) / self.mandate.p_decision

    def _fill_variance(self) -> float:
        if len(self.fill_history) < 2:
            return 0.0
        p_dec = self.mandate.p_decision
        return float(np.var(self.fill_history) / (p_dec * p_dec + EPSILON))

    def _inventory_time_penalty(self) -> float:
        rem_frac = self.remaining_volume / max(self.mandate.target_size, EPSILON)
        time_elapsed = time.monotonic() - self.time_start
        time_rem_frac = max(0.0, 1.0 - time_elapsed / max(self.time_budget, EPSILON))
        return rem_frac * (1.0 - time_rem_frac) ** 2

    def _force_terminal_fill(self, side: str) -> float:
        """
        Window expired with volume outstanding. Force market fill, double-penalty.
        """
        if self.curr_ob is None or self.remaining_volume <= 0:
            return 0.0
        # Walk the book at full aggression.
        book = self.curr_ob["asks"] if side == "buy" else self.curr_ob["bids"]
        rem = self.remaining_volume
        cost = 0.0
        filled = 0.0
        for price, size in book:
            if rem <= 0:
                break
            take = min(size, rem)
            cost += take * price
            filled += take
            rem -= take
        if filled <= 0:
            return -self.terminal_mult  # nothing left to fill against — heavy penalty
        avg = cost / filled
        if self.mandate.direction > 0:
            fee_adj = avg * (1.0 + TAKER_FEE)
            step_is = (fee_adj - self.mandate.p_decision) / self.mandate.p_decision
        else:
            fee_adj = avg * (1.0 - TAKER_FEE)
            step_is = (self.mandate.p_decision - fee_adj) / self.mandate.p_decision

        self.fill_history.append(fee_adj)
        self.fee_total += filled * avg * TAKER_FEE
        self.remaining_volume = max(0.0, rem)
        return -self.terminal_mult * step_is

    # ----------------------------------------------------- observation build
    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        ob = self.curr_ob
        if ob is None or not ob.get("bids") or not ob.get("asks"):
            return obs
        mid = 0.5 * (ob["bids"][0][0] + ob["asks"][0][0])
        if mid <= 0:
            return obs

        bids = ob["bids"][:10]
        asks = ob["asks"][:10]
        target = max(self.mandate.target_size, EPSILON)

        # Bids (slots 0..19)
        cum = 0.0
        for i in range(10):
            if i < len(bids):
                p, s = bids[i]
                obs[i] = (p - mid) / mid * 10_000.0   # bps (negative for bids)
                cum += s
                obs[10 + i] = cum / target
            else:
                obs[i] = 0.0
                obs[10 + i] = cum / target

        # Asks (slots 20..39)
        cum = 0.0
        for i in range(10):
            if i < len(asks):
                p, s = asks[i]
                obs[20 + i] = (p - mid) / mid * 10_000.0   # bps (positive)
                cum += s
                obs[30 + i] = cum / target
            else:
                obs[20 + i] = 0.0
                obs[30 + i] = cum / target

        rem_frac = self.remaining_volume / target
        time_elapsed = time.monotonic() - self.time_start
        time_rem_frac = max(0.0, 1.0 - time_elapsed / max(self.time_budget, EPSILON))
        obs[40] = rem_frac
        obs[41] = time_rem_frac
        return obs

    # -------------------------------------------------------- live order path
    def _live_submit(self, action: float, side: str) -> tuple[float, float, float, bool]:
        """Synchronous fallback for the live path. Real engine uses async."""
        # In the asyncio engine, ExecutionEnv.step is called from the strategy
        # task; the actual exchange.create_order is wrapped at the Engine layer.
        # This fallback returns no-fill so the env remains useable without a
        # connected exchange (e.g. in unit tests).
        return 0.0, 0.0, 0.0, False

    # ----------------------------------------------------- summary at episode end
    def episode_result(self) -> EpisodeResult:
        if self.fill_history:
            mean_is = float(
                np.mean([(p - self.mandate.p_decision) / self.mandate.p_decision
                         for p in self.fill_history])
            )
            std_p = float(np.std(self.fill_history) / max(self.mandate.p_decision, EPSILON))
        else:
            mean_is = 0.0
            std_p = 0.0
        return EpisodeResult(
            mean_is=mean_is,
            fill_price_std=std_p,
            forced_terminal=self.remaining_volume > MIN_ORDER_SIZE * 0.5,
            total_filled=self.mandate.target_size - self.remaining_volume,
            target_size=self.mandate.target_size,
            fee_paid_total=self.fee_total,
            n_fills=len(self.fill_history),
        )


# ============================================================================
# PPO Agent — split CNN encoders + actor/critic heads
# ============================================================================
class _BookEncoder(nn.Module):
    """Conv1d encoder for one side of the book. Input shape (B, 2, 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return x.flatten(1)            # (B, 320) — 32 ch × 10 spatial


class PPOAgent(nn.Module):
    """
    Inputs: 42-dim observation.
    Splits into: bid prices (10) + bid vols (10) → (B, 2, 10) → bid_encoder → (B, 320)
                 ask prices (10) + ask vols (10) → (B, 2, 10) → ask_encoder → (B, 320)
                 context: [rem_frac, time_rem_frac] → (B, 2)
    Fusion → (B, 642)
    Heads: actor (Normal mean + log_std), critic (V).

    NB: spec says 322-dim fusion based on Conv1d output (2,16,3) → (16,10) flattened
    is 160; (16,32,3) → (32,10) flattened is 320. The brief says "Flatten → (160,)"
    from one block, but with the two-conv stack producing 32 channels the flatten
    is actually 320 per side. We stay faithful to the architecture (two convs,
    final 32 channels) and let the fusion size follow from the math.
    """

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.bid_encoder = _BookEncoder()
        self.ask_encoder = _BookEncoder()
        per_side = 32 * 10
        fusion = per_side * 2 + 2

        self.actor_trunk = nn.Sequential(
            nn.Linear(fusion, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        self.critic = nn.Sequential(
            nn.Linear(fusion, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    # ------------------------------------------------------------ utilities
    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs (B, 42)
        bid_p = obs[:, :10]
        bid_v = obs[:, 10:20]
        ask_p = obs[:, 20:30]
        ask_v = obs[:, 30:40]
        ctx = obs[:, 40:42]
        bid = torch.stack([bid_p, bid_v], dim=1)   # (B, 2, 10)
        ask = torch.stack([ask_p, ask_v], dim=1)
        return bid, ask, ctx

    def _fuse(self, obs: torch.Tensor) -> torch.Tensor:
        bid, ask, ctx = self._split(obs)
        return torch.cat([self.bid_encoder(bid), self.ask_encoder(ask), ctx], dim=-1)

    # ----------------------------------------------------- forward / sample
    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        h = self._fuse(obs)
        mean = torch.tanh(self.actor_mean(self.actor_trunk(h)))   # (B, 1) in [-1, 1]
        log_std = self.actor_log_std.expand_as(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        value = self.critic(h).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        x = torch.from_numpy(obs[None, :]).float()
        dist, value = self.forward(x)
        if deterministic:
            a = dist.mean
        else:
            a = dist.sample()
        a_clipped = torch.clamp(a, -1.0, 1.0)
        log_prob = dist.log_prob(a).sum(-1)
        return (
            a_clipped.cpu().numpy().flatten(),
            float(log_prob.item()),
            float(value.item()),
        )


# ============================================================================
# PPO Trainer (CleanRL-style, from scratch)
# ============================================================================
@dataclass
class RolloutBuffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    """
    Minimal-but-correct PPO trainer. Designed to train against ExecutionEnv
    (single-env or vectorized via torch.multiprocessing — vectorization handled
    by the caller).
    """

    def __init__(
        self,
        agent: PPOAgent,
        lr: float = PPO_LR,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_eps: float = PPO_CLIP_EPS,
        epochs: int = PPO_EPOCHS,
        minibatch_size: int = PPO_MINIBATCH_SIZE,
        entropy_coef: float = PPO_ENTROPY_COEF,
        value_coef: float = PPO_VALUE_COEF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        device: str = "cpu",
    ) -> None:
        self.agent = agent.to(device)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

    # ----------------------------------------------- GAE advantage estimate
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            adv[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        returns = adv + values
        return adv, returns

    # ------------------------------------------------------------ optimize
    def update(
        self,
        rollout: RolloutBuffer,
        last_value: float,
    ) -> dict:
        obs = torch.tensor(np.stack(rollout.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack(rollout.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(rollout.log_probs, dtype=torch.float32, device=self.device)
        values_np = np.array(rollout.values, dtype=np.float32)
        rewards_np = np.array(rollout.rewards, dtype=np.float32)
        dones_np = np.array(rollout.dones, dtype=np.float32)

        adv_np, ret_np = self._compute_gae(rewards_np, values_np, dones_np, last_value)
        adv = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = torch.tensor(ret_np, dtype=torch.float32, device=self.device)

        N = obs.shape[0]
        idx = np.arange(N)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "n": 0}
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                mb_obs = obs[mb]
                mb_act = actions[mb]
                mb_old_lp = old_log_probs[mb]
                mb_adv = adv[mb]
                mb_ret = returns[mb]

                dist, value = self.agent(mb_obs)
                new_log_prob = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_log_prob - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_ret)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_lp - new_log_prob).mean().item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl"] += approx_kl
                metrics["n"] += 1

        if metrics["n"] > 0:
            for k in ("policy_loss", "value_loss", "entropy", "kl"):
                metrics[k] /= metrics["n"]
        return metrics


# ============================================================================
# KL divergence between two PPOAgents — for Layer 4 stability gate
# ============================================================================
@torch.no_grad()
def estimate_kl(
    agent_p: PPOAgent,
    agent_q: PPOAgent,
    sample_obs: torch.Tensor,
) -> float:
    """KL(p || q) estimated over a sample batch of observations."""
    dist_p, _ = agent_p(sample_obs)
    dist_q, _ = agent_q(sample_obs)
    # KL between two Normals with same shape; closed-form per-dim, then sum and mean.
    kl = torch.distributions.kl_divergence(dist_p, dist_q).sum(-1).mean()
    return float(kl.item())
