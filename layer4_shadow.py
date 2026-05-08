"""
layer4_shadow.py — Continuous shadow training of the execution agent.

Runs forever in the background on GPU. Replays the last 24h of (state, action,
reward) tuples from the live ExecutionEnv into a SHADOW PPOAgent. After each
training pass, runs the three-condition stability gate:

    1. KL(live || shadow) < KL_THRESHOLD
    2. HMM regime in training window matches the current live regime
    3. ≥ MIN_SHADOW_STEPS gradient steps since last promotion

If all three pass, weights are pushed to the live agent via Polyak averaging
with τ = POLYAK_TAU. Live agent never sees a hard cutover.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from config import (
    KL_THRESHOLD,
    MIN_SHADOW_STEPS,
    POLYAK_TAU,
    REPLAY_BUFFER_HOURS,
)
from layer3_execution import PPOAgent, PPOTrainer, RolloutBuffer, estimate_kl

log = logging.getLogger(__name__)


# ============================================================================
# Replay buffer
# ============================================================================
@dataclass
class ReplayEntry:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    value: float
    log_prob: float
    done: bool
    regime: int
    timestamp: float


class ShadowReplayBuffer:
    """
    Rolling 24h replay of live transitions. Every step in ExecutionEnv emits one
    entry. The buffer is bounded by wall-clock time, not entry count.
    """

    def __init__(self, hours: float = REPLAY_BUFFER_HOURS) -> None:
        self.window_sec = hours * 3600.0
        self._buf: deque = deque()

    def append(self, entry: ReplayEntry) -> None:
        self._buf.append(entry)
        cutoff = time.monotonic() - self.window_sec
        while self._buf and self._buf[0].timestamp < cutoff:
            self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    def to_rollout(self) -> RolloutBuffer:
        rb = RolloutBuffer()
        for e in self._buf:
            rb.obs.append(e.obs)
            rb.actions.append(e.action)
            rb.log_probs.append(e.log_prob)
            rb.rewards.append(e.reward)
            rb.values.append(e.value)
            rb.dones.append(e.done)
        return rb

    def regime_distribution(self) -> dict[int, float]:
        if not self._buf:
            return {}
        total = len(self._buf)
        out: dict[int, int] = {}
        for e in self._buf:
            out[e.regime] = out.get(e.regime, 0) + 1
        return {k: v / total for k, v in out.items()}

    def sample_obs_batch(self, n: int = 256) -> np.ndarray:
        if not self._buf:
            return np.zeros((0, 42), dtype=np.float32)
        idxs = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        items = list(self._buf)
        return np.stack([items[i].obs for i in idxs], axis=0)


# ============================================================================
# Stability Gate
# ============================================================================
@dataclass
class GateResult:
    passed: bool
    kl: float
    kl_pass: bool
    regime_pass: bool
    steps_pass: bool
    shadow_dominant_regime: int
    live_regime: int
    steps_since_last_push: int


class StabilityGate:
    """Three-condition gate from the spec."""

    def __init__(
        self,
        kl_threshold: float = KL_THRESHOLD,
        min_steps: int = MIN_SHADOW_STEPS,
    ) -> None:
        self.kl_threshold = kl_threshold
        self.min_steps = min_steps

    def check(
        self,
        live_agent: PPOAgent,
        shadow_agent: PPOAgent,
        sample_obs: torch.Tensor,
        shadow_regime_dist: dict[int, float],
        live_regime: int,
        steps_since_last_push: int,
    ) -> GateResult:
        kl = estimate_kl(live_agent, shadow_agent, sample_obs) if sample_obs.numel() else float("inf")
        kl_pass = kl < self.kl_threshold

        if shadow_regime_dist:
            shadow_dominant = max(shadow_regime_dist, key=shadow_regime_dist.get)
        else:
            shadow_dominant = -1
        regime_pass = shadow_dominant == live_regime

        steps_pass = steps_since_last_push >= self.min_steps
        return GateResult(
            passed=kl_pass and regime_pass and steps_pass,
            kl=kl,
            kl_pass=kl_pass,
            regime_pass=regime_pass,
            steps_pass=steps_pass,
            shadow_dominant_regime=shadow_dominant,
            live_regime=live_regime,
            steps_since_last_push=steps_since_last_push,
        )


# ============================================================================
# Polyak averaging
# ============================================================================
@torch.no_grad()
def polyak_update(
    live_agent: PPOAgent,
    shadow_agent: PPOAgent,
    tau: float = POLYAK_TAU,
) -> None:
    for live_p, shadow_p in zip(live_agent.parameters(), shadow_agent.parameters()):
        live_p.data.copy_(tau * shadow_p.data + (1.0 - tau) * live_p.data)


# ============================================================================
# Shadow Simulator
# ============================================================================
class ShadowSimulator:
    """
    Background process driving continuous training of the shadow agent.

    The live engine pushes ReplayEntry objects into `replay`. This class wakes
    periodically, runs PPO updates against the buffer, and attempts to push
    weights to the live agent through the StabilityGate.
    """

    def __init__(
        self,
        live_agent: PPOAgent,
        shadow_agent: PPOAgent,
        live_regime_provider,                 # callable → int
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        train_interval_sec: float = 30.0,
    ) -> None:
        self.replay = ShadowReplayBuffer()
        self.live_agent = live_agent
        self.shadow_agent = shadow_agent.to(device)
        self.trainer = PPOTrainer(self.shadow_agent, device=device)
        self.gate = StabilityGate()
        self.live_regime_provider = live_regime_provider
        self.device = torch.device(device)
        self.train_interval_sec = train_interval_sec
        self.steps_since_last_push = 0
        self.running = False

    def push_transition(self, entry: ReplayEntry) -> None:
        self.replay.append(entry)

    async def run(self) -> None:
        self.running = True
        log.info("ShadowSimulator started on %s", self.device)
        while self.running:
            await asyncio.sleep(self.train_interval_sec)
            if len(self.replay) < 1024:
                continue
            try:
                self._train_one_pass()
                self._maybe_promote()
            except Exception as e:
                log.exception("ShadowSimulator iteration failed: %s", e)

    def _train_one_pass(self) -> None:
        rollout = self.replay.to_rollout()
        if len(rollout) == 0:
            return
        # Last value bootstrap: 0.0 (we don't carry value across the rolling window).
        metrics = self.trainer.update(rollout, last_value=0.0)
        self.steps_since_last_push += metrics.get("n", 0)
        log.debug("shadow update: %s", metrics)

    def _maybe_promote(self) -> None:
        sample = self.replay.sample_obs_batch(256)
        sample_t = torch.from_numpy(sample).float().to(self.device)
        result = self.gate.check(
            live_agent=self.live_agent,
            shadow_agent=self.shadow_agent,
            sample_obs=sample_t,
            shadow_regime_dist=self.replay.regime_distribution(),
            live_regime=self.live_regime_provider(),
            steps_since_last_push=self.steps_since_last_push,
        )
        if result.passed:
            polyak_update(self.live_agent, self.shadow_agent)
            self.steps_since_last_push = 0
            log.info("Promoted shadow → live. KL=%.4f", result.kl)
        else:
            log.debug("gate blocked: %s", result)

    def stop(self) -> None:
        self.running = False
