"""
layer2_alpha.py — Alpha generation.

The ONLY place in the system where directional opinions are formed.

Components:
    TCNSpikePredictor   — temporal convnet → turbulence index
    HeatEquationSolver  — attrition-adjusted equilibrium price + PDE diffusion
    AlphaEngine         — orchestrator that produces TradeMandate objects
    TradeMandate        — typed payload consumed by Layer 3
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BASE_EXECUTION_WINDOW,
    CAPTURE_RATIO,
    EPSILON,
    EXCHANGE_ID,
    FD_DOMAIN_PCT,
    FD_STABILITY_FACTOR,
    LAMBDA_ATTRITION,
    LAMBDA_SENSITIVITY_DELTA,
    MAX_POSITION_LIMIT,
    MIN_ORDER_SIZE,
    P_EQ_CONFIDENCE_BAND_BPS,
    SYMBOL,
    TCN_DILATIONS,
    TCN_HIDDEN_CHANNELS,
    TCN_INPUT_CHANNELS,
    TCN_INPUT_LENGTH,
    TICK_SIZE,
    TURBULENCE_THRESHOLD,
)

log = logging.getLogger(__name__)


# ============================================================================
# TradeMandate
# ============================================================================
@dataclass
class TradeMandate:
    direction: int                  # +1 long, -1 short
    target_size: float              # base currency
    p_decision: float               # mid at mandate generation (fixed IS reference)
    execution_window_seconds: float
    vpin: float
    alpha_calibrated: float
    viscosity: float
    symbol: str = SYMBOL


# ============================================================================
# TCN Spike Predictor
# ============================================================================
class CausalConv1d(nn.Module):
    """Causal 1D convolution. Pads only on the left so no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Two stacked dilated causal convs with a residual connection."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x + residual


class TCNSpikePredictor(nn.Module):
    """
    Input:  (batch, C, 60) where C is TCN_INPUT_CHANNELS (3 for equities,
            5 for Hyperliquid per Path D / §13.5 of LAYER2_TRAINING.md).
    Output: (batch,) — turbulence_index in (0, 1).

    forward() returns sigmoided probabilities (for inference).
    forward_logits() returns raw logits (for BCE/Focal/AUCM training paths
    that expect logits — see train_stream.py).
    """

    def __init__(
        self,
        in_channels: int = TCN_INPUT_CHANNELS,
        hidden_channels: int = TCN_HIDDEN_CHANNELS,
        dilations: tuple = TCN_DILATIONS,
        seq_len: int = TCN_INPUT_LENGTH,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            TCNBlock(hidden_channels, dilation=d) for d in dilations
        )
        self.head = nn.Linear(hidden_channels, 1)
        self.seq_len = seq_len

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → raw logits (B,)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        # Take the LAST timestep (causal, so it summarizes the full window).
        x_last = x[:, :, -1]
        return self.head(x_last).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


# ============================================================================
# Transformer Spike Predictor — Path C / P3.5 (2026-05-10)
# ============================================================================
class TransformerSpikePredictor(nn.Module):
    """Causal Transformer encoder alternative to the TCN.

    Path C / P3.5 tests whether the TCN's fixed-grid causal-conv inductive
    bias is the remaining binding constraint after AUCM (Path A) and
    crypto-native features (Path D) have addressed loss geometry and
    feature inadequacy. Attention removes the dilation-equidistance
    assumption and lets the model learn arbitrary token-to-token
    relationships across the 60-tick window.

    Drop-in shape compatibility with TCNSpikePredictor:
        Input:  (B, C, T)  where C = TCN_INPUT_CHANNELS, T = TCN_INPUT_LENGTH
        Output: (B,)       sigmoided probability via forward();
                           raw logit via forward_logits().

    Sized to roughly match TCN parameter count (~30K trainable). Tunable
    via init args. Sinusoidal positional encoding is index-based — the
    irregular cadence is exposed to the model through the existing
    feature channels (mlofi/vamp/kyles_lambda already encode tick-rate
    pressure indirectly via the Path D harvest). A future iteration
    could add explicit Δt as a 6th input channel.

    Causal mask: upper-triangular True (i.e. token i can attend only to
    tokens ≤ i). Same no-future-leakage guarantee the TCN's causal conv
    provides.
    """

    def __init__(
        self,
        in_channels: int = TCN_INPUT_CHANNELS,
        d_model: int = TCN_HIDDEN_CHANNELS,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 128,
        seq_len: int = TCN_INPUT_LENGTH,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(in_channels, d_model)
        # Sinusoidal positional encoding; frozen.
        self.register_buffer("pos_enc", self._sinusoidal_pe(seq_len, d_model))
        # Upper-triangular causal mask: position i can attend to j ≤ i only.
        # nn.TransformerEncoderLayer with is_causal=True interprets True = mask out.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm — more stable for small models at bf16
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, T, d_model)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → transpose to (B, T, C) for TransformerEncoder.
        x = x.transpose(1, 2)
        h = self.input_proj(x) + self.pos_enc[:, : x.size(1)]
        h = self.encoder(h, mask=self.causal_mask, is_causal=True)
        # Take the last token (causal — summarizes the full window).
        return self.head(h[:, -1]).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


# ============================================================================
# Mamba Spike Predictor — Path C / P3.5 (2026-05-11)
# ============================================================================
# Hand-rolled minimal Mamba (Gu & Dao 2023) in pure PyTorch. The official
# mamba-ssm package requires a fused CUDA kernel build (nvcc + Python ≤3.12);
# the dev environment is Python 3.14 + no nvcc + RTX 5070 Ti, so we
# implement the selective scan directly. Slower per-step than the fused
# kernel but mathematically identical and device-agnostic.
#
# The key innovation tested here: input-dependent dynamics. The model
# computes per-tick step size Δ from the input, then evolves a hidden
# state via h_t = exp(Δ_t · A) · h_{t-1} + Δ_t · B_t · x_t. Bursty data
# with millisecond-irregular cadence is exactly the regime Mamba's
# selection mechanism was designed for — vs. TCN's fixed dilation grid
# and Transformer's index-based positional encoding.


class MambaBlock(nn.Module):
    """Single Mamba block: selective state-space scan with input-dependent Δ.

    Structure (Gu & Dao 2023):
      x → LayerNorm → Linear (split into x, z)
      x → causal Conv1d → SiLU → SSM (with input-dep Δ, B, C) → gate by z
        → Linear out → residual add

    For numerical stability, the recurrent scan runs in fp32 even under
    bf16 autocast (exp/softplus of large Δ underflows in bf16's small
    mantissa).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise Conv1d on the inner channel — short-range context
        # before the selective scan. Pad d_conv-1 on each side, trim to L.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # x → [Δ_raw, B, C] projection
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False,
        )
        # Δ_raw → Δ projection (bias init so softplus(0) gives ~1)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        # Initialize dt_proj bias so initial Δ ≈ 1 (softplus(0.541) ≈ 1)
        with torch.no_grad():
            self.dt_proj.bias.fill_(math.log(math.expm1(1.0)))

        # State-space matrix A (initialized as -arange(1, d_state+1) per row)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # Skip-connection scalar D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, d_model) → (B, L, d_model). Residual added by caller."""
        B, L, _ = h.shape
        residual = h
        h = self.norm(h)

        xz = self.in_proj(h)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Causal Conv1d
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # trim right-padding → causal
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        # Selective scan (fp32 for stability)
        y = self._selective_scan(x)

        # Gate by z, project out
        y = y * F.silu(z)
        out = self.out_proj(y)
        return residual + out

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Recurrent selective scan in fp32.

        x: (B, L, d_inner)
        Returns: (B, L, d_inner)
        """
        orig_dtype = x.dtype
        x = x.float()

        B, L, _ = x.shape
        d_inner, d_state = self.d_inner, self.d_state

        # Compute input-dependent Δ, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta_raw, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, d_inner)

        # Discretize A and B using Δ
        # A: (d_inner, d_state)
        # delta: (B, L, d_inner)
        A = -torch.exp(self.A_log.float())
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB_u = (
            delta.unsqueeze(-1) * B_param.unsqueeze(2) * x.unsqueeze(-1)
        )  # (B, L, d_inner, d_state)

        # Recurrent scan
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y_i = (C_param[:, i].unsqueeze(1) * h).sum(-1)  # (B, d_inner)
            ys.append(y_i)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Skip-connection D
        y = y + x * self.D

        return y.to(orig_dtype)


class MambaSpikePredictor(nn.Module):
    """Mamba (selective state-space) model for spike prediction.

    Tests the H2 hypothesis from `tcn-failure-investigation.md`: TCN's
    fixed-grid inductive bias is the binding constraint after H4 (loss
    geometry, Path A) and H3 (feature set, Path D) are addressed.

    Drop-in shape compatibility with TCN/Transformer predictors:
        Input:  (B, C, T)   C = TCN_INPUT_CHANNELS
        Output: (B,)        forward() = sigmoid(forward_logits())

    Default sizing aims for parameter parity with TCN (~30K). Use
    --model mamba in train_stream.py.
    """

    def __init__(
        self,
        in_channels: int = TCN_INPUT_CHANNELS,
        d_model: int = TCN_HIDDEN_CHANNELS,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        seq_len: int = TCN_INPUT_LENGTH,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            )
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)  # residual added inside block
        h = self.norm_out(h)
        return self.head(h[:, -1]).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


# ============================================================================
# Heat Equation Solver
# ============================================================================
class HeatEquationSolver:
    """
    Computes the post-shock equilibrium price by integrating attrition-adjusted
    book depth in the direction of the shock until the cumulative absorbed
    depth equals the imbalance volume Q.

    Also runs the attrition-sensitivity check: re-solving at λ ± δ. If the
    P_eq range exceeds the confidence band, the result is None and Layer 2
    suppresses the mandate.

    The PDE is used to bound the diffusion timescale (and to enforce the
    CFL stability condition on the explicit scheme), but the trade decision
    only needs P_eq itself.
    """

    def __init__(
        self,
        lambda_attrition: float = LAMBDA_ATTRITION,
        lambda_delta: float = LAMBDA_SENSITIVITY_DELTA,
        confidence_band_bps: float = P_EQ_CONFIDENCE_BAND_BPS,
        domain_pct: float = FD_DOMAIN_PCT,
        tick_size: float = TICK_SIZE,
    ) -> None:
        self.lambda_attrition = lambda_attrition
        self.lambda_delta = lambda_delta
        self.confidence_band_bps = confidence_band_bps
        self.domain_pct = domain_pct
        self.tick_size = tick_size

    @staticmethod
    def _shock_volume_and_direction(ob: dict) -> tuple[float, int]:
        """
        Q is the net order-book imbalance at the triggering tick, signed:
        positive Q with direction=+1 means buy pressure / shock will push price up.
        Returns (|Q|, direction).
        """
        bid_total = sum(sz for _, sz in ob["bids"])
        ask_total = sum(sz for _, sz in ob["asks"])
        net = bid_total - ask_total
        direction = 1 if net > 0 else -1
        return abs(net), direction

    def _solve_equilibrium_price(
        self,
        ob: dict,
        Q: float,
        direction: int,
        lambda_attr: float,
        vpin: float,
        p_current: float,
    ) -> Optional[float]:
        """
        Walk the side of the book that absorbs the shock, applying the attrition
        discount adjusted_depth_i = raw_i * exp(-λ * vpin * d_i_bps), and find
        the price at which the cumulative adjusted depth equals Q.
        """
        if Q <= 0:
            return None

        # If direction == +1 (buy shock pushing price up), the ASK side absorbs.
        side = ob["asks"] if direction == +1 else ob["bids"]
        if not side:
            return None

        cumulative = 0.0
        prev_price = p_current
        for price, size in side:
            d_bps = abs(price - p_current) / p_current * 10_000.0
            discount = np.exp(-lambda_attr * vpin * d_bps)
            adj = size * discount
            if cumulative + adj >= Q:
                # Linear interpolation within this level.
                needed = Q - cumulative
                frac = needed / max(adj, EPSILON)
                p_eq = prev_price + frac * (price - prev_price)
                return float(p_eq)
            cumulative += adj
            prev_price = price

        # Q exceeds available adjusted depth in domain — clamp to last level price
        # but also enforce the FD domain ceiling.
        domain_lim = (
            p_current * (1.0 + self.domain_pct) if direction == +1
            else p_current * (1.0 - self.domain_pct)
        )
        return float(min(prev_price, domain_lim) if direction == +1
                     else max(prev_price, domain_lim))

    def solve(
        self,
        ob: dict,
        physics_state,
        p_current: float,
    ) -> Optional[dict]:
        """
        Returns dict with {p_eq, direction, target_size, diffusion_timescale} or
        None if mandate should be suppressed (attrition-uncertainty).
        """
        Q, direction = self._shock_volume_and_direction(ob)
        if Q < MIN_ORDER_SIZE:
            return None

        vpin = physics_state.vpin

        # Sensitivity check: solve at λ-δ, λ, λ+δ.
        p_eq_low = self._solve_equilibrium_price(
            ob, Q, direction, self.lambda_attrition - self.lambda_delta, vpin, p_current
        )
        p_eq_base = self._solve_equilibrium_price(
            ob, Q, direction, self.lambda_attrition, vpin, p_current
        )
        p_eq_high = self._solve_equilibrium_price(
            ob, Q, direction, self.lambda_attrition + self.lambda_delta, vpin, p_current
        )
        if any(p is None for p in (p_eq_low, p_eq_base, p_eq_high)):
            log.info("MANDATE_SUPPRESSED: P_eq solution missing for one or more λ values")
            return None

        # Confidence band check (in bps from current price).
        p_lo = min(p_eq_low, p_eq_base, p_eq_high)
        p_hi = max(p_eq_low, p_eq_base, p_eq_high)
        range_bps = (p_hi - p_lo) / p_current * 10_000.0
        if range_bps > self.confidence_band_bps:
            log.info(
                f"MANDATE_SUPPRESSED_ATTRITION_UNCERTAINTY: "
                f"P_eq range = {range_bps:.2f} bps (limit {self.confidence_band_bps}); "
                f"λ={self.lambda_attrition} δ={self.lambda_delta}"
            )
            return None

        # Diffusion timescale from CFL: Δt = FD_STABILITY_FACTOR * Δx² / α_calibrated.
        alpha = max(EPSILON, physics_state.alpha_calibrated)
        dt = FD_STABILITY_FACTOR * (self.tick_size ** 2) / alpha

        # Target size derivation.
        target_size = min(Q * CAPTURE_RATIO, MAX_POSITION_LIMIT)
        # Round to lot size (assume MIN_ORDER_SIZE is the increment).
        target_size = max(0.0, round(target_size / MIN_ORDER_SIZE) * MIN_ORDER_SIZE)
        if target_size < MIN_ORDER_SIZE:
            return None

        return {
            "p_eq": p_eq_base,
            "direction": direction,
            "target_size": target_size,
            "diffusion_timescale_sec": dt,
            "Q": Q,
        }


# ============================================================================
# AlphaEngine — orchestrator
# ============================================================================
class AlphaEngine:
    """
    On each tick:
      1. Append latest [ce_ratio, obi, liquidation_rate] to rolling buffer.
      2. If buffer is full (60 ticks) and ood_flag is False, run the TCN.
      3. If turbulence_index > threshold, run the heat solver.
      4. Solver returns either P_eq (mandate) or None (suppressed).
      5. Build TradeMandate.
    """

    def __init__(
        self,
        tcn: Optional[TCNSpikePredictor] = None,
        solver: Optional[HeatEquationSolver] = None,
        device: str = "cpu",
        turbulence_threshold: float = TURBULENCE_THRESHOLD,
    ) -> None:
        self.device = torch.device(device)
        self.tcn = (tcn or TCNSpikePredictor()).to(self.device).eval()
        self.solver = solver or HeatEquationSolver()
        self.turbulence_threshold = turbulence_threshold
        self._buffer: deque = deque(maxlen=TCN_INPUT_LENGTH)
        # Rolling log of evaluation outcomes — read by the dashboard.
        self.decision_log: deque = deque(maxlen=200)
        self.last_turbulence: float = 0.0

    def _record(self, outcome: str, **fields) -> None:
        rec = {"ts": int(time.time() * 1000), "outcome": outcome, **fields}
        self.decision_log.append(rec)

    def _push_features(self, physics_state) -> None:
        # Feature scaling MUST match train_stream.py so trained weights and
        # live inputs share scale. Path D (P3.6 / §13.4) — Hyperliquid uses
        # 5 crypto-native channels; everything else keeps the original 3.
        if EXCHANGE_ID == "hyperliquid":
            feats = np.array([
                physics_state.ce_ratio / 10.0,
                physics_state.obi,
                physics_state.mlofi,
                physics_state.vamp / 10.0,
                physics_state.kyles_lambda * 100.0,
            ], dtype=np.float32)
        else:
            feats = np.array([
                physics_state.ce_ratio / 10.0,
                physics_state.obi,
                physics_state.liquidation_rate,
            ], dtype=np.float32)
        self._buffer.append(feats)

    def _tcn_predict(self) -> float:
        if len(self._buffer) < TCN_INPUT_LENGTH:
            return 0.0
        # (T, C) → (1, C, T)
        arr = np.stack(self._buffer, axis=0).T[None, :, :]
        x = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            y = self.tcn(x).item()
        return float(y)

    def evaluate(self, physics_state) -> Optional[TradeMandate]:
        """Returns a TradeMandate if a shock signature is confirmed, else None."""
        self._push_features(physics_state)

        if physics_state.ood_flag:
            log.info("MANDATE_SUPPRESSED_OOD: mahal_dist=%.3f", physics_state.mahal_dist)
            self._record("suppressed_ood", mahal_dist=float(physics_state.mahal_dist))
            return None

        if len(self._buffer) < TCN_INPUT_LENGTH:
            self._record("warming_up", buffer_size=len(self._buffer))
            return None

        turbulence = self._tcn_predict()
        self.last_turbulence = turbulence
        if turbulence < self.turbulence_threshold:
            self._record("below_turbulence", turbulence=float(turbulence))
            return None

        ob = physics_state.ob_snapshot
        if ob is None:
            self._record("no_orderbook")
            return None
        p_current = 0.5 * (ob["bids"][0][0] + ob["asks"][0][0])

        result = self.solver.solve(ob, physics_state, p_current)
        if result is None:
            self._record("suppressed_attrition", turbulence=float(turbulence))
            return None

        # Execution window shrinks with VPIN (high VPIN → tight window).
        exec_window = BASE_EXECUTION_WINDOW * (1.0 - physics_state.vpin)
        exec_window = max(1.0, exec_window)

        mandate = TradeMandate(
            direction=result["direction"],
            target_size=result["target_size"],
            p_decision=p_current,
            execution_window_seconds=exec_window,
            vpin=physics_state.vpin,
            alpha_calibrated=physics_state.alpha_calibrated,
            viscosity=physics_state.viscosity,
            symbol=SYMBOL,
        )
        self._record(
            "mandate",
            direction=int(mandate.direction),
            target_size=float(mandate.target_size),
            p_decision=float(mandate.p_decision),
            p_eq=float(result["p_eq"]),
            turbulence=float(turbulence),
            vpin=float(physics_state.vpin),
        )
        return mandate

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.tcn.load_state_dict(state)
        self.tcn.eval()
