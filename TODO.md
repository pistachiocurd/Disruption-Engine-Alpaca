# Deferred Work

Durable record of items intentionally deferred during the HL prove-out work
(2026-05-09 → 2026-05-10). Each entry: what, why deferred, the trigger
condition for picking it back up, and rough effort estimate.

For higher-level open frontiers (PPO checkpoint persistence, eval harness
data providers, replay session-boundary detection) see [README.md "Open
work / collaboration"](README.md). For research-grade findings to
implement (BC warm-start, SAC migration, RevIN, etc.) see §A/§B/§C of the
deep-research review (synthesized in this session, in the plan file at
`~/.claude/plans/fresh-session-i-want-deep-lake.md`).

For the dedicated TCN failure investigation (P0/P0.5 below), see
`~/.claude/plans/tcn-failure-investigation.md` which includes the Gemini
Deep Research synthesis (2026-05-10) that drives the prioritization of
P0–P3.7.

## Mini-projects (parked research, separate branches)

- **Path G — Dimensionless microstructure features** (2026-05-11).
  Buckingham-π / dimensional-analysis approach to feature construction;
  goal is cross-coin generalization via universal correlations. Parked
  while Path E (SSL pretraining, H5) takes priority. Planning docs on
  branch `research/path-g-dimensionless` (commit 330a6bd) under
  `mini_projects/path_G_dimensionless/` — README, RESEARCH_PLAN,
  THEORY. Estimated effort: 4 days dedicated when picked up. Trigger:
  Path E completion, or interest in symbol-agnostic transfer.

---

## P0 — Path A: AUCM loss via LibAUC — **COMPLETED 2026-05-10** ✓

**Result.** F2 = 0.102 (up from §13.3 baseline 0.046; 2.2× improvement).
Non-degenerate sweep curve achieved — precision climbs smoothly 0.026 →
0.040 as threshold rises, no cliff at 0.45 like §13.3's degenerate runs.
Success criterion (F2 ≥ 0.10) met. See LAYER2_TRAINING.md §13.4 for the
full sweep table and analysis.

**What shipped.**
- `libauc>=1.4` added to [requirements.txt](requirements.txt).
- [train_stream.py](train_stream.py) — `--loss aucm` branch with PESG
  optimizer; `--aucm-margin` and `--aucm-lr` flags; sigmoid before AUCM
  forward; defensive `mkdir(parents=True, exist_ok=True)` before save.
- [train_stream.py](train_stream.py) — `ShuffledBufferDataset`
  reservoir-shuffle wrapper applied only under `--loss=aucm`.
  Required: AUCM returns 0 on zero-positive batches and ~95% of the
  streaming temporal-order batches had no positives. Wrapper guarantees
  ~48 positives per 2048-batch (P(zero-positives) ≈ 10⁻²¹).

**What's left unresolved.** Signal extraction is still weak — max
precision 0.040 is only 1.67× base rate. H4 (loss collapse) was
*necessary* but not *sufficient*. Path D (P3.6) is the next move.

**Artifacts (`calibration/`):** `tcn_weights_BTC_USDC_USDC.pt`,
`tcn_threshold_BTC_USDC_USDC.json` (thr=0.05, F2=0.102),
`tcn_threshold_sweep_BTC_USDC_USDC.csv`.

**Citations.** Yuan et al. 2023 ("Empirical X-Risk Minimization"); Qi et
al. 2021 ("Stochastic Optimization of AUPRC").

---

## P0.5 — Path B: Diagnostic recipe (signal-localization)

**Status (2026-05-10):** **Triggered if P0 doesn't clear F2=0.10.**
Empirical localization of the failure point instead of guessing which of
H1–H6 (from `tcn-failure-investigation.md`) is the binding constraint.

**What.** New `tests/diagnose_tcn.py` with five phases:
1. Logistic regression on un-windowed raw ticks (signal at instant level?)
2. Temporal resolution sweep — retrain TCN at `seq_len ∈ {5, 10, 20, 30, 60}`, F2 vs L curve (does the 60-tick window blur the signal?)
3. Per-channel activation variance + Shapiro-Wilk after each TCNBlock (channel collapse via normalization?)
4. Per-class gradient norm logging in train loop (pos/neg ratio → 0 = loss is collapsing)
5. Multi-seed (≥5) F2 distribution (structural vs. stochastic failure?)

**Trigger.** F2 from P0 stays at noise floor (< 0.10) after AUCM swap.

**Effort.** ~4–6 hours. Pure diagnostic, no production-code changes.

---

## P1 — HL liquidation channel (real source)

**What.** Wire a working historical liquidation feed for Hyperliquid so
`liquidation_rate > 0` in harvested CSVs. Today the channel is identically 0
and Σ_diag[2]=0 (regularized to 1e-6 jitter), creating a degenerate axis in
the OOD detector that can produce inflated Mahalanobis values from
floating-point noise alone. See research review §A row 5.

**Why deferred.** Three external sources investigated; all three are dead
ends for fresh data:
1. `dir`-field substring match — verified empirically via
   [tests/inspect_hl_liquidations.py](tests/inspect_hl_liquidations.py),
   no `dir` value contains "Liquidate".
2. `trade_dir_override` field — 100% missing in current archives.
3. [`hyperliquid-dex/historical_data/liquidations.csv`](https://github.com/hyperliquid-dex/historical_data) —
   only 62 events total, dates 2023-02 through 2023-04. One-time research
   snapshot, not a maintained feed.

The infrastructure is in place — `LiquidationCascadeTracker.record()`
public injection hook, `SensorArray` enables the tracker for
`EXCHANGE_ID=hyperliquid`, harvester heap-merges liquidation events into
the chronological replay. We just don't have a working source.

**Trigger to pick back up.** Either:
- (a) Multi-coin training (P0 — running now) does NOT get F2 above 0.30,
  meaning feature-set inadequacy is plausibly the binding constraint and a
  real liquidation channel is worth pursuing.
- (b) We move to live deployment and start seeing OOD-storm symptoms
  (MANDATE_SUPPRESSED_OOD floods like [LAYER2_TRAINING.md §11.1](LAYER2_TRAINING.md)
  saw on equities). At that point the dead-channel-noise-amplification
  problem becomes operationally critical.

**Candidate sources to investigate.**
- HL info API — `https://api.hyperliquid.xyz/info` with type `userFills`
  filtered to known liquidator addresses, OR a dedicated liquidations
  endpoint if one exists. Need to read HL API docs.
- HLP (Hyperliquid LP) vault counterparty detection — HLP backstops
  liquidations. Identify HLP's address(es), then flag any trade in
  `node_fills_by_block` where one side_info[*].user is an HLP address.
  Need to confirm address(es).
- Funding-rate-as-stress proxy — `asset_ctxs/<date>.csv.lz4` in the HL
  archive bucket has hourly funding. High funding → directional pressure
  → liquidation correlate. Different feature semantically but might
  occupy the same OOD-channel slot. Cheapest implementation.

**Effort estimate.** 2-4 hours including verification and adapter code.
Funding-rate proxy is the fastest (~1 hour); HLP-counterparty detection
is medium (~3 hours, contingent on finding addresses); info-API is
longest (~4 hours, requires API discovery).

---

## P2 — Multi-coin trade-file download deduplication

**What.** `node_fills_by_block/hourly/<date>/<hour>.lz4` files multiplex
ALL coins. The current harvester downloads each trade file *once per
coin* in the `--coins` list. For BTC,ETH,SOL × 14 days × 24 hours that's
1008 GETs total when 336 would suffice — 3× wasted egress and time.

**Why deferred.** Saves ~$0.30 and ~2 hours on a 14-day 3-coin harvest.
Real but not blocking. The current implementation produces correct
output; the optimization is purely cost/time.

**Trigger.** When the user wants to harvest more than ~30 coin-days
total (e.g. 7 coins × 14 days, or scaling beyond BTC/ETH/SOL).

**Effort.** ~1 hour. Refactor `cmd_harvest` so the trade key list is
fetched once and a multi-coin event filter runs in a single pass; would
also need per-coin SensorArray + FeatureDumper instances and a dispatch
on `fill["coin"]`.

---

## P3 — Multi-coin training generalization assumption — TESTED NEGATIVE

**Status (2026-05-10):** **Tested with negative result.** Multi-coin
pooling on 7d × {BTC, ETH, SOL} produced F2=0.046, marginally above
single-coin's 0.027 but still at noise-floor levels. The per-coin
held-out cross-symbol test (the originally-planned validation of the
generalization assumption) was not run because the in-pool F2 already
failed to clear the noise floor — there's no signal to generalize.

**DR update (2026-05-10):** Gemini Deep Research demoted "needs more
positive samples" (H1 in `tcn-failure-investigation.md`) as a standalone
cause. 154K positives is robust *if* optimization is managed correctly.
The failure is geometric (loss collapse — see P0) and architectural
(TCN cadence mismatch on bursty data — see P3.5), not sample-count.
Re-harvesting more coin-days is now a fallback after P0/P3.5/P3.6/P3.7
are exhausted, not a primary path.

**Real finding:** the binding constraint on HL perp TCN training is
not multi-coin generalization. It's something more fundamental about
the architecture / loss / feature combination, per §13.3 of
LAYER2_TRAINING.md. Five training configurations all produced F2 in
[0.01, 0.05] regardless of single-coin vs multi-coin, regime vs shock
labels, BCE vs focal, H=30 vs H=10, ep=1 vs ep=10.

**Trigger to revisit.** Only if the dedicated TCN research session
(see `~/.claude/plans/tcn-failure-investigation.md`) identifies an
architecture/loss change that breaks through to F2 ≥ 0.30 on single-coin.
At that point retest multi-coin generalization with the working
architecture.

**Effort.** N/A — superseded by the TCN failure investigation.

---

## P3.5 — Path C: Architecture swap — **TESTED NEGATIVE 2026-05-11**

**Status (2026-05-11):** **Tested with negative result.** H2 (TCN's
fixed-grid inductive bias is the binding constraint) is not supported
by the data. Three architectures tested under matched protocol (7d ×
3-coin, AUCM, PESG lr=0.01):

| Architecture | F2 | Max prec | Max prec / base rate |
|---|---:|---:|---:|
| TCN (Path D) | 0.121 | 0.050 | 1.81× |
| Transformer 3-ep | 0.126 | 0.052 | 1.88× |
| Mamba 1-ep | 0.126 | 0.032 | 1.15× |

All three converge to F2 ≈ 0.12. Mamba's input-dependent selective
scan — the canonical fix for irregular cadence — produced the *worst*
discrimination. TCN's sweep shape is actually the cleanest. See
LAYER2_TRAINING.md §13.6 for full sweep tables and analysis.

**What shipped (preserved infrastructure).**
- `layer2_alpha.py` — added `TransformerSpikePredictor` (causal
  encoder, sinusoidal PE, pre-norm) and `MambaSpikePredictor` (pure-PyTorch
  selective scan, fp32 recurrence, depthwise causal conv). Both expose
  a common `forward_logits()` method matching the refactored TCN.
- `train_stream.py` — `--model {tcn,transformer,mamba}` flag; model
  factory dispatches; training loop uses the common interface.

**Install attempts that failed.**
- `mamba-ssm` + `causal-conv1d` — `NameError: bare_metal_version`
  (setup.py probes for nvcc; none in env). Python 3.14 also has no
  pre-built wheels.
- `torchcde` installed cleanly but skipped (ODE solver 5-10× per-batch
  overhead made it impractical for 1-epoch iteration).

**Interpretation.** The selective scan addresses irregular sampling
*intervals*; once the harvest collapses bursty trade arrivals into
per-tick rows, tick-index distance no longer carries cadence info
that Δ could exploit. Adding explicit Δt as a 6th input channel
might recover this signal — possible follow-up, but lower priority
than P3.7 since H5 (labels) is the remaining unaddressed hypothesis.

**Forward pointer.** P3.7 (Path E) — SSL pretext pretraining. With H4
(loss), H3 (features), and now H2 (architecture) tested, label noise
(H5) is the highest-leverage remaining hypothesis. Dense next-tick
prediction → freeze encoder → fine-tune with AUCM should insulate
against heuristic-label noise.

**Citations.** Kidger et al. 2020 (Neural CDEs); Gu & Dao 2023 (Mamba).

---

## P3.6 — Path D: Crypto-native microstructure features — **COMPLETED 2026-05-10** ✓

**Result.** F2 = 0.121 (up from Path A's 0.102; +18%). Max precision in
the sweep climbed from 0.040 → 0.050 (+25%); base-rate-adjusted gain is
+6% relative. **Recall at thr=0.05 jumped 0.377 → 0.723** — the most
striking signal. The new features (MLOFI/VAMP/Kyle's λ) carry signal,
but the model still extracts it only weakly (max precision is 1.81×
base rate vs TSLA-equity's >20×). See LAYER2_TRAINING.md §13.5 for the
full A/D comparison table and sweep details.

**What shipped.**
- `layer1_sensors.py` — new `MLOFITracker`, `VAMPDeviationComputer`,
  `KylesLambdaTracker`; `PhysicsState` gained `mlofi`, `vamp`,
  `kyles_lambda` fields; `FeatureDumper` extended to 12-column schema.
- `config.py` — `MLOFI_DEPTH_LEVELS=5`, `KYLES_LAMBDA_LOOKBACK_TICKS=100`;
  `TCN_INPUT_CHANNELS` conditional (5 for HL, 3 elsewhere).
- `train_stream.py` + `layer2_alpha.AlphaEngine._push_features` —
  conditional feature stacking that mirrors at train + serve.
- `tests/test_replay.py` — reads new columns with `.get(col, 0.0)` for
  backward-compat with old equity CSVs. 5/5 tests still pass.
- 7d × 3-coin re-harvest produced 6.54M rows × 12 cols. Path A CSVs
  preserved as `feature_history_<COIN>.pathA.csv`.

**What's left unresolved.** Max precision 0.050 ≈ 1.81× base rate. The
remaining gap is plausibly architectural (H2): TCN's fixed-grid
inductive bias is wrong for HL's millisecond-irregular bursty cadence.

**Forward pointer.** Either P3.5 (Path C: Mamba / Neural CDE) or P3.7
(Path E: SSL pretext pretraining). Path C is the higher-likelihood bet
since H4 (loss) and H3 (features) are now addressed, leaving H2
(cadence) as the most plausible unfixed contributor.

**Citations.** Xu et al. 2018 (MLOFI); Stoikov 2018 (VAMP);
Kyle 1985 (price-impact-per-unit-flow).

**What.** Replace OBI + ce_ratio + dead liquidation_rate with
crypto-specific features:
- **MLOFI** — Multi-Level Order Flow Imbalance across top 5–10 book
  depth levels (combats top-of-book spoofing)
- **VAMP** — Volume-Adjusted Mid Price (queue-position-weighted)
- **Kyle's λ** — rolling 100-tick regression of return on signed volume;
  proxy for current liquidity regime / price-impact sensitivity

**Files.**
- `fetch_history_hyperliquid.py` — new feature computation; schema break
  on `feature_history_*.csv`
- [config.py](config.py) — bump `TCN_INPUT_CHANNELS` 3 → 5 (or keep 3
  and just replace dead `liquidation_rate` with MLOFI)
- [train_stream.py](train_stream.py) lines 190–194 — feature stacking
- [layer2_alpha.py](layer2_alpha.py) `_push_features` (lines 312–321) —
  match at inference

**Friction.** Re-harvest 3 coins × 7 days. ~$1 S3 + 3 hours wall. Old
HL TCN weights invalidated (different input channel count or semantics).

**Effort.** 1.5–2 days including re-harvest and verification.

**Citations.** Xu et al. 2018 (MLOFI); Philip et al. 2022 (Queue
Position).

---

## P3.7 — Path E: Self-supervised pretext pretraining — **TESTED NEGATIVE 2026-05-11**

**Status (2026-05-11):** **Tested with negative result.** F2 = 0.126,
indistinguishable from Path D (0.121) within ±0.02 single-seed
variance. Max precision 0.048 = 1.73× base rate, same regime as all
prior paths. H5 (label noise) does not appear to be the binding
constraint either — SSL-pretrained encoder + frozen fine-tune produces
the same signal-extraction ceiling as every other intervention. See
§13.8 of LAYER2_TRAINING.md for full sweep and analysis.

**What shipped.**
- `train_stream.py` — `--pretext` flag (dense MSE on next-tick
  log-return in bps), `--load-pretrained PATH` (load_state_dict
  strict=False), `--freeze-encoder` (only `head.*` trainable).
- `StreamingTCNDataset` + `MultiCsvTCNDataset` — `pretext` param that
  emits next-tick log-return as target instead of binary shock label.
- Pretrain: Adam lr=1e-3 + MSELoss, skips threshold sweep, saves to
  `calibration/pretrain_weights_<SLUG>.pt`.
- Fine-tune: any classification loss + `--freeze-encoder` — only the
  head's 33 parameters update.

**Result.** F2=0.126, same threshold-sweep shape as Path D/C. SSL
phase 1 loss curve 19.6 → 0.35 (bps² MSE) so the encoder did learn
*something* about next-tick dynamics, but that representation didn't
yield a stronger classifier than end-to-end-trained models.

**Implication.** Five hypotheses from `tcn-failure-investigation.md`
now exhausted (H4 ✓, H3 ✓, H2 ✗, H5 ✗, H1 untested but DR-demoted).
The F2 ≈ 0.12 / max-prec ≈ 1.7-1.8× base-rate band is the effective
signal-extraction ceiling under the current shock-labeling scheme.

**Forward.** Either pivot away from path-list iteration (ship Path D's
F2=0.121, focus on live deployment), or pursue:
- **H1** — harvest 30+ coin-days and re-run Path D. ~$5 S3, ~15 h wall.
- **Path G** — dimensionless features on `research/path-g-dimensionless`.
  4-day research project. Tests cross-coin invariance, not just F2.
- **P0.5 step 1** — logistic regression on un-windowed raw ticks. If
  AUROC ≈ 0.55, confirms intrinsic data ceiling regardless of model.

**Citations.** Fang 2023 (Temporal Bag-of-Features); Wallbridge 2020
(Transformers for LOBs).

**What.** Two-phase training:
1. **Pretext task** — predict next-tick OBI (or mid-return) via MSE.
   Dense supervision on 1.087M rows × 3 coins.
2. **Freeze encoder, attach fresh classification head, fine-tune** on
   sparse shock labels — ideally with AUCM (P0) layered on top.

**Files.**
- New `pretrain_tcn.py` OR `--pretext` flag in
  [train_stream.py](train_stream.py)
- Pretext head: linear projection of last-tick hidden state to scalar
  next-tick OBI prediction
- Fine-tune head: existing classification head reset, encoder frozen
  during fine-tune

**Trigger.** AUCM works but caps below F2=0.30 — diagnosis: labels too
noisy, encoder needs to learn microstructure manifold from dense signal
first.

**Effort.** ~1 day. Pipelines two phases.

**Citations.** Fang 2023 (Temporal Bag-of-Features); Wallbridge 2020
(Transformers for LOBs).

---

## P4 — RevIN input normalization on TCN

**What.** Research review §A row 6 / failure mode #4: the TCN has no
input normalization, only a static `ce_ratio / 10` divisor. RevIN
(Reversible Instance Normalization) at the input projection would
auto-adapt to regime shifts in feature distributions.

**Why deferred.** The current TSLA epoch-3-onward plateau ([§5 of
LAYER2_TRAINING.md](LAYER2_TRAINING.md)) is consistent with the
saturation pathology RevIN is meant to fix, but we haven't isolated
whether RevIN actually helps or whether the plateau is just rare-class
learning bottleneck. Adding RevIN before establishing baseline
multi-coin F2 confounds two changes.

**Trigger.** After multi-coin training establishes a baseline F2.
RevIN is then a controlled experiment: same data, same loss, just adds
RevIN at input projection — easy to A/B.

**Effort.** ~2 hours including the implementation and A/B run.

---

## P5 — HMM emission refit on HL data

**What.** [§11.4 of LAYER2_TRAINING.md](LAYER2_TRAINING.md) lists "HMM
cold-start prior" as deferred operator work. On HL we're using
`StudentTHMM.from_default_priors()` which gives noisy regime
classification (verified: row 1 = regime 0, row 2 = regime 2 with no
gradual transitions). This is why `--label-source shock` was needed
during training — `regime` column from cold-start HMM was unreliable.

**Why deferred.** HMM emission fitting requires a labeled state
sequence to train against. Producing those labels (offline EM or
hand-tuned heuristic) is non-trivial. The shock-based labels are a
reasonable workaround until HMM is fitted.

**Trigger.** When we want to train Layer 2 with `--label-source regime`
(faster + denser than shock-based labels per the docs). Or when the
engine deploys and noisy regime classifications confuse downstream
gates.

**Effort.** ~half a day. Requires labeling work + EM fit + verification.

---

## P6 — `read_csv_batched` deprecation

**What.** Polars deprecated `pl.read_csv_batched`; recommends
`pl.scan_csv().collect_batches()`. Currently `train_stream.py` line ~74
uses the deprecated form (visible warning in every training run).

**Why deferred.** Cosmetic. Functional in current polars version.

**Trigger.** When polars actually removes the deprecated API.

**Effort.** ~10 min find/replace + smoke test.
