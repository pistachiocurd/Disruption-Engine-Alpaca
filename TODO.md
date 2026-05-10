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
