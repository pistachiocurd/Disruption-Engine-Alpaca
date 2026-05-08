# Disruption Arbitrage Engine

Physics-informed market-making / shock-arbitrage engine. Detects pre-shock
microstructure signatures, computes a post-shock equilibrium price via
attrition-adjusted heat-equation diffusion, and executes the resulting
mandate through a fee-aware PPO agent that is continuously retrained in
the background.

**Primary target: US equities via Alpaca** (NVDA by default; SPY also
supported). The original crypto path (Coinbase Advanced spot, optional
Binance liquidation feed) is retained behind the same interfaces and
selected automatically when `SYMBOL` contains a `/`.

For the full design rationale, see `IMPLEMENTATION.md`.

## Quickstart (US equities — same-day TCN training)

```powershell
# 1. Create a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (CPU torch). alpaca-py is in requirements.txt.
pip install -r requirements.txt

# 3. (Optional) Install CUDA torch for the shadow trainer
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Run the test suite
pytest

# 5. Set Alpaca paper credentials
$env:EXCHANGE_API_KEY = "<paper-key>"
$env:EXCHANGE_SECRET  = "<paper-secret>"
$env:SYMBOL = "NVDA"

# 6. Harvest 14 trading days of NVDA quotes + trades into a feature_history
#    CSV that is byte-identical to what the live FeatureDumper writes. This
#    drives the actual SensorArray code path on historical Alpaca data so
#    you can train the TCN today instead of waiting for live capture.
python fetch_history_alpaca.py --symbol NVDA --days 14

# 7. Verify enough shock events were captured, fit OOD, train the TCN.
python check_shocks.py
python fit_ood_from_csv.py
python train_tcn.py

# 8. Live shadow-mode smoke test (during market hours).
python engine.py
```

## Quickstart (crypto — original Coinbase path)

```powershell
$env:EXCHANGE_ID     = "coinbaseadvanced"
$env:SYMBOL          = "BTC/USD"
$env:EXCHANGE_API_KEY = "..."
$env:EXCHANGE_SECRET  = "..."
python engine.py
```

The presence of `/` in `SYMBOL` switches the engine to the ccxt.pro path
and re-enables the Binance liquidation feed if `EXCHANGE_ID=binance`.

## Critical environment variables

| Variable             | Purpose                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `EXCHANGE_ID`        | `alpaca` (default), `coinbaseadvanced`, `binance`. Auto-selected by `SYMBOL` shape if unset.  |
| `SYMBOL`             | Equity ticker (e.g. `NVDA`, `SPY`) or crypto pair (e.g. `BTC/USD`). Default `NVDA`.           |
| `ALPACA_DATA_FEED`   | `iex` (free, default) or `sip` (Algo Trader Plus). Equities only.                             |
| `EXCHANGE_API_KEY`   | Alpaca paper/live API key, or Coinbase CDP key name. Disable withdrawal scopes.               |
| `EXCHANGE_SECRET`    | Matching secret. Stored in env, never on disk.                                                |
| `EXCHANGE_LIVE`      | Set to literal `"true"` to disable paper / sandbox.                                           |
| `LOG_LEVEL`          | `DEBUG`, `INFO` (default), `WARNING`.                                                         |

## Operational toggles in `config.py`

| Flag                       | Default | Purpose                                                       |
| -------------------------- | ------- | ------------------------------------------------------------- |
| `SHADOW_MODE`              | `True`  | Route orders to `LocalMatchingEngine` (paper).                |
| `TRADING_ENABLED`          | `True`  | Global kill switch. Set `False` from outside to halt.         |
| `CALIBRATION_STALE`        | `False` | Set `True` by drift monitor; suspends mandates.               |
| `IS_EQUITY`                | derived | Auto-set from `SYMBOL`. Drives every code-path branch.        |
| `SESSION_WARMUP_SECONDS`   | `30`    | OOD gate quarantined for N seconds after each market open.    |
| `SESSION_RESET_HMM/_KALMAN/_VPIN` | `True` | Whether each filter is reset at every session open.           |

## Market session handling (equities only)

The engine pulls the trading calendar (including half-days and holidays)
from Alpaca's `/v2/calendar` endpoint, refreshed daily. When the market
closes:

1. Open orders are cancelled.
2. The strategy loop awaits the next session open.
3. On the next open, `SensorArray.reset_session()` re-initializes the HMM
   forward variable to a uniform prior, re-initializes the Kalman state,
   clears the VPIN buckets, and resets the C/E proxy.
4. The OOD gate is suspended for `SESSION_WARMUP_SECONDS` so the first
   morning quotes don't trip a false-positive shock signal while filters
   re-warm.

This avoids feeding the overnight gap to the Kalman as a 1-tick spread
innovation or scoring the first morning observation against yesterday's
regime distribution — both of which would otherwise produce an OOD storm
at the open.

Crypto runs 24/7; the `SessionManager` is not instantiated for crypto
symbols.

## Calibration workflow (equities, same-day path)

```powershell
# 1. Harvest history. Replays trades + quotes through the live SensorArray
#    code path so the CSV is byte-identical to live FeatureDumper output.
python fetch_history_alpaca.py --symbol NVDA --days 14

# 2. Verify shock count.
python check_shocks.py

# 3. Fit OOD distribution; writes calibration/latest_NVDA.json.
python fit_ood_from_csv.py

# 4. Train the TCN; writes calibration/tcn_weights_NVDA.pt and
#    tcn_threshold_NVDA.json.
python train_tcn.py
```

The crypto pipeline (online L2 capture, `identify_shock_events()`, EWLS
fit for `ALPHA_CALIBRATION_C`) is unchanged and documented in
`IMPLEMENTATION.md` §4.

## File layout

```
disruption_arbitrage_engine/
├── engine.py                 # asyncio orchestrator, session manager, kill switches
├── layer1_sensors.py         # VPIN (Lee-Ready), Kalman, HMM, OOD,
│                             #   CEInferenceEngine (crypto L2),
│                             #   QuoteCancellationProxy (equities L1)
├── layer2_alpha.py           # TCN, heat solver, AlphaEngine
├── layer3_execution.py       # ExecutionEnv, PPOAgent, PPOTrainer
├── layer4_shadow.py          # ShadowSimulator, stability gate, Polyak update
├── matching_engine.py        # LocalMatchingEngine — paper fills
├── calibration.py            # shock identification, EWLS, drift monitor, OOD fit
├── fetch_history_alpaca.py   # historical harvester for the equities path
├── train_tcn.py              # supervised TCN training
├── check_shocks.py           # validates feature_history shock density
├── fit_ood_from_csv.py       # offline OOD μ, Σ fit from feature_history
├── hpo.py                    # Optuna HPO over reward weights
├── synthetic_data.py         # synthetic feature_history generator
├── config.py                 # all hyperparameters + CALIBRATION_REGISTRY
├── requirements.txt
├── README.md                 # this file
├── IMPLEMENTATION.md         # full architectural rationale and runbooks
└── tests/
    ├── test_sensors.py
    ├── test_calibration.py
    ├── test_matching_engine.py
    ├── test_alpha.py
    └── test_execution.py
```

## Safety constraints

- The matching engine **cannot** place real orders. Structural assertion at import.
- API keys must have **withdrawals disabled**.
- `EXCHANGE_LIVE=true` is required to leave paper / sandbox.
- Session drawdown > 2% halts trading via the drawdown kill switch.
- Calibration drift (rolling MAPE > 0.35) suspends mandate generation.
- OOD detector blocks mandates when the live observation is outside the
  training manifold by Mahalanobis distance.
- Equities: no mandates fire while the market is closed; sensor state is
  reset at every market open to prevent overnight-gap drift.

## Required validation before going live

1. ≥ 30 historical shock events in the calibration set
   (`MIN_CALIBRATION_EVENTS` in config). For NVDA on IEX-only data, this
   typically requires 7–14 trading days; bump `--days` if `check_shocks.py`
   reports too few.
2. Drift monitor inactive.
3. ≥ 72 hours of `SHADOW_MODE=True` operation against live data
   (equities: this spans ~10 trading sessions).
4. HPO winning trial validated on the SUBSEQUENT week.

Only then flip `SHADOW_MODE=False` and `EXCHANGE_LIVE=true`.
