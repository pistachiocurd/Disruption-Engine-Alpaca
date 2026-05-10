"""
config.py — All hyperparameters, calibration registry, and runtime flags.

Single source of truth. Every other module imports constants from here.
Do not hardcode values elsewhere; if a value is policy, it lives in this file.

CRITICAL FLAGS:
- SHADOW_MODE: True routes orders to LocalMatchingEngine; False sends real orders.
- TRADING_ENABLED: Global kill switch. Set False on any unhandled exception.
- CALIBRATION_STALE: Set True by the drift monitor to suspend mandate generation.
"""
import os

# ============================================================================
# Exchange & Symbol
# ============================================================================
EXCHANGE_ID = os.environ.get("EXCHANGE_ID", "alpaca")
SYMBOL = os.environ.get("SYMBOL", "NVDA")

# Per-pair tick + lot sizes. Override via env if needed.
_PAIR_DEFAULTS = {
    # US equities (Alpaca). 1-share min, fractional supported via Alpaca but the
    # PPO action space discretizes on integer shares.
    "NVDA":     {"tick": 0.01,    "min_order": 1.0,    "max_position": 200.0},
    "SPY":      {"tick": 0.01,    "min_order": 1.0,    "max_position": 500.0},
    "TSLA":     {"tick": 0.01,    "min_order": 1.0,    "max_position": 100.0},
    # Crypto pairs retained for back-compat / regression tests.
    "BTC/USD":  {"tick": 0.01,    "min_order": 0.001,  "max_position": 0.5},
    "ETH/USD":  {"tick": 0.01,    "min_order": 0.01,   "max_position": 5.0},
    "SOL/USD":  {"tick": 0.01,    "min_order": 0.01,   "max_position": 50.0},
    "AVAX/USD": {"tick": 0.001,   "min_order": 0.1,    "max_position": 200.0},
    "DOGE/USD": {"tick": 0.0001,  "min_order": 1.0,    "max_position": 50_000.0},
    "LINK/USD": {"tick": 0.001,   "min_order": 0.1,    "max_position": 500.0},
    # Hyperliquid perps (CCXT format BTC/USDC:USDC). Position caps sized
    # for prove-out — small notional, scale via MAX_POSITION_LIMIT env if
    # leveraging up. Tick sizes mirror HL's published price increments.
    "BTC/USDC:USDC":  {"tick": 0.5,    "min_order": 0.0001, "max_position": 0.05},
    "ETH/USDC:USDC":  {"tick": 0.05,   "min_order": 0.001,  "max_position": 1.0},
    "SOL/USDC:USDC":  {"tick": 0.001,  "min_order": 0.01,   "max_position": 50.0},
    "HYPE/USDC:USDC": {"tick": 0.0001, "min_order": 0.1,    "max_position": 1000.0},
}
_pair = _PAIR_DEFAULTS.get(SYMBOL, {"tick": 0.01, "min_order": 0.01, "max_position": 50.0})
TICK_SIZE = float(os.environ.get("TICK_SIZE", _pair["tick"]))

# Filesystem-safe slug used by per-symbol calibration / training files.
SYMBOL_SLUG = SYMBOL.replace("/", "_").replace(":", "_")

# Cosmetic flag — many sites need to branch on "is this a crypto pair or an
# equity?" without grepping for slashes. Kept here so the source of truth is
# config, not strewn ``"/" in symbol`` checks across modules.
IS_EQUITY = "/" not in SYMBOL

# Per-symbol artifact paths. Engine + tools read from these by default so that
# data and weights for one coin never bleed into another's runs.
FEATURE_DUMP_PATH = f"./calibration/feature_history_{SYMBOL_SLUG}_balanced.csv"
OOD_CALIBRATION_PATH = f"./calibration/latest_{SYMBOL_SLUG}.json"
TCN_WEIGHTS_PATH = f"./calibration/tcn_weights_{SYMBOL_SLUG}.pt"
TCN_THRESHOLD_PATH = f"./calibration/tcn_threshold_{SYMBOL_SLUG}.json"

# Credentials are read at runtime, never hardcoded.
EXCHANGE_API_KEY = os.environ.get("EXCHANGE_API_KEY", "")
EXCHANGE_SECRET = os.environ.get("EXCHANGE_SECRET", "")

# Live-trading guard. EXCHANGE_LIVE must be the literal string "true" to allow
# real order flow. Any other value (including unset) keeps the system in testnet.
EXCHANGE_LIVE = os.environ.get("EXCHANGE_LIVE", "false").lower() == "true"

# Replay mode. Empty (default) → live exchange feed. Non-empty path → engine
# uses ReplaySensorArray (test_replay.py) to drive Layers 2/3/4 + dashboard
# from a feature_history CSV. Single-variable swap: unset to return to live.
REPLAY_CSV = os.environ.get("REPLAY_CSV", "")

# ============================================================================
# Runtime Mode Flags
# ============================================================================
SHADOW_MODE = True            # True: route via LocalMatchingEngine (no real orders).
TRADING_ENABLED = True        # Global kill switch.
CALIBRATION_STALE = False     # Set True by drift monitor; pauses mandate generation.

# ============================================================================
# Fees
# ============================================================================
# Equity defaults — zero-commission broker (Alpaca). Asymmetric regulatory
# fees apply only on sells; the execution env adds them on a per-fill basis.
# Crypto fee structure (Coinbase Advanced spot, 60/40 bps) is retained below
# and selected when SYMBOL is a crypto pair.
if IS_EQUITY:
    TAKER_FEE = 0.0
    MAKER_FEE = 0.0
elif EXCHANGE_ID == "hyperliquid":
    # Hyperliquid perps default (non-VIP). HL offers rebates at higher tiers;
    # these are conservative for prove-out so trained policy doesn't overfit
    # to optimistic fees. Override the constants if your tier differs.
    TAKER_FEE = 0.00045       # 4.5 bps
    MAKER_FEE = 0.00015       # 1.5 bps
else:
    TAKER_FEE = 0.0060        # Coinbase Advanced spot — 60 bps
    MAKER_FEE = 0.0040        # 40 bps
FEE_AGGRESSION_THRESHOLD = 0.0  # Action below threshold treated as taker.

# Equities-only regulatory fees (sells only). Both are zero for crypto.
SEC_FEE_BPS_ON_SELL = 0.00229       # ~$22.90 / $1M principal, sells only
FINRA_TAF_PER_SHARE_SELL = 0.000166 # capped at $8.30 per execution

# ============================================================================
# Layer 1 — Robust statistical models
# ============================================================================
KALMAN_DOF = 4.0              # Student-t ν for spread filter. Frozen for live.

# Student-t HMM degrees of freedom per state. Fitted offline; never updated live.
HMM_DOF_LAMINAR = 9.0
HMM_DOF_TRANSITION = 5.0
HMM_DOF_TURBULENT = 3.5

# Sensor cadence
SENSOR_LOOP_INTERVAL_SEC = 0.1     # 100ms

# VPIN
VPIN_BUCKETS = 50                  # Trailing volume buckets
VPIN_VOLUME_AVG_WINDOW_SEC = 3600  # Rolling 1h volume for bucket sizing
EPSILON = 1e-8

# C/E ratio
CE_RATIO_MAX = 50.0
CE_ROLLING_WINDOW_SEC = 5.0

# Liquidation cascade
LIQUIDATION_WINDOW_SEC = 10.0

# OOD detector
OOD_THRESHOLD = 4.5

# ============================================================================
# Layer 1 → 2 Calibration
# ============================================================================
CALIBRATION_REGISTRY = {
    # Equities — placeholder values; refit via fit_ood_from_csv.py once the
    # historical harvester has produced a feature_history_*.csv.
    "NVDA": 0.25,
    "SPY":  0.20,
    # Crypto
    "BTC/USD": 0.18,           # Coinbase Advanced spot
    "ETH/USD": 0.22,
    "SOL/USD": 0.28,           # higher-beta alt; default est until fitted
    "AVAX/USD": 0.30,
    "DOGE/USD": 0.32,
    "LINK/USD": 0.26,
    "BTC/USDT:USDT": 0.18,     # Legacy perp values, retained for back-compat
    "ETH/USDT:USDT": 0.22,
    # Hyperliquid perps — initial estimates; refit via fit_ood_from_csv.py
    # once fetch_history_hyperliquid.py has produced a feature_history CSV.
    "BTC/USDC:USDC": 0.18,
    "ETH/USDC:USDC": 0.22,
    "SOL/USDC:USDC": 0.28,
    "HYPE/USDC:USDC": 0.30,
}
ALPHA_CALIBRATION_C = CALIBRATION_REGISTRY.get(SYMBOL, 0.25)

# Shock event identification
SHOCK_PRICE_MOVE_PCT = 0.0025           # DEFAULT: 0.3%
MIN_SHOCK_SPACING_SECONDS = 60
MAX_DIFFUSION_TICKS = 6000              #DEFAULT 20

# TCN leading-classifier horizon (used by train_stream.py). For each shock
# event at tick t, the H ticks in [t-H, t) are labeled positive — i.e. the
# model is trained to answer "will a shock start within the next H ticks?".
# 30 = 30 seconds at Alpaca's 1Hz IEX feed (actionable execution window;
# stays well within session boundaries). Crypto pipelines using train_tcn.py
# still consume MAX_DIFFUSION_TICKS above for back-compat.
TCN_LABEL_HORIZON_TICKS = 30
EQUILIBRIUM_BAND_PCT = 0.002           # DEFAULT 0.1% (0.001)
EQUILIBRIUM_STABILITY_TICKS = 300        #DEFAULT: 10
MIN_CALIBRATION_EVENTS = 30
TURBULENCE_THRESHOLD = 0.5              #DEFAULT: 0.9

# Drift detection
DRIFT_THRESHOLD = 0.35                 # MAPE threshold
N_DRIFT_WINDOW = 10

# Calibration cadence (offline job)
CALIBRATION_INTERVAL_HOURS = 48
EWLS_DECAY = 0.95                      # Exponentially weighted OLS decay

# ============================================================================
# Layer 2 — Alpha generation
# ============================================================================
TCN_INPUT_LENGTH = 60
TCN_INPUT_CHANNELS = 3                 # [ce_ratio, obi, liquidation_rate]
TCN_HIDDEN_CHANNELS = 32
TCN_DILATIONS = (1, 2, 4, 8, 16)

# Heat equation solver
LAMBDA_ATTRITION = 0.15
LAMBDA_SENSITIVITY_DELTA = 0.05
P_EQ_CONFIDENCE_BAND_BPS = 15.0
FD_DOMAIN_PCT = 0.03                   # ±3% domain
FD_STABILITY_FACTOR = 0.5              # CFL safety

# Mandate sizing — per-pair defaults from _PAIR_DEFAULTS above.
CAPTURE_RATIO = 0.05
MAX_POSITION_LIMIT = float(os.environ.get("MAX_POSITION_LIMIT", _pair["max_position"]))
MIN_ORDER_SIZE = float(os.environ.get("MIN_ORDER_SIZE", _pair["min_order"]))
BASE_EXECUTION_WINDOW = 20.0           # seconds
MAX_PASSIVE_OFFSET_BPS = 5.0

# ============================================================================
# Layer 3 — Execution agent
# ============================================================================
OBS_DIM = 42
ACTION_DIM = 1

# PPO
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 64
PPO_ROLLOUT_LENGTH = 2048

# Reward weights — tune via Optuna HPO
ETA = 0.3                              # Fill variance weight
GAMMA_INV = 0.5                        # Inventory urgency weight
TERMINAL_PENALTY_MULTIPLIER = 2.0

# Execution safety
EXECUTION_HARD_STOP_MULTIPLIER = 1.5   # Cancel limit orders after window * 1.5

# ============================================================================
# HPO
# ============================================================================
HPO_TRAIN_STEPS = 500_000
HPO_DEGRADATION_THRESHOLD = 0.20
HPO_VALIDATION_HOURS = 6
HPO_N_TRIALS = 50

# ============================================================================
# Layer 4 — Shadow simulator
# ============================================================================
POLYAK_TAU = 0.05
KL_THRESHOLD = 0.1
MIN_SHADOW_STEPS = 50_000
REPLAY_BUFFER_HOURS = 24
SHADOW_PARALLEL_ENVS = 8

# ============================================================================
# Risk & Constraints
# ============================================================================
# Drawdown circuit-breaker: hybrid USD + percentage gate.
#
# The percentage gate alone — drawdown / peak_pnl — has unbounded leverage
# when peak_pnl is small. A $1 winner followed by a $10 loss is a 1000%
# drawdown ratio and trips any reasonable percent threshold instantly. So:
#
#   1. MAX_SESSION_DRAWDOWN_USD: absolute dollar cap, always active.
#      Trips when (peak - current) > USD limit, regardless of peak size.
#      This is the primary safety stop.
#
#   2. MAX_SESSION_DRAWDOWN_PCT: percentage cap, active only once peak >=
#      DRAWDOWN_PCT_MIN_PEAK_USD. Below that floor the ratio is pathological
#      and we rely on the USD cap.
MAX_SESSION_DRAWDOWN_USD = float(
    os.environ.get("MAX_SESSION_DRAWDOWN_USD", "1000.0")
)
MAX_SESSION_DRAWDOWN_PCT = float(
    os.environ.get("MAX_SESSION_DRAWDOWN_PCT", "0.02")
)
DRAWDOWN_PCT_MIN_PEAK_USD = float(
    os.environ.get("DRAWDOWN_PCT_MIN_PEAK_USD", "100.0")
)

# MTM (mark-to-market) drawdown — same hybrid USD + percent gate, but on
# theoretical_pnl (cash + position * mid). The IS-based gate above only sees
# accumulated execution-shortfall costs (~$10s of dollars), so a position
# accumulating $20K of directional exposure loss can sail past the IS gate
# undetected. MTM gate catches that.
#
# Defaults split by mode:
#   - LIVE (REPLAY_CSV unset): tight production caps. A 100-share TSLA position
#     (~$44K notional) trips at $20K USD (~45% adverse move) or 10% peak-relative.
#   - REPLAY: loose caps. The harvested CSV concatenates 73 sessions; replay
#     never hits a "market open" event, so _position and _cash never reset
#     across CSV session boundaries. Mid can jump 40%+ at a session join,
#     which trips the production caps almost immediately. Loosening for
#     replay-mode demos is a workaround for the missing reset; live RTH gets
#     full production safety.
# Env vars override either default explicitly.
if REPLAY_CSV:
    MAX_MTM_DRAWDOWN_USD = float(
        os.environ.get("MAX_MTM_DRAWDOWN_USD", "1_000_000.0")
    )
    MAX_MTM_DRAWDOWN_PCT = float(
        os.environ.get("MAX_MTM_DRAWDOWN_PCT", "10.0")
    )
else:
    MAX_MTM_DRAWDOWN_USD = float(
        os.environ.get("MAX_MTM_DRAWDOWN_USD", "20000.0")
    )
    MAX_MTM_DRAWDOWN_PCT = float(
        os.environ.get("MAX_MTM_DRAWDOWN_PCT", "0.10")
    )
KILL_SWITCH_LOG_PATH = "./logs/kill_switch.log"

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_PATH = os.environ.get("LOG_PATH", "./logs/engine.log")

# ============================================================================
# Market session (equities only — crypto runs 24/7)
# ============================================================================
# Regular trading hours; half-days and holidays are resolved at runtime via
# Alpaca's /v2/calendar so we don't hardcode them here.
MARKET_TZ = "America/New_York"
SESSION_OPEN_LOCAL = "09:30"
SESSION_CLOSE_LOCAL = "16:00"

# After the open we suspend the OOD gate briefly so the first few quotes don't
# trip a false-positive shock signal while filters re-warm.
SESSION_WARMUP_SECONDS = 30

# Which sensors get reset_session() called on them at each market open.
SESSION_RESET_HMM = True
SESSION_RESET_KALMAN = True
SESSION_RESET_VPIN = True

# Alpaca data feed selector. Free tier = "iex"; Algo Trader Plus = "sip".
ALPACA_DATA_FEED = os.environ.get("ALPACA_DATA_FEED", "iex")
