"""Central configuration. Everything that defines a 'setup' lives here so that training, paper trading
and grading all agree on the same definitions."""
from __future__ import annotations
from dataclasses import dataclass, field

# ---- universe ---------------------------------------------------------------------------------------
CRYPTO = {            # Kraken pair -> display symbol
    "XBTUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "XRPUSD": "XRP-USD",
    "DOGEUSD": "DOGE-USD",
}
STOCKS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMZN", "MSFT", "META", "AMD", "IWM"]
INTERVAL = "1h"
BAR_MINUTES = 60
CRYPTO_BAR_MINUTES = 240   # crypto is modelled on 4h bars: hourly moves on Kraken do not cover the 0.6% round-trip fee

# ---- setup definition (triple barrier) -------------------------------------------------------------
HORIZON_BARS = 24          # a setup resolves within 24 hourly bars (one day) if neither barrier is hit
TARGET_ATR = 1.5           # default target = entry + 1.5 * ATR(14)
STOP_ATR = 1.0             # default stop   = entry - 1.0 * ATR(14)   -> reward:risk 1.5 before costs
# per-market barriers: crypto pays ~0.6% round trip on Kraken, so its barriers must be wider or fees eat the edge
BARRIERS = {"crypto": (2.0, 1.2), "stock": (1.5, 1.0)}   # (target_atr, stop_atr)
def barriers(market: str) -> tuple[float, float]:
    return BARRIERS.get(market, (TARGET_ATR, STOP_ATR))
MIN_HISTORY_BARS = 400     # bars needed before a symbol can produce a setup

# ---- costs ------------------------------------------------------------------------------------------
FEES_BPS = {"crypto": 26.0, "stock": 2.0}      # round-trip in basis points (Kraken taker ~0.26% each side is 52 bps; we model entry+exit separately)
SLIPPAGE_BPS = {"crypto": 5.0, "stock": 2.0}   # per side
# per-side fee used by the paper engine
FEE_SIDE_BPS = {"crypto": 26.0, "stock": 1.0}

# ---- models -----------------------------------------------------------------------------------------
NN_SEEDS = [11, 23, 37, 51, 73]
NN_HIDDEN = [(32, 16), (48, 24), (32, 32), (64, 16), (24, 24, 12)]
TRAIN_MIN_SETUPS = 300
WALK_FORWARD_FOLDS = 5
CALIBRATION_BINS = 10

# ---- trade filter -----------------------------------------------------------------------------------
@dataclass
class StrategyMode:
    name: str
    min_prob: float          # calibrated win probability needed
    max_disagreement: float  # max std-dev across the 7 voters
    min_rr_after_costs: float
    max_hold_bars: int

MODES = {
    "swing":    StrategyMode("swing",    0.58, 0.12, 1.10, HORIZON_BARS),
    "momentum": StrategyMode("momentum", 0.60, 0.10, 1.20, HORIZON_BARS),
    "position": StrategyMode("position", 0.56, 0.15, 1.00, HORIZON_BARS * 3),
}
DEFAULT_MODE = "swing"

# ---- risk -------------------------------------------------------------------------------------------
START_CASH = 10_000.0
RISK_PER_TRADE = 0.01       # 1% of equity at risk per trade (distance to stop)
MAX_OPEN_TRADES = 6
MAX_PER_SYMBOL = 1
MAX_DAILY_LOSS = 0.03       # stop opening trades for the day after -3%

# ---- learning ---------------------------------------------------------------------------------------
CHALLENGER_MIN_RESOLVED = 40    # resolved setups needed before a challenger can be judged
PROMOTE_BRIER_GAIN = 0.004      # challenger must beat champion Brier by this much on recent resolved setups
