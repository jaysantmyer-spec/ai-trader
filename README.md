# AI Trader — paper-trading research pipeline

The pipeline from the diagram, as working code:

| Stage | Module | What it does |
|---|---|---|
| 1 Market data | `ai_trader/data.py` | Kraken public OHLC (crypto, 1h) + Yahoo Finance (stocks, 1h), cached under `data/bars/` |
| 2 Data preparation | `ai_trader/features.py` | EMA, RSI, MACD, ATR, Bollinger, returns/momentum, volatility, support/resistance, candle shape, time-of-day |
| 3 Forecasting ensemble | `ai_trader/features.py` | naive, linear regression, Holt trend, AR(1), EMA drift, mean reversion → expected move in ATR units |
| 4 AI brain | `ai_trader/brain.py` | 5 neural nets + logistic regression + gradient-boosted trees vote on P(target before stop); walk-forward validated; Platt-calibrated |
| 5 Signal & filter | `ai_trader/trading.py` | strategy mode, probability threshold, voter disagreement, R:R after fees+slippage, EV, risk limits → BUY / WAIT with reasons |
| 6 Paper engine | `ai_trader/trading.py` | 1% risk sizing, stop / target / time exits, fees, slippage, P&L ledger in `data/trades.csv` |
| 7 Feedback | `ai_trader/pipeline.py` | every evaluated setup is pre-registered in `data/setups.csv` and graded when it resolves; daily challenger vs champion by Brier; promote or keep |

**Setup definition.** A long entered at the close of bar *t*; target = +1.5 ATR (crypto +2.5), stop = −1.0 ATR (crypto −1.5), resolved within 24 bars. Crypto barriers are wider because Kraken's taker fee (0.26% a side) would otherwise eat most of a 1-ATR hourly stop.

**Honesty rules.** Every accuracy number shown is out-of-sample. Predictions are written to the ledger *before* the outcome is known. The dashboard reports Brier score against the base rate on *all* setups, not just the ones that were traded. Nothing is ever executed with real money.

See `DEPLOY.md` to run it. Command line: `python -m ai_trader [run|train|backtest|refresh]`.

This is a research tool. Better behaviour is not the same as profitability.
