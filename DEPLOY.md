# Deploying AI Trader

## 1. First run (GitHub)
Repo → **Actions** → enable workflows if asked → **hourly run** → **Run workflow**.
The first run downloads ~2 years of hourly bars for 15 symbols, trains the first champion (walk-forward, ~5 min), evaluates
the newest bar for every symbol and commits `data/` and `models/`. After that it runs by itself every hour.

## 2. Dashboard (Streamlit Community Cloud)
share.streamlit.io → **Create app** → this repo, branch `main`, main file `app.py`, Python 3.11 → Deploy.
The app only reads committed files, so it stays fast and never trains on its own. It redeploys when the bot commits.

## 3. Weekly backtest
**weekly backtest** workflow runs Mondays; run it by hand any time. It writes `data/backtest.json` for the Backtest tab.

## Tuning
`ai_trader/config.py` holds everything: universe, barriers (target/stop in ATR), fees, strategy modes (probability thresholds,
disagreement, R:R), risk limits, and the promotion rule. Change, commit, and the next hourly run uses it.

## What "good" looks like
After ~2 weeks: 300+ resolved setups. Model Brier below base Brier by 0.005+, BUY win rate within a few points of the
probabilities quoted, positive average R after costs. If those aren't there, the honest conclusion is that this feature set has
no edge on these markets at this timeframe — and the system will have told you so for free.
