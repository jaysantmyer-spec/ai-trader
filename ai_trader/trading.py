"""Stage 5 - signal & trade filter.  Stage 6 - paper-trading engine.
The filter turns a probability into BUY / WAIT with explicit reasons; the engine simulates fills, stops,
targets, time exits, fees and slippage, and keeps the ledger in data/trades.csv."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from . import config as C
from .data import market_of

TRADES = "data/trades.csv"
SETUPS = "data/setups.csv"       # every evaluated setup (pre-registered prediction), traded or not
TRADE_COLS = ["id", "symbol", "market", "mode", "opened", "entry", "stop", "target", "qty", "risk_usd", "p", "disagree",
              "fee_in", "closed", "exit", "reason", "pnl", "r", "bars", "model"]
SETUP_COLS = ["time", "symbol", "market", "p", "p_raw", "disagree", "f_mean", "rr_net", "decision", "reasons", "model",
              "resolved", "y", "y_r", "close_price"]


def costs_bps(market: str) -> float:
    return 2 * (C.FEE_SIDE_BPS[market] + C.SLIPPAGE_BPS[market])   # round-trip


def rr_after_costs(entry: float, atr: float, market: str) -> float:
    tg, sp = C.barriers(market)
    gross_win = tg * atr
    gross_loss = sp * atr
    cost = entry * costs_bps(market) / 1e4
    return (gross_win - cost) / (gross_loss + cost)


def decide(p: float, disagree: float, rr_net: float, f_mean: float, mode: C.StrategyMode,
           open_count: int, open_symbol: int, day_pnl_pct: float) -> tuple[str, list[str]]:
    reasons = []
    if p < mode.min_prob:               reasons.append(f"p {p:.2f} < {mode.min_prob}")
    if disagree > mode.max_disagreement: reasons.append(f"voters disagree {disagree:.2f} > {mode.max_disagreement}")
    if rr_net < mode.min_rr_after_costs: reasons.append(f"R:R after costs {rr_net:.2f} < {mode.min_rr_after_costs}")
    if f_mean < -0.25:                   reasons.append(f"forecasters lean down ({f_mean:.2f})")
    if open_count >= C.MAX_OPEN_TRADES:  reasons.append("max open trades")
    if open_symbol >= C.MAX_PER_SYMBOL:  reasons.append("already in symbol")
    if day_pnl_pct <= -C.MAX_DAILY_LOSS: reasons.append("daily loss limit hit")
    # expected value check: p*win - (1-p)*loss in R units must be positive with margin
    ev = p * rr_net - (1 - p) * 1.0
    if ev < 0.05:                        reasons.append(f"EV {ev:+.2f}R too small")
    return ("BUY" if not reasons else "WAIT"), reasons


# ---- ledger ------------------------------------------------------------------------------------------------
def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES):
        return pd.DataFrame(columns=TRADE_COLS)
    return pd.read_csv(TRADES, parse_dates=["opened", "closed"])


def save_trades(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(TRADES, index=False)


def load_setups() -> pd.DataFrame:
    if not os.path.exists(SETUPS):
        return pd.DataFrame(columns=SETUP_COLS)
    return pd.read_csv(SETUPS, parse_dates=["time"])


def save_setups(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(SETUPS, index=False)


def equity(trades: pd.DataFrame) -> float:
    closed = trades[trades["closed"].notna()]
    return C.START_CASH + float(closed["pnl"].sum()) if len(closed) else C.START_CASH


# ---- engine ------------------------------------------------------------------------------------------------
class PaperEngine:
    def __init__(self):
        self.trades = load_trades()

    def open_trade(self, symbol: str, now: pd.Timestamp, entry_ref: float, atr: float, p: float, disagree: float,
                   mode: str, model: str) -> dict:
        market = market_of(symbol)
        slip = entry_ref * C.SLIPPAGE_BPS[market] / 1e4
        entry = entry_ref + slip
        tg, sp = C.barriers(market)
        stop = entry_ref - sp * atr
        target = entry_ref + tg * atr
        eq = equity(self.trades)
        risk_usd = eq * C.RISK_PER_TRADE
        qty = risk_usd / max(entry - stop, 1e-9)
        fee_in = qty * entry * C.FEE_SIDE_BPS[market] / 1e4
        tid = int(self.trades["id"].max()) + 1 if len(self.trades) else 1
        row = {"id": tid, "symbol": symbol, "market": market, "mode": mode, "opened": now, "entry": entry, "stop": stop,
               "target": target, "qty": qty, "risk_usd": risk_usd, "p": p, "disagree": disagree, "fee_in": fee_in,
               "closed": pd.NaT, "exit": np.nan, "reason": "", "pnl": np.nan, "r": np.nan, "bars": 0, "model": model}
        self.trades = pd.concat([self.trades, pd.DataFrame([row])], ignore_index=True)
        return row

    def update(self, bars: dict[str, pd.DataFrame], max_hold: dict[str, int]):
        """Walk each open trade forward through the bars that arrived after it was opened."""
        for i, t in self.trades[self.trades["closed"].isna()].iterrows():
            df = bars.get(t["symbol"])
            if df is None or df.empty:
                continue
            after = df[df["time"] > t["opened"]]
            if after.empty:
                continue
            market = t["market"]
            nbars = int(t["bars"])
            for _, b in after.iloc[nbars:].iterrows():
                nbars += 1
                exit_px = None; reason = None
                if b["low"] <= t["stop"]:
                    exit_px = t["stop"] - t["stop"] * C.SLIPPAGE_BPS[market] / 1e4; reason = "stop"
                elif b["high"] >= t["target"]:
                    exit_px = t["target"]; reason = "target"
                elif nbars >= max_hold.get(t["mode"], C.HORIZON_BARS):
                    exit_px = b["close"] - b["close"] * C.SLIPPAGE_BPS[market] / 1e4; reason = "time"
                if exit_px is not None:
                    fee_out = t["qty"] * exit_px * C.FEE_SIDE_BPS[market] / 1e4
                    pnl = (exit_px - t["entry"]) * t["qty"] - t["fee_in"] - fee_out
                    self.trades.loc[i, ["closed", "exit", "reason", "pnl", "r", "bars"]] = [b["time"], exit_px, reason, pnl, pnl / t["risk_usd"], nbars]
                    break
            else:
                self.trades.loc[i, "bars"] = nbars
        save_trades(self.trades)

    def open_positions(self) -> pd.DataFrame:
        return self.trades[self.trades["closed"].isna()]

    def day_pnl_pct(self, now: pd.Timestamp) -> float:
        c = self.trades[self.trades["closed"].notna()]
        today = c[pd.to_datetime(c["closed"], utc=True).dt.date == now.date()]
        return float(today["pnl"].sum()) / equity(self.trades) if len(today) else 0.0

    def scorecard(self) -> dict:
        c = self.trades[self.trades["closed"].notna()].copy()
        if c.empty:
            return {"trades": 0}
        c = c.sort_values("closed")
        eq = C.START_CASH + c["pnl"].cumsum()
        peak = eq.cummax(); dd = ((eq - peak) / peak).min()
        wins = c[c["pnl"] > 0]; losses = c[c["pnl"] <= 0]
        pf = float(wins["pnl"].sum() / -losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() < 0 else float("inf")
        return {
            "trades": int(len(c)), "win_rate": float((c["pnl"] > 0).mean()), "avg_r": float(c["r"].mean()),
            "total_pnl": float(c["pnl"].sum()), "return_pct": float(c["pnl"].sum() / C.START_CASH), "profit_factor": pf,
            "max_drawdown": float(dd), "avg_bars": float(c["bars"].mean()),
            "by_reason": c["reason"].value_counts().to_dict(),
            "by_market": c.groupby("market")["pnl"].agg(["count", "sum"]).round(2).to_dict("index"),
            "brier_on_trades": float(np.mean((c["p"] - (c["pnl"] > 0).astype(float)) ** 2)),
            "last_closed": str(c["closed"].iloc[-1]),
        }
