"""AI Trader - Streamlit dashboard. Reads only what the GitHub Actions run commits under data/ and models/;
it never fetches or trains on its own."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Trader (paper)", page_icon="📈", layout="wide")


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def csv(p, **kw):
    try:
        return pd.read_csv(p, **kw)
    except Exception:
        return pd.DataFrame()


S = jload("data/summary.json", {})
champ = jload("models/champion.json", {})
setups = csv("data/setups.csv", parse_dates=["time"])
trades = csv("data/trades.csv", parse_dates=["opened", "closed"])
bt = jload("data/backtest.json", {})

st.title("AI Trader — paper-trading research pipeline")
st.caption("Data → indicators → 6 forecasters → 5 NNs + logit + trees → filter → paper trades → grading → champion/challenger. "
           "Nothing here is advice; a forecast, a BUY signal, a high probability and a profitable trade are four different things.")

if not S:
    st.warning("No run yet. In GitHub → Actions → **hourly run** → Run workflow. The first run downloads history and trains (≈5 min).")
    st.stop()

ts = pd.Timestamp(S["ts"]); age = (pd.Timestamp.utcnow() - ts).total_seconds() / 60
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Equity (paper)", f"${S['equity']:,.2f}", f"{(S['equity']/10000-1)*100:+.2f}%")
c2.metric("Open trades", S["open_trades"])
sc = S.get("scorecard", {})
c3.metric("Closed trades", sc.get("trades", 0), f"win {sc.get('win_rate', 0)*100:.0f}%" if sc.get("trades") else None)
c4.metric("Setups graded", f"{S['setups_resolved']}/{S['setups_total']}")
c5.metric("Brier (setups)", f"{S['setup_brier']:.4f}" if S.get("setup_brier") else "—",
          f"base {S['setup_base_brier']:.4f}" if S.get("setup_base_brier") else None, delta_color="off")
c6.metric("Last run", f"{age:.0f} min ago", S.get("mode"))

tab_today, tab_book, tab_score, tab_brain, tab_bt, tab_learn, tab_data = st.tabs(
    ["Today", "Paper book", "Scorecard", "AI brain", "Backtest", "Learning", "Data"])

# ---- Today -------------------------------------------------------------------------------------------------
with tab_today:
    st.subheader("Latest evaluation of every symbol")
    if len(setups):
        last = setups.sort_values("time").groupby("symbol").tail(1).sort_values("p", ascending=False)
        show = last[["symbol", "market", "time", "p", "disagree", "rr_net", "f_mean", "decision", "reasons"]].copy()
        show["p"] = (show["p"] * 100).round(1); show["disagree"] = show["disagree"].round(3); show["rr_net"] = show["rr_net"].round(2); show["f_mean"] = show["f_mean"].round(2)
        st.dataframe(show.rename(columns={"p": "win prob %", "disagree": "voter std", "rr_net": "R:R net", "f_mean": "forecasters (ATR)"}),
                     use_container_width=True, hide_index=True)
        buys = last[last["decision"] == "BUY"]
        if len(buys):
            st.success(f"{len(buys)} BUY signal(s): " + ", ".join(f"{r.symbol} ({r.p*100:.0f}%)" for r in buys.itertuples()))
        else:
            st.info("No BUY signals on the latest bar. WAIT is the normal state: the filter needs calibrated p ≥ threshold, voter agreement, R:R after costs and positive EV all at once.")
    else:
        st.info("No setups evaluated yet.")

# ---- Paper book ---------------------------------------------------------------------------------------------
with tab_book:
    st.subheader("Open positions")
    op = trades[trades["closed"].isna()] if len(trades) else pd.DataFrame()
    if len(op):
        st.dataframe(op[["id", "symbol", "mode", "opened", "entry", "stop", "target", "qty", "risk_usd", "p", "bars"]].round(4), use_container_width=True, hide_index=True)
    else:
        st.write("None.")
    st.subheader("Closed trades")
    cl = trades[trades["closed"].notna()].sort_values("closed", ascending=False) if len(trades) else pd.DataFrame()
    if len(cl):
        st.dataframe(cl[["id", "symbol", "market", "opened", "closed", "entry", "exit", "reason", "pnl", "r", "p", "bars"]].round(4), use_container_width=True, hide_index=True)
        eq = 10000 + cl.sort_values("closed")["pnl"].cumsum()
        st.line_chart(pd.DataFrame({"equity": eq.values}, index=cl.sort_values("closed")["closed"]))
    else:
        st.write("None yet.")

# ---- Scorecard ----------------------------------------------------------------------------------------------
with tab_score:
    st.subheader("How to judge this system")
    st.markdown("""
- **Brier on setups** vs **base Brier**: the model's probabilities against the naive base rate, on *every* pre-registered setup (not just trades). Lower is better; if it isn't below base after a few hundred setups, the brain has no skill.
- **BUY win rate** vs the probability it quoted: a 60% signal should win about 60% of the time.
- **Avg R / profit factor / max drawdown** on closed paper trades, after fees and slippage.
- Judge nothing on fewer than ~100 resolved setups and ~30 closed trades.
""")
    if sc.get("trades"):
        a, b, c, d = st.columns(4)
        a.metric("Avg R", f"{sc['avg_r']:+.2f}"); b.metric("Profit factor", f"{sc['profit_factor']:.2f}" if np.isfinite(sc['profit_factor']) else "∞")
        c.metric("Max drawdown", f"{sc['max_drawdown']*100:.1f}%"); d.metric("Brier on trades", f"{sc['brier_on_trades']:.4f}")
        st.write("Exits:", sc.get("by_reason")); st.write("By market:", sc.get("by_market"))
    res = setups[setups["resolved"] == True] if len(setups) else pd.DataFrame()
    if len(res):
        st.subheader(f"Reliability on {len(res)} resolved setups")
        res = res.copy(); res["bin"] = (res["p"] * 10).clip(0, 9).astype(int) / 10
        rel = res.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
        st.dataframe(rel.round(3), hide_index=True)
        b = res[res["decision"] == "BUY"]
        if len(b):
            st.write(f"BUY signals resolved: {len(b)} · quoted avg {b['p'].mean()*100:.1f}% · actual win {b['y'].mean()*100:.1f}%")

# ---- Brain -------------------------------------------------------------------------------------------------
with tab_brain:
    st.subheader("Champion model (walk-forward, out-of-sample)")
    if champ:
        a, b, c, d = st.columns(4)
        a.metric("Trained on", f"{champ['n']:,} setups"); b.metric("OOS Brier (cal)", f"{champ['oos_brier_cal']:.4f}", f"base {champ['oos_brier_base']:.4f}", delta_color="off")
        c.metric("OOS accuracy", f"{champ['oos_acc']*100:.1f}%", f"base rate {champ['base_rate']*100:.1f}%", delta_color="off"); d.metric("Trained", champ["trained_at"][:16].replace("T", " "))
        st.write("Per-voter OOS Brier (lower is better):")
        st.dataframe(pd.DataFrame([champ["voter_brier"]]).T.rename(columns={0: "brier"}).round(4))
        st.write("Walk-forward folds:"); st.dataframe(pd.DataFrame(champ["folds"]).round(4), hide_index=True)
        st.write("Calibration (OOS):"); st.dataframe(pd.DataFrame(champ["reliability"]).round(3), hide_index=True)
        hc = champ.get("oos_hi_conf") or {}
        if hc.get("n"):
            st.write(f"When the model said ≥60% out-of-sample ({hc['n']} cases) it was right {hc['win']*100:.1f}% of the time.")
    else:
        st.write("Not trained yet.")

# ---- Backtest -----------------------------------------------------------------------------------------------
with tab_bt:
    st.subheader("Offline walk-forward backtest of the full pipeline")
    if bt:
        st.json(bt)
        bs = csv("data/backtest_signals.csv", parse_dates=["time"])
        if len(bs):
            st.line_chart(pd.DataFrame({"cumulative R": bs.sort_values("time")["r"].cumsum().values}, index=bs.sort_values("time")["time"]))
    else:
        st.write("Run `python -m ai_trader backtest` (the weekly workflow does this).")

# ---- Learning -----------------------------------------------------------------------------------------------
with tab_learn:
    st.subheader("Champion / challenger")
    st.markdown("Every day a challenger is retrained on the newest data. It replaces the champion only if its Brier on the "
                "last few hundred *resolved, pre-registered* setups is better by a clear margin. Lucky weeks don't get promoted.")
    for p in (S.get("state", {}).get("promotions") or [])[::-1][:20]:
        st.write(p)

# ---- Data ---------------------------------------------------------------------------------------------------
with tab_data:
    st.subheader("Data status")
    ds = S.get("data_status", {})
    if ds:
        st.dataframe(pd.DataFrame(ds).T, use_container_width=True)
    st.caption("Crypto: Kraken public OHLC (1h). Stocks: Yahoo Finance (1h, regular session). Bars are cached in data/bars/.")
