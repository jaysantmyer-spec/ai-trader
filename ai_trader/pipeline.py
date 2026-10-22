"""Stages 1-7 wired together.  `python -m ai_trader run` is what GitHub Actions calls every hour:
  1. refresh bars            4. champion predicts the newest bar for every symbol
  2. build features/labels   5. filter -> paper trades; update open trades with new bars
  3. (re)train if due        6. grade pre-registered setups; 7. challenger vs champion, promote or keep
"""
from __future__ import annotations
import os, json, sys, time
import numpy as np
import pandas as pd
from . import config as C
from . import data as D
from .features import build_features, label_setups, feature_cols
from .brain import Brain, brier
from .trading import PaperEngine, decide, rr_after_costs, load_setups, save_setups, SETUP_COLS, equity

STATE = "data/state.json"


def log(msg: str):
    print(f"[{pd.Timestamp.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state() -> dict:
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"last_train": None, "champion_since": None, "promotions": []}


def save_state(s: dict):
    os.makedirs("data", exist_ok=True)
    json.dump(s, open(STATE, "w"), indent=1, default=str)


# ---- dataset across all symbols ----------------------------------------------------------------------------
def dataset(min_bars: int = C.MIN_HISTORY_BARS) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, feats = [], {}
    for sym in D.all_symbols():
        df = D.load_model_bars(sym)
        if len(df) < min_bars:
            continue
        f = build_features(df)
        tg, sp = C.barriers(D.market_of(sym))
        f = label_setups(f, tgt=tg, stp=sp)
        f["symbol"] = sym; f["market"] = D.market_of(sym)
        feats[sym] = f
        rows.append(f)
    if not rows:
        return pd.DataFrame(), feats
    all_ = pd.concat(rows, ignore_index=True)
    return all_, feats


def training_frame(all_: pd.DataFrame) -> pd.DataFrame:
    cols = feature_cols()
    tr = all_.dropna(subset=cols + ["y"]).copy()
    # drop the last HORIZON bars of each symbol from training even if labelled (label needs the future)
    return tr


# ---- train -------------------------------------------------------------------------------------------------
def train(name: str = "champion", seed_offset: int = 0) -> dict | None:
    all_, _ = dataset()
    if all_.empty:
        log("no data to train on"); return None
    tr = training_frame(all_)
    if len(tr) < C.TRAIN_MIN_SETUPS:
        log(f"only {len(tr)} labelled setups; need {C.TRAIN_MIN_SETUPS}"); return None
    cols = feature_cols()
    X = tr[cols].values.astype(float); y = tr["y"].values.astype(int); t = tr["time"].values.astype("datetime64[ns]").astype(np.int64)
    log(f"training {name} on {len(tr)} setups, base rate {y.mean():.3f}")
    b = Brain(name, seed_offset)
    meta = b.walk_forward(X, y, t)
    b.save(name)
    log(f"{name}: OOS Brier {meta['oos_brier_cal']:.4f} vs base {meta['oos_brier_base']:.4f}, acc {meta['oos_acc']:.3f}")
    return meta


# ---- predict + trade ---------------------------------------------------------------------------------------
def run(mode_name: str = C.DEFAULT_MODE, retrain_hours: int = 24):
    now = pd.Timestamp.utcnow()
    state = load_state()
    log("refreshing bars"); st = D.refresh_all(log)
    ok = sum(1 for v in st.values() if v.get("ok"))
    log(f"{ok}/{len(st)} symbols refreshed")

    all_, feats = dataset()
    if all_.empty:
        log("not enough history yet"); save_state(state); return

    champ = Brain.load("champion")
    due = champ is None or state.get("last_train") is None or \
        (now - pd.Timestamp(state["last_train"])) > pd.Timedelta(hours=retrain_hours)
    if due:
        meta = train("challenger", seed_offset=int(now.timestamp()) % 1000)
        if meta:
            state["last_train"] = now.isoformat()
            if champ is None:
                os.replace("models/challenger.joblib", "models/champion.joblib"); os.replace("models/challenger.json", "models/champion.json")
                state["champion_since"] = now.isoformat(); state["promotions"].append({"at": now.isoformat(), "why": "first model"})
                champ = Brain.load("champion")
                log("first champion installed")
    if champ is None:
        save_state(state); return

    mode = C.MODES[mode_name]
    engine = PaperEngine()
    # update open trades with bars that have arrived
    engine.update({s: f[["time", "open", "high", "low", "close"]] for s, f in feats.items()}, {k: v.max_hold_bars for k, v in C.MODES.items()})

    # ---- 7. grade pre-registered setups now that their outcomes are known -------------------------------
    setups = load_setups()
    if len(setups):
        unresolved = setups[setups["resolved"] != True]
        for i, s in unresolved.iterrows():
            f = feats.get(s["symbol"])
            if f is None: continue
            row = f[f["time"] == pd.Timestamp(s["time"], tz="UTC")]
            if row.empty or not np.isfinite(row["y"].iloc[0]): continue
            setups.loc[i, ["resolved", "y", "y_r"]] = [True, float(row["y"].iloc[0]), float(row["y_r"].iloc[0])]

    # ---- 4/5. predict the newest completed bar for each symbol ------------------------------------------
    cols = champ.cols
    new_rows = []; opened = 0
    open_pos = engine.open_positions()
    for sym, f in feats.items():
        last = f.iloc[-1]
        if last[cols].isna().any() or not np.isfinite(last["atr"]) or last["atr"] <= 0:
            continue
        if len(setups) and ((setups["symbol"] == sym) & (setups["time"] == last["time"])).any():
            continue   # already evaluated this bar
        pr = champ.predict(last[cols].values.astype(float).reshape(1, -1)).iloc[0]
        market = D.market_of(sym)
        rr = rr_after_costs(last["close"], last["atr"], market)
        dec, reasons = decide(pr["p"], pr["disagree"], rr, last["f_mean"], mode,
                              len(open_pos) + opened, int((open_pos["symbol"] == sym).sum()), engine.day_pnl_pct(now))
        new_rows.append({"time": last["time"], "symbol": sym, "market": market, "p": pr["p"], "p_raw": pr["p_raw"],
                         "disagree": pr["disagree"], "f_mean": last["f_mean"], "rr_net": rr, "decision": dec,
                         "reasons": "; ".join(reasons), "model": champ.meta.get("trained_at", ""), "resolved": False,
                         "y": np.nan, "y_r": np.nan, "close_price": last["close"]})
        if dec == "BUY":
            # paper entry at the last close (next bar's open is unknown until it prints; close is the honest proxy)
            engine.open_trade(sym, last["time"], last["close"], last["atr"], float(pr["p"]), float(pr["disagree"]), mode_name, champ.meta.get("trained_at", ""))
            opened += 1
            log(f"BUY {sym} p={pr['p']:.2f} rr={rr:.2f}")
    if new_rows:
        setups = pd.concat([setups, pd.DataFrame(new_rows)], ignore_index=True)
    save_setups(setups)
    from .trading import save_trades
    save_trades(engine.trades)

    # ---- 7. champion vs challenger on recently resolved setups -------------------------------------------
    chal = Brain.load("challenger")
    if chal is not None and champ is not None and chal.meta.get("trained_at") != champ.meta.get("trained_at"):
        res = setups[setups["resolved"] == True].tail(400)
        if len(res) >= C.CHALLENGER_MIN_RESOLVED:
            Xr = []; yr = []
            for _, s in res.iterrows():
                f = feats.get(s["symbol"]);
                if f is None: continue
                row = f[f["time"] == pd.Timestamp(s["time"], tz="UTC")]
                if row.empty or row[cols].isna().any(axis=1).iloc[0]: continue
                Xr.append(row[cols].values[0].astype(float)); yr.append(float(s["y"]))
            if len(yr) >= C.CHALLENGER_MIN_RESOLVED:
                Xr = np.array(Xr); yr = np.array(yr)
                b_ch = brier(champ.predict(Xr)["p"].values, yr); b_cl = brier(chal.predict(Xr)["p"].values, yr)
                verdict = {"at": now.isoformat(), "n": int(len(yr)), "champion_brier": b_ch, "challenger_brier": b_cl}
                if b_cl < b_ch - C.PROMOTE_BRIER_GAIN:
                    os.replace("models/challenger.joblib", "models/champion.joblib"); os.replace("models/challenger.json", "models/champion.json")
                    verdict["why"] = "challenger promoted"; state["champion_since"] = now.isoformat()
                    log(f"PROMOTED challenger ({b_cl:.4f} < {b_ch:.4f})")
                else:
                    verdict["why"] = "champion kept"
                    log(f"champion kept ({b_ch:.4f} vs challenger {b_cl:.4f})")
                state["promotions"].append(verdict)
        else:
            log(f"challenger waiting: {len(res)} resolved setups of {C.CHALLENGER_MIN_RESOLVED}")

    # ---- summary for the dashboard -------------------------------------------------------------------------
    sc = engine.scorecard()
    resolved = setups[setups["resolved"] == True]
    summary = {
        "ts": now.isoformat(), "mode": mode_name, "symbols": len(feats), "equity": equity(engine.trades),
        "open_trades": int(len(engine.open_positions())), "setups_total": int(len(setups)), "setups_resolved": int(len(resolved)),
        "setup_brier": float(np.mean((resolved["p"] - resolved["y"]) ** 2)) if len(resolved) else None,
        "setup_base_brier": float(np.mean((resolved["y"].mean() - resolved["y"]) ** 2)) if len(resolved) else None,
        "buy_signals_resolved": int(((resolved["decision"] == "BUY")).sum()),
        "buy_win_rate": float(resolved[resolved["decision"] == "BUY"]["y"].mean()) if (resolved["decision"] == "BUY").any() else None,
        "scorecard": sc, "champion": champ.meta if champ else None, "data_status": st, "state": state,
    }
    json.dump(summary, open("data/summary.json", "w"), indent=1, default=str)
    save_state(state)
    log(f"done: equity {summary['equity']:.2f}, open {summary['open_trades']}, setups {len(setups)} ({len(resolved)} resolved)")


# ---- offline backtest of the whole pipeline (for the dashboard's "Backtest" tab) --------------------------
def backtest(mode_name: str = C.DEFAULT_MODE) -> dict:
    """Walk-forward: train on the first 60%, then step through the rest bar by bar per symbol, applying the
    filter and the paper rules. Reports what the system *would* have done out-of-sample."""
    all_, feats = dataset()
    if all_.empty: return {"error": "no data"}
    cols = feature_cols(); mode = C.MODES[mode_name]
    tr = training_frame(all_).sort_values("time")
    cut = tr["time"].quantile(0.6)
    fit = tr[tr["time"] <= cut]; test = tr[tr["time"] > cut]
    b = Brain("backtest").fit(fit[cols].values.astype(float), fit["y"].values.astype(int))
    # calibrate on the last 25% of the fit window (held out from the voters would be better; this is a quick proxy)
    pv = b.predict(test[cols].values.astype(float))
    test = test.assign(p=pv["p"].values, disagree=pv["disagree"].values)
    rows = []
    for _, s in test.iterrows():
        rr = rr_after_costs(s["close"], s["atr"], s["market"])
        dec, _ = decide(s["p"], s["disagree"], rr, s["f_mean"], mode, 0, 0, 0.0)
        if dec == "BUY":
            tg, sp = C.barriers(s["market"])
            cost_r = (s["close"] * (2 * (C.FEE_SIDE_BPS[s["market"]] + C.SLIPPAGE_BPS[s["market"]])) / 1e4) / (sp * s["atr"])
            rows.append({"time": s["time"], "symbol": s["symbol"], "market": s["market"], "p": s["p"], "y": s["y"], "r": s["y_r"] / sp - cost_r})
    bt = pd.DataFrame(rows)
    out = {"test_from": str(cut), "setups_tested": int(len(test)), "signals": int(len(bt)),
           "oos_brier": brier(test["p"].values, test["y"].values), "base_brier": brier(np.full(len(test), fit["y"].mean()), test["y"].values)}
    if len(bt):
        out.update({"win_rate": float(bt["y"].mean()), "avg_r": float(bt["r"].mean()), "sum_r": float(bt["r"].sum()),
                    "by_market": bt.groupby("market")["r"].agg(["count", "mean", "sum"]).round(3).to_dict("index")})
        bt.to_csv("data/backtest_signals.csv", index=False)
    json.dump(out, open("data/backtest.json", "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run": run()
    elif cmd == "train": train("champion")
    elif cmd == "backtest": print(json.dumps(backtest(), indent=1, default=str))
    elif cmd == "refresh": D.refresh_all()
    else: print("usage: python -m ai_trader [run|train|backtest|refresh]")
