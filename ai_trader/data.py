"""Stage 1 - market data. Kraken public OHLC for crypto (no key, works from US servers),
Yahoo Finance hourly for stocks. Everything is cached as CSV under data/bars/ so the Streamlit app
never fetches on its own."""
from __future__ import annotations
import os, time, json
import pandas as pd
import numpy as np
import requests
from . import config as C

BARS_DIR = "data/bars"
COLS = ["time", "open", "high", "low", "close", "volume"]


def _path(sym: str) -> str:
    os.makedirs(BARS_DIR, exist_ok=True)
    return os.path.join(BARS_DIR, f"{sym}.csv")


def load_bars(sym: str) -> pd.DataFrame:
    p = _path(sym)
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(p, parse_dates=["time"])
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate hourly bars into larger bars (e.g. 4h) for markets whose hourly moves can't pay the fees."""
    if df.empty or minutes <= 60:
        return df
    g = df.set_index("time").resample(f"{minutes}min", label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()}).dropna(subset=["open"]).reset_index()
    return out[COLS]


def load_model_bars(sym: str) -> pd.DataFrame:
    """Bars at the timeframe the model trades for this symbol."""
    df = load_bars(sym)
    return resample(df, C.CRYPTO_BAR_MINUTES) if market_of(sym) == "crypto" else df


def save_bars(sym: str, df: pd.DataFrame) -> None:
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df.to_csv(_path(sym), index=False)


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return new
    both = pd.concat([old, new], ignore_index=True)
    both = both.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    # keep ~2 years of hourly bars
    return both.tail(24 * 365 * 2)


# ---- Kraken --------------------------------------------------------------------------------------------
def fetch_kraken(pair: str, since: int | None = None, interval: int = C.BAR_MINUTES) -> pd.DataFrame:
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}
    if since:
        params["since"] = since
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(f"kraken {pair}: {j['error']}")
    key = [k for k in j["result"] if k != "last"][0]
    rows = j["result"][key]
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    # the last row is the still-forming candle: drop it
    return df[COLS].iloc[:-1].reset_index(drop=True)


# ---- Yahoo ---------------------------------------------------------------------------------------------
def fetch_yahoo(sym: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["time"] = pd.to_datetime(df.index, utc=True)
    df = df[COLS].reset_index(drop=True)
    # drop the current (incomplete) bar if the market is open
    now = pd.Timestamp.utcnow()
    df = df[df["time"] < now.floor("h")]
    return df


# ---- refresh everything ----------------------------------------------------------------------------------
def refresh_all(log=print) -> dict:
    status = {}
    for pair, sym in C.CRYPTO.items():
        try:
            old = load_bars(sym)
            if len(old) < 2000:
                # Kraken only serves the last 720 candles; backfill two years from Yahoo, then Kraken keeps the edge fresh
                try:
                    old = merge(old, fetch_yahoo(sym, period="729d", interval=C.INTERVAL))
                except Exception as e:
                    log(f"{sym} yahoo backfill: {e}")
            since = int(old["time"].iloc[-1].timestamp()) if not old.empty else None
            new = fetch_kraken(pair, since)
            df = merge(old, new)
            save_bars(sym, df)
            status[sym] = {"ok": True, "bars": len(df), "new": len(new), "last": str(df["time"].iloc[-1]) if len(df) else None}
            time.sleep(1.1)   # Kraken public rate limit
        except Exception as e:
            status[sym] = {"ok": False, "error": str(e)[:200]}
            log(f"{sym}: {e}")
    for sym in C.STOCKS:
        try:
            old = load_bars(sym)
            new = fetch_yahoo(sym, period="60d" if not old.empty else "729d")
            df = merge(old, new)
            save_bars(sym, df)
            status[sym] = {"ok": True, "bars": len(df), "new": len(new), "last": str(df["time"].iloc[-1]) if len(df) else None}
        except Exception as e:
            status[sym] = {"ok": False, "error": str(e)[:200]}
            log(f"{sym}: {e}")
    os.makedirs("data", exist_ok=True)
    with open("data/data_status.json", "w") as f:
        json.dump({"ts": pd.Timestamp.utcnow().isoformat(), "symbols": status}, f, indent=1)
    return status


def market_of(sym: str) -> str:
    return "crypto" if sym in C.CRYPTO.values() else "stock"


def all_symbols() -> list[str]:
    return list(C.CRYPTO.values()) + C.STOCKS
