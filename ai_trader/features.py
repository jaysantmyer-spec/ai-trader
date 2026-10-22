"""Stage 2 - indicators and features. All features at bar t use only bars <= t (no look-ahead).
Stage 3 - the six statistical forecasters, each producing an expected forward return."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C


# ---- indicators ------------------------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(close: pd.Series):
    m = ema(close, 12) - ema(close, 26)
    sig = ema(m, 9)
    return m, sig, m - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


# ---- six statistical forecasters (each returns expected return over HORIZON_BARS, in ATR units) ---------
def forecasters(close: pd.Series, atr_s: pd.Series, h: int = C.HORIZON_BARS) -> pd.DataFrame:
    """Rolling, causal versions of: naive, linear regression, Holt trend, AR(1), EMA drift, mean reversion."""
    out = pd.DataFrame(index=close.index)
    lc = np.log(close)
    r = lc.diff()
    scale = (atr_s / close).replace(0, np.nan)          # ATR as a fraction of price -> normalise forecasts

    # 1. naive: last bar's move repeats
    out["f_naive"] = (r * h) / scale
    # 2. linear regression slope over 48 bars, projected h bars ahead
    n = 48
    x = np.arange(n); xm = x.mean(); xv = ((x - xm) ** 2).sum()
    slope = lc.rolling(n).apply(lambda y: ((x - xm) * (y - y.mean())).sum() / xv, raw=True)
    out["f_linreg"] = (slope * h) / scale
    # 3. Holt (double exponential smoothing) trend
    level = lc.ewm(alpha=0.3, adjust=False).mean()
    trend = level.diff().ewm(alpha=0.1, adjust=False).mean()
    out["f_holt"] = (trend * h) / scale
    # 4. AR(1) on returns: phi estimated over 96 bars
    phi = r.rolling(96).apply(lambda y: np.corrcoef(y[:-1], y[1:])[0, 1] if np.std(y[:-1]) > 0 and np.std(y[1:]) > 0 else 0, raw=True)
    out["f_ar1"] = (phi * r * (1 - phi ** h) / (1 - phi).replace(0, np.nan)).fillna(0) / scale
    # 5. EMA drift: fast EMA vs slow EMA gap, as a projected drift
    out["f_emadrift"] = ((ema(lc, 12) - ema(lc, 48)) * (h / 24)) / scale
    # 6. mean reversion: distance from 48-bar mean, expected to close half-way
    out["f_meanrev"] = (-(lc - lc.rolling(48).mean()) * 0.5) / scale
    return out.replace([np.inf, -np.inf], np.nan)


# ---- feature matrix ------------------------------------------------------------------------------------------
FEATURE_COLS: list[str] = []   # filled on first call


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: time, open, high, low, close, volume (sorted). Returns df with feature columns appended."""
    d = df.copy()
    c = d["close"]
    d["atr"] = atr(d, 14)
    d["atr_pct"] = d["atr"] / c
    d["ema20"] = ema(c, 20); d["ema50"] = ema(c, 50); d["ema200"] = ema(c, 200)
    d["x_ema20"] = (c - d["ema20"]) / d["atr"]
    d["x_ema50"] = (c - d["ema50"]) / d["atr"]
    d["x_ema200"] = (c - d["ema200"]) / d["atr"]
    d["x_ema_slope"] = d["ema20"].diff(5) / d["atr"]
    d["x_rsi"] = (rsi(c) - 50) / 25
    m, s, hst = macd(c)
    d["x_macd"] = m / d["atr"]; d["x_macd_hist"] = hst / d["atr"]
    mid, up, lo = bollinger(c)
    d["x_bb_pos"] = ((c - mid) / (up - lo).replace(0, np.nan)).clip(-2, 2)
    d["x_bb_width"] = ((up - lo) / mid).clip(0, 1)
    d["x_ret1"] = c.pct_change(1) / d["atr_pct"]
    d["x_ret6"] = c.pct_change(6) / d["atr_pct"]
    d["x_ret24"] = c.pct_change(24) / d["atr_pct"]
    d["x_ret72"] = c.pct_change(72) / d["atr_pct"]
    d["x_vol_ratio"] = (d["atr"] / d["atr"].rolling(100).mean()).clip(0, 4)
    d["x_volu_ratio"] = (d["volume"] / d["volume"].rolling(48).mean().replace(0, np.nan)).clip(0, 6).fillna(1)
    hi48 = d["high"].rolling(48).max(); lo48 = d["low"].rolling(48).min()
    d["x_res_dist"] = (hi48 - c) / d["atr"]            # distance to resistance
    d["x_sup_dist"] = (c - lo48) / d["atr"]            # distance to support
    d["x_range_pos"] = ((c - lo48) / (hi48 - lo48).replace(0, np.nan)).clip(0, 1)
    # simple candle patterns (last bar)
    body = (d["close"] - d["open"]).abs(); rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["x_body_ratio"] = (body / rng).fillna(0)
    d["x_upper_wick"] = ((d["high"] - d[["open", "close"]].max(axis=1)) / rng).fillna(0)
    d["x_lower_wick"] = ((d[["open", "close"]].min(axis=1) - d["low"]) / rng).fillna(0)
    d["x_hour_sin"] = np.sin(2 * np.pi * d["time"].dt.hour / 24)
    d["x_hour_cos"] = np.cos(2 * np.pi * d["time"].dt.hour / 24)
    d["x_dow"] = d["time"].dt.dayofweek / 6
    f = forecasters(c, d["atr"])
    for col in f.columns:
        d[col] = f[col].clip(-5, 5)
    d["f_mean"] = f.clip(-5, 5).mean(axis=1)
    d["f_agree"] = (np.sign(f.clip(-5, 5)).sum(axis=1)) / 6
    global FEATURE_COLS
    FEATURE_COLS = [x for x in d.columns if x.startswith("x_") or x.startswith("f_")]
    return d


def feature_cols() -> list[str]:
    if not FEATURE_COLS:
        # build once on dummy data to populate the list
        t = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
        p = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 300))
        build_features(pd.DataFrame({"time": t, "open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1000}))
    return list(FEATURE_COLS)


# ---- labels: triple barrier -----------------------------------------------------------------------------------
def label_setups(d: pd.DataFrame, h: int = C.HORIZON_BARS, tgt: float = C.TARGET_ATR, stp: float = C.STOP_ATR) -> pd.DataFrame:
    """For a LONG entered at the next bar's open after bar t, did price reach entry+tgt*ATR before entry-stp*ATR
    within h bars?  y=1 win, y=0 loss/timeout. Rows that cannot be resolved yet get y=NaN."""
    n = len(d)
    entry = d["open"].shift(-1).values
    a = d["atr"].values
    hi = d["high"].values; lo = d["low"].values; cl = d["close"].values
    y = np.full(n, np.nan); bars = np.full(n, np.nan); outcome = np.full(n, np.nan)   # outcome = realised R multiple
    for i in range(n - 2):
        e = entry[i]; at = a[i]
        if not np.isfinite(e) or not np.isfinite(at) or at <= 0:
            continue
        T = e + tgt * at; S = e - stp * at
        end = min(n, i + 1 + h)
        res = None
        for j in range(i + 1, end):
            if lo[j] <= S:          # conservative: stop checked first within a bar
                res = (0.0, j - i, -stp); break
            if hi[j] >= T:
                res = (1.0, j - i, tgt); break
        if res is None:
            if end < n or (i + 1 + h) <= n - 1:
                last = cl[end - 1]
                res = (1.0 if last > e else 0.0, h, (last - e) / at)
            else:
                continue   # still open
        y[i], bars[i], outcome[i] = res
    d = d.copy()
    d["y"] = y; d["y_bars"] = bars; d["y_r"] = outcome
    return d
