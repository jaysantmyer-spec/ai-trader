"""Stage 4 - the AI brain. Five neural networks + logistic regression + gradient-boosted trees vote on the
probability that a long setup reaches its target before its stop. Trained walk-forward (expanding window,
time-ordered folds) so every reported accuracy is out-of-sample. A Platt/binned calibration maps the raw
ensemble average to an honest probability."""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from . import config as C
from .features import feature_cols

MODELS_DIR = "models"


def _voters(seed_offset: int = 0):
    vs = {}
    for i, (seed, hid) in enumerate(zip(C.NN_SEEDS, C.NN_HIDDEN)):
        vs[f"nn{i+1}"] = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hid, alpha=1e-3, max_iter=400,
                                                                     early_stopping=True, random_state=seed + seed_offset))
    vs["logit"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
    vs["trees"] = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                 random_state=7 + seed_offset)
    return vs


def _fit_calibration(p_raw: np.ndarray, y: np.ndarray) -> dict:
    """Platt scaling on the logit of the raw ensemble probability."""
    eps = 1e-4
    z = np.log(np.clip(p_raw, eps, 1 - eps) / (1 - np.clip(p_raw, eps, 1 - eps)))
    lr = LogisticRegression(C=10.0).fit(z.reshape(-1, 1), y)
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def _apply_cal(p_raw: np.ndarray, cal: dict) -> np.ndarray:
    eps = 1e-4
    z = np.log(np.clip(p_raw, eps, 1 - eps) / (1 - np.clip(p_raw, eps, 1 - eps)))
    return 1 / (1 + np.exp(-(cal["a"] * z + cal["b"])))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


class Brain:
    def __init__(self, name: str = "champion", seed_offset: int = 0):
        self.name = name
        self.seed_offset = seed_offset
        self.voters = _voters(seed_offset)
        self.cal = {"a": 1.0, "b": 0.0}
        self.cols = feature_cols()
        self.meta: dict = {}

    # ---- prediction ---------------------------------------------------------------------------------------
    def vote(self, X: np.ndarray) -> pd.DataFrame:
        out = {}
        for k, m in self.voters.items():
            out[k] = m.predict_proba(X)[:, 1]
        return pd.DataFrame(out)

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        v = self.vote(X)
        raw = v.mean(axis=1).values
        v["p_raw"] = raw
        v["p"] = _apply_cal(raw, self.cal)
        v["disagree"] = v[[c for c in v.columns if c not in ("p_raw", "p")]].std(axis=1).values
        return v

    # ---- training -----------------------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        for m in self.voters.values():
            m.fit(X, y)
        return self

    def walk_forward(self, X: np.ndarray, y: np.ndarray, times: np.ndarray, folds: int = C.WALK_FORWARD_FOLDS) -> dict:
        """Expanding-window, time-ordered. Returns OOS metrics and fits calibration on OOS predictions,
        then refits all voters on the full data."""
        n = len(y); order = np.argsort(times); X, y, times = X[order], y[order], times[order]
        edges = np.linspace(int(n * 0.4), n, folds + 1).astype(int)
        oos_p, oos_y, per_voter = [], [], {k: [] for k in self.voters}
        fold_rows = []
        for k in range(folds):
            tr, te = slice(0, edges[k]), slice(edges[k], edges[k + 1])
            if edges[k + 1] - edges[k] < 20:
                continue
            fresh = Brain(self.name, self.seed_offset); fresh.fit(X[tr], y[tr])
            v = fresh.vote(X[te])
            p = v.mean(axis=1).values
            oos_p.append(p); oos_y.append(y[te])
            for c in v.columns:
                per_voter[c].append(brier(v[c].values, y[te]))
            fold_rows.append({"fold": k + 1, "train": int(edges[k]), "test": int(edges[k + 1] - edges[k]),
                              "brier_raw": brier(p, y[te]), "acc": float(np.mean((p >= .5) == (y[te] == 1))),
                              "base_rate": float(y[te].mean())})
        oos_p = np.concatenate(oos_p); oos_y = np.concatenate(oos_y)
        self.cal = _fit_calibration(oos_p, oos_y)
        p_cal = _apply_cal(oos_p, self.cal)
        # reliability table
        bins = np.linspace(0, 1, C.CALIBRATION_BINS + 1)
        rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p_cal >= lo) & (p_cal < hi)
            if m.sum():
                rel.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "pred": float(p_cal[m].mean()), "actual": float(oos_y[m].mean())})
        self.meta = {
            "trained_at": pd.Timestamp.utcnow().isoformat(), "n": int(n), "base_rate": float(y.mean()),
            "folds": fold_rows, "oos_brier_raw": brier(oos_p, oos_y), "oos_brier_cal": brier(p_cal, oos_y),
            "oos_brier_base": brier(np.full_like(oos_y, oos_y.mean(), dtype=float), oos_y),
            "oos_acc": float(np.mean((p_cal >= .5) == (oos_y == 1))),
            "voter_brier": {k: float(np.mean(v)) for k, v in per_voter.items()},
            "calibration": self.cal, "reliability": rel, "features": self.cols,
            "oos_hi_conf": {"n": int((p_cal >= .6).sum()), "win": float(oos_y[p_cal >= .6].mean()) if (p_cal >= .6).any() else None},
        }
        self.fit(X, y)
        return self.meta

    # ---- persistence --------------------------------------------------------------------------------------
    def save(self, name: str | None = None):
        os.makedirs(MODELS_DIR, exist_ok=True)
        name = name or self.name
        joblib.dump({"voters": self.voters, "cal": self.cal, "cols": self.cols, "meta": self.meta, "seed_offset": self.seed_offset},
                    os.path.join(MODELS_DIR, f"{name}.joblib"))
        with open(os.path.join(MODELS_DIR, f"{name}.json"), "w") as f:
            json.dump(self.meta, f, indent=1)

    @classmethod
    def load(cls, name: str = "champion") -> "Brain | None":
        p = os.path.join(MODELS_DIR, f"{name}.joblib")
        if not os.path.exists(p):
            return None
        d = joblib.load(p)
        b = cls(name, d.get("seed_offset", 0)); b.voters = d["voters"]; b.cal = d["cal"]; b.cols = d["cols"]; b.meta = d["meta"]
        return b
