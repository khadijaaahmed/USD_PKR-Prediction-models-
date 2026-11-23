"""PART A - CatBoost forecast.
Stage 1: forecast every factor.  Stage 2: factors -> interbank USD/PKR."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import (SEED, N_SEEDS, WARMUP, FC_PARAMS, VARS, FACTORS, LOG_VARS, MOM_COLS)


def change(df, v, k):
    return np.log(df[v]).diff(k) if v in LOG_VARS else df[v].diff(k)


def make_features(df):
    f = pd.DataFrame(index=df.index)
    for v in VARS:
        f[v] = df[v]
        d1 = change(df, v, 1)
        f[v + "_d1"] = d1
        f[v + "_d1_lag1"] = d1.shift(1)
        for k in (3, 6, 12):
            f[f"{v}_d{k}"] = change(df, v, k)
        f[v + "_mean3"] = d1.rolling(3, min_periods=1).mean()
        f[v + "_mean6"] = d1.rolling(6, min_periods=1).mean()
    f["policy_spread"] = df["sbp"] - df["fed"]
    f["real_policy_rate"] = df["sbp"] - df["pk_cpi"]
    f["usdpkr_dev12"] = np.log(df["usdpkr"]) - np.log(df["usdpkr"].rolling(12, min_periods=3).mean())
    f["month"] = df.index.month
    return f.fillna(0.0)


def future_change(df, v, h):
    if v in LOG_VARS:
        return np.log(df[v]).shift(-h) - np.log(df[v])
    return df[v].shift(-h) - df[v]


def stage2_features(cur, fut):
    X = pd.DataFrame(index=cur.index)
    for v in FACTORS:
        X[v + "_chg_h"] = (np.log(fut[v]) - np.log(cur[v])) if v in LOG_VARS else (fut[v] - cur[v])
    X["infl_diff_fut"] = fut["pk_cpi"] - fut["us_cpi"]
    X["infl_diff_now"] = cur["pk_cpi"] - cur["us_cpi"]
    X["policy_spread_fut"] = fut["sbp"] - fut["fed"]
    X["real_policy_fut"] = fut["sbp"] - fut["pk_cpi"]
    return X


def make_model(seed=42):
    return CatBoostRegressor(**FC_PARAMS, random_seed=seed, verbose=False, allow_writing_files=False)


def fit_ens(X, y, n_seeds=N_SEEDS):
    return [make_model(SEED + k).fit(X, y) for k in range(n_seeds)]


def pred_ens(ens, X):
    return float(np.mean([m.predict(X)[0] for m in ens]))


def train_forecast(df, h, n_seeds=N_SEEDS, last_target_row=None):
    """last_target_row keeps the back-test test-months out of training."""
    n = len(df)
    feats = make_features(df)
    stop = (n - h) if last_target_row is None else min(n - h, last_target_row - h)
    rows = list(range(WARMUP, stop))
    if len(rows) < 20:
        raise ValueError(f"A {h}-month horizon is too long for this data (only {len(rows)} training "
                         f"rows). Choose a nearer month or add more data.")
    idx = df.index[rows]
    X = feats.loc[idx]
    stage1 = {v: fit_ens(X, future_change(df, v, h).loc[idx], n_seeds) for v in FACTORS}
    fut, cur = df[FACTORS].shift(-h).loc[idx], df[FACTORS].loc[idx]
    X2 = pd.concat([stage2_features(cur, fut), feats.loc[idx, MOM_COLS]], axis=1)
    stage2 = fit_ens(X2, future_change(df, "usdpkr", h).loc[idx], n_seeds)
    return {"stage1": stage1, "stage2": stage2}


def predict_forecast(models, df):
    feats = make_features(df)
    X, last = feats.iloc[[-1]], df.iloc[-1]
    out = {}
    for v in FACTORS:                                           # STAGE 1: factors
        step = pred_ens(models["stage1"][v], X)
        out[v] = float(last[v] * np.exp(step) if v in LOG_VARS else last[v] + step)
        lo, hi = df[v].min(), df[v].max()
        pad = 0.05 * (hi - lo)
        out[v] = float(np.clip(out[v], lo - pad, hi + pad))
    out["sbp"], out["fed"] = max(out["sbp"], 0.0), max(out["fed"], 0.0)
    out["infl_diff"] = out["pk_cpi"] - out["us_cpi"]
    fut = pd.DataFrame([{v: out[v] for v in FACTORS}], index=X.index)
    cur = pd.DataFrame([{v: last[v] for v in FACTORS}], index=X.index)
    X2 = pd.concat([stage2_features(cur, fut), feats[MOM_COLS].iloc[[-1]]], axis=1)   # STAGE 2
    out["usdpkr"] = float(last["usdpkr"] * np.exp(pred_ens(models["stage2"], X2)))
    return out
