"""PART B - Random Forest risk: volatility, 95 % quantile (worst case), high-volatility probability."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .config import SEED, VARS, RISK_PARAMS
from .quantile_forest import QuantileForest


def reg():
    return RandomForestRegressor(**RISK_PARAMS, random_state=SEED, n_jobs=-1)


def clf():
    return RandomForestClassifier(**RISK_PARAMS, class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)


def risk_features(df, h):
    W = 6
    S = df["usdpkr"]
    ret = np.log(S).diff()
    F = pd.DataFrame(index=df.index)
    F["ret"] = ret
    for k in (1, 2, 3):
        F[f"abs_ret_lag{k}"] = ret.abs().shift(k)
    F["hist_vol"] = ret.rolling(W).std()
    F["hist_vol_short"] = ret.rolling(3).std()
    F["ewma_vol"] = np.sqrt((ret ** 2).ewm(alpha=0.06, adjust=False).mean())
    F["mean_ret_W"] = ret.rolling(W).mean()
    F["mean_abs_ret_W"] = ret.abs().rolling(W).mean()
    F["fx_level"] = S
    for c in VARS:
        if c != "usdpkr":
            F[c] = df[c]
            F[c + "_chg"] = df[c].diff()
    F["policy_spread"] = df["sbp"] - df["fed"]
    F["fwd_premium"] = np.log((1 + df["sbp"] / 100 * h / 12) / (1 + df["fed"] / 100 * h / 12))
    F = F.replace([np.inf, -np.inf], np.nan)
    return F, ret


def train_risk(df, h):
    Hw = max(h, 3)
    F, ret = risk_features(df, h)
    logS = np.log(df["usdpkr"])
    fwd_vol = ret.rolling(Hw).std().shift(-Hw) * 100                 # monthly volatility %
    fwd_ret = logS.shift(-h) - logS                                   # depreciation over h months
    lab = F.index[F["hist_vol"].notna() & fwd_vol.notna() & fwd_ret.notna()]
    n = len(lab)
    if n < 25:
        raise ValueError(f"Only {n} rows are left for the risk model (need at least 25). "
                         f"Choose a nearer month or add more data.")
    F = F.fillna(0.0)
    X, yv, yr = F.loc[lab], fwd_vol.loc[lab], fwd_ret.loc[lab]

    ycls = (yv > np.percentile(yv, 75)).astype(int)
    m = {"xnow": F.iloc[[-1]]}
    m["vol"] = reg().fit(X, yv)
    m["var"] = QuantileForest(0.95).fit(X, yr)
    m["mean"] = reg().fit(X, yr)
    m["cls"] = clf().fit(X, ycls) if ycls.nunique() > 1 else None
    return m


def risk_summary(rm, spot, mean_h):
    """Turn trained risk models into the numbers shown to the user."""
    xnow = rm["xnow"]
    vol = float(max(rm["vol"].predict(xnow)[0], 0))                                   # monthly volatility %
    var_spread = float(rm["var"].predict(xnow)[0] - rm["mean"].predict(xnow)[0])      # RF 95 % - mean
    p_high = 0.0 if rm["cls"] is None else float(rm["cls"].predict_proba(xnow)[0, 1])
    worst_rate = spot * float(np.exp(mean_h + max(var_spread, 0.0)))                  # RF worst case
    return vol, p_high, worst_rate
