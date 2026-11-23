"""BACK-TEST on the unseen last 20 % of the data."""
import numpy as np

from .config import TEST_FRACTION, Z_NORMAL_95, WORST_CASE_LEVEL, VOL_WINDOW
from .utils import diff_nan
from .system import SVMSystem


def run_backtest(z, logp, dates):
    n = len(logp)
    split = n - int(round(TEST_FRACTION * n))
    system = SVMSystem(z, logp, split)           # trained on first 80 % only
    r_all = diff_nan(logp)

    pred_price, actual_price, pred_r, act_r, sig, zres = [], [], [], [], [], []
    for t in range(split, n):
        zh = {k: v[:t] for k, v in z.items()}
        out = system.forecast(zh, logp[:t], 1)
        p_r = out["r"][-1]
        s, _ = system.risk_inputs(zh, logp[:t])
        pred_r.append(p_r)
        act_r.append(r_all[t])
        pred_price.append(float(np.exp(logp[t - 1] + p_r)))
        actual_price.append(float(np.exp(logp[t])))
        sig.append(s)
        zres.append((r_all[t] - p_r) / s)

    pred_price, actual_price = np.array(pred_price), np.array(actual_price)
    err = pred_price - actual_price
    naive = np.exp(logp[split - 1:n - 1])
    metrics = {
        "Test months": len(err),
        "MAE (PKR)": float(np.mean(np.abs(err))),
        "RMSE (PKR)": float(np.sqrt(np.mean(err ** 2))),
        "MAPE (%)": float(np.mean(np.abs(err) / actual_price) * 100),
        "Naive (no-change) MAE (PKR)": float(np.mean(np.abs(naive - actual_price))),
        "Directional accuracy (%)": float(np.mean(np.sign(pred_r) == np.sign(act_r)) * 100),
    }
    z95 = max(Z_NORMAL_95, float(np.quantile(zres, WORST_CASE_LEVEL)))

    # classifier check on the unseen months (only where the forward vol is known)
    bal = np.nan
    if system.vol.clf is not None:
        from sklearn.metrics import balanced_accuracy_score
        yt, yp = [], []
        for t in range(split - 1, n - 1 - VOL_WINDOW):
            zh = {k: v[:t + 1] for k, v in z.items()}
            _, p = system.risk_inputs(zh, logp[:t + 1])
            fwd = np.std(r_all[t + 1:t + 1 + VOL_WINDOW], ddof=1)
            yt.append(int(fwd > system.vol.threshold))
            yp.append(int(p >= 0.5))
        if len(set(yt)) > 1:
            bal = float(balanced_accuracy_score(yt, yp))
    metrics["High-vol classifier balanced accuracy (%)"] = bal * 100 if bal == bal else np.nan

    return {"dates": dates[split:], "actual": actual_price, "pred": pred_price,
            "metrics": metrics, "z95": z95}
