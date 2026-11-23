"""STEP 4 (risk) - SVR -> expected volatility, SVC -> probability of high volatility."""
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ALL_KEYS, VOL_WINDOW, HIGH_VOL_QUANTILE, SEED
from .utils import diff_nan, tune_svr


def vol_features(d, r, t):
    return np.array([r[t], abs(r[t]),
                     np.std(r[t - 2:t + 1], ddof=1),
                     np.std(r[t - 5:t + 1], ddof=1)] + [d[k][t] for k in ALL_KEYS])


class VolatilityModel:
    """SVR -> expected volatility ;  SVC -> probability of high volatility."""

    def fit(self, z, r, n_end):
        d = {k: diff_nan(z[k][:n_end]) for k in ALL_KEYS}
        rows = range(6, n_end - VOL_WINDOW)
        X = np.array([vol_features(d, r, t) for t in rows])
        fwd = np.array([np.std(r[t + 1:t + 1 + VOL_WINDOW], ddof=1) for t in rows])
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        ylog = np.log(fwd + 1e-5)
        self.y_mean, self.y_std = float(ylog.mean()), float(ylog.std()) + 1e-12
        self.reg = tune_svr(Xs, (ylog - self.y_mean) / self.y_std)

        self.threshold = float(np.quantile(fwd, HIGH_VOL_QUANTILE))
        lab = (fwd > self.threshold).astype(int)
        self.clf = None
        if lab.min() != lab.max():
            params = {"C": 1.0, "gamma": "scale"}
            try:
                gs = GridSearchCV(
                    SVC(kernel="rbf", class_weight="balanced"),
                    {"C": [0.5, 5, 50], "gamma": ["scale", 0.02, 0.1]},
                    cv=TimeSeriesSplit(4), scoring="balanced_accuracy",
                    error_score=np.nan, refit=False)
                gs.fit(Xs, lab)
                params = gs.best_params_
            except Exception:
                pass
            self.clf = SVC(kernel="rbf", probability=True, class_weight="balanced",
                           random_state=SEED, **params).fit(Xs, lab)
        return self

    def predict(self, d, r, t):
        x = self.scaler.transform(vol_features(d, r, t)[None, :])
        sigma = float(np.exp(self.reg.predict(x)[0] * self.y_std + self.y_mean) - 1e-5)
        prob = float(self.clf.predict_proba(x)[0, 1]) if self.clf is not None else np.nan
        return max(sigma, 1e-4), prob
