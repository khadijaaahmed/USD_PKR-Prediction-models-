"""STEP 2 - relationship factors -> Interbank (learned inside the code)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from .config import ALL_KEYS, LABELS, MAX_LAG, CORR_MIN, MIN_FACTORS, FEATURE_CLIP
from .utils import diff_nan, tune_svr


class InterbankRelation:
    """
    1. Spearman correlation between every factor change (lags 0..MAX_LAG) and
       the Interbank monthly log-return.
    2. Keep the best lag of each factor if |rho| >= CORR_MIN.
    3. Features are weighted by |rho| (strong relations count more inside the
       RBF kernel) and an SVR maps them to the Interbank log-return.
    """

    def fit(self, z, r):
        n = len(r)
        d = {k: diff_nan(z[k][:n]) for k in ALL_KEYS}
        t0 = MAX_LAG + 1
        rows = []
        for k in ALL_KEYS:
            for lag in range(MAX_LAG + 1):
                x, y = d[k][t0 - lag:n - lag], r[t0:n]
                rho = spearmanr(x, y)[0]
                rows.append((k, lag, 0.0 if np.isnan(rho) else float(rho)))
        x, y = r[t0 - 1:n - 1], r[t0:n]                      # Interbank own momentum
        rho = spearmanr(x, y)[0]
        rows.append(("own", 1, 0.0 if np.isnan(rho) else float(rho)))
        self.corr = pd.DataFrame(rows, columns=["factor", "lag", "rho"])

        best = (self.corr.assign(a=self.corr.rho.abs())
                .sort_values("a", ascending=False)
                .drop_duplicates("factor"))
        keep = best[best.a >= CORR_MIN]
        if len(keep) < MIN_FACTORS:
            keep = best.head(MIN_FACTORS)
        self.sel = [(r_.factor, int(r_.lag), float(r_.rho)) for r_ in keep.itertuples()]
        w = np.array([abs(s[2]) for s in self.sel])
        self.w = w / w.mean()

        X = np.array([self.build_x(d, r, t) for t in range(t0, n)])
        self.scaler = StandardScaler().fit(X)
        Xs = np.clip(self.scaler.transform(X), -FEATURE_CLIP, FEATURE_CLIP) * self.w
        self.y_scale = float(r[t0:n].std()) + 1e-12
        self.model = tune_svr(Xs, r[t0:n] / self.y_scale)
        return self

    def build_x(self, d, r, t):
        return np.array([r[t - 1] if f == "own" else d[f][t - lag]
                         for f, lag, _ in self.sel])

    def predict(self, d, r, t):
        x = np.clip(self.scaler.transform(self.build_x(d, r, t)[None, :]),
                    -FEATURE_CLIP, FEATURE_CLIP) * self.w
        return float(self.model.predict(x)[0]) * self.y_scale

    def summary(self):
        lines = []
        for f, lag, rho in sorted(self.sel, key=lambda s: -abs(s[2])):
            name = "Interbank own momentum" if f == "own" else LABELS[f]
            lines.append(f"    {name:<36s} lag {lag} month(s)   Spearman = {rho:+.2f}")
        return "\n".join(lines)
