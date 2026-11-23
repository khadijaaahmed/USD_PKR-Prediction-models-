"""Quantile Regression Forest (Meinshausen 2006) - the Random Forest 'worst case' (VaR) model."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .config import SEED, RISK_PARAMS, QUANTILE_LEAF


class QuantileForest:
    """A Random Forest whose leaves are used to weight the training targets; the 95 % quantile
    is the weighted 95th percentile of them."""

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        params = dict(RISK_PARAMS, min_samples_leaf=QUANTILE_LEAF)
        self.rf = RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.y = np.asarray(y, dtype=float)
        self.leaves = self.rf.apply(X)                      # (n_samples, n_trees)
        return self

    def predict(self, X):
        lx = self.rf.apply(X)
        out = []
        for row in lx:
            w = np.zeros(len(self.y))
            for t, leaf in enumerate(row):
                same = self.leaves[:, t] == leaf
                w[same] += 1.0 / same.sum()
            w /= w.sum()
            order = np.argsort(self.y)
            cum = np.cumsum(w[order])
            out.append(self.y[order][min(np.searchsorted(cum, self.alpha), len(order) - 1)])
        return np.array(out)
