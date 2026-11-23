"""STEP 1 - one SVR (RBF) per macro factor."""
import numpy as np
from sklearn.preprocessing import StandardScaler

from .utils import tune_svr


class FactorSVR:
    """SVR (RBF) that predicts next month's change of one factor."""

    @staticmethod
    def features(z):
        c = np.diff(z)
        w = z[-12:]
        return np.array([c[-1], c[-2], c[-3], c[-6:].mean(),
                         (z[-1] - w.mean()) / (w.std() + 1e-9)])

    def __init__(self, key):
        self.key = key

    def fit(self, z):
        z = np.asarray(z, float)
        X = np.array([self.features(z[:i + 1]) for i in range(12, len(z) - 1)])
        y = np.array([z[i + 1] - z[i] for i in range(12, len(z) - 1)])
        self.c_std = float(np.std(np.diff(z))) + 1e-12
        self.y_scale = float(y.std()) + 1e-12
        self.scaler = StandardScaler().fit(X)
        self.model = tune_svr(self.scaler.transform(X), y / self.y_scale)
        return self

    def predict_next(self, z):
        x = self.scaler.transform(self.features(np.asarray(z, float))[None, :])
        step = float(self.model.predict(x)[0]) * self.y_scale
        return float(np.clip(step, -4 * self.c_std, 4 * self.c_std))
