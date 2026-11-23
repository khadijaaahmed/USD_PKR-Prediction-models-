"""Small helpers shared by the models."""
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR


def diff_nan(a):
    a = np.asarray(a, float)
    out = np.full(len(a), np.nan)
    out[1:] = np.diff(a)
    return out


def tune_svr(X, y, n_splits=4):
    """RBF-SVR with a time-series grid search."""
    grid = {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.3],
            "gamma": ["scale", 0.02, 0.1, 0.3]}
    cv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(y) // 12)))
    gs = GridSearchCV(SVR(kernel="rbf"), grid, cv=cv,
                      scoring="neg_mean_squared_error", n_jobs=1)
    gs.fit(X, y)
    return gs.best_estimator_
