"""RELATION STEP (internal, no output).
Measures how each factor's h-month change is related to the h-month INTERBANK USD/PKR change
and builds one 'relation score' that links all factors together."""
import numpy as np

from .config import FACTORS


def fit_relation(X2, y2):
    """Correlation of every factor's h-month change with the USD/PKR change (training rows only)."""
    cols = [v + "_chg_h" for v in FACTORS]
    rel = {}
    for col in cols:
        sd = float(X2[col].std())
        corr = float(np.corrcoef(X2[col], y2)[0, 1]) if sd > 0 else 0.0
        rel[col] = (corr if np.isfinite(corr) else 0.0, float(X2[col].mean()), sd if sd > 0 else 1.0)
    return rel


def add_relation(X2, rel):
    """Links the factors: relation_score = sum( correlation x standardised factor change )."""
    X2 = X2.copy()
    X2["relation_score"] = sum(c * (X2[col] - m) / sd for col, (c, m, sd) in rel.items())
    return X2
