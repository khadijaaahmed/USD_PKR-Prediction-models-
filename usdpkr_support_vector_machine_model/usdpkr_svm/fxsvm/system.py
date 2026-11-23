"""WHOLE SVM SYSTEM - joins factor models, relation model and volatility model."""
import numpy as np

from .config import ALL_KEYS, BASE_KEYS, BOUNDS, FACTOR_DAMPING
from .utils import diff_nan
from .factor_model import FactorSVR
from .relation_model import InterbankRelation
from .volatility_model import VolatilityModel


class SVMSystem:
    def __init__(self, z, logp, n_end):
        """Train every model on the first n_end months only."""
        self.n_end = n_end
        zt = {k: v[:n_end] for k, v in z.items()}
        r = diff_nan(logp)[:n_end]
        self.factor_models = {k: FactorSVR(k).fit(zt[k]) for k in BASE_KEYS}
        self.relation = InterbankRelation().fit(zt, r)
        self.vol = VolatilityModel().fit(z, diff_nan(logp), n_end)

    def forecast(self, z_hist, logp_hist, h):
        """Recursive h-month forecast: factors -> relation -> Interbank."""
        z = {k: list(np.asarray(v, float)) for k, v in z_hist.items()}
        r = list(diff_nan(logp_hist))
        lp = list(np.asarray(logp_hist, float))
        for step in range(h):
            damp = FACTOR_DAMPING ** step
            for k in BASE_KEYS:
                nxt = z[k][-1] + damp * self.factor_models[k].predict_next(np.array(z[k]))
                if BOUNDS[k] is not None:
                    nxt = float(np.clip(nxt, *BOUNDS[k]))
                z[k].append(nxt)
            z["infl_diff"].append(z["pk_cpi"][-1] - z["us_cpi"][-1])
            d = {k: diff_nan(z[k]) for k in ALL_KEYS}
            t = len(r)
            r_new = self.relation.predict(d, np.array(r + [np.nan]), t)
            r.append(r_new)
            lp.append(lp[-1] + r_new)
        return {"z": {k: np.array(v) for k, v in z.items()},
                "r": np.array(r), "logp": np.array(lp)}

    def risk_inputs(self, z_hist, logp_hist):
        r = diff_nan(logp_hist)
        d = {k: diff_nan(z_hist[k]) for k in ALL_KEYS}
        return self.vol.predict(d, r, len(r) - 1)
