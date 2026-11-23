"""All constants and settings in ONE place. Change values here only."""
from pathlib import Path

DEFAULT_DATA_NAME = "risk_train_data.xlsx"
DEFAULT_OUT_DIR = Path.home() / "Downloads" / "FYP_Risk_Output"

TEST_FRACTION = 0.20        # last 20 % of months = unseen test data
MAX_LAG = 3                 # factor->Interbank lags examined (months)
CORR_MIN = 0.10             # minimum |Spearman| for a factor to be used
MIN_FACTORS = 3             # always keep at least this many factors
HIGH_VOL_QUANTILE = 0.75    # "high volatility" = top 25 % of historical vol
WORST_CASE_LEVEL = 0.95     # worst-case confidence
Z_NORMAL_95 = 1.645         # floor for the residual quantile
VOL_WINDOW = 3              # forward volatility window (months)
MAX_HORIZON = 60            # maximum months ahead
FACTOR_DAMPING = 0.75       # multi-step factor trends fade out (damped momentum)
FEATURE_CLIP = 3.0          # relation inputs limited to +/-3 std (no extrapolation)
SEED = 42

# key, label, keywords to find the column, transform, (lower, upper) bounds
FACTORS = [
    ("reserves", "Foreign reserves",                ["foreign", "reserve"],       "log", None),
    ("reer",     "REER",                            ["reer"],                     "log", None),
    ("brent",    "Brent Crude Oil Price",           ["brent"],                    "log", None),
    ("sbp",      "SBP Policy rate (%)",             ["sbp"],                      "lin", (0.0, 40.0)),
    ("ffr",      "Federal Funds Rate (%)",          ["federal", "funds"],         "lin", (0.0, 15.0)),
    ("pk_cpi",   "Pakistan CPI Inflation YoY (%)",  ["pakistan", "cpi"],          "lin", (-5.0, 60.0)),
    ("us_cpi",   "US CPI YoY Inflation (%)",        ["us cpi"],                   "lin", (-3.0, 15.0)),
    ("infl_diff", "Inflation rate differential",    ["differential"],             "lin", None),
]
BASE_KEYS = [f[0] for f in FACTORS if f[0] != "infl_diff"]
ALL_KEYS = [f[0] for f in FACTORS]
LABELS = {f[0]: f[1] for f in FACTORS}
KINDS = {f[0]: f[3] for f in FACTORS}
BOUNDS = {f[0]: f[4] for f in FACTORS}
