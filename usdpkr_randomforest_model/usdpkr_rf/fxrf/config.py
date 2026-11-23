"""All constants and settings in ONE place. Change values here only."""

SEED = 42
WARMUP = 12
TRAIN_RATIO = 0.80
MIN_ROWS = 48                 # minimum monthly rows required in the data file
LONG_HORIZON_WARNING = 36     # months after which a warning is printed
OUTPUT_FOLDER_NAME = "FYP_Risk_Output"
BACKTEST_HORIZONS = (1, 3, 6, 12)

# Forecast model: Random Forest on the h-month CHANGE (not the level), so the trees never
# have to extrapolate beyond the training range. Shallow trees + min_samples_leaf and
# max_features < 1 keep the forest from over-fitting on ~100 monthly rows.
FC_PARAMS = dict(n_estimators=300, max_depth=6, min_samples_leaf=3, max_features=0.5,
                 bootstrap=True, n_jobs=-1)

# Risk models (volatility / mean / classifier share these; quantile forest has its own leaf size)
RISK_PARAMS = dict(n_estimators=300, max_depth=5, min_samples_leaf=3, max_features=0.5)
QUANTILE_LEAF = 5

VARS = ["reserves", "reer", "brent", "sbp", "fed", "pk_cpi", "us_cpi", "infl_diff", "usdpkr"]
FACTORS = ["reserves", "reer", "brent", "sbp", "fed", "pk_cpi", "us_cpi"]
LOG_VARS = {"reserves", "reer", "brent", "usdpkr"}
NICE = {"reserves": "Foreign reserves", "reer": "REER", "brent": "Brent Crude Oil Price",
        "sbp": "SBP Policy rate (%)", "fed": "Federal Funds Rate (%)",
        "pk_cpi": "Pakistan CPI Inflation YoY (%)", "us_cpi": "US CPI YoY Inflation (%)",
        "infl_diff": "Inflation rate differential", "usdpkr": "USD/PKR (interbank close)"}

# Column names are matched by keywords (extra columns in your file are ignored)
KEYWORDS = [("infl_diff", ["differential"]), ("usdpkr", ["interbank", "usd/pkr"]),
            ("reserves", ["reserve"]), ("reer", ["reer", "real effective"]),
            ("brent", ["brent"]), ("sbp", ["sbp", "policy rate"]),
            ("fed", ["federal fund", "fed fund"]),
            ("pk_cpi", ["pakistan cpi", "pak cpi", "pakistan inflation"]),
            ("us_cpi", ["us cpi", "u.s. cpi"])]
MOM_COLS = ["usdpkr_d1", "usdpkr_d3", "usdpkr_d6", "usdpkr_d12", "usdpkr_dev12"]
