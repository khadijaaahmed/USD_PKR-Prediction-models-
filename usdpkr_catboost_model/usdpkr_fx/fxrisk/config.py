"""All constants and settings in ONE place. Change values here only."""

SEED = 42
N_SEEDS = 3
WARMUP = 12
TRAIN_RATIO = 0.80
MIN_ROWS = 48                 # minimum monthly rows required in the data file
LONG_HORIZON_WARNING = 36     # months after which a warning is printed
OUTPUT_FOLDER_NAME = "FYP_Risk_Output"
BACKTEST_HORIZONS = (1, 3, 6, 12)

# Forecast model settings (grid-searched on the back-test; Huber loss is robust to the
# 2023 devaluation shock, strong L2 + shallow trees stop over-fitting on ~100 months)
FC_PARAMS = dict(loss_function="Huber:delta=0.02", iterations=300, learning_rate=0.05,
                 depth=3, l2_leaf_reg=30, random_strength=1.0, bagging_temperature=0.5)

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
