"""ONE PAYMENT DATE - joins forecast + risk + reporting. Every payment gets its own files."""
import numpy as np

from .config import VARS, NICE, LONG_HORIZON_WARNING
from .forecast import train_forecast, predict_forecast
from .risk import train_risk, risk_summary
from .reporting import write_ab_excel, save_forecast_chart
from .utils import unique_path, print_table


class Context:
    """Holds the data, the trained-model caches and the output folder for one run."""
    def __init__(self, df, out_dir):
        self.df = df
        self.out_dir = out_dir
        self.fc = {}      # horizon -> forecast models
        self.rk = {}      # horizon -> risk models


def run_one(c, target):
    last = c.df.index[-1]
    h = (target.year - last.year) * 12 + (target.month - last.month)
    if h > LONG_HORIZON_WARNING:
        print(f"  Note: {h} months is far away; very long forecasts are much less reliable.")
    print("  (Training CatBoost, please wait ...)", flush=True)
    if h not in c.fc:
        c.fc[h] = train_forecast(c.df, h)
    if h not in c.rk:
        c.rk[h] = train_risk(c.df, h)
    fc = predict_forecast(c.fc[h], c.df)          # factors first, then USD/PKR from those factors

    spot = float(c.df["usdpkr"].iloc[-1])
    exp_rate = fc["usdpkr"]
    mean_h = float(np.log(exp_rate / spot))
    vol, p_high, worst_rate = risk_summary(c.rk[h], spot, mean_h)

    result = [
        ("PAYMENT", [("Payment month", f"{target:%b-%Y}"), ("Months ahead", h),
                     ("Last data month", f"{last:%b-%Y}")]),
        ("CATBOOST FORECAST", [(f"{NICE[v]} - forecast", round(fc[v], 3)) for v in VARS]),
        ("CATBOOST RISK", [("Last USD/PKR (spot)", round(spot, 2)),
                           ("Expected USD/PKR", round(exp_rate, 2)),
                           ("Volatility % (monthly)", round(vol, 2)),
                           ("Worst-case USD/PKR (95 %, CatBoost quantile)", round(worst_rate, 2)),
                           ("Probability of high volatility %", round(p_high * 100, 1))]),
    ]
    print("\n" + "=" * 70)
    print(f"  PAYMENT: {target:%b-%Y}  ({h} months ahead)")
    print("=" * 70)
    for title, items in result:
        print(f"  {title}")
        print_table(["Item", "Value"], [[k, v] for k, v in items])
        print()
    print("=" * 70)

    tag = f"{target:%b%Y}"
    xl = unique_path(c.out_dir, f"USDPKR_CatBoost_Forecast_{tag}", ".xlsx")
    try:
        write_ab_excel(xl, {"Sheet1": result})
        print(f"  Excel saved : {xl}")
    except Exception as e:
        print("  [Warning] Could not save the Excel file (is it open in Excel?):", e)

    png = unique_path(c.out_dir, f"USDPKR_Forecast_Chart_{tag}", ".png")
    try:
        save_forecast_chart(png, c.df["usdpkr"], target, exp_rate, worst_rate)
        print(f"  Chart saved : {png}")
    except Exception as e:
        print("  [Warning] Could not save the chart:", e)
