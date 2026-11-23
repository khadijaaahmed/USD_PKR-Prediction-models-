"""ONE PAYMENT DATE - joins forecast + risk + reporting."""
import numpy as np
import pandas as pd

from .config import ALL_KEYS, LABELS, KINDS
from .data_loader import back_transform
from .reporting import save_excel, save_forecast_chart


def run_payment(system, bt, df, z, logp, h, pay_date, out_dir, bt_path=None):
    n = len(df)
    last_date = df["date"].iloc[-1]
    label = f"{pay_date:%b-%Y}"
    out = system.forecast(z, logp, h)

    fx_path = np.exp(out["logp"][n:])
    expected = float(fx_path[-1])
    last_spot = float(np.exp(logp[-1]))
    mu_h = float(out["r"][n:].sum())

    sigma, prob = system.risk_inputs(z, logp)
    sigma_h = sigma * np.sqrt(h)
    worst = float(last_spot * np.exp(mu_h + bt["z95"] * sigma_h))

    rows = [("PAYMENT", None),
            ("Payment month", label),
            ("Months ahead", h),
            ("Last data month", f"{last_date:%b-%Y}"),
            ("SVM FORECAST", None)]
    for k in ALL_KEYS:
        val = float(back_transform(out["z"][k][-1], KINDS[k]))
        rows.append((f"{LABELS[k]} - forecast", round(val, 3)))
    rows += [("USD/PKR (interbank close) - forecast", round(expected, 3)),
             ("SVM RISK", None),
             ("Last USD/PKR (spot)", round(last_spot, 2)),
             ("Expected USD/PKR", round(expected, 2)),
             ("Volatility % (monthly)", round(sigma * 100, 2)),
             ("Worst-case USD/PKR (95 %, SVM residual quantile)", round(worst, 2)),
             ("Probability of high volatility %",
              round(prob * 100, 1) if prob == prob else "n/a")]

    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / f"USDPKR_SVM_Forecast_{label.replace('-', '')}.xlsx"
    png = out_dir / f"USDPKR_SVM_Forecast_Chart_{label.replace('-', '')}.png"
    save_excel(rows, xlsx, bt_path)
    fut_dates = [last_date + pd.DateOffset(months=i) for i in range(1, h + 1)]
    save_forecast_chart(df["date"], np.exp(logp), fut_dates, fx_path, worst, label, png)

    print("\n" + "=" * 66)
    for item, val in rows:
        print(f"\n{item}" if val is None else f"  {item:<52s} {val}")
    print("\n" + "=" * 66)
    print(f"Excel saved : {xlsx}")
    print(f"Chart saved : {png}")
