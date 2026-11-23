"""PART C - Back-test: 80 % train / 20 % test (time order), Random Forest vs naive."""
import numpy as np

from .config import TRAIN_RATIO, FACTORS, NICE, BACKTEST_HORIZONS
from .forecast import train_forecast, predict_forecast
from .reporting import save_backtest_chart
from .utils import unique_path, print_table


def backtest(df, out_dir, horizons=BACKTEST_HORIZONS):
    n = len(df)
    split = int(TRAIN_RATIO * n)
    print(f"\nBACK-TEST: train = {df.index[0]:%b-%Y} to {df.index[split-1]:%b-%Y} ({split} months) | "
          f"test = {df.index[split]:%b-%Y} to {df.index[-1]:%b-%Y} ({n-split} months)")
    mape = lambda a, p: float(np.mean(np.abs((a - p) / a)) * 100)
    rows, frows, plot = [], [], None
    for h in horizons:
        try:
            models = train_forecast(df, h, last_target_row=split)
        except ValueError:
            continue
        act, pred, naive, dates = [], [], [], []
        fa = {v: [] for v in FACTORS}; fp = {v: [] for v in FACTORS}; fn = {v: [] for v in FACTORS}
        for t in range(split - h, n - h):
            hist = df.iloc[: t + 1]
            out = predict_forecast(models, hist)
            act.append(df["usdpkr"].iloc[t + h]); pred.append(out["usdpkr"])
            naive.append(hist["usdpkr"].iloc[-1]); dates.append(df.index[t + h])
            for v in FACTORS:
                fa[v].append(df[v].iloc[t + h]); fp[v].append(out[v]); fn[v].append(hist[v].iloc[-1])
        act, pred, naive = map(np.array, (act, pred, naive))
        rows.append([f"{h} month", f"{mape(act, pred):.2f} %", f"{mape(act, naive):.2f} %",
                     f"{np.sqrt(np.mean((act - pred) ** 2)):.2f}",
                     f"{np.mean(np.sign(pred - naive) == np.sign(act - naive)) * 100:.0f} %"])
        for kind, src in (("RandomForest", fp), ("Naive", fn)):
            frows.append([f"{h} month", kind] + [f"{np.mean(np.abs(np.array(fa[v]) - np.array(src[v]))):.3f}"
                                                  for v in FACTORS])
        if plot is None:
            plot = (h, dates, act, pred)
    print("\n  USD/PKR (predicted factors -> USD/PKR)")
    print_table(["Horizon", "RF MAPE", "Naive MAPE", "RMSE (Rs)", "Direction hit"], rows)
    print("\n  FACTORS - Mean Absolute Error on test part (lower is better)")
    print_table(["Horizon", "Model"] + [NICE[v] for v in FACTORS], frows)
    if plot:
        h, dates, act, pred = plot
        try:
            p = unique_path(out_dir, "USDPKR_RF_Backtest_Chart", ".png")
            save_backtest_chart(p, h, dates, act, pred)
            print(f"\n  Back-test chart saved : {p}")
        except Exception as e:
            print("  [Warning] Could not save the back-test chart:", e)
