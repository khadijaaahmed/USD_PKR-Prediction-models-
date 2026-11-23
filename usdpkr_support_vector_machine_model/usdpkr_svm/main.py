#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USD/PKR -- SVM (Support Vector Machine) Forecast + Volatility Risk Model
ENTRY POINT.

  STEP 1  Forecast every macro factor           (fxsvm/factor_model.py)
  STEP 2  Learn factor -> Interbank relation    (fxsvm/relation_model.py)
  STEP 3  Forecast Interbank USD/PKR            (fxsvm/system.py)
  STEP 4  Risk: volatility, probability, worst  (fxsvm/volatility_model.py)

Usage:
  python main.py
  python main.py --data risk_train_data.xlsx --out MyOutputFolder
  python main.py --payment Dec-2026          (single run, no prompt)

Requirements:  pip install -r requirements.txt
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

from fxsvm.config import ALL_KEYS, KINDS, DEFAULT_OUT_DIR, SEED
from fxsvm.data_loader import find_data_file, load_data, transform
from fxsvm.backtest import run_backtest
from fxsvm.system import SVMSystem
from fxsvm.payment_input import parse_payment
from fxsvm.reporting import save_backtest_chart
from fxsvm.pipeline import run_payment

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="USD/PKR SVM forecast + risk model")
    ap.add_argument("--data", help="path to risk_train_data.xlsx")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--payment", help="payment month or months ahead (skips the prompt)")
    args = ap.parse_args()

    np.random.seed(SEED)
    data_path = find_data_file(args.data)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    z = {k: transform(df[k], KINDS[k]) for k in ALL_KEYS}
    logp = np.log(df["fx"].to_numpy(float))
    last_date = df["date"].iloc[-1]
    print(f"Data loaded : {data_path}  ({len(df)} months, "
          f"{df['date'].iloc[0]:%b-%Y} to {last_date:%b-%Y})")

    print("\n(Training SVM models + back-test, please wait ...)")
    bt = run_backtest(z, logp, df["date"])
    bt_path = out_dir / "USDPKR_SVM_Backtest_Chart.png"
    save_backtest_chart(bt, bt_path)
    print("\nBack-test results (unseen test data, 1-month ahead):")
    for k, v in bt["metrics"].items():
        print(f"    {k:<46s} {v:.2f}" if v == v else f"    {k:<46s} n/a")
    print(f"\nBack-test chart saved : {bt_path}")

    system = SVMSystem(z, logp, len(df))          # final models: all data
    print("\nFactor -> Interbank relationships learned by the code:")
    print(system.relation.summary())
    print(f"\nHigh-volatility threshold (monthly) : {system.vol.threshold * 100:.2f} %")

    def one(text):
        h, pay = parse_payment(text, last_date)
        run_payment(system, bt, df, z, logp, h, pay, out_dir, bt_path)

    if args.payment:
        one(args.payment)
        return
    while True:
        try:
            text = input(
                "\nWhen is the payment? Type the number of MONTHS ahead (e.g. 6)\n"
                f"or a MONTH-YEAR (e.g. Dec-2026)   [data ends {last_date:%b-%Y} "
                "| q = quit]: ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in {"q", "quit", "exit"}:
            break
        try:
            one(text)
        except ValueError as e:
            print(f"  ! {e}")


if __name__ == "__main__":
    main()
