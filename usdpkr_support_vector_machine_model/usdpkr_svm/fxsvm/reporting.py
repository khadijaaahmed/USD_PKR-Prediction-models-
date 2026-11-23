"""OUTPUTS - Excel file and charts."""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # file-only backend: fixes the Tkinter "main thread is not in main loop" error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


def save_backtest_chart(bt, path):
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.plot(bt["dates"], bt["actual"], color="royalblue", lw=2, label="Actual USD/PKR (test)")
    ax.plot(bt["dates"], bt["pred"], color="darkorange", lw=2, ls="--",
            label="SVM (1-month ahead)")
    ax.set_title("USD/PKR back-test (unseen 20 % test data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/PKR")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_excel(rows, path, backtest_png=None):
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Item", "Value"])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78")
    for item, val in rows:
        ws.append([item, val])
        if val is None:
            ws.cell(ws.max_row, 1).font = Font(bold=True)
            for c in ws[ws.max_row]:
                c.fill = PatternFill("solid", fgColor="DDEBF7")
        else:
            ws.cell(ws.max_row, 2).alignment = Alignment(horizontal="right")
    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 18
    if backtest_png is not None and Path(backtest_png).exists():   # back-test graph (once)
        from openpyxl.drawing.image import Image as XLImage
        ws2 = wb.create_sheet("Back-test")
        ws2["A1"] = "USD/PKR back-test (unseen 20 % test data)"
        ws2["A1"].font = Font(bold=True)
        img = XLImage(str(backtest_png))
        img.width, img.height = 960, 435
        ws2.add_image(img, "A3")
    wb.save(path)


def save_forecast_chart(dates, price, fut_dates, path_prices, worst, label, path):
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.plot(dates, price, color="royalblue", lw=2, label="Historical USD/PKR")
    ax.plot([dates.iloc[-1]] + list(fut_dates), [price[-1]] + list(path_prices),
            color="darkorange", lw=1.8, ls="--", label="SVM forecast")
    pay = fut_dates[-1]
    ax.plot([pay, pay], [path_prices[-1], worst], color="gray", lw=1.5)
    ax.scatter([pay], [path_prices[-1]], color="darkorange", s=45, zorder=5)
    ax.scatter([pay], [worst], color="red", s=45, zorder=5, label="Worst case (95%)")
    ax.annotate(f"{path_prices[-1]:.1f}", (pay, path_prices[-1]),
                textcoords="offset points", xytext=(8, -16), fontsize=10)
    ax.set_title(f"USD/PKR forecast and worst case for {label} (SVM)")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/PKR")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_xlim(pd.Timestamp("2015-07-01"), pay + pd.DateOffset(months=6))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
