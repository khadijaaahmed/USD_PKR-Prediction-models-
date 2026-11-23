"""DATA - choose the Excel file, load it, transform columns."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FACTORS, ALL_KEYS


def find_data_file(user_path):
    """Open a file-selection window so the user can pick the training Excel file."""
    if user_path:                                   # optional: --data <path>
        p = Path(user_path.strip().strip('"'))
        if p.exists():
            return p
        raise FileNotFoundError(f"Data file not found: {p}")

    chosen = ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)           # bring the window to the front
        chosen = filedialog.askopenfilename(
            title="Select the training data file (risk_train_data.xlsx)",
            initialdir=str(Path.home() / "Downloads"),
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        root.update()
        root.destroy()
        del root
    except Exception:                               # no GUI available -> text fallback
        chosen = input("Paste the full path of the training Excel file: ").strip().strip('"')

    if not chosen:
        raise SystemExit("No file selected. Program stopped.")
    return Path(chosen)


def find_col(df, keywords):
    for c in df.columns:
        name = str(c).lower()
        if all(k in name for k in keywords):
            return c
    raise KeyError(f"Column with keywords {keywords} not found in the data file.")


def load_data(path):
    raw = pd.read_excel(path)
    date_col = find_col(raw, ["date"])
    fx_col = find_col(raw, ["interbank"])
    cols = {k: find_col(raw, kw) for k, _, kw, _, _ in FACTORS}
    df = pd.DataFrame({"date": pd.to_datetime(raw[date_col])})
    for k in ALL_KEYS:
        df[k] = pd.to_numeric(raw[cols[k]], errors="coerce")
    df["fx"] = pd.to_numeric(raw[fx_col], errors="coerce")
    df = df.dropna(subset=["date", "fx"]).sort_values("date").reset_index(drop=True)
    df[ALL_KEYS] = df[ALL_KEYS].ffill().bfill()
    return df


def transform(series, kind):
    s = np.asarray(series, float)
    return np.log(s) if kind == "log" else s.copy()


def back_transform(z, kind):
    return np.exp(z) if kind == "log" else z
