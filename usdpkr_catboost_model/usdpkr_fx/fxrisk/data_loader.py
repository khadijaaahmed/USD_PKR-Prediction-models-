"""PART 1 - File selection, data loading and payment-date input."""
import os
import re
import pandas as pd

from .config import VARS, LOG_VARS, KEYWORDS, NICE, MIN_ROWS


def ask_file(arg_path=None):
    if arg_path and os.path.exists(arg_path):
        return arg_path
    path = ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select your TRAIN DATA file (xlsx / csv)",
            filetypes=[("Excel / CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")])
        root.destroy()
    except Exception:
        path = ""
    if not path:
        path = input("Type / paste the full path of your train data file: ").strip().strip('"')
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path


def load_data(path):
    raw = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Date" not in raw.columns:
        raise ValueError("The file must have a column named 'Date'.")
    mapping = {}
    for key, words in KEYWORDS:
        for col in raw.columns:
            if col == "Date" or col in mapping.values():
                continue
            if any(w in col.lower() for w in words):
                mapping[key] = col
                break
    missing = [v for v in VARS if v not in mapping and v != "infl_diff"]
    if missing:
        raise ValueError("These columns were not found in the file: " + ", ".join(NICE[m] for m in missing))
    df = pd.DataFrame({"Date": pd.to_datetime(raw["Date"], errors="coerce")})
    for key in VARS:
        if key in mapping:
            df[key] = pd.to_numeric(raw[mapping[key]], errors="coerce")
    if "infl_diff" not in df.columns:
        df["infl_diff"] = df["pk_cpi"] - df["us_cpi"]
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date").set_index("Date")
    df.index = df.index.to_period("M").to_timestamp()
    df = df[~df.index.duplicated(keep="last")]
    df = df[VARS].interpolate().ffill().bfill()
    if (df[list(LOG_VARS)] <= 0).any().any():
        raise ValueError("Reserves / REER / Brent / USD-PKR must be positive numbers.")
    if len(df) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} monthly rows are required.")
    return df


def parse_target(text, last_date):
    """A number (= months ahead) or a month-year such as Dec-2026, 12/2026, 'Dec 2026'."""
    text = str(text).strip()
    if text.isdigit():
        n = int(text)
        return (last_date + pd.DateOffset(months=n)).to_period("M").to_timestamp() if n > 0 else None
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{4})", text)
    if m and 1 <= int(m.group(1)) <= 12:
        return pd.Timestamp(int(m.group(2)), int(m.group(1)), 1)
    try:
        return pd.to_datetime(text).to_period("M").to_timestamp()
    except Exception:
        return None


def ask_payment(last_date):
    while True:
        t = input(f"\nWhen is the payment? Type the number of MONTHS ahead (e.g. 6)\n"
                  f"or a MONTH-YEAR (e.g. Dec-2026)   [data ends {last_date:%b-%Y} | q = quit]: ")
        if t.strip().lower() in ("q", "quit", "exit"):
            return None
        target = parse_target(t, last_date)
        if target is not None and target > last_date:
            return target
        print(f"  Please enter a valid month after {last_date:%b-%Y}.")
