"""PAYMENT INPUT - understands 6 | Dec-2026 | December 2026 | 9-2024 | 09/2024 | 2024-09."""
import calendar
import re

import pandas as pd

from .config import MAX_HORIZON


def parse_payment(text, last_date):
    s = text.strip()
    if re.fullmatch(r"\d{1,3}", s):
        h = int(s)
    else:
        m = None
        mm = re.fullmatch(r"([A-Za-z]{3,9})[\s\-/,]*(\d{4})", s)
        if mm:
            names = {calendar.month_abbr[i].lower(): i for i in range(1, 13)}
            names.update({calendar.month_name[i].lower(): i for i in range(1, 13)})
            m = names.get(mm.group(1).lower())
            y = int(mm.group(2))
        else:
            mm = re.fullmatch(r"(\d{1,2})[\s\-/](\d{4})", s)
            if mm:
                m, y = int(mm.group(1)), int(mm.group(2))
            else:
                mm = re.fullmatch(r"(\d{4})[\s\-/](\d{1,2})", s)
                if mm:
                    y, m = int(mm.group(1)), int(mm.group(2))
        if m is None or not 1 <= m <= 12:
            raise ValueError("Could not understand the input. Use e.g. 6 or Dec-2026.")
        h = (y - last_date.year) * 12 + (m - last_date.month)
    if h < 1:
        raise ValueError(f"The payment month must be after the last data month "
                         f"({last_date:%b-%Y}).")
    if h > MAX_HORIZON:
        raise ValueError(f"Maximum horizon is {MAX_HORIZON} months.")
    return h, last_date + pd.DateOffset(months=h)
