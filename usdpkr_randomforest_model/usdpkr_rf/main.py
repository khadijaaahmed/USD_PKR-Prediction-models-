# =====================================================================
#  AI-Enabled FX Hedging Decision System for USD/PKR
#  Using NDF and Forward Premium Analysis for Treasury Risk Management
#  ---------------------------------------------------------------------
#  RANDOM FOREST: FORECAST + RISK.   ENTRY POINT.  Run:
#     python main.py                            (file window opens)
#     python main.py data.xlsx                  (file given, asks the date)
#     python main.py data.xlsx 2 Dec-2026       (no questions)
#     python main.py data.xlsx --skip-backtest
# =====================================================================
import os
import sys
import argparse

from fxrf import bootstrap
bootstrap.setup()          # installs missing packages, sets matplotlib backend (must be first)

from fxrf.config import OUTPUT_FOLDER_NAME
from fxrf.data_loader import ask_file, load_data, parse_target, ask_payment
from fxrf.backtest import backtest
from fxrf.pipeline import Context, run_one


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("file", nargs="?", default=None)
    ap.add_argument("targets", nargs="*", help="months ahead (6) or month-year (Dec-2026)")
    ap.add_argument("--skip-backtest", action="store_true")
    args, _ = ap.parse_known_args()

    print("=" * 70)
    print("  USD/PKR  Random Forest Forecast + Risk")
    print("=" * 70)
    path = ask_file(args.file)
    df = load_data(path)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(path)), OUTPUT_FOLDER_NAME)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.getcwd()
    c = Context(df, out_dir)
    last = df.index[-1]
    print(f"\nData: {len(df)} months ({df.index[0]:%b-%Y} to {last:%b-%Y}). "
          f"Factor values are taken from the last row of the file.")
    print(f"Output folder: {out_dir}")

    if not args.skip_backtest and not args.targets:
        try:
            print("Running back-test (about 1-3 minutes) ...")
            backtest(df, out_dir)
        except Exception as e:
            print(f"(back-test skipped: {e})")

    def process(target):
        try:
            run_one(c, target)
        except Exception as e:
            print("  [ERROR]", e)

    if args.targets:                                        # non-interactive mode
        for txt in args.targets:
            t = parse_target(txt, last)
            if t is None or t <= last:
                print(f"  '{txt}' is not valid (give a month after {last:%b-%Y}).")
                continue
            process(t)
        return

    while True:
        target = ask_payment(last)
        if target is None:
            break
        process(target)
    print("\nThank you! Program closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n[ERROR]", e)
        if sys.stdin and sys.stdin.isatty():
            input("\nPress Enter to close ...")
