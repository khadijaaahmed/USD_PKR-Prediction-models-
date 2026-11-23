"""OUTPUT - Excel files and charts."""
import matplotlib.pyplot as plt


def write_ab_excel(path, sheets):
    """sheets = {sheet_name: [(section_title, [(item, value), ...]), ...]}
    Column A = item, column B = value."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    wb = Workbook()
    wb.remove(wb.active)
    thin = Side(style="thin", color="999999")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    for name, sections in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(["Item", "Value"])
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.border = box
        for title, items in sections:
            ws.append([title, None])
            r = ws.max_row
            for cell in ws[r]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="DDEBF7")
                cell.border = box
            for item, value in items:
                ws.append([item, value])
                r = ws.max_row
                ws.cell(row=r, column=1).border = box
                ws.cell(row=r, column=2).border = box
                ws.cell(row=r, column=2).alignment = Alignment(horizontal="right")
        ws.column_dimensions["A"].width = 50
        ws.column_dimensions["B"].width = 20
    wb.save(path)


def save_forecast_chart(path, hist, target, exp_rate, worst_rate):
    fig = plt.figure(figsize=(11, 5))
    plt.plot(hist.index, hist.values, color="royalblue", lw=2, label="Historical USD/PKR")
    plt.plot([hist.index[-1], target], [hist.iloc[-1], exp_rate], color="darkorange", ls="--",
             lw=1.5, label="Random Forest forecast")
    plt.scatter([target], [exp_rate], color="darkorange", zorder=5)
    plt.scatter([target], [worst_rate], color="red", zorder=5, label="Worst case (95%)")
    plt.plot([target, target], [exp_rate, worst_rate], color="gray", alpha=.5)
    plt.annotate(f"{exp_rate:.1f}", (target, exp_rate), textcoords="offset points",
                 xytext=(6, -12), fontsize=9)
    plt.title(f"USD/PKR forecast and worst case for {target:%b-%Y} (Random Forest)")
    plt.xlabel("Date"); plt.ylabel("USD/PKR"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_backtest_chart(path, h, dates, act, pred):
    fig = plt.figure(figsize=(11, 5))
    plt.plot(dates, act, color="royalblue", lw=2, label="Actual USD/PKR (test)")
    plt.plot(dates, pred, color="darkorange", ls="--", lw=2, label=f"Random Forest ({h}-month ahead)")
    plt.title("USD/PKR back-test (unseen 20 % test data)")
    plt.xlabel("Date"); plt.ylabel("USD/PKR"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
