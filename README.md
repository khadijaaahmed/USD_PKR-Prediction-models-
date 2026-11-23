# USD_PKR-Prediction-models-
Compare Three  Prediction models to Predict Upcoming USD Direction and Exact rates for Pakistan  Interbank conversions
# USD/PKR Forecast + Risk: CatBoost, Random Forest and SVM

Part of: **AI-Enabled FX Hedging Decision System for USD/PKR, using NDF and Forward Premium Analysis for Treasury Risk Management.**

Three versions of the same tool forecast the **INTERBANK USD/PKR** rate for a payment month and measure the risk around that forecast. They use the same data file, the same workflow and the same three outputs. Only the model family changes.

> The **hedge part (NDF / forward premium) is not included yet.** First the three models are compared, then the best one is used for the hedge decision.

![How the three models work]<img width="1890" height="1762" alt="USDPKR_Models_Flowchart" src="https://github.com/user-attachments/assets/2439bb4e-5314-4d07-a45f-07e4960f6942" />


---

## 1. Quick comparison

| | **CatBoost** | **Random Forest** | **SVM** |
|---|---|---|---|
| Project folder / package | `usdpkr_fx` / `fxrisk` | `usdpkr_rf` / `fxrf` | `usdpkr_svm` / `fxsvm` |
| Factor forecast (7 factors) | CatBoost, 3 seeds per factor, Huber loss, predicts the h-month change | One Random Forest per factor, predicts the h-month change | One SVR (RBF) per factor, next-month change, recursive, trend fades x0.75 per month |
| Link factors to USD/PKR | Inside Stage 2 (no separate step) | Internal relation step: Pearson correlation gives a `relation_score` | Spearman correlation at lags 0-3, keep \|rho\| >= 0.10, weight by \|rho\|, then SVR |
| Interbank forecast | Forecast factors + USD/PKR momentum, h-month change | Forecast factors + `relation_score`, h-month change | Factor changes to monthly log-return, summed up to the payment month |
| Volatility (monthly %) | CatBoost regressor | RF regressor | SVR on 3-month volatility, scaled by sqrt(h) |
| Worst case (95 %) | CatBoost quantile model | Quantile Regression Forest | 95th percentile of standardised back-test errors (floor 1.645) |
| High-volatility probability | CatBoost classifier | RF classifier | SVC (calibrated probability) |
| Back-test horizons | 1, 3, 6, 12 months | 1, 3, 6, 12 months | 1 month ahead |
| Longest payment month | Warns above 36 months | Warns above 36 months | Refuses above 60 months |

Inflation differential = Pakistan CPI - US CPI in all three. All three predict **changes**, not levels, because USD/PKR rose from about 105 to about 280 in the data and tree models cannot predict outside the range they have seen.

---

## 2. Data file

A **monthly** Excel (or CSV for CatBoost / Random Forest) file, at least **48 rows**, with a `Date` column. Columns are found by keywords, so exact names are not needed and extra columns are ignored:

- Foreign reserves
- REER
- Brent Crude Oil Price
- SBP Policy rate
- Federal Funds Rate
- Pakistan CPI Inflation YoY (%)
- US CPI YoY Inflation (%)
- Inflation rate differential (optional for CatBoost / Random Forest; calculated if missing)
- INTERBANK USD/PKR - Close

The SVM version expects the same columns as `risk_train_data.xlsx`, including the differential column.

---

## 3. Install and run

Python 3.9 or newer. Each project has its own `requirements.txt` (`pip install -r requirements.txt`). CatBoost and Random Forest also install missing packages automatically on the first run. Run from **inside** the project folder.

```
# CatBoost
cd usdpkr_fx
python main.py                                  # file window opens
python main.py data.xlsx                        # asks the payment date
python main.py data.xlsx 2 Dec-2026             # no questions
python main.py data.xlsx --skip-backtest

# Random Forest
cd usdpkr_rf
python main.py data.xlsx 1 6 Dec-2026           # three payment dates, back-test skipped

# SVM
cd usdpkr_svm
python main.py --data data.xlsx --out MyOutputFolder
python main.py --data data.xlsx --payment Dec-2026
```

**Payment date:** months ahead (`6`) or a month-year (`Dec-2026`, `12/2026`, `9-2024`). Type `q` to quit. The payment month must be after the last data month (Jul-2024 in the sample data).

---

## 4. Output (three things per model)

| Model | Excel (one per payment month) | Forecast chart (one per payment month) | Back-test chart (once) |
|---|---|---|---|
| CatBoost | `USDPKR_CatBoost_Forecast_Sep2024.xlsx` | `USDPKR_Forecast_Chart_Sep2024.png` | `USDPKR_Backtest_Chart.png` |
| Random Forest | `USDPKR_RandomForest_Forecast_Sep2024.xlsx` | `USDPKR_RF_Forecast_Chart_Sep2024.png` | `USDPKR_RF_Backtest_Chart.png` |
| SVM | `USDPKR_SVM_Forecast_Sep2024.xlsx` | `USDPKR_SVM_Forecast_Chart_Sep2024.png` | `USDPKR_SVM_Backtest_Chart.png` |

- **Folder:** CatBoost and Random Forest save in `FYP_Risk_Output` next to the data file and never overwrite (`_2`, `_3` are added to a repeated month). SVM saves in `Downloads/FYP_Risk_Output` unless `--out` is given, and a repeated month overwrites the earlier file.
- **Excel layout:** column A = item, column B = value. Sections: PAYMENT (month, months ahead, last data month), FORECAST (all factors and USD/PKR), RISK (last spot, expected USD/PKR, monthly volatility %, worst-case USD/PKR at 95 %, probability of high volatility %). The SVM workbook also has a second sheet with the back-test chart.
- No relation file or relation table is exported. The relation step runs inside the code only (the SVM version prints its strongest relations in the console).

---

## 5. How to read the result

| Item | Meaning |
|---|---|
| Factor forecasts | Forecast value of each factor for the payment month |
| Expected USD/PKR | Forecast interbank rate for the payment month |
| Volatility % (monthly) | Expected monthly volatility of USD/PKR |
| Worst-case USD/PKR (95 %) | A rate the model expects to be exceeded only about 5 % of the time |
| Probability of high volatility % | Chance that future volatility falls in the top 25 % of the history |

---

## 6. Back-test and accuracy snapshot

The last **20 %** of the months (Nov-2022 to Jul-2024, 21 months) are never used for training. The forecast is compared with a naive "no change" forecast.

**CatBoost (MAPE, lower is better)**

| Horizon | CatBoost | Naive |
|---|---|---|
| 1 month | 1.86 % | 2.02 % |
| 3 months | 4.57 % | 4.38 % |
| 6 months | 8.03 % | 8.67 % |
| 12 months | 16.46 % | 18.88 % |

**Random Forest:** beat naive at 6 and 12 months and was about level at 1 month on the sample file.

**SVM (1-month ahead):** MAE 5.26 PKR (naive 5.50), RMSE 8.42 PKR, MAPE 1.92 %, directional accuracy 57 %.

**How to compare them fairly:** the three back-tests use the same 80/20 time split, but the SVM reports only 1-month results and the Random Forest README gives no numbers. For a final comparison, run each project and compare MAPE vs naive at the same horizons. The CatBoost settings were chosen by a grid search on this same test period, so its numbers are somewhat optimistic.

---

## 7. Limitations

- Only about 100 monthly observations, and the test period contains the 2022-23 devaluation shock.
- CatBoost does not beat naive at 3 months. Its raw worst case tends to be too low (real rates exceeded it more often than the intended 5 %), so treat it as an optimistic worst case.
- SVM factor forecasts are model estimates, and their errors pass into the Interbank forecast. Its worst case is a statistical estimate, not a guarantee.
- Forecasts beyond about 12 months are unreliable. A very far payment month may be refused with a message asking for a nearer month or more data.
- If the Excel file is open in Excel when the program saves, close it and run again.
- Academic / research tool. **Not financial advice.**

---


Compare the three models on the same horizons, pick the best one, and add the hedge module (NDF / forward premium) on top of it.
