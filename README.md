# BESS Peak-Shaving Load Forecasting

This case study builds an interpretable 15-minute load forecasting baseline for a German C&I site. The pipeline parses the German-locale CSV, regularizes the time grid, flags physically implausible readings, preserves valid zero-load shutdowns, interpolates only isolated invalid intervals, and evaluates a weekly seasonal-naive forecast with a conservative 80th-percentile residual uplift.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py
pytest -q
```

Outputs are written to `outputs/`: the prepared dataset, EDA summary, holdout metrics, and saved figures under `outputs/plots/`. The notebook `bess_peak_shaving_case_study.ipynb` presents the narrative analysis and visualizations.

For readability, the headline metrics use the complete final 56-day chronological holdout, including holidays (15. Nov - 31. Dec). The forecast comparison graphic uses the first complete Monday-Sunday week in the holdout selected by calendar and data availability, without using its load values to choose the example.

## Modeling choices

The validation set is the final 56 calendar days, preserving temporal order. Metrics include MAE/RMSE for scale, under-forecast rate, peak under-forecast rate, and a cost-weighted absolute error that assigns twice the weight to under-forecasting. The production recommendation is receding-horizon operation: refresh a 15-minute to 4-hour forecast at each interval, use the conservative estimate to protect a demand-charge threshold, and retain a reserve for forecast error.

The baseline intentionally excludes weather, customer production schedules, and site-specific tariff data. German holiday flags are available as features, horizon scoring uses rolling forecast origins, and the dispatch demo tracks SOC with simple illustrative opportunity cost. The highest-value next additions are site-specific tariff rules, production schedules, market prices, and a fuller rolling backtest across multiple validation windows.
