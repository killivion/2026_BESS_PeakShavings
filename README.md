# BESS Peak-Shaving Load Forecasting

This case study builds a 15-minute load forecasting pipeline for a German C&I site, evaluated against the asymmetric cost of missing a demand-charge peak. The pipeline parses the German-locale CSV, regularizes the time grid, flags physically implausible readings, preserves valid zero-load shutdowns, interpolates only isolated invalid intervals, and evaluates four forecasting models under one leakage-safe chronological harness: weekly seasonal-naive, a conservative 80th-percentile residual uplift, a ridge regression on load and calendar features, and a gradient-boosted quantile model (LightGBM, pinball loss).

## Run

```powershell
python -m pip install -r requirements.txt
python main.py --input case_inputs/load_timeseries_2025_case_study.csv --output-dir outputs
pytest -q
```

Outputs are written to `outputs/`: `prepared_load.csv`, `eda_summary.csv`, `forecast_metrics.csv`, `rolling_validation_metrics.csv`, `dispatch_scenarios.csv` / `dispatch_simulation.csv` (conservative q80 dispatch), `dispatch_scenarios_gbm.csv` / `dispatch_simulation_gbm.csv` (GBM quantile dispatch), and saved figures under `outputs/plots/` (data quality, forecast comparison, and both dispatch graphics). The notebook `bess_peak_shaving_case_study.ipynb` presents the narrative analysis and visualizations.

For readability, the headline metrics use the complete final 56-day chronological holdout, including holidays (15. Nov - 31. Dec). The forecast comparison graphic uses the first complete Monday-Sunday week in the holdout selected by calendar and data availability, without using its load values to choose the example.

The notebook runs against `.venv-1`; `main.py` and `pytest` run against the base interpreter. Both environments pin the same `lightgbm` version, so aggregate metrics match to the displayed precision across the two, but a single illustrative dispatch figure can drift by well under 1 kW between them (a floating-point artifact of GBM's tree splits, not a modeling inconsistency).

## Modeling choices

The validation set is the final 56 calendar days, preserving temporal order, plus three rolling 28-day folds for stability checking. Metrics include MAE/RMSE for scale, under-forecast rate, peak under-forecast rate (how often the forecast falls short of a true peak — the metric that actually drives missed demand-charge protection), peak recall, and a cost-weighted absolute error that assigns twice the weight to under-forecasting.

Four models are evaluated head to head:

- **Weekly seasonal-naive** — the transparent governance baseline. No training; forecasts the same quarter-hour one week earlier.
- **Conservative q80** — the naive forecast plus a single flat historical uplift (80th percentile of week-over-week deltas). Transparent, but the uplift is applied everywhere and all the time, not just near peaks.
- **Ridge regression** — load-only lags (15 min, 1 h, daily, weekly), rolling means, time-of-day, weekday, and German holiday flags. The strongest fully-linear, fully-interpretable option; best raw MAE of the four in this data.
- **GBM quantile (recommended for peak protection)** — the same feature set as ridge, but LightGBM trained directly on pinball loss at the 0.8 quantile, so it targets the peak-protective quantile itself rather than approximating it with a flat shift.

On the headline holdout (15-minute horizon), the GBM quantile model holds an MAE of 6.1 kW (close to ridge's 5.5 kW, far below naive/q80's 16-22 kW) **and** the lowest peak under-forecast rate of the four (42% vs ridge's 79%, q80's 61%, naive's 81%) — it does not trade one for the other the way the other three do. The same pattern holds across all three rolling folds. At the 4-hour horizon specifically, ridge's peak recall collapses to 0% in this data while the GBM model still recovers roughly three-quarters of true peaks, making it the only retained model that holds up across the full 15-minute-to-4-hour operational range. In the dispatch illustration, the GBM forecast recovers a real, double-digit-kW peak reduction at a threshold where the flat q80 uplift achieves none, and — because it isn't inflated by a constant applied even during genuinely quiet hours — also unlocks substantially more legitimate charging headroom overnight, addressing both sides of the case's asymmetric-cost framing at once.

The production recommendation is receding-horizon operation: refresh a 15-minute to 4-hour forecast at each interval, use the GBM quantile forecast to drive the peak-shaving threshold, and keep ridge and weekly-naive as interpretable, low-cost cross-checks. The trade-off is real: GBM is a tree ensemble, not a set of readable coefficients, and adds `lightgbm` as a project dependency with roughly double the run time of a ridge-only pipeline.

See [model_research.md](model_research.md) for the model-family comparison, expected accuracy, advantages, drawbacks, peak-risk fit, and the client information that would change the ranking.

Future development depends on client information that is not available in this case: site operating schedules and planned shutdowns, holiday calendars specific to the customer, battery power/energy/SOC history, demand-charge tariff and billing windows, submetered process loads, and market prices for valuing arbitrage opportunity cost. Battery cycle-life/degradation cost is also unmodeled: the GBM-driven dispatch triggers noticeably more charge/discharge activity than q80's in this simulation, a genuine efficiency gain here but one that would need weighing against real cycling costs before going live. These should be added only with time-aware validation and client-specific cost weights.
