# Load-Forecasting Model Research

## Context

The available data is one year of 15-minute load for one site: about 35,000 regular intervals, with no weather, production schedule, battery state of charge, tariff, or price data. The target is not only average accuracy: missed high-load events can create demand charges, while excessive forecasts can reserve battery capacity that could otherwise earn arbitrage value.

The comparison below therefore separates:

- **Point accuracy:** MAE and RMSE.
- **Peak protection:** under-forecast rate, top-decile peak miss rate, and asymmetric cost.
- **Operational usability:** whether the forecast is available at the decision time, stable across time windows, and explainable to an operator.

## Model Matrix

| Model family | Expected accuracy here | Peak-risk fit | Advantages | Drawbacks | Recommendation |
|---|---:|---:|---|---|---|
| Persistence / last value | Low to medium for 15-minute control | Weak for sudden peaks | Almost free, strong immediate benchmark, no training | Misses schedule changes and recurring weekly structure | Keep as a sanity-check baseline |
| Daily seasonal naive | Medium | Weak to medium | Captures same time yesterday; easy to explain | Sunday/Monday and holiday behavior can be poor | Keep as an additional baseline |
| Weekly seasonal naive | Medium | Medium | Captures weekday and weekly operating pattern; highly auditable | Cannot adapt to process changes or unusual peaks | Keep as the primary governance baseline |
| Conservative seasonal naive | Medium point accuracy, lower under-forecasting | Medium to strong | Directly encodes asymmetric risk through a residual uplift | Can waste battery capacity through over-forecasting; fixed uplift may be crude | Use when demand-charge protection is prioritized |
| OLS calendar/lag regression | Medium | Weak to medium | Interpretable coefficients and fast training | Sensitive to correlated features and linearity assumptions | Keep as a diagnostic benchmark |
| Ridge calendar/lag regression | Medium to high | Medium | Stabilizes correlated lags, handles calendar effects, remains explainable | Still linear; may smooth away sharp peaks; alpha must be validated temporally | Interpretable cross-check on the GBM quantile model below |
| Quantile regression | Medium to high for risk-aware forecasts | Strong | Directly estimates a high conditional quantile; aligns with asymmetric peak protection | Requires enough peak examples and careful calibration; one quantile is not a full dispatch policy | High-value next experiment |
| Gradient-boosted trees (mean/absolute-error objective) | High potential on tabular lag/calendar data | Strong if trained with peak weights or quantile loss | Learns nonlinear interactions and regime effects; can use missing values; often strong on tabular data | Less transparent, easier to overfit one site, needs strict time-based validation | Superseded by the quantile-loss variant below, which is what's actually implemented |
| Gradient-boosted quantile model | **Implemented and validated** — MAE 6.1 kW, peak under-forecast rate 42% at the headline holdout (vs ridge 5.5 kW / 79%) | Very strong — lowest peak under-forecast rate of all four retained models, consistent across the headline holdout and all three rolling folds | Produces an upper conditional forecast instead of adding one global uplift; only retained model that still recovers most peaks at the 4-hour horizon | Less transparent than ridge; adds `lightgbm` as a dependency; roughly doubles pipeline run time | **Recommended model for peak-protection decisions** |
| SARIMAX / dynamic regression | Medium to high if dynamics are stable | Medium | Explicit autocorrelation, seasonality, and exogenous regressors; statistical diagnostics | Computationally heavier at 15-minute seasonal period; order selection can overfit; less convenient for nonlinear peaks | Consider after adding reliable exogenous variables |
| Random forest / ExtraTrees | Medium to high | Medium | Nonlinear, robust, easy to prototype | Usually weaker extrapolation and less targeted than boosting; feature importance can mislead | Secondary tree benchmark, not first choice |
| LSTM / Transformer / TFT | Unknown; potentially high with much more data | Potentially strong | Can model complex multi-step patterns and covariates | One site/year is too little; opaque; expensive to tune and validate; operational overkill here | Defer until multi-site or multi-year data exists |

## Practical Ranking

### 1. Keep as governance baselines

- Weekly seasonal naive.
- Conservative weekly seasonal naive.
- Persistence and daily seasonal naive as simple sanity checks.

These establish whether a more complex model adds value and make model drift visible.

### 2. Kept as the interpretable cross-check

**Ridge regression with load-only features** remains the most defensible fully-linear model for this case, now serving as a coefficient-level-readable sanity check on the GBM quantile model in section 3 rather than the primary challenger:

- Short lags: 15 minutes and 1 hour.
- Seasonal lags: 1 day and 1 week.
- Shifted rolling means: 1 hour and 24 hours.
- Time-of-day sine/cosine.
- Weekday and German public-holiday flags.

Ridge is preferable to unregularized OLS because these lag and rolling features are correlated. The regularization penalty shrinks unstable coefficients rather than allowing a single split to decide model behavior. This is the **operational feature set** — valid at the 15 min-4 h horizons this case targets, since it leans on what happened in the last 15-60 minutes; see section 4 for why it stops being valid a day out.

### 3. Now implemented as the primary peak-protection model

A shallow gradient-boosted tree model, trained on the same operational feature set as ridge above, on pinball loss at the 0.8 quantile, with conservative hyperparameters (limited leaf count, large minimum leaf size, low learning rate, `deterministic`/`force_row_wise` for reproducible splits), is now the recommended model for peak-shaving threshold decisions. Validated on the same leakage-safe headline holdout and three rolling folds used for the other models, it holds the lowest peak under-forecast rate of all four retained models while matching ridge's average accuracy — see [README.md](README.md) for the exact figures. Hyperparameters were kept intentionally conservative rather than tuned on the final holdout, consistent with the guidance below.

### 4. Day-ahead horizon: tested, and not solved by a feature-set swap

The obvious fix for a day-ahead (24 h) forecast is to drop the short lags above — they're stale by then — and keep only same-day-yesterday, same-time-last-week, and calendar features (`gbm_day_ahead_forecast`, `build_day_ahead_features`). Tested directly at the 96-slot horizon: it does **not** work. It underperforms both the operational feature set *and* the trivial weekly-seasonal-naive baseline on every metric (MAE, peak under-forecast rate, peak recall). Dropping the short lags removes real, if indirect, signal about the site's current operating regime without anything of comparable value to replace it. This confirms rather than contradicts the horizon-strategy guidance elsewhere in this case: genuine day-ahead accuracy needs additional inputs (weather, a production schedule) or a different kind of approach (forecasting the day's shape/peak risk rather than each 15-minute point) — not a feature-set change to the same short-horizon model family. Kept in the codebase, tested, and documented as evidence for that conclusion.

## Why Not Choose the Most Complex Model Immediately?

The data contains many rows but only one customer and one calendar year. The effective number of independent operating regimes is much smaller than 35,000 because neighboring 15-minute readings are correlated. A flexible model can therefore produce an impressive holdout score by learning site-specific quirks, holidays, or isolated events that will not repeat.

A model should be promoted only when it improves:

1. Average rolling-fold asymmetric cost.
2. Top-decile peak miss rate.
3. Stability across folds and horizons.
4. Operational interpretability and forecast availability.

## Client Data That Would Change the Ranking

Additional client information could make more advanced models worthwhile:

- **Operating schedule and planned shutdowns:** improves calendar/regime features and holiday handling.
- **Customer-specific holidays and production calendar:** prevents treating shutdowns as unexpected noise.
- **Battery SOC, power, energy capacity, and efficiency:** enables a true forecast-to-dispatch optimization rather than a load-only simulation.
- **Demand-charge tariff, billing windows, and ratchets:** turns asymmetric error weights into euros per missed kW.
- **Submetered process loads:** separates predictable base load from volatile production events.
- **Weather and temperature:** useful if HVAC or weather-sensitive processes materially drive demand.
- **Market prices and imbalance costs:** values the opportunity cost of reserving or discharging the battery.
- **Multiple sites or multiple years:** makes boosting, quantile models, and deep learning less prone to site-specific overfitting.

## References

- scikit-learn Ridge and regularized linear models: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
- scikit-learn gradient-boosted trees, quantile loss, missing values, and regularization: https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
- statsmodels SARIMAX API and seasonal/exogenous-regressor specification: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
- LightGBM objectives including regression and quantile loss: https://lightgbm.readthedocs.io/en/latest/Parameters.html
