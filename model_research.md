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
| Ridge calendar/lag regression | Medium to high | Medium | Stabilizes correlated lags, handles calendar effects, remains explainable | Still linear; may smooth away sharp peaks; alpha must be validated temporally | Current preferred challenger |
| Quantile regression | Medium to high for risk-aware forecasts | Strong | Directly estimates a high conditional quantile; aligns with asymmetric peak protection | Requires enough peak examples and careful calibration; one quantile is not a full dispatch policy | High-value next experiment |
| Gradient-boosted trees | High potential on tabular lag/calendar data | Strong if trained with peak weights or quantile loss | Learns nonlinear interactions and regime effects; can use missing values; often strong on tabular data | Less transparent, easier to overfit one site, needs strict time-based validation | Best accuracy candidate after ridge |
| Gradient-boosted quantile model | High potential for peak decisions | Very strong | Produces an upper conditional forecast instead of adding one global uplift | Quantile calibration and hyperparameters require multiple rolling folds; may be unstable with limited regimes | Recommended production research direction |
| SARIMAX / dynamic regression | Medium to high if dynamics are stable | Medium | Explicit autocorrelation, seasonality, and exogenous regressors; statistical diagnostics | Computationally heavier at 15-minute seasonal period; order selection can overfit; less convenient for nonlinear peaks | Consider after adding reliable exogenous variables |
| Random forest / ExtraTrees | Medium to high | Medium | Nonlinear, robust, easy to prototype | Usually weaker extrapolation and less targeted than boosting; feature importance can mislead | Secondary tree benchmark, not first choice |
| LSTM / Transformer / TFT | Unknown; potentially high with much more data | Potentially strong | Can model complex multi-step patterns and covariates | One site/year is too little; opaque; expensive to tune and validate; operational overkill here | Defer until multi-site or multi-year data exists |

## Practical Ranking

### 1. Keep as governance baselines

- Weekly seasonal naive.
- Conservative weekly seasonal naive.
- Persistence and daily seasonal naive as simple sanity checks.

These establish whether a more complex model adds value and make model drift visible.

### 2. Use now as the transparent challenger

**Ridge regression with load-only features** is the most defensible next model for this case:

- Short lags: 15 minutes and 1 hour.
- Seasonal lags: 1 day and 1 week.
- Shifted rolling means: 1 hour and 24 hours.
- Time-of-day sine/cosine.
- Weekday and German public-holiday flags.

Ridge is preferable to unregularized OLS because these lag and rolling features are correlated. The regularization penalty shrinks unstable coefficients rather than allowing a single split to decide model behavior.

### 3. Best next accuracy experiment

A shallow gradient-boosted tree model is the strongest realistic accuracy candidate with the current information. Start with a small feature set and conservative settings:

- Limited tree depth or leaf count.
- Large minimum leaf size.
- Low learning rate.
- Early stopping using a chronological validation fold.
- Absolute-error or quantile loss rather than unconstrained squared error.
- Compare median and 0.80-0.90 quantile forecasts.

Do not select hyperparameters on the final holdout. Use earlier rolling folds for selection and reserve the final block for one final report.

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
