"""Reusable pipeline for the BESS peak-shaving case study."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY


def initialize_case(root: str | Path, input_path: str | Path | None = None) -> tuple[Path, Path, Path, pd.DataFrame, pd.DataFrame]:
    """Load the case data and return paths, prepared data, and raw data."""
    root = Path(root)
    if input_path is not None:
        data_path = Path(input_path)
    else:
        candidates = (
            root / "load_timeseries_2025_case_study.csv",
            root / "case_inputs" / "load_timeseries_2025_case_study.csv",
        )
        data_path = next((path for path in candidates if path.exists()), candidates[0])
    output_dir = root / "outputs"
    plot_dir = output_dir / "plots"
    frame = add_calendar_features(load_and_prepare(data_path))
    raw = pd.read_csv(data_path, sep=";", decimal=",")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return data_path, output_dir, plot_dir, frame, raw


def load_and_prepare(path: str | Path) -> pd.DataFrame:
    """Parse the German-locale input and regularize it to a 15-minute grid.

    Values outside a conservative physical envelope are flagged as artifacts.
    Only isolated invalid readings are linearly interpolated; longer outages
    remain missing so they cannot manufacture or conceal a peak.
    """
    raw = pd.read_csv(path, sep=";", decimal=",")
    raw["timestamp"] = pd.to_datetime(
        raw["Timestamps"], format="%d.%m.%Y %H:%M", errors="raise"
    )
    raw["load_kw_raw"] = pd.to_numeric(raw["Load_kw"], errors="coerce")
    raw = raw.set_index("timestamp").sort_index()[["load_kw_raw"]]
    raw["is_artifact"] = raw["load_kw_raw"].isna() | raw["load_kw_raw"].lt(0) | raw["load_kw_raw"].gt(200)

    full_index = pd.date_range(raw.index.min(), raw.index.max(), freq="15min")
    prepared = raw.reindex(full_index)
    prepared.index.name = "timestamp"
    prepared["was_observed"] = prepared["load_kw_raw"].notna()
    prepared["is_artifact"] = prepared["is_artifact"].eq(True) | prepared["load_kw_raw"].isna()
    prepared["load_kw"] = prepared["load_kw_raw"].where(~prepared["is_artifact"])

    # Interpolate only one missing interval at a time. Outages remain NaN.
    missing_before = prepared["load_kw"].isna()
    prepared["load_kw"] = prepared["load_kw"].interpolate(
        method="time", limit=1, limit_area="inside"
    )
    prepared["was_interpolated"] = missing_before & prepared["load_kw"].notna()
    prepared["is_scoring_eligible"] = prepared["load_kw"].notna() & ~prepared["was_interpolated"]
    prepared["weekday"] = prepared.index.dayofweek < 5
    prepared["hour"] = prepared.index.hour + prepared.index.minute / 60
    prepared["date"] = prepared.index.date
    return prepared


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["slot"] = result.index.hour * 4 + result.index.minute // 15
    result["day_of_week"] = result.index.dayofweek
    result["lag_day_kw"] = result["load_kw"].shift(SLOTS_PER_DAY)
    result["lag_week_kw"] = result["load_kw"].shift(SLOTS_PER_WEEK)
    return result


def build_regression_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build load-only, leakage-safe calendar, lag, and rolling features."""
    result = add_german_holiday_flag(frame)
    slots = result.index.hour * 4 + result.index.minute // 15
    result["hour_sin"] = np.sin(2 * np.pi * slots / SLOTS_PER_DAY)
    result["hour_cos"] = np.cos(2 * np.pi * slots / SLOTS_PER_DAY)
    result["weekday_flag"] = (result.index.dayofweek < 5).astype(float)
    result["holiday_flag"] = result["is_public_holiday"].astype(float)
    for lag in (1, 4, SLOTS_PER_DAY, SLOTS_PER_WEEK):
        result[f"lag_{lag}_kw"] = result["load_kw"].shift(lag)
    result["rolling_1h_mean_kw"] = result["load_kw"].shift(1).rolling(4).mean()
    result["rolling_24h_mean_kw"] = result["load_kw"].shift(1).rolling(SLOTS_PER_DAY).mean()
    return result


def seasonal_forecast(
    frame: pd.DataFrame,
    horizon_slots: int = 1,
    residual_quantile: float = 0.0,
    train_end: pd.Timestamp | None = None,
) -> pd.Series:
    """Forecast from the same quarter-hour one week earlier plus calibration uplift."""
    # The returned series is indexed by target time. Horizon is reported by
    # the evaluator; the comparable seasonal benchmark remains one week lagged.
    base = frame["load_kw"].shift(SLOTS_PER_WEEK)
    if residual_quantile:
        history_mask = frame["is_scoring_eligible"]
        if train_end is not None:
            history_mask = history_mask & (frame.index < train_end)
        history = frame["load_kw"].where(history_mask)
        residual = history - history.shift(SLOTS_PER_WEEK)
        uplift = residual.dropna().quantile(residual_quantile)
    else:
        uplift = 0.0
    return base + uplift


def asymmetric_metrics(
    actual: pd.Series,
    forecast: pd.Series,
    peak_quantile: float = 0.90,
    peak_cutoff_kw: float | None = None,
) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), forecast.rename("forecast")], axis=1).dropna()
    error = aligned["forecast"] - aligned["actual"]
    under = error < 0
    peak_cutoff = peak_cutoff_kw if peak_cutoff_kw is not None else aligned["actual"].quantile(peak_quantile)
    peak = aligned["actual"] >= peak_cutoff
    peak_hits = (aligned["forecast"] >= peak_cutoff) & peak
    return {
        "n": float(len(aligned)),
        "mae_kw": float(error.abs().mean()),
        "rmse_kw": float(np.sqrt((error.pow(2)).mean())),
        "underforecast_rate": float(under.mean()),
        "underforecast_mae_kw": float((-error[under]).mean()) if under.any() else 0.0,
        "peak_underforecast_rate": float((under & peak).sum() / peak.sum()) if peak.any() else 0.0,
        "peak_recall": float(peak_hits.sum() / peak.sum()) if peak.any() else 0.0,
        "peak_underforecast_mae_kw": float((-error[under & peak]).mean()) if (under & peak).any() else 0.0,
        "peak_forecast_bias_kw": float(error[peak].mean()) if peak.any() else 0.0,
        "weighted_absolute_error_kw": float((error.abs() * np.where(error < 0, 2.0, 1.0)).mean()),
        "peak_cutoff_kw": float(peak_cutoff),
    }


def summarize_rolling_metrics(rolling_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average fold-level metrics for a side-by-side model comparison."""
    numeric_columns = rolling_metrics.select_dtypes(include="number").columns.difference(["fold", "n"])
    summary = rolling_metrics.groupby("model", as_index=False)[numeric_columns].mean()
    folds = rolling_metrics.groupby("model", as_index=False)["fold"].nunique().rename(columns={"fold": "folds"})
    observations = rolling_metrics.groupby("model", as_index=False)["n"].sum()
    observations = observations.rename(columns={"n": "observations"})
    return summary.merge(folds, on="model").merge(observations, on="model")


def evaluate_holdout(frame: pd.DataFrame, holdout_days: int = 56) -> pd.DataFrame:
    """Evaluate target times reached from origins in the final time block.

    The seasonal lag is known at each forecast origin. For horizon ``h``, the
    first scored target is ``holdout_start + h``; this prevents the horizon
    labels from scoring exactly the same target set.
    """
    cutoff = frame.index.max() - pd.Timedelta(days=holdout_days) + pd.Timedelta(minutes=15)
    training_load = frame.loc[(frame.index < cutoff) & frame["is_scoring_eligible"], "load_kw"]
    peak_cutoff = float(training_load.quantile(.90))
    rows: list[dict[str, float | str]] = []
    for horizon in (1, 4, 16):
        for quantile, label in ((0.0, "weekly_seasonal_naive"), (0.8, "conservative_q80")):
            origins = frame.index[(frame.index >= cutoff) & (frame.index <= frame.index.max() - pd.Timedelta(minutes=15 * horizon))]
            target_index = origins + pd.Timedelta(minutes=15 * horizon)
            source_index = target_index - pd.Timedelta(days=7)
            forecast = pd.Series(frame["load_kw"].reindex(source_index).to_numpy(), index=target_index)
            if quantile:
                history = frame["load_kw"].where((frame.index < cutoff) & frame["is_scoring_eligible"])
                uplift = (history - history.shift(SLOTS_PER_WEEK)).dropna().quantile(quantile)
                forecast = forecast + uplift
            actual = frame["load_kw"].reindex(target_index)
            eligible = frame["is_scoring_eligible"].reindex(target_index).fillna(False)
            metrics = asymmetric_metrics(actual[eligible], forecast[eligible], peak_cutoff_kw=peak_cutoff)
            rows.append({"horizon_15min": horizon, "model": label, **metrics})
    return pd.DataFrame(rows)


def peak_summary(frame: pd.DataFrame) -> dict[str, float | str]:
    eligible = frame.loc[frame["is_scoring_eligible"], "load_kw"]
    daily = eligible.resample("D").agg(["max", "mean"])
    peak_time = eligible.idxmax()
    return {
        "mean_kw": float(eligible.mean()),
        "p95_kw": float(eligible.quantile(0.95)),
        "max_kw": float(eligible.max()),
        "load_factor_vs_observed_max": float(eligible.mean() / eligible.max()),
        "weekday_mean_kw": float(frame.loc[frame["weekday"] & frame["is_scoring_eligible"], "load_kw"].mean()),
        "weekend_mean_kw": float(frame.loc[(~frame["weekday"]) & frame["is_scoring_eligible"], "load_kw"].mean()),
        "median_daily_peak_kw": float(daily["max"].median()),
        "peak_timestamp": str(peak_time),
        "peak_hour": float(peak_time.hour + peak_time.minute / 60),
    }


def select_representative_week(
    frame: pd.DataFrame,
    first_start: str = "2025-01-06",
    last_start: str = "2025-12-22",
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """Return the complete week closest to the median weekly load profile."""
    starts = pd.date_range(first_start, last_start, freq="7D")
    weeks: list[pd.Timestamp] = []
    profiles: list[np.ndarray] = []
    for start in starts:
        index = pd.date_range(start, periods=SLOTS_PER_WEEK, freq="15min")
        values = frame["load_kw"].reindex(index)
        if values.notna().all():
            weeks.append(start)
            profiles.append(values.to_numpy())
    matrix = np.vstack(profiles)
    median_profile = np.median(matrix, axis=0)
    distances = np.mean(np.abs(matrix - median_profile), axis=1)
    position = int(np.argmin(distances))
    end = weeks[position] + pd.Timedelta(days=6, hours=23, minutes=45)
    return weeks[position], end, float(distances[position])


def save_plot(fig, output_dir: str | Path, filename: str) -> Path:
    """Save a matplotlib figure using consistent presentation settings."""
    import matplotlib.pyplot as plt

    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eda(frame: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Create and save the two curated EDA figures."""
    import matplotlib.pyplot as plt

    eligible = frame.loc[frame["is_scoring_eligible"], "load_kw"]
    start, end, distance = select_representative_week(frame)
    week = frame.loc[start:end, "load_kw"]
    daily = eligible.resample("D").agg(mean_kw="mean", max_kw="max")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    frame.load_kw.plot(ax=axes[0, 0], lw=0.35, color="#1f4e5f", title="Cleaned 15-minute load: 2025")
    week.plot(ax=axes[0, 1], color="#d95f02", title=f"Representative week: {start:%d %b} to {end:%d %b %Y}")
    week_days = pd.date_range(start, end.normalize(), freq="D")
    axes[0, 1].set_xticks(week_days)
    axes[0, 1].set_xticklabels([day.strftime("%A\n%d %b") for day in week_days])
    axes[0, 1].set_xlabel("Day of week")
    profile = frame.assign(day_type=np.where(frame.weekday, "Weekday", "Weekend"), slot=frame.index.hour * 4 + frame.index.minute // 15).groupby(["day_type", "slot"]).load_kw.mean().unstack(0)
    profile.plot(ax=axes[1, 0], color={"Weekday": "#1b9e77", "Weekend": "#7570b3"}, title="Average intraday profile")
    axes[1, 0].set_xticks(np.arange(0, 96, 16))
    axes[1, 0].set_xticklabels([f"{hour:02d}:00" for hour in range(0, 24, 4)])
    axes[1, 0].set_xlabel("Time of day")
    daily.max_kw.plot(ax=axes[1, 1], color="#e7298a", title="Daily maximum load")
    for axis in axes.flat:
        axis.set_ylabel("kW")
    first = save_plot(fig, output_dir, "01_load_overview_and_profiles.png")

    peaks = frame[frame.load_kw.ge(eligible.quantile(.90)) & frame.is_scoring_eligible]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    axes[0].scatter(peaks.index, peaks.load_kw, s=8, alpha=.5, color="#e41a1c")
    axes[0].set_title("Top-decile load events")
    axes[0].set_ylabel("kW")
    heat = frame.assign(day=frame.index.dayofweek, slot=frame.index.hour * 4 + frame.index.minute // 15).pivot_table(index="day", columns="slot", values="load_kw", aggfunc="mean")
    image = axes[1].imshow(heat, aspect="auto", cmap="YlOrRd")
    axes[1].set_title("Average load by weekday and quarter-hour")
    axes[1].set_xticks(np.arange(0, 96, 16))
    axes[1].set_xticklabels([f"{hour:02d}:00" for hour in range(0, 24, 4)])
    axes[1].set_xlabel("Time of day")
    axes[1].set_yticks(range(7))
    axes[1].set_yticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    axes[1].set_ylabel("Day of week")
    fig.colorbar(image, ax=axes[1], label="kW")
    second = save_plot(fig, output_dir, "02_peak_events_and_weekday_heatmap.png")
    return first, second


def plot_forecast_comparison(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
    rolling_metrics: pd.DataFrame | None = None,
) -> tuple[Path, ...]:
    """Save validation error, model comparison, and a holdout-week comparison."""
    import matplotlib.pyplot as plt

    cutoff = frame.index.max() - pd.Timedelta(days=56) + pd.Timedelta(minutes=15)
    first_monday = cutoff + pd.Timedelta(days=(7 - cutoff.dayofweek) % 7)
    comparison_start = first_monday
    while comparison_start <= pd.Timestamp("2025-12-15"):
        candidate_index = pd.date_range(comparison_start, periods=SLOTS_PER_WEEK, freq="15min")
        if frame["load_kw"].reindex(candidate_index).notna().all():
            break
        comparison_start += pd.Timedelta(days=7)
    comparison_end = comparison_start + pd.Timedelta(days=6, hours=23, minutes=45)
    actual = frame.loc[comparison_start:comparison_end, "load_kw"]
    naive = seasonal_forecast(frame, 1, 0.0, cutoff).loc[actual.index]
    conservative = seasonal_forecast(frame, 1, .8, cutoff).loc[actual.index]
    model_order = ["weekly_seasonal_naive", "conservative_q80", "ridge_load_calendar"]
    horizon = metrics.groupby(["horizon_15min", "model"], as_index=False).weighted_absolute_error_kw.first()
    horizon["model"] = pd.Categorical(horizon["model"], categories=model_order, ordered=True)
    horizon = horizon.sort_values(["horizon_15min", "model"])
    fig, axis = plt.subplots(figsize=(8, 4))
    for name, group in horizon.groupby("model", observed=True):
        axis.plot(group.horizon_15min * 15 / 60, group.weighted_absolute_error_kw, marker="o", label=name)
    axis.set_xticks([.25, 1, 4], ["15m", "1h", "4h"])
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Cost-weighted absolute error (kW)")
    axis.set_title("Error trade-off by operational horizon")
    axis.legend()
    first = save_plot(fig, output_dir, "03_horizon_error_tradeoff.png")
    paths = [first]
    if rolling_metrics is not None:
        rolling_summary = rolling_metrics.groupby("model", as_index=False)[
            ["mae_kw", "peak_recall"]
        ].mean()
        rolling_summary["model"] = pd.Categorical(rolling_summary["model"], categories=model_order, ordered=True)
        rolling_summary = rolling_summary.sort_values("model")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        colors = {"weekly_seasonal_naive": "#377eb8", "conservative_q80": "#e41a1c", "ridge_load_calendar": "#1b9e77"}
        axes[0].bar(rolling_summary["model"], rolling_summary["mae_kw"], color=[colors[name] for name in rolling_summary["model"]])
        axes[0].set_title("Average rolling-fold MAE")
        axes[0].set_ylabel("MAE (kW)")
        axes[1].bar(rolling_summary["model"], rolling_summary["peak_recall"], color=[colors[name] for name in rolling_summary["model"]])
        axes[1].set_title("Average rolling-fold peak recall")
        axes[1].set_ylabel("Recall")
        axes[1].set_ylim(0, 1)
        for axis in axes:
            axis.tick_params(axis="x", labelrotation=25)
        comparison_path = save_plot(fig, output_dir, "05_rolling_model_comparison.png")
        paths.append(comparison_path)
    fig, axis = plt.subplots(figsize=(14, 4))
    actual.plot(ax=axis, color="black", lw=1, label="actual")
    naive.plot(ax=axis, color="#377eb8", alpha=.8, label="weekly naive")
    conservative.plot(ax=axis, color="#e41a1c", alpha=.8, label="conservative q80")
    axis.set_ylabel("kW")
    axis.set_title(f"Predeclared holdout week: {comparison_start:%d %b} to {comparison_end:%d %b %Y}")
    axis.legend()
    second = save_plot(fig, output_dir, "04_typical_week_forecast_comparison.png")
    paths.append(second)
    return tuple(paths)


def add_german_holiday_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Add German public-holiday flags when the optional dependency is installed."""
    result = frame.copy()
    try:
        import holidays

        calendar = holidays.country_holidays("DE", years=sorted(set(result.index.year)))
        result["is_public_holiday"] = pd.Index(result.index.date).isin(calendar)
    except ImportError:
        result["is_public_holiday"] = False
    return result


def run_regression_baseline(frame: pd.DataFrame) -> tuple[dict[str, float], pd.Series]:
    """Fit the small calendar/lag regression on pre-holdout data only."""
    features = frame[["load_kw", "lag_day_kw", "lag_week_kw"]].copy()
    features["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
    features["weekday"] = (frame.index.dayofweek < 5).astype(int)
    model_data = features.dropna()
    cutoff = frame.index.max() - pd.Timedelta(days=56) + pd.Timedelta(minutes=15)
    train = (model_data.index < cutoff) & frame.loc[model_data.index, "is_scoring_eligible"].to_numpy()
    columns = ["lag_day_kw", "lag_week_kw", "hour_sin", "hour_cos", "weekday"]
    design_train = np.c_[np.ones(train.sum()), model_data.loc[train, columns]]
    coefficients = np.linalg.lstsq(design_train, model_data.loc[train, "load_kw"], rcond=None)[0]
    prediction = pd.Series(np.c_[np.ones(len(model_data)), model_data[columns]] @ coefficients, index=model_data.index)
    peak_cutoff = float(model_data.loc[train, "load_kw"].quantile(.90))
    score = asymmetric_metrics(
        model_data.loc[model_data.index >= cutoff, "load_kw"],
        prediction.loc[prediction.index >= cutoff],
        peak_cutoff_kw=peak_cutoff,
    )
    return score, pd.Series(coefficients, index=["intercept"] + columns).sort_values(key=abs, ascending=False)


def run_ridge_baseline(
    frame: pd.DataFrame,
    alpha: float = 10.0,
    train_end: pd.Timestamp | None = None,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
) -> tuple[dict[str, float], pd.Series]:
    """Fit a small standardized ridge model using only pre-holdout data."""
    data = build_regression_features(frame)
    cutoff = train_end or (frame.index.max() - pd.Timedelta(days=56) + pd.Timedelta(minutes=15))
    evaluation_start = eval_start or cutoff
    evaluation_end = eval_end or frame.index.max()
    columns = [column for column in data.columns if column.startswith(("lag_", "rolling_"))] + ["hour_sin", "hour_cos", "weekday_flag", "holiday_flag"]
    data = data.dropna(subset=["load_kw"] + columns)
    train = (data.index < cutoff) & frame.loc[data.index, "is_scoring_eligible"].to_numpy()
    mean = data.loc[train, columns].mean()
    scale = data.loc[train, columns].std().replace(0, 1)
    x_train = (data.loc[train, columns] - mean) / scale
    x_all = (data[columns] - mean) / scale
    design_train = np.c_[np.ones(len(x_train)), x_train.to_numpy()]
    penalty = np.diag([0.0] + [alpha] * len(columns))
    coefficients = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ data.loc[train, "load_kw"].to_numpy())
    prediction = pd.Series(np.c_[np.ones(len(x_all)), x_all.to_numpy()] @ coefficients, index=data.index)
    evaluation = (
        (data.index >= evaluation_start)
        & (data.index <= evaluation_end)
        & frame.loc[data.index, "is_scoring_eligible"].to_numpy()
    )
    peak_cutoff = float(data.loc[train, "load_kw"].quantile(.90))
    score = asymmetric_metrics(data.loc[evaluation, "load_kw"], prediction.loc[evaluation], peak_cutoff_kw=peak_cutoff)
    return score, pd.Series(coefficients[1:], index=columns).sort_values(key=abs, ascending=False)


def evaluate_rolling_validation(frame: pd.DataFrame, folds: int = 3, validation_days: int = 28) -> pd.DataFrame:
    """Evaluate retained baselines over several chronological validation windows."""
    rows: list[dict[str, float | int | str]] = []
    last_target = frame.index.max()
    for fold in range(1, folds + 1):
        validation_end = last_target - pd.Timedelta(days=validation_days * (fold - 1))
        cutoff = validation_end - pd.Timedelta(days=validation_days) + pd.Timedelta(minutes=15)
        target_index = frame.index[(frame.index >= cutoff) & (frame.index <= validation_end)]
        actual = frame["load_kw"].reindex(target_index)
        eligible = frame["is_scoring_eligible"].reindex(target_index).fillna(False)
        training_load = frame.loc[(frame.index < cutoff) & frame["is_scoring_eligible"], "load_kw"]
        peak_cutoff = float(training_load.quantile(.90))
        forecasts = {
            "weekly_seasonal_naive": seasonal_forecast(frame, train_end=cutoff),
            "conservative_q80": seasonal_forecast(frame, residual_quantile=.8, train_end=cutoff),
        }
        for name, forecast in forecasts.items():
            rows.append({"fold": fold, "model": name, **asymmetric_metrics(actual[eligible], forecast.reindex(target_index)[eligible], peak_cutoff_kw=peak_cutoff)})
        ridge_score, _ = run_ridge_baseline(frame.loc[:validation_end], train_end=cutoff, eval_start=cutoff, eval_end=validation_end)
        rows.append({"fold": fold, "model": "ridge_load_calendar", **ridge_score})
    return pd.DataFrame(rows)


def dispatch(load_kw: pd.Series, forecast_kw: pd.Series, threshold_kw: float = 80.0, battery_kwh: float = 100.0, power_kw: float = 50.0, efficiency: float = .92, initial_soc_kwh: float | None = None) -> pd.DataFrame:
    """Clip forecasted load above a threshold with explicit SOC tracking."""
    load_kw = pd.Series(load_kw).astype(float)
    forecast_kw = pd.Series(forecast_kw, index=load_kw.index).astype(float)
    soc = battery_kwh if initial_soc_kwh is None else min(initial_soc_kwh, battery_kwh)
    rows = []
    for timestamp in load_kw.index:
        requested_kw = min(max(forecast_kw.loc[timestamp] - threshold_kw, 0), power_kw)
        requested_kw = min(requested_kw, max(load_kw.loc[timestamp], 0))
        max_from_soc_kw = soc * efficiency / .25
        discharge_kw = min(requested_kw, max_from_soc_kw)
        energy_from_soc = discharge_kw * .25 / efficiency
        soc -= energy_from_soc
        rows.append((timestamp, discharge_kw, soc))
    dispatch_data = pd.DataFrame(rows, columns=["timestamp", "discharge_kw", "soc_kwh"]).set_index("timestamp")
    return pd.DataFrame({"load_kw": load_kw, "forecast_kw": forecast_kw}).join(dispatch_data).assign(net_load_kw=lambda data: data["load_kw"] - data["discharge_kw"], energy_discharged_kwh=lambda data: data["discharge_kw"] * .25)


def run_dispatch_demo(frame: pd.DataFrame, forecast: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the illustrative dispatch demo on the first complete holdout week."""
    cutoff = frame.index.max() - pd.Timedelta(days=56) + pd.Timedelta(minutes=15)
    start = cutoff + pd.Timedelta(days=(7 - cutoff.dayofweek) % 7)
    end = start + pd.Timedelta(days=6, hours=23, minutes=45)
    actual = frame.loc[start:end, "load_kw"]
    simulation = dispatch(actual, forecast.loc[actual.index])
    thresholds = pd.DataFrame({"threshold_kw": [70, 80, 90], "risk_posture": ["aggressive protection", "balanced", "arbitrage preserving"]})
    thresholds["peak_after_dispatch_kw"] = [dispatch(actual, forecast.loc[actual.index], threshold).net_load_kw.max() for threshold in thresholds["threshold_kw"]]
    thresholds["energy_discharged_kwh"] = [dispatch(actual, forecast.loc[actual.index], threshold).energy_discharged_kwh.sum() for threshold in thresholds["threshold_kw"]]
    thresholds["illustrative_arbitrage_value_eur"] = thresholds["energy_discharged_kwh"] * 0.10
    thresholds["rule"] = thresholds.apply(lambda row: f"Weekdays 08:00-18:00: clip forecast above {row.threshold_kw} kW ({row.risk_posture})", axis=1)
    return simulation, thresholds


def run_forecast_analysis(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    """Run holdout scoring, regression comparison, and saved forecast plots."""
    metrics = evaluate_holdout(frame)
    regression_metrics, coefficients = run_regression_baseline(frame)
    ridge_metrics, ridge_coefficients = run_ridge_baseline(frame)
    rolling_metrics = evaluate_rolling_validation(frame)
    paths = plot_forecast_comparison(frame, metrics, output_dir, rolling_metrics)
    cutoff = frame.index.max() - pd.Timedelta(days=56) + pd.Timedelta(minutes=15)
    start = cutoff + pd.Timedelta(days=(7 - cutoff.dayofweek) % 7)
    end = start + pd.Timedelta(days=6, hours=23, minutes=45)
    actual = frame.loc[start:end, "load_kw"]
    forecast = seasonal_forecast(frame, 1, .8, cutoff).loc[actual.index]
    return {
        "metrics": metrics,
        "regression_metrics": regression_metrics,
        "coefficients": coefficients,
        "ridge_metrics": ridge_metrics,
        "ridge_coefficients": ridge_coefficients,
        "rolling_metrics": rolling_metrics,
        "plot_paths": paths,
        "actual": actual,
        "forecast": forecast,
    }


def run_pipeline(input_path: str | Path, output_dir: str | Path = "outputs") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared = add_calendar_features(load_and_prepare(input_path))
    metrics = evaluate_holdout(prepared)
    summary = peak_summary(prepared)
    prepared.to_csv(out / "prepared_load.csv")
    metrics.to_csv(out / "forecast_metrics.csv", index=False)
    pd.Series(summary, name="value").to_csv(out / "eda_summary.csv")
    return prepared, metrics, summary
