"""Reusable pipeline for the BESS peak-shaving case study."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY
HOLDOUT_DAYS = 56

# Illustrative-only: no client tariff was supplied with this case. Germany's
# demand-charge ("Leistungspreis") component commonly falls in this range for
# a C&I connection; replace with the customer's actual grid-tariff rate.
ILLUSTRATIVE_DEMAND_CHARGE_EUR_PER_KW_YEAR = 100.0
ILLUSTRATIVE_ARBITRAGE_EUR_PER_KWH = 0.10


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
    frame = load_and_prepare(data_path)
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


def _add_calendar_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day/weekday/holiday features shared by both feature sets below."""
    result = add_german_holiday_flag(frame)
    slots = result.index.hour * 4 + result.index.minute // 15
    result["hour_sin"] = np.sin(2 * np.pi * slots / SLOTS_PER_DAY)
    result["hour_cos"] = np.cos(2 * np.pi * slots / SLOTS_PER_DAY)
    result["weekday_flag"] = (result.index.dayofweek < 5).astype(float)
    result["holiday_flag"] = result["is_public_holiday"].astype(float)
    return result


def build_regression_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build load-only, leakage-safe calendar, lag, and rolling features.

    Includes short-horizon lags (last 15/60 minutes), so this feature set is
    only informative at the operational 15 min-4 h horizons -- see
    ``build_day_ahead_features`` for horizons where those go stale.
    """
    result = _add_calendar_flags(frame)
    for lag in (1, 4, SLOTS_PER_DAY, SLOTS_PER_WEEK):
        result[f"lag_{lag}_kw"] = result["load_kw"].shift(lag)
    result["rolling_1h_mean_kw"] = result["load_kw"].shift(1).rolling(4).mean()
    result["rolling_24h_mean_kw"] = result["load_kw"].shift(1).rolling(SLOTS_PER_DAY).mean()
    return result


def build_day_ahead_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build calendar and seasonal-lag features that stay valid a day ahead.

    Drops the short-horizon lags and rolling means from
    ``build_regression_features``: "what happened in the last hour" carries
    no signal 24 hours out. Keeps only same-day-yesterday (``lag_96_kw``),
    same-time-last-week (``lag_672_kw``), and calendar features, none of
    which depend on recent observations.
    """
    result = _add_calendar_flags(frame)
    result["lag_96_kw"] = result["load_kw"].shift(SLOTS_PER_DAY)
    result["lag_672_kw"] = result["load_kw"].shift(SLOTS_PER_WEEK)
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


def evaluate_holdout(frame: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> pd.DataFrame:
    """Evaluate target times reached from origins in the final time block.

    The seasonal lag is known at each forecast origin. For horizon ``h``, the
    first scored target is ``holdout_start + h``; this prevents the horizon
    labels from scoring exactly the same target set.
    """
    cutoff = holdout_cutoff(frame, holdout_days)
    training_load = frame.loc[(frame.index < cutoff) & frame["is_scoring_eligible"], "load_kw"]
    peak_cutoff = float(training_load.quantile(.90))
    rows: list[dict[str, float | str]] = []
    for horizon in (1, 4, 16):
        for quantile, label in ((0.0, "weekly_seasonal_naive"), (0.8, "conservative_q80")):
            origins = frame.index[(frame.index >= cutoff) & (frame.index <= frame.index.max() - pd.Timedelta(minutes=15 * horizon))]
            target_index = origins + pd.Timedelta(minutes=15 * horizon)
            forecast = seasonal_forecast(frame, horizon_slots=horizon, residual_quantile=quantile, train_end=cutoff).reindex(target_index)
            actual = frame["load_kw"].reindex(target_index)
            eligible = frame["is_scoring_eligible"].reindex(target_index).fillna(False)
            metrics = asymmetric_metrics(actual[eligible], forecast[eligible], peak_cutoff_kw=peak_cutoff)
            rows.append({"horizon_15min": horizon, "model": label, **metrics})
        ridge = ridge_forecast(frame, horizon_slots=horizon, train_end=cutoff, target_index=target_index)
        ridge_metrics = asymmetric_metrics(actual[eligible], ridge[eligible], peak_cutoff_kw=peak_cutoff)
        rows.append({"horizon_15min": horizon, "model": "ridge_load_calendar", **ridge_metrics})
        gbm = gbm_quantile_forecast(frame, horizon_slots=horizon, quantile=0.8, train_end=cutoff, target_index=target_index)
        gbm_metrics = asymmetric_metrics(actual[eligible], gbm[eligible], peak_cutoff_kw=peak_cutoff)
        rows.append({"horizon_15min": horizon, "model": "gbm_quantile", **gbm_metrics})
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


def holdout_cutoff(frame: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> pd.Timestamp:
    """First scored timestamp of the chronological holdout block."""
    return frame.index.max() - pd.Timedelta(days=holdout_days) + pd.Timedelta(minutes=15)


def select_predeclared_holdout_week(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    search_start_offset_days: int = HOLDOUT_DAYS // 2,
    search_limit: pd.Timestamp = pd.Timestamp("2025-12-15"),
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first complete Monday-Sunday week starting well inside the holdout.

    The search begins ``search_start_offset_days`` after cutoff rather than
    immediately at it: the week right next to the train/test boundary is the
    most favorable position a frozen model's recency-based lag features can
    be in, and would make results look stronger than a normal week deeper
    into the holdout. Offsetting by half the holdout window keeps the week
    a genuine mid-holdout example without reaching for a specific one.
    Selected by calendar completeness alone, never by load values, so the
    comparison week still cannot be cherry-picked to flatter or disadvantage
    any model. Shared by every plot and demo that needs "the" holdout week,
    so they always agree on which week that is -- including dispatch.
    """
    search_start = cutoff + pd.Timedelta(days=search_start_offset_days)
    start = search_start + pd.Timedelta(days=(7 - search_start.dayofweek) % 7)
    while start <= search_limit:
        candidate_index = pd.date_range(start, periods=SLOTS_PER_WEEK, freq="15min")
        if frame["load_kw"].reindex(candidate_index).notna().all():
            break
        start += pd.Timedelta(days=7)
    end = start + pd.Timedelta(days=6, hours=23, minutes=45)
    return start, end


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
    ridge_week: pd.Series | None = None,
    gbm_week: pd.Series | None = None,
) -> tuple[Path, ...]:
    """Save validation error, model comparison, and a holdout-week comparison.

    ``ridge_week``/``gbm_week`` let a caller that already fit these models for
    the same predeclared week (e.g. ``run_forecast_analysis``) pass the
    forecasts through instead of paying for a second, redundant fit.
    """
    import matplotlib.pyplot as plt

    cutoff = holdout_cutoff(frame)
    comparison_start, comparison_end = select_predeclared_holdout_week(frame, cutoff)
    actual = frame.loc[comparison_start:comparison_end, "load_kw"]
    naive = seasonal_forecast(frame, 1, 0.0, cutoff).loc[actual.index]
    conservative = seasonal_forecast(frame, 1, .8, cutoff).loc[actual.index]
    ridge = ridge_week if ridge_week is not None else ridge_forecast(frame, horizon_slots=1, train_end=cutoff, target_index=actual.index)
    gbm = gbm_week if gbm_week is not None else gbm_quantile_forecast(frame, horizon_slots=1, quantile=0.8, train_end=cutoff, target_index=actual.index)
    model_order = ["weekly_seasonal_naive", "conservative_q80", "ridge_load_calendar", "gbm_quantile"]
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
        colors = {"weekly_seasonal_naive": "#377eb8", "conservative_q80": "#e41a1c", "ridge_load_calendar": "#1b9e77", "gbm_quantile": "#ff7f00"}
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
    # One panel per model (actual + that model only) instead of five lines
    # stacked on one axis, so each forecast's behavior is easy to trace on
    # its own rather than lost in overlapping color/alpha.
    panels = [
        ("Weekly seasonal-naive", naive, "#377eb8"),
        ("Conservative q80", conservative, "#e41a1c"),
        ("Ridge load + calendar", ridge, "#1b9e77"),
        ("GBM quantile", gbm, "#ff7f00"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharex=True, sharey=True, constrained_layout=True)
    for panel_axis, (name, series, color) in zip(axes.flat, panels):
        actual.plot(ax=panel_axis, color="black", lw=1.1, label="actual")
        series.plot(ax=panel_axis, color=color, lw=1.3, label=name)
        panel_axis.set_title(name, fontsize=11)
        panel_axis.set_ylabel("kW")
        panel_axis.legend(loc="upper left", fontsize=8, framealpha=0.85)
    fig.suptitle(f"Predeclared holdout week: {comparison_start:%d %b} to {comparison_end:%d %b %Y}")
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


def ridge_forecast(
    frame: pd.DataFrame,
    horizon_slots: int = 1,
    train_end: pd.Timestamp | None = None,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Forecast targets from origin features, fitting only before ``train_end``."""
    data = build_regression_features(frame)
    columns = [column for column in data.columns if column.startswith(("lag_", "rolling_"))] + ["hour_sin", "hour_cos", "weekday_flag", "holiday_flag"]
    target = data["load_kw"].shift(-horizon_slots)
    target_eligible = frame["is_scoring_eligible"].astype("boolean").shift(-horizon_slots).fillna(False).astype(bool)
    data = data.assign(target=target, target_eligible=target_eligible).dropna(subset=columns + ["target"])
    cutoff = train_end or holdout_cutoff(frame)
    # Gate on the *target* time, not the origin: an origin just before cutoff
    # with a multi-step horizon still has a target inside [cutoff, train_end),
    # i.e. a label that would not exist yet at the moment this model is
    # frozen for deployment. Excluding it keeps every horizon leakage-free.
    target_time = data.index + pd.Timedelta(minutes=15 * horizon_slots)
    train = (target_time < cutoff) & frame.loc[data.index, "is_scoring_eligible"].to_numpy() & data["target_eligible"].to_numpy()
    mean = data.loc[train, columns].mean()
    scale = data.loc[train, columns].std().replace(0, 1)
    x_train = (data.loc[train, columns] - mean) / scale
    x_all = (data[columns] - mean) / scale
    design_train = np.c_[np.ones(len(x_train)), x_train.to_numpy()]
    penalty = np.diag([0.0] + [10.0] * len(columns))
    coefficients = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ data.loc[train, "target"].to_numpy())
    prediction = pd.Series(np.c_[np.ones(len(x_all)), x_all.to_numpy()] @ coefficients, index=data.index)
    if target_index is None:
        target_index = frame.index[frame.index >= cutoff]
    origin_index = target_index - pd.Timedelta(minutes=15 * horizon_slots)
    return prediction.reindex(origin_index).set_axis(target_index)


def _fit_gbm_quantile(
    frame: pd.DataFrame,
    data: pd.DataFrame,
    columns: list[str],
    horizon_slots: int,
    quantile: float,
    train_end: pd.Timestamp | None,
    target_index: pd.DatetimeIndex | None,
) -> pd.Series:
    """Shared leakage-safe LightGBM quantile fit/predict routine.

    Gates training on the *target* time, not the origin: an origin just
    before cutoff with a multi-step horizon still has a target inside
    ``[cutoff, train_end)``, i.e. a label that would not exist yet at the
    moment this model is frozen for deployment. Excluding it keeps every
    horizon and every feature set leakage-free.
    """
    import lightgbm as lgb

    target = data["load_kw"].shift(-horizon_slots)
    target_eligible = frame["is_scoring_eligible"].astype("boolean").shift(-horizon_slots).fillna(False).astype(bool)
    data = data.assign(target=target, target_eligible=target_eligible).dropna(subset=columns + ["target"])
    cutoff = train_end or holdout_cutoff(frame)
    target_time = data.index + pd.Timedelta(minutes=15 * horizon_slots)
    train = (target_time < cutoff) & frame.loc[data.index, "is_scoring_eligible"].to_numpy() & data["target_eligible"].to_numpy()

    params = {
        "objective": "quantile",
        "alpha": quantile,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    train_set = lgb.Dataset(data.loc[train, columns], label=data.loc[train, "target"])
    booster = lgb.train(params, train_set, num_boost_round=300)
    prediction = pd.Series(booster.predict(data[columns]), index=data.index)

    if target_index is None:
        target_index = frame.index[frame.index >= cutoff]
    origin_index = target_index - pd.Timedelta(minutes=15 * horizon_slots)
    return prediction.reindex(origin_index).set_axis(target_index)


def gbm_quantile_forecast(
    frame: pd.DataFrame,
    horizon_slots: int = 1,
    quantile: float = 0.8,
    train_end: pd.Timestamp | None = None,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Forecast a conservative quantile of load directly from origin features.

    Same leakage-safe origin/target construction as ``ridge_forecast``, but
    trained on pinball loss so the model targets the upper quantile itself
    instead of shifting a mean forecast by a flat historical uplift. Uses
    the operational feature set (short lags included) -- valid at 15 min-4 h
    horizons; see ``gbm_day_ahead_forecast`` for longer horizons.
    """
    data = build_regression_features(frame)
    columns = [column for column in data.columns if column.startswith(("lag_", "rolling_"))] + ["hour_sin", "hour_cos", "weekday_flag", "holiday_flag"]
    return _fit_gbm_quantile(frame, data, columns, horizon_slots, quantile, train_end, target_index)


def gbm_day_ahead_forecast(
    frame: pd.DataFrame,
    horizon_slots: int = SLOTS_PER_DAY,
    quantile: float = 0.8,
    train_end: pd.Timestamp | None = None,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Fit a GBM quantile forecast using only day-ahead-valid features.

    Same leakage-safe fit routine as ``gbm_quantile_forecast``, but built on
    ``build_day_ahead_features`` (same-day-yesterday and same-time-last-week
    lags plus calendar features only, no short lags) on the theory that
    short-horizon features are stale a day out.

    Tested at the 1-day horizon (96 slots), this does **not** solve
    day-ahead forecasting: it underperforms both the operational feature
    set and the trivial weekly-seasonal-naive baseline on every metric
    (worse MAE, worse peak under-forecast rate). Dropping the short lags
    removes real, if indirect, signal about the site's current operating
    regime without anything of comparable value to replace it. Genuine
    day-ahead accuracy needs additional inputs (weather, a production
    schedule) or a different kind of approach (e.g. forecasting the day's
    shape/peak risk rather than each 15-minute point), not a feature-set
    change to the same short-horizon model. Kept here, tested, and
    documented as evidence for that conclusion, not as a recommended
    forecaster.
    """
    data = build_day_ahead_features(frame)
    columns = [column for column in data.columns if column.startswith(("lag_", "rolling_"))] + ["hour_sin", "hour_cos", "weekday_flag", "holiday_flag"]
    return _fit_gbm_quantile(frame, data, columns, horizon_slots, quantile, train_end, target_index)


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
            "ridge_load_calendar": ridge_forecast(frame, horizon_slots=1, train_end=cutoff),
            "gbm_quantile": gbm_quantile_forecast(frame, horizon_slots=1, quantile=0.8, train_end=cutoff),
        }
        for name, forecast in forecasts.items():
            rows.append({"fold": fold, "model": name, **asymmetric_metrics(actual[eligible], forecast.reindex(target_index)[eligible], peak_cutoff_kw=peak_cutoff)})
    return pd.DataFrame(rows)


def dispatch(load_kw: pd.Series, forecast_kw: pd.Series, threshold_kw: float = 80.0, battery_kwh: float = 100.0, power_kw: float = 50.0, efficiency: float = .92, initial_soc_kwh: float | None = None, charge_power_kw: float | None = None) -> pd.DataFrame:
    """Charge below and discharge above a threshold with explicit SOC tracking."""
    load_kw = pd.Series(load_kw).astype(float)
    forecast_kw = pd.Series(forecast_kw, index=load_kw.index).astype(float)
    charge_power_kw = power_kw if charge_power_kw is None else charge_power_kw
    soc = battery_kwh if initial_soc_kwh is None else min(initial_soc_kwh, battery_kwh)
    rows = []
    for timestamp in load_kw.index:
        forecast_at_timestamp = forecast_kw.loc[timestamp]
        if pd.isna(forecast_at_timestamp):
            # No forecast for this interval (e.g. a rolling-window feature
            # crossed a data gap upstream): hold position rather than act on
            # missing information. Critically, this keeps `soc` numeric --
            # letting NaN through here would poison every later timestep,
            # since `float('nan')` propagates through all subsequent
            # arithmetic and Python's max(nan, 0) returns nan, not 0.
            rows.append((timestamp, 0.0, 0.0, soc))
            continue
        requested_discharge_kw = min(max(forecast_at_timestamp - threshold_kw, 0), power_kw)
        requested_discharge_kw = min(requested_discharge_kw, max(load_kw.loc[timestamp], 0))
        max_from_soc_kw = soc * efficiency / .25
        discharge_kw = min(requested_discharge_kw, max_from_soc_kw)
        energy_from_soc = discharge_kw * .25 / efficiency
        soc -= energy_from_soc
        forecast_headroom_kw = max(threshold_kw - forecast_at_timestamp, 0)
        observed_headroom_kw = max(threshold_kw - load_kw.loc[timestamp], 0)
        requested_charge_kw = min(forecast_headroom_kw, observed_headroom_kw, charge_power_kw)
        max_into_soc_kw = max((battery_kwh - soc) / (.25 * efficiency), 0)
        charge_kw = min(requested_charge_kw, max_into_soc_kw) if discharge_kw == 0 else 0.0
        soc += charge_kw * .25 * efficiency
        rows.append((timestamp, charge_kw, discharge_kw, soc))
    dispatch_data = pd.DataFrame(rows, columns=["timestamp", "charge_kw", "discharge_kw", "soc_kwh"]).set_index("timestamp")
    return pd.DataFrame({"load_kw": load_kw, "forecast_kw": forecast_kw}).join(dispatch_data).assign(net_load_kw=lambda data: data["load_kw"] + data["charge_kw"] - data["discharge_kw"], energy_charged_kwh=lambda data: data["charge_kw"] * .25, energy_discharged_kwh=lambda data: data["discharge_kw"] * .25)


def run_dispatch_demo(frame: pd.DataFrame, forecast: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the illustrative dispatch demo on the predeclared holdout week."""
    cutoff = holdout_cutoff(frame)
    start, end = select_predeclared_holdout_week(frame, cutoff)
    actual = frame.loc[start:end, "load_kw"]
    simulation = dispatch(actual, forecast.loc[actual.index])
    peak_before_kw = float(actual.max())
    thresholds = pd.DataFrame({"threshold_kw": [70, 80, 90], "risk_posture": ["aggressive protection", "balanced", "arbitrage preserving"]})
    thresholds["peak_before_kw"] = peak_before_kw
    thresholds["peak_after_dispatch_kw"] = [dispatch(actual, forecast.loc[actual.index], threshold).net_load_kw.max() for threshold in thresholds["threshold_kw"]]
    thresholds["peak_reduction_kw"] = thresholds["peak_before_kw"] - thresholds["peak_after_dispatch_kw"]
    thresholds["energy_discharged_kwh"] = [dispatch(actual, forecast.loc[actual.index], threshold).energy_discharged_kwh.sum() for threshold in thresholds["threshold_kw"]]
    thresholds["energy_charged_kwh"] = [dispatch(actual, forecast.loc[actual.index], threshold).energy_charged_kwh.sum() for threshold in thresholds["threshold_kw"]]
    thresholds["illustrative_arbitrage_value_eur"] = thresholds["energy_discharged_kwh"] * ILLUSTRATIVE_ARBITRAGE_EUR_PER_KWH
    thresholds["illustrative_demand_charge_value_eur_per_year"] = thresholds["peak_reduction_kw"] * ILLUSTRATIVE_DEMAND_CHARGE_EUR_PER_KW_YEAR
    thresholds["rule"] = thresholds.apply(lambda row: f"Discharge whenever the forecast exceeds {row.threshold_kw} kW ({row.risk_posture}); charge whenever both forecast and actual load are below it. Runs every interval, 24/7 -- no time-of-day or weekday restriction.", axis=1)
    return simulation, thresholds


def plot_dispatch_simulation(
    simulation: pd.DataFrame,
    output_dir: str | Path,
    threshold_kw: float = 80.0,
    forecast_label: str = "GBM quantile forecast",
    filename: str = "06_dispatch_over_time.png",
) -> Path:
    """Plot load, forecast, dispatch, and SOC for the simulated week."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    load_axis, battery_axis = axes
    simulation["load_kw"].plot(ax=load_axis, color="black", lw=1, label="actual load")
    simulation["forecast_kw"].plot(ax=load_axis, color="#e41a1c", alpha=.8, label=forecast_label)
    load_axis.axhline(threshold_kw, color="#d95f02", ls="--", label=f"threshold ({threshold_kw:.0f} kW)")
    load_axis.set_ylabel("Load (kW)")
    load_axis.set_title("Load and forecast")
    load_axis.legend(loc="upper left")
    simulation["charge_kw"].plot(ax=battery_axis, color="#984ea3", alpha=.75, label="battery charge")
    simulation["discharge_kw"].plot(ax=battery_axis, color="#1b9e77", alpha=.75, label="battery discharge")
    battery_axis.set_ylabel("Battery power (kW)")
    battery_axis.set_title("Battery operation")
    soc_axis = battery_axis.twinx()
    simulation["soc_kwh"].plot(ax=soc_axis, color="#7570b3", alpha=.75, label="battery SOC")
    soc_axis.set_ylabel("SOC (kWh)")
    handles, labels = battery_axis.get_legend_handles_labels()
    secondary_handles, secondary_labels = soc_axis.get_legend_handles_labels()
    battery_axis.legend(handles + secondary_handles, labels + secondary_labels, loc="upper left")
    figure.suptitle(f"Dispatch simulation over holdout week: {simulation.index.min():%d %b} to {simulation.index.max():%d %b %Y}")
    return save_plot(figure, output_dir, filename)


def run_forecast_analysis(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    """Run holdout scoring and saved forecast plots."""
    metrics = evaluate_holdout(frame)
    rolling_metrics = evaluate_rolling_validation(frame)
    cutoff = holdout_cutoff(frame)
    start, end = select_predeclared_holdout_week(frame, cutoff)
    actual = frame.loc[start:end, "load_kw"]
    forecast = seasonal_forecast(frame, 1, .8, cutoff).loc[actual.index]
    ridge_forecast_week = ridge_forecast(frame, horizon_slots=1, train_end=cutoff, target_index=actual.index)
    gbm_forecast = gbm_quantile_forecast(frame, horizon_slots=1, quantile=0.8, train_end=cutoff, target_index=actual.index)
    # Reuse this week's already-fit ridge/GBM forecasts instead of letting
    # plot_forecast_comparison refit them a second time for the same week.
    paths = plot_forecast_comparison(
        frame, metrics, output_dir, rolling_metrics,
        ridge_week=ridge_forecast_week, gbm_week=gbm_forecast,
    )
    ridge_metrics = metrics.loc[
        (metrics["horizon_15min"] == 1) & (metrics["model"] == "ridge_load_calendar")
    ].iloc[0].to_dict()
    gbm_metrics = metrics.loc[
        (metrics["horizon_15min"] == 1) & (metrics["model"] == "gbm_quantile")
    ].iloc[0].to_dict()
    return {
        "metrics": metrics,
        "ridge_metrics": ridge_metrics,
        "gbm_metrics": gbm_metrics,
        "rolling_metrics": rolling_metrics,
        "plot_paths": paths,
        "actual": actual,
        "forecast": forecast,
        "gbm_forecast": gbm_forecast,
    }


def run_pipeline(input_path: str | Path, output_dir: str | Path = "outputs") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared = load_and_prepare(input_path)
    metrics = evaluate_holdout(prepared)
    summary = peak_summary(prepared)
    prepared.to_csv(out / "prepared_load.csv")
    metrics.to_csv(out / "forecast_metrics.csv", index=False)
    pd.Series(summary, name="value").to_csv(out / "eda_summary.csv")
    return prepared, metrics, summary
