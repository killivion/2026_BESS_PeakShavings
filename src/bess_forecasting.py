"""Reusable pipeline for the BESS peak-shaving case study."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY


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
        history = frame.loc[frame.index < train_end, "load_kw"] if train_end is not None else frame["load_kw"]
        residual = history - history.shift(SLOTS_PER_WEEK)
        uplift = residual.dropna().quantile(residual_quantile)
    else:
        uplift = 0.0
    return base + uplift


def asymmetric_metrics(actual: pd.Series, forecast: pd.Series, peak_quantile: float = 0.90) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), forecast.rename("forecast")], axis=1).dropna()
    error = aligned["forecast"] - aligned["actual"]
    under = error < 0
    peak_cutoff = aligned["actual"].quantile(peak_quantile)
    peak = aligned["actual"] >= peak_cutoff
    return {
        "n": float(len(aligned)),
        "mae_kw": float(error.abs().mean()),
        "rmse_kw": float(np.sqrt((error.pow(2)).mean())),
        "underforecast_rate": float(under.mean()),
        "underforecast_mae_kw": float((-error[under]).mean()) if under.any() else 0.0,
        "peak_underforecast_rate": float((under & peak).sum() / peak.sum()) if peak.any() else 0.0,
        "peak_underforecast_mae_kw": float((-error[under & peak]).mean()) if (under & peak).any() else 0.0,
        "weighted_absolute_error_kw": float((error.abs() * np.where(error < 0, 2.0, 1.0)).mean()),
        "peak_cutoff_kw": float(peak_cutoff),
    }


def evaluate_holdout(frame: pd.DataFrame, holdout_days: int = 56) -> pd.DataFrame:
    """Evaluate one-step and 4-hour-ahead forecasts on the final time block."""
    cutoff = frame.index.max() - pd.Timedelta(days=holdout_days) + pd.Timedelta(minutes=15)
    rows: list[dict[str, float | str]] = []
    for horizon in (1, 4, 16):
        for quantile, label in ((0.0, "weekly_seasonal_naive"), (0.8, "conservative_q80")):
            forecast = seasonal_forecast(frame, horizon, quantile, cutoff)
            mask = (frame.index >= cutoff) & frame["is_scoring_eligible"]
            metrics = asymmetric_metrics(frame.loc[mask, "load_kw"], forecast.loc[mask])
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


def plot_forecast_comparison(frame: pd.DataFrame, metrics: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Save horizon error and a representative holdout-week comparison."""
    import matplotlib.pyplot as plt

    cutoff = frame.index.max() - pd.Timedelta(days=56) + pd.Timedelta(minutes=15)
    first_monday = cutoff + pd.Timedelta(days=(7 - cutoff.dayofweek) % 7)
    typical_start, typical_end, _ = select_representative_week(
        frame, first_start=first_monday.strftime("%Y-%m-%d"), last_start="2025-12-15"
    )
    actual = frame.loc[typical_start:typical_end, "load_kw"]
    naive = seasonal_forecast(frame, 1, 0.0, cutoff).loc[actual.index]
    conservative = seasonal_forecast(frame, 1, .8, cutoff).loc[actual.index]
    horizon = metrics.groupby(["horizon_15min", "model"], as_index=False).weighted_absolute_error_kw.first()
    fig, axis = plt.subplots(figsize=(8, 4))
    for name, group in horizon.groupby("model"):
        axis.plot(group.horizon_15min * 15 / 60, group.weighted_absolute_error_kw, marker="o", label=name)
    axis.set_xticks([.25, 1, 4], ["15m", "1h", "4h"])
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Cost-weighted absolute error (kW)")
    axis.set_title("Error trade-off by operational horizon")
    axis.legend()
    first = save_plot(fig, output_dir, "03_horizon_error_tradeoff.png")
    fig, axis = plt.subplots(figsize=(14, 4))
    actual.plot(ax=axis, color="black", lw=1, label="actual")
    naive.plot(ax=axis, color="#377eb8", alpha=.8, label="weekly naive")
    conservative.plot(ax=axis, color="#e41a1c", alpha=.8, label="conservative q80")
    axis.set_ylabel("kW")
    axis.set_title(f"Typical week comparison: {typical_start:%d %b} to {typical_end:%d %b %Y}")
    axis.legend()
    second = save_plot(fig, output_dir, "04_typical_week_forecast_comparison.png")
    return first, second


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
