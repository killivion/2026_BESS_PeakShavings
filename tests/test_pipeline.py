from pathlib import Path

import pandas as pd
import pytest

from src.bess_forecasting import (
    SLOTS_PER_DAY,
    asymmetric_metrics,
    dispatch,
    evaluate_holdout,
    gbm_day_ahead_forecast,
    gbm_quantile_forecast,
    holdout_cutoff,
    load_and_prepare,
    plot_dispatch_simulation,
    ridge_forecast,
    seasonal_forecast,
    summarize_rolling_metrics,
)


ROOT = Path(__file__).parents[1]
DATA = ROOT / "load_timeseries_2025_case_study.csv"
if not DATA.exists():
    DATA = ROOT / "case_inputs" / "load_timeseries_2025_case_study.csv"


def test_prepare_parses_locale_and_preserves_valid_zeros():
    frame = load_and_prepare(DATA)
    assert frame.index.freq == pd.tseries.frequencies.to_offset("15min")
    assert frame["load_kw"].eq(0).sum() > 3000
    assert frame["load_kw_raw"].gt(200).sum() == 4
    assert frame["was_interpolated"].sum() == 22


def test_asymmetric_metric_penalizes_underforecast_more():
    actual = pd.Series([100.0, 100.0])
    under = pd.Series([90.0, 90.0])
    over = pd.Series([110.0, 110.0])
    assert asymmetric_metrics(actual, under)["weighted_absolute_error_kw"] > asymmetric_metrics(actual, over)["weighted_absolute_error_kw"]


def test_asymmetric_metrics_reports_peak_recall_and_bias():
    actual = pd.Series([50.0, 100.0, 120.0, 130.0])
    forecast = pd.Series([50.0, 90.0, 105.0, 145.0])
    metrics = asymmetric_metrics(actual, forecast, peak_quantile=0.5)
    assert metrics["peak_recall"] == 0.5
    assert metrics["peak_forecast_bias_kw"] == 0.0


def test_peak_metrics_accept_a_cutoff_defined_before_evaluation():
    actual = pd.Series([80.0, 100.0])
    forecast = pd.Series([90.0, 90.0])
    metrics = asymmetric_metrics(actual, forecast, peak_cutoff_kw=95.0)
    assert metrics["peak_cutoff_kw"] == 95.0
    assert metrics["peak_recall"] == 0.0


def test_rolling_summary_averages_each_model_across_folds():
    rolling = pd.DataFrame(
        [
            {"fold": 1, "model": "ridge", "n": 10, "mae_kw": 10.0, "peak_recall": 0.5},
            {"fold": 2, "model": "ridge", "n": 10, "mae_kw": 14.0, "peak_recall": 1.0},
            {"fold": 1, "model": "naive", "n": 10, "mae_kw": 12.0, "peak_recall": 0.75},
        ]
    )
    summary = summarize_rolling_metrics(rolling)
    ridge = summary.loc[summary["model"] == "ridge"].iloc[0]
    assert ridge["folds"] == 2
    assert ridge["mae_kw"] == 12.0
    assert ridge["peak_recall"] == 0.75


def test_horizon_evaluation_starts_after_forecast_origin():
    frame = load_and_prepare(DATA)
    metrics = evaluate_holdout(frame)
    counts = metrics.groupby("horizon_15min")["n"].first()
    assert counts.loc[1] > counts.loc[4] > counts.loc[16]


def test_ridge_is_evaluated_at_each_horizon_without_future_target_features():
    frame = load_and_prepare(DATA)
    cutoff = holdout_cutoff(frame)
    target_index = pd.date_range(cutoff, periods=8, freq="15min")
    forecast = ridge_forecast(frame, horizon_slots=4, train_end=cutoff, target_index=target_index)
    assert forecast.index.equals(target_index)
    assert forecast.notna().all()
    scored = evaluate_holdout(frame)
    assert set(scored.loc[scored["model"] == "ridge_load_calendar", "horizon_15min"]) == {1, 4, 16}


def test_gbm_quantile_is_evaluated_at_each_horizon_and_conservative_vs_mean():
    frame = load_and_prepare(DATA)
    cutoff = holdout_cutoff(frame)
    target_index = pd.date_range(cutoff, periods=8, freq="15min")
    forecast = gbm_quantile_forecast(frame, horizon_slots=4, quantile=0.8, train_end=cutoff, target_index=target_index)
    assert forecast.index.equals(target_index)
    assert forecast.notna().all()
    scored = evaluate_holdout(frame)
    assert set(scored.loc[scored["model"] == "gbm_quantile", "horizon_15min"]) == {1, 4, 16}
    median_forecast = gbm_quantile_forecast(frame, horizon_slots=4, quantile=0.5, train_end=cutoff, target_index=target_index)
    assert forecast.mean() >= median_forecast.mean()


def test_gbm_day_ahead_features_run_but_do_not_beat_naive_at_one_day_horizon():
    """Documents an empirical finding, not a requirement to preserve.

    Dropping short-horizon lags in favor of only same-day-yesterday and
    same-time-last-week features does not fix day-ahead forecasting -- it
    underperforms even the trivial naive baseline here. See
    ``gbm_day_ahead_forecast``'s docstring for why. If a future change makes
    this pass with a *lower* MAE than naive, that's a genuine improvement
    worth noting explicitly -- update this test to say so rather than
    silently deleting it.
    """
    frame = load_and_prepare(DATA)
    cutoff = holdout_cutoff(frame)
    horizon = SLOTS_PER_DAY
    origins = frame.index[(frame.index >= cutoff) & (frame.index <= frame.index.max() - pd.Timedelta(minutes=15 * horizon))]
    target_index = origins + pd.Timedelta(minutes=15 * horizon)
    actual = frame["load_kw"].reindex(target_index)
    eligible = frame["is_scoring_eligible"].reindex(target_index).fillna(False)

    naive = seasonal_forecast(frame, horizon_slots=horizon, train_end=cutoff).reindex(target_index)
    day_ahead = gbm_day_ahead_forecast(frame, horizon_slots=horizon, quantile=0.8, train_end=cutoff, target_index=target_index)

    naive_mae = asymmetric_metrics(actual[eligible], naive[eligible])["mae_kw"]
    day_ahead_mae = asymmetric_metrics(actual[eligible], day_ahead[eligible])["mae_kw"]
    assert day_ahead_mae > naive_mae


def test_ridge_and_gbm_are_not_affected_by_future_load_values():
    """A model frozen at ``cutoff`` must be unaffected by anything at or after it.

    Query a fixed origin far enough before cutoff that its own features can't
    reach into the mutated region; any change in the prediction there can
    only come from the *trained model itself* having absorbed a label built
    from mutated (i.e. leaked) future data. This is a stronger check than
    scoring-set overlap: it directly proves the fit, not just the report.
    """
    frame = load_and_prepare(DATA)
    cutoff = holdout_cutoff(frame)
    safe_origin = pd.DatetimeIndex([cutoff - pd.Timedelta(days=30)])

    mutated = frame.copy()
    mutated.loc[mutated.index >= cutoff, "load_kw"] = 99999.0

    for horizon in (1, 4, 16):
        baseline_ridge = ridge_forecast(frame, horizon_slots=horizon, train_end=cutoff, target_index=safe_origin)
        perturbed_ridge = ridge_forecast(mutated, horizon_slots=horizon, train_end=cutoff, target_index=safe_origin)
        assert baseline_ridge.iloc[0] == pytest.approx(perturbed_ridge.iloc[0]), f"ridge leaks future data at horizon={horizon}"

        baseline_gbm = gbm_quantile_forecast(frame, horizon_slots=horizon, quantile=0.8, train_end=cutoff, target_index=safe_origin)
        perturbed_gbm = gbm_quantile_forecast(mutated, horizon_slots=horizon, quantile=0.8, train_end=cutoff, target_index=safe_origin)
        assert baseline_gbm.iloc[0] == pytest.approx(perturbed_gbm.iloc[0]), f"gbm leaks future data at horizon={horizon}"


def test_dispatch_holds_position_and_preserves_soc_when_forecast_is_missing():
    """A single missing forecast value must not corrupt the rest of the run.

    Regression test: ``float('nan')`` propagates through Python's own
    max()/min() (max(nan, 0) is nan, not 0), so one NaN forecast value used
    to permanently poison the running SOC variable for every later
    timestep -- silently disabling the SOC-based discharge cap and
    corrupting the SOC trace for the remainder of the simulated week.
    """
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    actual = pd.Series(100.0, index=index)
    forecast = pd.Series(100.0, index=index)
    forecast.iloc[2] = float("nan")
    # A large battery relative to power_kw so it never fully depletes --
    # isolates "did the NaN step break resumption" from "the battery ran out".
    result = dispatch(actual, forecast, threshold_kw=80, battery_kwh=100, power_kw=20)
    assert result["soc_kwh"].notna().all()
    assert result.loc[index[2], "charge_kw"] == 0.0
    assert result.loc[index[2], "discharge_kw"] == 0.0
    # SOC at the missing interval should be unchanged from the step before.
    assert result.loc[index[2], "soc_kwh"] == pytest.approx(result.loc[index[1], "soc_kwh"])
    # Dispatch should resume normally afterwards, not stay stuck at zero.
    assert result.loc[index[3]:, "discharge_kw"].gt(0).any()


def test_dispatch_respects_soc_and_energy_balance():
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    actual = pd.Series(100.0, index=index)
    forecast = pd.Series(100.0, index=index)
    result = dispatch(actual, forecast, threshold_kw=0, battery_kwh=10, power_kw=20)
    assert result.soc_kwh.ge(0).all()
    assert result.discharge_kw.le(20).all()
    assert result.energy_discharged_kwh.sum() <= 10 * .92 + 1e-9


def test_dispatch_can_charge_when_forecast_is_below_threshold():
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    actual = pd.Series(20.0, index=index)
    forecast = pd.Series(20.0, index=index)
    result = dispatch(actual, forecast, threshold_kw=80, battery_kwh=10, power_kw=20, initial_soc_kwh=0)
    assert result.charge_kw.gt(0).any()
    assert result.discharge_kw.eq(0).all()
    assert result.soc_kwh.iloc[-1] > result.soc_kwh.iloc[0]
    assert result.net_load_kw.le(80).all()
    assert result.soc_kwh.iloc[-1] <= 10
    assert result.energy_charged_kwh.sum() <= 10 / .92 + 1e-9


def test_dispatch_plot_is_saved(tmp_path):
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    actual = pd.Series(100.0, index=index)
    forecast = pd.Series(100.0, index=index)
    simulation = dispatch(actual, forecast, threshold_kw=80, battery_kwh=10, power_kw=20)
    path = plot_dispatch_simulation(simulation, tmp_path, threshold_kw=80)
    assert path.name == "06_dispatch_over_time.png"
    assert path.exists()
