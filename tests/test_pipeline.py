from pathlib import Path

import pandas as pd

from src.bess_forecasting import (
    asymmetric_metrics,
    dispatch,
    evaluate_holdout,
    load_and_prepare,
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


def test_dispatch_respects_soc_and_energy_balance():
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    actual = pd.Series(100.0, index=index)
    forecast = pd.Series(100.0, index=index)
    result = dispatch(actual, forecast, threshold_kw=0, battery_kwh=10, power_kw=20)
    assert result.soc_kwh.ge(0).all()
    assert result.discharge_kw.le(20).all()
    assert result.energy_discharged_kwh.sum() <= 10 * .92 + 1e-9
