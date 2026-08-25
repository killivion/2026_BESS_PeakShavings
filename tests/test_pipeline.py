from pathlib import Path

import pandas as pd

from src.bess_forecasting import asymmetric_metrics, dispatch, evaluate_holdout, load_and_prepare


DATA = Path(__file__).parents[1] / "load_timeseries_2025_case_study.csv"


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
