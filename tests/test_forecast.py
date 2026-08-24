"""Tests for the price loader and the forecasting models.

These cover the parts of the system that produce numbers a user might act on.
The tests check properties that must hold regardless of the data — a continuous
calendar, a band that widens with the horizon, a scenario that moves price the
right way — rather than pinning exact values, which would break on every data
refresh without catching a single real defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neftegaz.forecast.data import load_prices_from_frame
from neftegaz.forecast.models import arima_forecast, forecast, simple_exponential_smoothing
from neftegaz.tools.forecast_tool import PRICE_ELASTICITY, apply_supply_scenario


# ── loader ─────────────────────────────────────────────────────────────────


def test_loader_sorts_deduplicates_and_fills_calendar():
    """Unsorted input, a duplicate date and a gap all normalise to one series."""
    raw = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-01", "2026-01-01", "2026-01-08"],
            "close": ["78.5", "76.25", "", "80.0"],
        }
    )
    result = load_prices_from_frame(raw)

    assert list(result.columns) == ["close"]
    assert result.index.name == "date"
    assert result.index.is_monotonic_increasing
    # 1 through 8 January inclusive, with 2, 3, 4, 6, 7 filled in.
    assert len(result) == 8
    assert result["close"].notna().all()
    assert result["close"].dtype == "float64"


def test_loader_keeps_last_duplicate():
    """A revised row appended after the original must win."""
    raw = pd.DataFrame(
        {"date": ["2026-01-01", "2026-01-02", "2026-01-01"], "close": ["70.0", "71.0", "99.0"]}
    )
    result = load_prices_from_frame(raw)
    assert result.loc["2026-01-01", "close"] == 99.0


def test_loader_backward_fills_leading_gap():
    """A blank first observation is filled from the future, not left as NaN."""
    raw = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "close": ["", "71.0"]})
    result = load_prices_from_frame(raw)
    assert result["close"].iloc[0] == 71.0


def test_loader_rejects_missing_columns():
    with pytest.raises(KeyError):
        load_prices_from_frame(pd.DataFrame({"date": ["2026-01-01"], "price": [70.0]}))


# ── models ─────────────────────────────────────────────────────────────────


@pytest.fixture
def series() -> pd.Series:
    index = pd.date_range("2025-01-01", periods=200, freq="D", name="date")
    # A gentle upward drift plus noise: enough structure for ARIMA to fit,
    # enough noise that a zero-variance band would be obviously wrong.
    rng = np.random.default_rng(seed=20260824)
    values = 70 + np.linspace(0, 10, 200) + rng.normal(0, 1.5, 200)
    return pd.Series(values, index=index, name="close")


def test_ses_band_widens_with_horizon(series):
    result = simple_exponential_smoothing(series, horizon=30)
    widths = result.frame["upper"] - result.frame["lower"]
    assert widths.is_monotonic_increasing
    assert (widths > 0).all()


def test_ses_point_forecast_is_flat(series):
    """A level model has no trend term; the flat line is the method, not a bug."""
    result = simple_exponential_smoothing(series, horizon=10)
    assert result.frame["forecast"].nunique() == 1


def test_forecast_frame_shape_is_uniform_across_methods(series):
    """The agent swaps methods without knowing which ran, so shapes must match."""
    ses = simple_exponential_smoothing(series, horizon=15).frame
    arima = arima_forecast(series, horizon=15).frame
    assert list(ses.columns) == list(arima.columns) == ["forecast", "lower", "upper"]
    assert ses.index.equals(arima.index)
    assert ses.index.name == "date"


def test_forecast_continues_the_day_after_history(series):
    result = forecast(series, horizon=5)
    assert result.frame.index[0] == series.index[-1] + pd.Timedelta(days=1)


def test_band_contains_point_estimate(series):
    result = forecast(series, horizon=20)
    assert (result.frame["lower"] <= result.frame["forecast"]).all()
    assert (result.frame["forecast"] <= result.frame["upper"]).all()


def test_auto_falls_back_when_arima_cannot_fit():
    """Two identical points: ARIMA has nothing to identify, SES still answers."""
    index = pd.date_range("2026-01-01", periods=2, freq="D", name="date")
    flat = pd.Series([80.0, 80.0], index=index)
    result = forecast(flat, horizon=3)
    assert len(result.frame) == 3
    # Every column, not just the point estimate: a NaN band next to a finite
    # forecast is exactly the shape of defect this test exists to catch, and
    # checking only `forecast` let it through once already.
    assert np.isfinite(result.frame.to_numpy()).all()


@pytest.mark.parametrize("length", [2, 3, 5])
def test_band_is_finite_on_short_series(length):
    """The sample sigma is undefined for one residual; the band must not be."""
    index = pd.date_range("2026-01-01", periods=length, freq="D", name="date")
    values = pd.Series([80.0 + i for i in range(length)], index=index)
    result = simple_exponential_smoothing(values, horizon=3)
    assert np.isfinite(result.frame.to_numpy()).all()
    assert np.isfinite(result.residual_sigma)


def test_horizon_must_be_positive(series):
    with pytest.raises(ValueError):
        forecast(series, horizon=0)


def test_short_series_is_rejected():
    index = pd.date_range("2026-01-01", periods=1, freq="D", name="date")
    with pytest.raises(ValueError):
        simple_exponential_smoothing(pd.Series([80.0], index=index), horizon=5)


# ── scenario ───────────────────────────────────────────────────────────────


def test_supply_cut_raises_price(series):
    base = simple_exponential_smoothing(series, horizon=10)
    cut = apply_supply_scenario(base, supply_change_mb_d=-1.0)
    assert (cut.frame["forecast"] > base.frame["forecast"]).all()


def test_supply_increase_lowers_price(series):
    base = simple_exponential_smoothing(series, horizon=10)
    glut = apply_supply_scenario(base, supply_change_mb_d=2.0)
    assert (glut.frame["forecast"] < base.frame["forecast"]).all()


def test_scenario_magnitude_matches_stated_elasticity(series):
    """The documented elasticity must be what the code actually applies."""
    base = simple_exponential_smoothing(series, horizon=5)
    cut = apply_supply_scenario(base, supply_change_mb_d=-1.02)  # exactly 1% of supply
    ratio = float(cut.frame["forecast"].iloc[0] / base.frame["forecast"].iloc[0])
    assert ratio == pytest.approx(1.0 + PRICE_ELASTICITY / 100.0, rel=1e-6)


def test_zero_scenario_is_identity(series):
    base = simple_exponential_smoothing(series, horizon=5)
    assert apply_supply_scenario(base, 0.0) is base


def test_absurd_scenario_is_refused(series):
    """Rather than print a negative oil price."""
    base = simple_exponential_smoothing(series, horizon=5)
    with pytest.raises(ValueError):
        apply_supply_scenario(base, supply_change_mb_d=50.0)
