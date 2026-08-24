"""Two independent price-forecasting methods, both returning a confidence band.

The assignment asks for at least two methods. They are deliberately different
in kind rather than two flavours of the same idea:

* :func:`simple_exponential_smoothing` — a hand-written level model with no
  dependencies. It always works, it is auditable line by line, and it is the
  fallback when the statistical stack is unavailable or the series is too
  short to identify anything richer.
* :func:`arima_forecast` — a fitted ARIMA from statsmodels, which can express
  trend and autocorrelation that the level model cannot.

Both return the same frame shape, so the agent can swap them without knowing
which one ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ForecastResult", "simple_exponential_smoothing", "arima_forecast", "forecast"]

# 95% two-sided normal quantile. Named because a bare 1.96 in the middle of an
# expression is exactly the kind of constant that later gets "tidied" to 2.
Z_95 = 1.96


@dataclass(frozen=True)
class ForecastResult:
    """A forecast plus everything needed to explain it.

    ``frame`` has a daily DatetimeIndex named ``date`` and columns
    ``forecast`` / ``lower`` / ``upper``.
    """

    frame: pd.DataFrame
    method: str
    params: dict
    residual_sigma: float
    n_observations: int

    def interpretation(self, instrument: str = "Brent") -> str:
        """One short paragraph a human can read without opening the frame."""
        first = self.frame.iloc[0]
        last = self.frame.iloc[-1]
        horizon = len(self.frame)
        width_start = float(first["upper"] - first["lower"])
        width_end = float(last["upper"] - last["lower"])
        return (
            f"{instrument}: forecast {last['forecast']:.2f} USD/bbl at horizon "
            f"{horizon} days ({self.method}, fitted on {self.n_observations} "
            f"observations). The 95% band widens from {width_start:.2f} to "
            f"{width_end:.2f} USD/bbl as the horizon grows — uncertainty scales "
            f"with the square root of time, so a distant point estimate carries "
            f"far less information than a near one. Residual sigma "
            f"{self.residual_sigma:.2f} USD/bbl."
        )


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Daily index continuing one day past the end of ``series``."""
    return pd.date_range(
        series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D", name="date"
    )


def _residual_sigma(residuals: list[float]) -> float:
    """Standard deviation of residuals, defined for every input length.

    The sample estimator (ddof=1) divides by n-1, which is NaN for a single
    residual — and a NaN sigma propagates silently into the confidence band,
    producing a forecast whose interval is NaN while the point estimate looks
    fine. That is a worse failure than a crude number, because it survives a
    casual glance. Below two residuals we fall back to the population
    estimator: on one residual it reports 0, meaning "no dispersion observed",
    which is what a two-point series actually tells us.
    """
    array = np.asarray(residuals, dtype="float64")
    if array.size == 0:
        return 0.0
    return float(np.std(array, ddof=1 if array.size >= 2 else 0))


def _validate(series: pd.Series, horizon: int) -> None:
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 day, got {horizon}")
    if len(series) < 2:
        raise ValueError(f"need at least 2 observations, got {len(series)}")


def simple_exponential_smoothing(
    series: pd.Series, horizon: int, alpha: float = 0.3
) -> ForecastResult:
    """Exponential smoothing of the level, with a widening confidence band.

    The recursion is ``level_t = alpha * y_t + (1 - alpha) * level_{t-1}``,
    seeded with ``level_0 = y_0``. A level model has no trend term, so the
    point forecast is flat — that is a property of the method, not a bug, and
    it is exactly why the band matters more than the line.

    The band comes from the standard deviation of the one-step-ahead residuals
    accumulated during fitting, scaled by ``sqrt(h)``: h independent one-step
    errors compound as a random walk.
    """
    _validate(series, horizon)
    observations = series.to_numpy(dtype="float64")

    level = float(observations[0])
    residuals: list[float] = []
    for value in observations[1:]:
        residuals.append(float(value) - level)
        level = alpha * float(value) + (1.0 - alpha) * level

    sigma = _residual_sigma(residuals)
    steps = np.arange(1, horizon + 1, dtype="float64")
    half_width = Z_95 * sigma * np.sqrt(steps)

    frame = pd.DataFrame(
        {
            "forecast": np.full(horizon, level, dtype="float64"),
            "lower": level - half_width,
            "upper": level + half_width,
        },
        index=_future_index(series, horizon),
    )
    return ForecastResult(
        frame=frame,
        method="simple exponential smoothing",
        params={"alpha": alpha},
        residual_sigma=sigma,
        n_observations=len(observations),
    )


def arima_forecast(
    series: pd.Series, horizon: int, order: tuple[int, int, int] = (1, 1, 1)
) -> ForecastResult:
    """ARIMA forecast with the model's own 95% prediction interval.

    Defaults to ARIMA(1,1,1): one difference because price levels are not
    stationary, one AR and one MA term because daily oil prices are close to a
    random walk with mild short-memory structure. Raises ImportError if
    statsmodels is absent so the caller can fall back deliberately rather than
    silently getting a different model than it asked for.
    """
    _validate(series, horizon)
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "arima_forecast requires statsmodels; install it or use "
            "simple_exponential_smoothing instead"
        ) from exc

    values = series.astype("float64")
    fitted = ARIMA(values.to_numpy(), order=order).fit()
    prediction = fitted.get_forecast(steps=horizon)
    band = prediction.conf_int(alpha=0.05)

    # conf_int returns a plain ndarray for ndarray input: column 0 lower, 1 upper.
    lower = np.asarray(band)[:, 0]
    upper = np.asarray(band)[:, 1]

    frame = pd.DataFrame(
        {
            "forecast": np.asarray(prediction.predicted_mean, dtype="float64"),
            "lower": lower.astype("float64"),
            "upper": upper.astype("float64"),
        },
        index=_future_index(series, horizon),
    )
    return ForecastResult(
        frame=frame,
        method=f"ARIMA{order}",
        params={"order": order},
        residual_sigma=_residual_sigma(list(np.asarray(fitted.resid, dtype="float64"))),
        n_observations=len(values),
    )


def forecast(
    series: pd.Series, horizon: int, method: str = "auto", **kwargs
) -> ForecastResult:
    """Dispatch to a named method.

    ``method="auto"`` prefers ARIMA and falls back to exponential smoothing if
    statsmodels is missing or the fit fails — a failed fit on a short or
    degenerate series is normal, and the caller still needs an answer.
    """
    if method == "ses":
        return simple_exponential_smoothing(series, horizon, **kwargs)
    if method == "arima":
        return arima_forecast(series, horizon, **kwargs)
    if method != "auto":
        raise ValueError(f"unknown forecast method: {method!r}")

    try:
        return arima_forecast(series, horizon)
    except Exception:  # noqa: BLE001 - any fit failure is a fallback trigger
        return simple_exponential_smoothing(series, horizon)
