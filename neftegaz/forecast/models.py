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

# Масштаб логарифма цены при подгонке ARIMA: log-цена, умноженная на 100, это
# лог-доходность в процентах.
#
# ★Число НЕ косметическое, оно про сходимость. На нашем ряде Brent подгонка по
# голому логарифму (значения около 4.5, приращения около 0.01) обрывается на
# ПЕРВОЙ итерации с converged=False — оптимизатору дефолтные допуски кажутся
# достигнутыми сразу. На той же серии, умноженной на 100, подгонка сходится за
# 14 итераций. ⚠Больше — хуже: множитель 1000 разваливает подгонку в
# переполнение (прогноз около 4.7e19 долл./барр.), так что масштаб выбран
# замером, а не по вкусу.
LOG_SCALE = 100.0


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


def _multiplicative_band(
    point: np.ndarray, log_sigma: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Доверительный коридор, построенный в ЛОГАРИФМАХ цены.

    ★Почему не ``точка ± z·σ·√h``, как было раньше. Аддитивный коридор растёт
    как корень из горизонта без всякой границы, а цена ограничена снизу нулём —
    и на длинном горизонте модель печатала ОТРИЦАТЕЛЬНУЮ цену: на нашем ряде
    Brent сглаживание давало нижнюю границу −1.98 долл./барр. уже на годе и
    −119.54 на пяти годах. Это не косметика вывода, а неверная форма модели:
    нормальное распределение на положительной величине неверно ровно там, где
    интервал широк.

    Здесь коридор строится на логарифме цены и возвращается в уровни
    экспонентой (геометрическое броуновское движение — стандартная модель для
    цены актива). Два следствия, и оба верные по существу:

    * Границы ПОЛОЖИТЕЛЬНЫ по построению, а не обрезаны по нулю постфактум.
      Обрезание чинило бы печать, оставляя неверной саму ширину.
    * Коридор АСИММЕТРИЧЕН: вверх цена может уйти в разы, вниз только до нуля.
      Аддитивная форма утверждала обратное.
    """
    steps = np.arange(1, horizon + 1, dtype="float64")
    half_width = Z_95 * log_sigma * np.sqrt(steps)
    return point * np.exp(-half_width), point * np.exp(half_width)


def _log_residual_sigma(observations: np.ndarray, alpha: float) -> float:
    """Сигма одношаговых остатков, посчитанная на логарифме цены.

    Отдельно от :func:`_residual_sigma`, который остаётся в долларах: он
    печатается пользователю с единицей измерения, и подменять его безразмерной
    величиной значило бы соврать в подписи.

    Неположительные наблюдения делают логарифм неопределённым. Для цены нефти
    это не бывает, но проверка стоит здесь, а не в вызывающем коде: функция
    обязана отвечать за то, что сама вычисляет.
    """
    if observations.size < 2 or float(np.min(observations)) <= 0.0:
        return 0.0
    logs = np.log(observations)
    level = float(logs[0])
    residuals: list[float] = []
    for value in logs[1:]:
        residuals.append(float(value) - level)
        level = alpha * float(value) + (1.0 - alpha) * level
    return _residual_sigma(residuals)


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
    errors compound as a random walk. ★Собирается он В ЛОГАРИФМАХ цены и
    возвращается в уровни экспонентой — см. :func:`_multiplicative_band`;
    аддитивная форма уходила в отрицательную цену на длинном горизонте.
    """
    _validate(series, horizon)
    observations = series.to_numpy(dtype="float64")

    level = float(observations[0])
    residuals: list[float] = []
    for value in observations[1:]:
        residuals.append(float(value) - level)
        level = alpha * float(value) + (1.0 - alpha) * level

    # Две сигмы намеренно: в долларах — та, что печатается пользователю с
    # единицей измерения; в логарифмах — та, из которой строится коридор.
    sigma = _residual_sigma(residuals)
    log_sigma = _log_residual_sigma(observations, alpha)

    point = np.full(horizon, level, dtype="float64")
    lower, upper = _multiplicative_band(point, log_sigma, horizon)

    frame = pd.DataFrame(
        {"forecast": point, "lower": lower, "upper": upper},
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

    ★Подгонка идёт по ЛОГАРИФМУ цены, а результат возвращается в уровни
    экспонентой. Причина та же, что у сглаживания: собственный доверительный
    интервал ARIMA в уровнях уходил в отрицательную цену на длинном горизонте
    (−33.06 долл./барр. на пяти годах нашего ряда). В логарифмах интервал
    положителен по построению и асимметричен, как и должен быть у цены. Побочно
    это лечит и вторую неверность: модель в уровнях считала дисперсию шага
    одинаковой при цене 20 и при цене 120, тогда как колеблется цена
    процентами, а не долларами.

    Если в ряду есть неположительные значения, логарифм неопределён и подгонка
    честно откатывается в уровни — с той самой слабостью, которая описана выше.
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
    observations = values.to_numpy()
    in_logs = bool(np.min(observations) > 0.0)

    target = np.log(observations) * LOG_SCALE if in_logs else observations
    fitted = ARIMA(target, order=order).fit()
    prediction = fitted.get_forecast(steps=horizon)
    band = prediction.conf_int(alpha=0.05)

    # conf_int returns a plain ndarray for ndarray input: column 0 lower, 1 upper.
    point = np.asarray(prediction.predicted_mean, dtype="float64")
    lower = np.asarray(band)[:, 0].astype("float64")
    upper = np.asarray(band)[:, 1].astype("float64")

    if in_logs:
        point = np.exp(point / LOG_SCALE)
        lower = np.exp(lower / LOG_SCALE)
        upper = np.exp(upper / LOG_SCALE)
        # Остатки для отчёта пересчитываются В ДОЛЛАРАХ: residual_sigma
        # печатается пользователю с единицей измерения, и подставить туда
        # безразмерную логарифмическую величину значило бы соврать в подписи.
        level_residuals = observations - np.exp(
            np.asarray(fitted.fittedvalues, dtype="float64") / LOG_SCALE
        )
    else:
        level_residuals = np.asarray(fitted.resid, dtype="float64")

    frame = pd.DataFrame(
        {"forecast": point, "lower": lower, "upper": upper},
        index=_future_index(series, horizon),
    )
    return ForecastResult(
        frame=frame,
        method=f"ARIMA{order}",
        params={
            "order": order,
            "fitted_on_logs": in_logs,
            # ★Сходимость едет в результате, а не теряется в предупреждении.
            # Несошедшаяся подгонка исключения НЕ бросает: `forecast(method="auto")`
            # её не заметит и вернёт числа, выглядящие как обычный ответ.
            "converged": bool(fitted.mle_retvals.get("converged", True)),
        },
        residual_sigma=_residual_sigma(list(level_residuals[np.isfinite(level_residuals)])),
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
