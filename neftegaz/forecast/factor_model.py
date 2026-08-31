"""Третий метод прогноза: цена из факторов, а не из собственной истории.

Две прежние модели — сглаживание и ARIMA — знают о нефти ровно одно: как её цена
менялась раньше. Они не знают ни сколько её добывают, ни сколько потребляют, ни
сколько лежит в хранилищах, и поэтому не могут ответить на вопрос, ради которого
аналитику и нужен прогноз: ПОЧЕМУ цена будет такой.

Здесь прогноз строится иначе. На квартальных рядах корпуса подгоняется модель
отклика цены на два фактора — сдвиг мировой добычи и изменение запасов OECD:

    Δln P = a + b·Δln S + c·ΔI

Затем в неё подставляются ПРОГНОЗНЫЕ значения этих факторов, которые EIA
печатает в тех же таблицах на шесть кварталов вперёд. Получается траектория
цены, выведенная из баланса рынка.

★ЧТО ЭТО ДАЁТ СВЕРХ ДВУХ ПРЕЖНИХ МЕТОДОВ

* **Прогноз объясним.** Не «так продолжается ряд», а «добыча восстанавливается
  на 6% за квартал, запасы пополняются, поэтому цена идёт вниз». Каждое число
  прослеживается до строки таблицы в отчёте.
* **Появляется вторая точка зрения на будущее.** Наш прогноз и прогноз EIA
  построены на одних и тех же факторах, но разными руками: расхождение между
  ними — содержательная величина, а не шум.
* **Метод отличается ПО ПРИРОДЕ.** Сглаживание и ARIMA — родственники: обе
  экстраполируют один ряд. Третий метод ломает это родство, а значит согласие
  трёх методов означает больше, чем согласие двух.

⚠ГРАНИЦЫ, КОТОРЫЕ НАДО НАЗВАТЬ ВСЛУХ

* Наблюдений девять. Модель с двумя факторами на девяти точках — предел
  разумного; третий фактор начал бы подгонять шум.
* Прогноз факторов взят у EIA. Если EIA ошибётся в добыче, ошибёмся и мы —
  наш метод не независим от их прогноза, он лишь переводит его в цену.
* Дальше последнего прогнозного квартала метод не работает вовсе и честно
  отказывается, вместо того чтобы продолжать линию в пустоту.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neftegaz.forecast.elasticity import price_response_model
from neftegaz.forecast.factors import load_factors, projection_error_sigma
from neftegaz.forecast.models import Z_95, ForecastResult, _future_index, _validate

__all__ = ["factor_forecast"]


def factor_forecast(
    series: pd.Series, horizon: int, reports_dir: str | None = None
) -> ForecastResult:
    """Прогноз цены по прогнозным факторам EIA.

    ``series`` нужен только как ЯКОРЬ: траектория строится от последней
    известной дневной цены, а не от квартальной средней — иначе прогноз начинал
    бы со среднего по прошедшему кварталу и на первом же дне расходился бы с
    рынком на несколько долларов без всякой причины.

    Бросает ``ValueError``, если корпус не даёт достаточно данных: это состояние
    развёртывания, и вызывающий (режим ``auto``) должен откатиться к методу,
    которому корпус не нужен, а не получить выдуманную линию.
    """
    _validate(series, horizon)

    factors = load_factors(reports_dir)
    if factors.frame.empty:
        raise ValueError(
            "factor forecast needs the report corpus: no readable STEO tables found in "
            f"{reports_dir or 'the configured reports directory'}"
        )

    actuals = factors.actuals
    model = price_response_model(actuals)
    if model is None:
        raise ValueError(
            f"factor forecast needs more history: {len(actuals)} actual quarters in the corpus"
        )

    frame = factors.frame.sort_index()
    supply_change = np.log(frame["production"]).diff()
    inventory_change = frame["oecd_inventories"].diff() / frame["consumption"] / 91.0

    last_actual = actuals.index.max()
    future = [ts for ts in frame.index if ts > last_actual]
    if not future:
        raise ValueError("factor forecast needs projected quarters; the corpus has none")

    # Траектория в логарифмах, по кварталам, от последней ДНЕВНОЙ цены.
    anchor = float(series.iloc[-1])
    level = np.log(anchor)
    quarter_dates: list[pd.Timestamp] = []
    quarter_levels: list[float] = []
    quarter_steps: list[int] = []
    for step, timestamp in enumerate(future, start=1):
        change = model.predict_log_change(
            float(supply_change.loc[timestamp]), float(inventory_change.loc[timestamp])
        )
        if not np.isfinite(change):
            break
        level += change
        quarter_dates.append(timestamp)
        quarter_levels.append(level)
        quarter_steps.append(step)

    if not quarter_levels:
        raise ValueError("factor forecast produced no usable quarters")

    index = _future_index(series, horizon)
    if index[-1] > quarter_dates[-1]:
        raise ValueError(
            f"factor forecast reaches only {quarter_dates[-1].date()} "
            f"(the corpus projects {len(quarter_dates)} quarters ahead), "
            f"but the horizon asks for {index[-1].date()}"
        )

    # Между квартальными точками интерполируем В ЛОГАРИФМАХ: цена движется
    # процентами, и линейная интерполяция уровней дала бы систематический сдвиг
    # внутри квартала. Первая точка привязана к якорю, чтобы прогноз начинался
    # от известной цены, а не прыгал в неё.
    known_x = np.array(
        [series.index[-1].value] + [ts.value for ts in quarter_dates], dtype="float64"
    )
    known_y = np.array([np.log(anchor)] + quarter_levels, dtype="float64")
    wanted_x = np.array([ts.value for ts in index], dtype="float64")
    log_path = np.interp(wanted_x, known_x, known_y)

    # Неопределённость копится по КВАРТАЛАМ, а не по дням: остаток модели — это
    # ошибка квартального шага, и √k шагов складывают её как случайное блуждание.
    steps_ahead = np.interp(
        wanted_x,
        known_x,
        np.array([0.0] + [float(s) for s in quarter_steps], dtype="float64"),
    )

    # ★ДВА ИСТОЧНИКА ОШИБКИ, А НЕ ОДИН. Остаток регрессии отвечает на вопрос
    # «насколько точно модель переводит факторы в цену» и молчит о том, что сами
    # факторы взяты из ПРОГНОЗА EIA, который тоже может не сбыться. Второй
    # источник измерен на нашем же корпусе — по расхождению между прогнозом
    # добычи на ближайший квартал и тем, чем этот квартал оказался, — и он
    # ВДВОЕ БОЛЬШЕ первого: 0.018 на добыче при коэффициенте −5.9 даёт 0.107 в
    # логарифме цены против 0.049 у остатка модели. Коридор, посчитанный по
    # одному лишь остатку, рисовал бы уверенность, которой нет.
    supply_sigma = projection_error_sigma(reports_dir)
    step_sigma = model.residual_sigma
    if supply_sigma is not None:
        step_sigma = float(np.hypot(step_sigma, model.supply_beta * supply_sigma))
    half_width = Z_95 * step_sigma * np.sqrt(np.maximum(steps_ahead, 0.0))

    point = np.exp(log_path)
    result = pd.DataFrame(
        {
            "forecast": point,
            "lower": np.exp(log_path - half_width),
            "upper": np.exp(log_path + half_width),
        },
        index=index,
    )
    return ForecastResult(
        frame=result,
        method="факторная модель (добыча и запасы из прогноза EIA)",
        # ★Природа метода названа полем, а не подстрокой ярлыка: на ней стои́т
        # решение «спрашивать ли второе мнение», и перевод ярлыка сломал бы его
        # молча.
        kind="factors",
        params={
            "supply_beta": model.supply_beta,
            "inventory_beta": model.inventory_beta,
            "r_squared": model.r_squared,
            "quarters_fitted": model.n_observations,
            "quarters_projected": len(quarter_dates),
            "anchor_price": anchor,
            "model_sigma": model.residual_sigma,
            "supply_projection_sigma": supply_sigma,
            "step_sigma": step_sigma,
        },
        residual_sigma=float(anchor * model.residual_sigma),
        n_observations=model.n_observations,
    )
