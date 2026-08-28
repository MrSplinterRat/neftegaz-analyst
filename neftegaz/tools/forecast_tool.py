"""The price-forecasting tool the agent calls.

Requirement 2.5: a separate callable the agent invokes for questions like
"forecast Brent for 3 months" or "estimate the price range if output is cut by
N mb/d". It loads history, runs a model, and returns a forecast with a
confidence interval and a short interpretation.

The scenario adjustment (a supply cut or increase) is applied through a
constant-elasticity demand curve — see :func:`price_multiplier`. Every number
that curve needs lives in the configuration, not here: the whole scenario
branch rests on assumptions we did not measure, so a reviewer must be able to
disagree with them without touching code.

Those assumptions travel with the answer. :func:`run_forecast` states the
elasticity, the horizon it was taken at, and the fact that it comes from the
literature rather than from our own data — a scenario number without its
premise is worse than no number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from neftegaz.config import settings
from neftegaz.forecast.data import load_prices
from neftegaz.forecast.models import ForecastResult, forecast

__all__ = [
    "ForecastReport",
    "run_forecast",
    "apply_supply_scenario",
    "elasticity_for_horizon",
    "price_multiplier",
]


def elasticity_for_horizon(horizon_days: int) -> float:
    """Эластичность спроса по цене для заданного горизонта.

    Возвращает отрицательное число: рост цены снижает потребление. Модуль
    РАСТЁТ с горизонтом — за месяцы потребитель почти не может изменить
    поведение, за годы может. Между короткой и длинной оценкой интерполируем
    линейно, за пределами — насыщение.

    Прежняя версия этого модуля горизонт игнорировала: прогноз на неделю и на
    год сдвигались одинаково. Это заведомо неверно для обоих концов.
    """
    short = settings.demand_elasticity_short
    long_run = settings.demand_elasticity_long
    h_short = settings.elasticity_short_days
    h_long = settings.elasticity_long_days

    if horizon_days <= h_short or h_long <= h_short:
        return short
    if horizon_days >= h_long:
        return long_run
    weight = (horizon_days - h_short) / (h_long - h_short)
    return short + (long_run - short) * weight


def price_multiplier(share: float, elasticity: float) -> float:
    """Во сколько раз меняется цена при сдвиге предложения на долю ``share``.

    Спрос с постоянной эластичностью: ``Q = A · P^ε``. Чтобы рынок поглотил
    сдвиг предложения, цена должна сместиться так, что

        (Q₀ + ΔQ) / Q₀ = (P₁ / P₀)^ε   ⟹   P₁/P₀ = (1 + share)^(1/ε)

    ★Почему не линейная форма ``1 − share·k``, которая стояла здесь раньше.
    Три её свойства были ложными, и все три лечит одна правильная функция:

    1. Она ЛОМАЛАСЬ на больших шоках: при ``share ≥ 0.1`` множитель обращался
       в ноль или уходил в минус, и код отказывался считать — ровно там, где
       вопрос интереснее всего. Степенная форма определена на всём диапазоне,
       где предложение остаётся положительным.
    2. Она была СИММЕТРИЧНОЙ: сокращение и наращивание на 1 млн барр./сут
       двигали цену одинаково по модулю. В жизни сокращение бьёт сильнее —
       заменить нефть немедленно нечем, а избыток упирается в стоимость
       хранения. Здесь асимметрия возникает САМА, без отдельного допущения:
       в логарифмах форма симметрична, в уровнях — нет.
    3. На малых шоках обе формы совпадают, поэтому прежняя калибровка не
       теряется: ±1 млн барр./сут даёт −9.3 % / +10.4 % против −9.8 % / +9.8 %.

    ``elasticity`` обязана быть отрицательной — положительная означала бы, что
    рост цены увеличивает спрос.
    """
    if elasticity >= 0:
        raise ValueError(
            f"elasticity must be negative (demand falls as price rises), got {elasticity}"
        )
    if 1.0 + share <= 0:
        raise ValueError(
            f"supply change of {share:+.3f} of world supply would remove all supply; "
            "no finite price expresses that"
        )
    return (1.0 + share) ** (1.0 / elasticity)


@dataclass(frozen=True)
class ForecastReport:
    """Everything the agent needs to answer a forecast question."""

    instrument: str
    horizon_days: int
    method: str
    last_price: float
    # Дата последнего наблюдения едет вместе с ценой намеренно. История лежит
    # в CSV, который обновляется отдельным скриптом, и цена без даты читается
    # как «цена сейчас» — а она может быть недельной давности. Дата делает
    # утверждение проверяемым и переводит вопрос свежести данных из
    # незаметного в явный.
    last_date: str
    point: float
    lower: float
    upper: float
    interpretation: str
    scenario: str | None
    frame: pd.DataFrame

    def as_text(self) -> str:
        lines = [
            f"Инструмент: {self.instrument}",
            f"Последняя известная цена: {self.last_price:.2f} долл./барр. "
            f"(последнее наблюдение в истории: {self.last_date})",
            f"Горизонт: {self.horizon_days} дн.",
            f"Метод: {self.method}",
            f"Прогноз на конец горизонта: {self.point:.2f} долл./барр.",
            f"95% доверительный интервал: {self.lower:.2f} — {self.upper:.2f} долл./барр.",
        ]
        if self.scenario:
            lines.append(f"Сценарий: {self.scenario}")
        lines.append("")
        lines.append(self.interpretation)
        return "\n".join(lines)


def apply_supply_scenario(
    result: ForecastResult,
    supply_change_mb_d: float,
    horizon_days: int | None = None,
) -> ForecastResult:
    """Сдвинуть прогноз под изменение предложения.

    Положительное ``supply_change_mb_d`` — нефти на рынке больше, цена ниже.

    ★Коридор не просто масштабируется, а РАСШИРЯЕТСЯ. Сценарий добавляет к
    прогнозу гипотезу об эластичности, а гипотеза не может сузить незнание:
    центральная линия идёт по нашей оценке эластичности, а граница коридора со
    стороны более сильного отклика — по краю литературного диапазона. Прежняя
    версия умножала весь кадр на одно число, то есть утверждала, что добавленное
    допущение ничего не стоит.

    ``horizon_days`` нужен, потому что эластичность зависит от горизонта; если
    не передан, берётся длина кадра прогноза.
    """
    if supply_change_mb_d == 0:
        return result

    share = supply_change_mb_d / settings.global_supply_mb_d
    horizon = horizon_days if horizon_days is not None else len(result.frame)
    elasticity = elasticity_for_horizon(horizon)

    central = price_multiplier(share, elasticity)
    # Край литературного диапазона: отклик сильнее нашей центральной оценки.
    edge = price_multiplier(share, settings.demand_elasticity_band)
    low, high = (central, edge) if central <= edge else (edge, central)

    shifted = result.frame.copy()
    shifted["forecast"] = shifted["forecast"] * central
    shifted["lower"] = shifted["lower"] * low
    shifted["upper"] = shifted["upper"] * high

    return ForecastResult(
        frame=shifted,
        method=f"{result.method} + сценарий предложения",
        params={
            **result.params,
            "supply_change_mb_d": supply_change_mb_d,
            "elasticity": elasticity,
            "elasticity_band": settings.demand_elasticity_band,
            "price_multiplier": central,
        },
        residual_sigma=result.residual_sigma * central,
        n_observations=result.n_observations,
    )


def run_forecast(
    horizon_days: int = 90,
    method: str = "auto",
    supply_change_mb_d: float = 0.0,
    instrument: str = "Brent",
    prices_csv: str | None = None,
) -> ForecastReport:
    """Load history, forecast, optionally apply a supply scenario.

    Raises FileNotFoundError with an actionable message when the price file is
    absent — that is a setup problem, and silently returning a made-up series
    would be the worst possible response for this particular product.
    """
    path = prices_csv or settings.prices_csv
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"price history not found at {path}. Run scripts/fetch_prices.py to "
            "download it, or set PRICES_CSV in .env to point at your own file."
        )

    history = load_prices(path)
    series = history["close"]
    result = forecast(series, horizon_days, method=method)

    scenario_text = None
    if supply_change_mb_d:
        result = apply_supply_scenario(result, supply_change_mb_d, horizon_days=horizon_days)
        direction = "сокращение" if supply_change_mb_d < 0 else "увеличение"
        elasticity = result.params["elasticity"]
        multiplier = result.params["price_multiplier"]
        share_pct = 100.0 * supply_change_mb_d / settings.global_supply_mb_d
        # ★Число не отпускается без указания, на чём оно стоит. Сценарная ветка
        # опирается на допущение, которого нет в наших данных, и пользователь
        # обязан видеть это рядом с цифрой, а не в документации.
        scenario_text = (
            f"{direction} предложения на {abs(supply_change_mb_d):.2f} млн барр./сут "
            f"({abs(share_pct):.1f}% мирового предложения, принятого равным "
            f"{settings.global_supply_mb_d:.0f} млн барр./сут) "
            f"⇒ цена ×{multiplier:.3f}. "
            f"Допущение: эластичность спроса по цене {elasticity:.2f} на горизонте "
            f"{horizon_days} дн. Это ОЦЕНКА ИЗ ЛИТЕРАТУРЫ, а не измерение на наших "
            f"данных; коридор расширен до края диапазона ({settings.demand_elasticity_band:.2f}), "
            f"поэтому граница со стороны сильного отклика отражает неопределённость "
            f"самого допущения."
        )

    last_row = result.frame.iloc[-1]
    return ForecastReport(
        instrument=instrument,
        horizon_days=horizon_days,
        method=result.method,
        last_price=float(series.iloc[-1]),
        # load_prices гарантирует DatetimeIndex, поэтому последняя метка —
        # это дата последней строки файла, а не заполненный выходной.
        last_date=series.index[-1].date().isoformat(),
        point=float(last_row["forecast"]),
        lower=float(last_row["lower"]),
        upper=float(last_row["upper"]),
        interpretation=result.interpretation(instrument),
        scenario=scenario_text,
        frame=result.frame,
    )
