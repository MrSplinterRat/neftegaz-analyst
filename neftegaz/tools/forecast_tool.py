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
from functools import lru_cache

import pandas as pd

from neftegaz.config import settings
from neftegaz.forecast.data import load_prices
from neftegaz.forecast.models import ForecastResult, forecast

__all__ = [
    "ForecastReport",
    "run_forecast",
    "apply_supply_scenario",
    "elasticity_for_horizon",
    "measured_elasticity",
    "price_multiplier",
]


@lru_cache(maxsize=1)
def measured_elasticity():
    """Эластичность клиринга, оценённая на корпусе, или ``None``.

    Кэшируется на процесс: оценка требует разбора всех отчётов корпуса (около
    трёх секунд), а меняется она только при пополнении корпуса — то есть между
    запусками, а не между вопросами.

    ``None`` возвращается в трёх случаях, и все три — законные состояния, а не
    сбои: режим ``literature`` выбран настройкой; корпуса нет или он мал;
    оценка вышла непригодной (положительная либо с интервалом через ноль).
    Тогда расчёт берёт литературное число и говорит об этом в ответе.
    """
    if settings.elasticity_source.strip().lower() != "measured":
        return None
    try:
        from neftegaz.forecast.elasticity import estimate_clearing_elasticity
        from neftegaz.forecast.factors import load_factors

        estimate = estimate_clearing_elasticity(load_factors().actuals)
    except Exception:  # noqa: BLE001 — нечитаемый корпус не должен ронять расчёт
        return None
    if estimate is None or not estimate.usable:
        return None
    return estimate


def elasticity_for_horizon(horizon_days: int) -> float:
    """Эластичность спроса по цене для заданного горизонта.

    Возвращает отрицательное число: рост цены снижает потребление. Модуль
    РАСТЁТ с горизонтом — за месяцы потребитель почти не может изменить
    поведение, за годы может. Между короткой и длинной оценкой интерполируем
    линейно, за пределами — насыщение.

    Прежняя версия этого модуля горизонт игнорировала: прогноз на неделю и на
    год сдвигались одинаково. Это заведомо неверно для обоих концов.
    """
    measured = measured_elasticity()
    # Измерение заменяет КОРОТКИЙ конец: оно получено на квартальных разностях,
    # то есть говорит именно о ближнем горизонте. Длинного горизонта в корпусе
    # нет, и подставлять туда квартальную оценку значило бы выдать измерение за
    # то, чем оно не является.
    short = measured.value if measured is not None else settings.demand_elasticity_short
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
    # Прогноз, построенный ДРУГИМ ПО ПРИРОДЕ методом — по балансу рынка, а не по
    # истории цены. Едет вместе с основным намеренно: два метода, экстраполирующих
    # один и тот же ряд, согласятся почти всегда, и их согласие ничего не
    # доказывает. Согласие с методом, который смотрит на добычу и запасы, —
    # доказывает; расхождение с ним столь же информативно и должно быть видно.
    second_opinion: str | None
    frame: pd.DataFrame

    def as_text(self) -> str:
        # Когда сценарий есть, числа ниже — УЖЕ сценарные. Пометка стоит прямо у
        # них, а не только в абзаце про сценарий: читатель берёт цифру из строки,
        # а условие её получения — абзацем позже, и связывает их не всегда.
        mark = " (со сценарием, множитель уже учтён)" if self.scenario else ""
        lines = [
            f"Инструмент: {self.instrument}",
            f"Последняя известная цена: {self.last_price:.2f} долл./барр. "
            f"(последнее наблюдение в истории: {self.last_date})",
            f"Горизонт: {self.horizon_days} дн.",
            f"Метод: {self.method}",
            f"Прогноз на конец горизонта: {self.point:.2f} долл./барр.{mark}",
            f"95% доверительный интервал: {self.lower:.2f} — {self.upper:.2f} долл./барр.{mark}",
        ]
        if self.scenario:
            lines.append(f"Сценарий: {self.scenario}")
        if self.second_opinion:
            lines.append(f"Второе мнение: {self.second_opinion}")
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
    # Край диапазона неопределённости: отклик сильнее центральной оценки. При
    # измерении берётся ближний конец доверительного интервала — то есть коридор
    # расширяется ровно на то, насколько плохо мы знаем саму эластичность.
    measured = measured_elasticity()
    band = measured.ci_high if measured is not None else settings.demand_elasticity_band
    edge = price_multiplier(share, band)
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
            "elasticity_band": band,
            "elasticity_measured": measured is not None,
            "price_multiplier": central,
            # Точка ДО сдвига. Нужна не для расчёта, а для отчёта: без неё
            # читатель видит только результат и множитель рядом, и естественным
            # образом умножает одно на другое ещё раз (так и случилось —
            # см. комментарий к scenario_text в run_forecast).
            "baseline_point": float(result.frame["forecast"].iloc[-1]),
        },
        residual_sigma=result.residual_sigma * central,
        n_observations=result.n_observations,
    )


def _second_opinion(series, horizon_days: int, primary) -> str | None:
    """Прогноз по балансу рынка рядом с прогнозом по истории цены.

    Возвращает ``None``, только когда факторный метод неприменим В ПРИНЦИПЕ —
    метод основной уже и есть факторный. Во всех остальных случаях возвращается
    строка: либо второй прогноз, либо причина, по которой его нет. ★Отсутствие
    второго мнения без объяснения читалось бы как «методы согласны», а это самое
    вредное из возможных умолчаний.
    """
    if primary.method.startswith("factor model"):
        return None
    try:
        from neftegaz.forecast.factor_model import factor_forecast

        other = factor_forecast(series, horizon_days)
    except Exception as exc:  # noqa: BLE001 — причина едет в ответ, а не в лог
        return f"прогноз по балансу рынка не построен ({exc})"

    row = other.frame.iloc[-1]
    mine = float(primary.frame.iloc[-1]["forecast"])
    theirs = float(row["forecast"])
    gap = (theirs - mine) / mine * 100.0
    return (
        f"по балансу рынка (добыча и запасы из прогноза EIA, R²={other.params['r_squared']:.2f} "
        f"на {other.params['quarters_fitted']} кварталах) на том же горизонте — "
        f"{theirs:.2f} долл./барр., 95% интервал {row['lower']:.2f} — {row['upper']:.2f}; "
        f"это на {gap:+.1f}% от прогноза по истории цены. "
        f"★Методы РАЗНЫЕ ПО ПРИРОДЕ: первый экстраполирует ряд цены, второй "
        f"переводит в цену прогноз добычи и запасов. Расхождение означает, что "
        f"рынок и балансовый прогноз EIA сейчас говорят разное, и это содержательный "
        f"факт, а не погрешность."
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
        measured = measured_elasticity()
        if measured is not None:
            origin = (
                f"ИЗМЕРЕНА на корпусе отчётов ({measured.method}, наблюдений: "
                f"{measured.n_observations}, 95% ДИ {measured.ci_low:.2f} … "
                f"{measured.ci_high:.2f})"
            )
        else:
            origin = "ОЦЕНКА ИЗ ЛИТЕРАТУРЫ, а не измерение на наших данных"
        band = result.params["elasticity_band"]
        baseline = result.params["baseline_point"]
        # ★МНОЖИТЕЛЬ НАЗЫВАЕТСЯ УЖЕ ПРИМЕНЁННЫМ, И ЭТО НЕ ПЕДАНТИЗМ. Прежняя
        # формулировка печатала «⇒ цена ×1.049» рядом с уже сдвинутыми числами,
        # не говоря, что сдвиг произведён. Читатель — в демо им оказалась сама
        # модель — умножил прогноз на множитель ВТОРОЙ раз и выдал 101.98 вместо
        # 97.21. Ошибка не в расчёте, а в подаче: величина и результат её
        # применения стояли рядом без указания порядка.
        scenario_text = (
            f"{direction} предложения на {abs(supply_change_mb_d):.2f} млн барр./сут "
            f"({abs(share_pct):.1f}% мирового предложения, принятого равным "
            f"{settings.global_supply_mb_d:.0f} млн барр./сут) "
            f"⇒ множитель к цене ×{multiplier:.3f}. "
            f"★Множитель УЖЕ ПРИМЕНЁН к прогнозу и границам, напечатанным выше: "
            f"без сценария прогноз на конец горизонта составил бы "
            f"{baseline:.2f} долл./барр. Умножать напечатанные числа на множитель "
            f"ещё раз НЕ НУЖНО. "
            f"Допущение: эластичность рыночного клиринга {elasticity:.2f} на горизонте "
            f"{horizon_days} дн. — на сколько процентов должна сдвинуться цена, чтобы "
            f"рынок поглотил сдвиг предложения на процент. Она {origin}. "
            f"★Это НЕ эластичность спроса ({settings.demand_elasticity_long:.2f} и подобные "
            f"числа из литературы): между ними стоит буфер запасов, который гасит ценовой "
            f"отклик, и подстановка одной вместо другой завышает движение цены втрое. "
            f"Коридор расширен до края диапазона ({band:.2f}), поэтому граница со стороны "
            f"сильного отклика отражает неопределённость самого допущения."
        )

    second_opinion = _second_opinion(series, horizon_days, result)

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
        second_opinion=second_opinion,
        frame=result.frame,
    )
