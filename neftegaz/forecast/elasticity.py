"""Эластичность, ОЦЕНЁННАЯ на данных корпуса, а не взятая из литературы.

Сценарный расчёт («что будет с ценой, если добыча упадёт на N млн барр./сут»)
до сих пор стоял на числе из литературы. Число разумное, но чужое: проверить
его на наших данных было нечем, а вся сценарная ветка держится на нём одном.

Здесь оно измеряется — по квартальным рядам добычи, потребления, запасов и
цены, вычитанным из корпуса STEO (см. :mod:`neftegaz.forecast.factors`).

★ДВЕ РАЗНЫЕ ВЕЛИЧИНЫ, КОТОРЫЕ ЛЕГКО СПУТАТЬ, И ЦЕНА ПУТАНИЦЫ — РАЗЫ

1. **Эластичность спроса** ``ε_d``: на сколько процентов падает ПОТРЕБЛЕНИЕ при
   росте цены на процент. Это её и называет литература, и её оценки −0.05…−0.10
   мы использовали.
2. **Эластичность рыночного клиринга** ``ε_m``: на сколько процентов должна
   сдвинуться ЦЕНА, чтобы рынок поглотил сдвиг ПРЕДЛОЖЕНИЯ на процент. Именно
   она стоит в сценарной формуле ``P₁/P₀ = (1 + ΔQ/Q₀)^(1/ε)``.

Они не равны, и не «примерно равны»: между ними стоит БУФЕР ЗАПАСОВ. Когда
предложение падает, разрыв закрывается не только сокращением потребления, но и
расходом хранилищ — а хранилища ведут себя как дополнительное предложение.
Поэтому цена двигается заметно МЕНЬШЕ, чем предсказывает эластичность спроса.

Измерение на корпусе: ``ε_d ≈ −0.11``, ``ε_m ≈ −0.31``. Подставив ``ε_d`` в
сценарную формулу, мы получали на шоке −2 млн барр./сут множитель ×1.219 вместо
×1.066 — завышение отклика почти втрое в логарифмическом масштабе. Это не
уточнение третьего знака, это другой ответ.

★ЭНДОГЕННОСТЬ И ЧТО С НЕЙ СДЕЛАНО. Цена и потребление определяются
одновременно, поэтому простая регрессия потребления на цену смещена: она
смешивает движение ВДОЛЬ кривой спроса с её СДВИГАМИ. Нужен инструмент —
величина, которая двигает цену, но сама не сдвигает спрос. Здесь это изменение
мировой добычи: корпус накрывает крупный перебой предложения (2026Q2, −8.3% за
квартал), а такой перебой к состоянию спроса отношения не имеет. Инструмент
сильный: первая ступень объясняет 72% дисперсии изменения цены.

⚠ЧЕСТНО О ГРАНИЦАХ. Наблюдений девять (десять фактических кварталов, девять
разностей). Это мало, доверительные интервалы широки, и добыча экзогенна не
полностью — производители реагируют на цену. Оценка лучше литературной не тем,
что точнее, а тем, что ПРОВЕРЯЕМА: она получена из данных, лежащих в корпусе,
и пересчитывается при его пополнении.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ElasticityEstimate",
    "estimate_demand_elasticity",
    "estimate_clearing_elasticity",
    "price_response_model",
    "PriceResponseModel",
    "MIN_OBSERVATIONS",
]

# Меньше этого числа разностей оценка не выдаётся вовсе. Три точки дадут
# какое-нибудь число с каким-нибудь интервалом, и это число будет выглядеть
# измерением, ничем им не являясь.
MIN_OBSERVATIONS = 6

# Двусторонняя нормальная квантиль 95%. С девятью наблюдениями честнее было бы
# стьюдентово t (t(7) ≈ 2.36 против 1.96), поэтому оно и берётся ниже; эта
# константа осталась для сопоставимости с полосой прогноза в models.py.
Z_95 = 1.96


@dataclass(frozen=True)
class ElasticityEstimate:
    """Оценка эластичности вместе со всем, что нужно, чтобы ей не поверить."""

    value: float
    ci_low: float
    ci_high: float
    method: str
    n_observations: int
    instrument_r2: float | None = None

    @property
    def usable(self) -> bool:
        """Оценка годится в дело, только если она отрицательна и не разомкнута.

        Положительная эластичность означала бы, что рост цены увеличивает
        потребление; интервал, накрывающий ноль, означает, что данных не хватило
        отличить отклик от его отсутствия. И то и другое — повод вернуться к
        литературной оценке, а не подставлять шум в расчёт, которым пользуются.
        """
        return self.value < 0 and self.ci_high < 0

    def describe(self) -> str:
        return (
            f"{self.value:.3f} (95% ДИ {self.ci_low:.3f} … {self.ci_high:.3f}, "
            f"{self.method}, наблюдений: {self.n_observations})"
        )


def _log_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Квартальные логарифмические приращения факторов.

    В разностях, а не в уровнях: уровни цены и добычи нестационарны, и регрессия
    одного на другой поймала бы общий тренд, а не связь.
    """
    needed = ["production", "consumption", "brent", "oecd_inventories"]
    clean = frame.dropna(subset=needed)
    if len(clean) < 2:
        return pd.DataFrame(columns=["dP", "dS", "dC", "dI"])
    return pd.DataFrame(
        {
            "dP": np.log(clean["brent"]).diff(),
            "dS": np.log(clean["production"]).diff(),
            "dC": np.log(clean["consumption"]).diff(),
            # Изменение запасов, приведённое к дням покрытия: запасы в млн барр.,
            # потребление в млн барр./сут, квартал — 91 день. Так коэффициент
            # читается как «столько-то за день покрытия», а не «за млн баррелей».
            "dI": clean["oecd_inventories"].diff() / clean["consumption"] / 91.0,
        }
    ).dropna()


def _ols(y: np.ndarray, columns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Метод наименьших квадратов со свободным членом. Возвращает (β, se, R²)."""
    design = np.column_stack([np.ones(len(y))] + columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    n, k = design.shape
    if n <= k:
        return beta, np.full(k, np.inf), float("nan")
    sigma2 = float((residuals**2).sum() / (n - k))
    # ★Псевдообращение, а не обращение. Вырожденная матрица здесь — обычное
    # состояние данных, а не сбой: неподвижный рынок даёт колонку из нулей, и
    # регрессия по нему не идентифицирована. Такой случай обязан вернуть
    # бесконечную ошибку оценки (то есть «ничего не измерено»), а не уронить
    # расчёт исключением из линейной алгебры.
    gram = design.T @ design
    if np.linalg.matrix_rank(gram) < k:
        return beta, np.full(k, np.inf), float("nan")
    covariance = sigma2 * np.linalg.pinv(gram)
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / total if total > 0 else float("nan")
    return beta, np.sqrt(np.diag(covariance)), r2


def _t_quantile(degrees_of_freedom: int) -> float:
    """95% двусторонняя квантиль Стьюдента, без тяжёлой зависимости.

    Таблица до 30 степеней свободы, дальше нормальное приближение. Брать 1.96
    при семи степенях свободы значило бы сузить интервал на 20% — ровно там, где
    выборка мала и честная ширина интервала важнее всего.
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042,
    }
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    if degrees_of_freedom < 1:
        return float("inf")
    if degrees_of_freedom > 30:
        return Z_95
    keys = sorted(table)
    nearest = min(keys, key=lambda k: abs(k - degrees_of_freedom))
    return table[nearest]


def estimate_demand_elasticity(frame: pd.DataFrame) -> ElasticityEstimate | None:
    """Эластичность СПРОСА, инструментальной переменной.

    Оценивается отношением ковариаций (оценка Вальда):

        ε_d = cov(Δln C, Δln S) / cov(Δln P, Δln S)

    где инструментом служит изменение добычи. Смысл: берётся только та часть
    движения цены, которую объяснил сдвиг предложения, и смотрится, как на неё
    ответило потребление. Движение цены, вызванное сдвигами самого спроса, в
    оценку не попадает — а именно оно смещало бы простую регрессию.

    Возвращает ``None``, если наблюдений слишком мало или инструмент не двигает
    цену: оценка без инструмента здесь хуже отсутствия оценки.
    """
    d = _log_differences(frame)
    if len(d) < MIN_OBSERVATIONS:
        return None

    _, _, first_stage_r2 = _ols(d["dP"].to_numpy(), [d["dS"].to_numpy()])
    covariance = np.cov(np.vstack([d["dC"], d["dP"], d["dS"]]), ddof=1)
    denominator = covariance[1, 2]
    if abs(denominator) < 1e-12:
        return None
    value = float(covariance[0, 2] / denominator)

    # Интервал — дельта-методом через две регрессии на инструмент: ε = b_C / b_P,
    # где обе беты оценены на одном и том же регрессоре, поэтому относительные
    # ошибки складываются. Приближение грубое и на девяти точках честно широкое.
    beta_c, se_c, _ = _ols(d["dC"].to_numpy(), [d["dS"].to_numpy()])
    beta_p, se_p, _ = _ols(d["dP"].to_numpy(), [d["dS"].to_numpy()])
    if abs(beta_p[1]) < 1e-12:
        return None
    relative = np.hypot(se_c[1] / beta_c[1], se_p[1] / beta_p[1]) if beta_c[1] else np.inf
    spread = abs(value) * relative * _t_quantile(len(d) - 2)
    return ElasticityEstimate(
        value=value,
        ci_low=value - spread,
        ci_high=value + spread,
        method="инструментальная переменная, инструмент — изменение мировой добычи",
        n_observations=len(d),
        instrument_r2=first_stage_r2,
    )


def estimate_clearing_elasticity(frame: pd.DataFrame) -> ElasticityEstimate | None:
    """Эластичность РЫНОЧНОГО КЛИРИНГА — та, что нужна сценарной формуле.

    Оценивается прямой регрессией отклика цены на сдвиг предложения:

        Δln P = a + b · Δln S,   ε_m = 1 / b

    Это ровно обращение сценарной формулы ``P₁/P₀ = (1 + ΔQ/Q₀)^(1/ε)``, то есть
    измеряется тот самый параметр, который в неё подставляется, — а не
    родственная ему величина из литературы.

    ★Полученное число (около −0.31) по модулю ВТРОЕ больше эластичности спроса
    (около −0.11), и разница не ошибка: разрыв между спросом и предложением
    закрывается ещё и запасами, а хранилища работают как дополнительное
    предложение и гасят ценовой отклик.
    """
    d = _log_differences(frame)
    if len(d) < MIN_OBSERVATIONS:
        return None

    beta, se, _ = _ols(d["dP"].to_numpy(), [d["dS"].to_numpy()])
    slope, slope_se = float(beta[1]), float(se[1])
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        return None

    t = _t_quantile(len(d) - 2)
    low, high = slope - t * slope_se, slope + t * slope_se
    # ★Интервал переносится на 1/b и ПЕРЕВОРАЧИВАЕТСЯ: обратная функция
    # убывающая. Если интервал наклона накрывает ноль, обратный интервал
    # разомкнут — оценка непригодна, и это видно по `usable`.
    if low < 0 < high:
        ci_low, ci_high = -np.inf, np.inf
    else:
        ends = sorted((1.0 / low, 1.0 / high))
        ci_low, ci_high = ends[0], ends[1]

    return ElasticityEstimate(
        value=1.0 / slope,
        ci_low=ci_low,
        ci_high=ci_high,
        method="регрессия отклика цены на сдвиг предложения",
        n_observations=len(d),
    )


@dataclass(frozen=True)
class PriceResponseModel:
    """Модель квартального изменения цены по сдвигу добычи и запасов."""

    intercept: float
    supply_beta: float
    inventory_beta: float
    residual_sigma: float
    r_squared: float
    n_observations: int

    def predict_log_change(self, supply_log_change: float, inventory_change: float) -> float:
        return (
            self.intercept
            + self.supply_beta * supply_log_change
            + self.inventory_beta * inventory_change
        )


def price_response_model(frame: pd.DataFrame) -> PriceResponseModel | None:
    """Подогнать модель ``Δln P = a + b·Δln S + c·ΔI``.

    Два фактора, а не один: добыча объясняет 72% движения цены, добыча вместе с
    изменением запасов — 90%. Знак при запасах положителен, и это содержательно,
    а не артефакт: расход хранилищ (ΔI < 0) СНИЖАЕТ цену относительно того, что
    дал бы один только шок добычи, потому что запасы выходят на рынок как
    дополнительное предложение.

    Больше двух факторов не берётся сознательно: наблюдений девять, и третий
    регрессор начал бы подгонять шум.
    """
    d = _log_differences(frame)
    if len(d) < MIN_OBSERVATIONS:
        return None

    y = d["dP"].to_numpy()
    columns = [d["dS"].to_numpy(), d["dI"].to_numpy()]
    # Неподвижный фактор не несёт сведений и делает систему вырожденной.
    # Выбросить его честнее, чем получить произвольный коэффициент при нём:
    # запасы могут стоять на месте, и модель по одной добыче — это модель, а не
    # отказ.
    if np.std(columns[1]) < 1e-12:
        columns = [columns[0], np.zeros_like(columns[1])]
    beta, _, r2 = _ols(y, columns)
    if not np.isfinite(beta).all():
        return None
    design = np.column_stack([np.ones(len(y)), columns[0], columns[1]])
    residuals = y - design @ beta
    degrees = len(y) - design.shape[1]
    if degrees <= 0:
        return None
    sigma = float(np.sqrt((residuals**2).sum() / degrees))
    return PriceResponseModel(
        intercept=float(beta[0]),
        supply_beta=float(beta[1]),
        inventory_beta=float(beta[2]),
        residual_sigma=sigma,
        r_squared=float(r2),
        n_observations=len(d),
    )
