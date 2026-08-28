"""Тесты рядов факторов, оценки эластичности и прогноза по балансу рынка.

Основная часть работает на СИНТЕТИЧЕСКОЙ панели: разбор PDF проверяется
отдельно и медленно, а свойства оценок — те, ради которых всё это строилось —
проверяются на данных с известным ответом. Тест, который на настоящем корпусе
подтверждает, что «эластичность отрицательная», не отличит правильную формулу от
случайно совпавшей.

Несколько тестов всё же ходят в настоящий корпус и помечены: они проверяют не
арифметику, а то, что тракт чтения таблиц всё ещё видит то, что видел.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neftegaz.forecast.elasticity import (
    MIN_OBSERVATIONS,
    estimate_clearing_elasticity,
    estimate_demand_elasticity,
    price_response_model,
)
from neftegaz.forecast.factors import load_factors, quarter_end, vintage_of

# ── метки периодов и выпусков ──────────────────────────────────────────────


def test_quarter_label_becomes_the_last_day_of_the_quarter():
    assert quarter_end("2026Q1") == pd.Timestamp("2026-03-31")
    assert quarter_end("2026Q4") == pd.Timestamp("2026-12-31")


def test_annual_columns_are_not_quarters():
    """В таблице рядом с кварталами стоят годовые итоги — их брать нельзя."""
    assert quarter_end("2026") is None
    assert quarter_end("") is None


def test_vintage_comes_from_the_file_name():
    assert vintage_of("/x/EIA_STEO_2026-07.pdf") == pd.Timestamp("2026-07-01")
    assert vintage_of("no-date-here.pdf") is None


# ── синтетическая панель с известным ответом ───────────────────────────────


def synthetic_panel(elasticity: float = -0.25, noise: float = 0.0, n: int = 12) -> pd.DataFrame:
    """Панель, в которой цена ПОСТРОЕНА по заданной эластичности клиринга.

    Добыча гуляет по заданной траектории, цена отвечает на неё ровно по формуле
    ``Δln P = Δln S / ε``. Значит оценка обязана вернуть ту самую ε — это и есть
    проверка, а не «похоже на правду».
    """
    rng = np.random.default_rng(seed=20260828)
    index = pd.date_range("2024-03-31", periods=n, freq="QE", name="quarter_end")
    supply_change = rng.normal(0, 0.02, n)
    supply_change[0] = 0.0
    production = 100.0 * np.exp(np.cumsum(supply_change))
    price = 80.0 * np.exp(np.cumsum(supply_change / elasticity + rng.normal(0, noise, n)))
    # Потребление отвечает на цену с эластичностью спроса, вдвое меньшей по
    # модулю: буфер запасов закрывает остаток.
    consumption = 99.0 * np.exp(np.cumsum(supply_change / elasticity * (elasticity / 2)))
    return pd.DataFrame(
        {
            "production": production,
            "consumption": consumption,
            "brent": price,
            # Запасы шевелятся: неподвижный фактор делает систему вырожденной,
            # и это отдельный тест, а не фон для всех остальных.
            "oecd_inventories": 2800.0 + np.cumsum(rng.normal(0, 20.0, n)),
        },
        index=index,
    )


def test_clearing_elasticity_recovers_the_value_it_was_built_from():
    frame = synthetic_panel(elasticity=-0.25)
    estimate = estimate_clearing_elasticity(frame)
    assert estimate is not None
    assert estimate.value == pytest.approx(-0.25, rel=1e-6)
    assert estimate.usable


def test_demand_elasticity_is_not_the_clearing_one():
    """Две величины, которые легко спутать, и цена путаницы — разы.

    В синтетике потребление отвечает вдвое слабее, чем требует клиринг: остаток
    закрывают запасы. Оценка спроса обязана это увидеть, а не вернуть ту же
    цифру, что оценка клиринга.
    """
    frame = synthetic_panel(elasticity=-0.25)
    clearing = estimate_clearing_elasticity(frame)
    demand = estimate_demand_elasticity(frame)
    assert clearing is not None and demand is not None
    assert abs(demand.value) < abs(clearing.value) / 1.5


def test_noise_widens_the_interval_rather_than_moving_the_estimate():
    """Шум обязан сказаться на ширине интервала, а не на самой оценке."""
    quiet = estimate_clearing_elasticity(synthetic_panel(noise=0.0))
    noisy = estimate_clearing_elasticity(synthetic_panel(noise=0.05))
    assert quiet is not None and noisy is not None
    quiet_width = quiet.ci_high - quiet.ci_low
    noisy_width = noisy.ci_high - noisy.ci_low
    assert noisy_width > quiet_width


def test_too_few_quarters_produce_no_estimate_at_all():
    """Три точки дадут число с интервалом, и это число будет выглядеть измерением."""
    short = synthetic_panel(n=MIN_OBSERVATIONS)  # разностей окажется на одну меньше
    assert estimate_clearing_elasticity(short.iloc[:3]) is None
    assert estimate_demand_elasticity(short.iloc[:3]) is None


def test_a_flat_market_yields_no_usable_estimate():
    """Без движения предложения инструмент ничего не идентифицирует."""
    index = pd.date_range("2024-03-31", periods=10, freq="QE", name="quarter_end")
    flat = pd.DataFrame(
        {
            "production": np.full(10, 100.0),
            "consumption": np.full(10, 99.0),
            "brent": np.full(10, 80.0),
            "oecd_inventories": np.full(10, 2800.0),
        },
        index=index,
    )
    estimate = estimate_clearing_elasticity(flat)
    assert estimate is None or not estimate.usable


def test_price_response_model_recovers_the_supply_coefficient():
    frame = synthetic_panel(elasticity=-0.25)
    model = price_response_model(frame)
    assert model is not None
    # Цена строилась как Δln S / ε, значит коэффициент при добыче равен 1/ε.
    assert model.supply_beta == pytest.approx(1.0 / -0.25, rel=1e-6)
    assert model.r_squared > 0.99


# ── настоящий корпус ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus():
    series = load_factors()
    if series.frame.empty:
        pytest.skip("корпус отчётов недоступен")
    return series


def test_corpus_yields_both_actuals_and_projections(corpus):
    """Разделение факта и прогноза — то, на чём стоит вся оценка.

    Смешать их значило бы оценивать связь между ценой и тем, что EIA о ней
    ДУМАЛ, вместо связи между ценой и тем, что произошло.
    """
    assert len(corpus.actuals) >= MIN_OBSERVATIONS
    assert not corpus.projections.empty
    assert corpus.actuals.index.max() < corpus.projections.index.min()


def test_corpus_columns_are_all_present_and_positive(corpus):
    """Пустая колонка прошла бы дальше как ряд из NaN и обнулила бы оценку."""
    for column in ("production", "consumption", "brent", "oecd_inventories"):
        values = corpus.actuals[column].dropna()
        assert len(values) >= MIN_OBSERVATIONS, column
        assert (values > 0).all(), column


def test_the_newest_vintage_wins_for_a_revised_quarter(corpus):
    """Прошлое пересматривается, и брать надо последнюю редакцию.

    Добыча за 2026Q1 в декабрьском выпуске — 106.50, в июльском — 103.86.
    Разница больше шока, который мы измеряем.
    """
    quarter = pd.Timestamp("2026-03-31")
    if quarter not in corpus.frame.index:
        pytest.skip("в корпусе нет этого квартала")
    row = corpus.frame.loc[quarter]
    assert row["vintage"] == corpus.frame["vintage"].max()


def test_measured_elasticity_on_the_corpus_is_usable_and_negative(corpus):
    estimate = estimate_clearing_elasticity(corpus.actuals)
    assert estimate is not None
    assert estimate.usable
    assert estimate.ci_high < 0 < estimate.n_observations


# ── прогноз по балансу рынка ───────────────────────────────────────────────


@pytest.fixture
def price_history() -> pd.Series:
    index = pd.date_range("2025-08-01", periods=400, freq="D", name="date")
    rng = np.random.default_rng(seed=20260828)
    return pd.Series(90 + rng.normal(0, 1.0, 400), index=index, name="close")


def test_factor_forecast_starts_at_the_last_known_price(price_history, corpus):
    """Прогноз обязан начинаться от рынка, а не от квартальной средней.

    Иначе первый же день расходится с последней ценой на несколько долларов без
    всякой причины, и читатель принимает разрыв за содержательный сигнал.
    """
    from neftegaz.forecast.factor_model import factor_forecast

    result = factor_forecast(price_history, horizon=30)
    first = float(result.frame["forecast"].iloc[0])
    assert first == pytest.approx(float(price_history.iloc[-1]), rel=0.05)


def test_factor_forecast_refuses_beyond_the_projected_quarters(price_history, corpus):
    """За горизонтом прогноза EIA метод отказывается, а не продолжает линию."""
    from neftegaz.forecast.factor_model import factor_forecast

    with pytest.raises(ValueError, match="reaches only"):
        factor_forecast(price_history, horizon=2000)


def test_factor_forecast_band_counts_both_sources_of_error(price_history, corpus):
    """Полоса обязана учитывать и ошибку модели, и ошибку прогноза факторов.

    Остаток регрессии отвечает лишь на вопрос «точно ли модель переводит факторы
    в цену». Сами факторы взяты из прогноза EIA, и на нашем корпусе ошибка этого
    прогноза даёт вклад ВДВОЕ больший. Коридор, посчитанный по одному остатку,
    рисовал бы уверенность, которой нет.
    """
    from neftegaz.forecast.factor_model import factor_forecast

    result = factor_forecast(price_history, horizon=90)
    assert result.params["supply_projection_sigma"] is not None
    assert result.params["step_sigma"] > result.params["model_sigma"]
    assert (result.frame["lower"] > 0).all()


def test_factor_forecast_is_not_reachable_through_auto(price_history):
    """Метод другой ПО ПРИРОДЕ, и подставлять его молча нельзя.

    Молчаливый переход означал бы, что пользователь получил ответ, построенный
    на балансе рынка, думая, что смотрит на экстраполяцию ряда цены.
    """
    from neftegaz.forecast.models import forecast

    auto = forecast(price_history, horizon=30, method="auto")
    assert not auto.method.startswith("factor model")


def test_unknown_method_is_still_refused(price_history):
    from neftegaz.forecast.models import forecast

    with pytest.raises(ValueError, match="unknown forecast method"):
        forecast(price_history, horizon=5, method="factor")
