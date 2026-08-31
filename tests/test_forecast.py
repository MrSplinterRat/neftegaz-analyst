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

from neftegaz.config import settings
from neftegaz.forecast.data import load_prices_from_frame
from neftegaz.forecast.models import arima_forecast, forecast, simple_exponential_smoothing
from neftegaz.tools.forecast_tool import (
    apply_supply_scenario,
    elasticity_for_horizon,
    price_multiplier,
)

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


@pytest.mark.parametrize("horizon", [90, 365, 1825, 5000])
@pytest.mark.parametrize("method", ["ses", "arima"])
def test_band_never_reaches_a_negative_price(series, method, horizon):
    """Цена ограничена снизу нулём, и коридор обязан это знать.

    Аддитивный интервал ``точка ± z·σ·√h`` рос без границы и печатал
    отрицательную цену: на реальном ряде Brent сглаживание давало нижнюю
    границу −1.98 долл./барр. уже на годе и −119.54 на пяти годах. Обрезать
    вывод по нулю было бы лечением печати, а не модели, поэтому интервал
    строится в логарифмах и положителен по построению.
    """
    result = forecast(series, horizon=horizon, method=method)
    assert (result.frame["lower"] > 0).all()


def test_band_is_asymmetric_around_the_point(series):
    """Вверх цена может уйти в разы, вниз только до нуля.

    Аддитивная форма утверждала обратное — что отклонения равновероятны в обе
    стороны на любую величину.
    """
    result = simple_exponential_smoothing(series, horizon=365)
    row = result.frame.iloc[-1]
    up = float(row["upper"] - row["forecast"])
    down = float(row["forecast"] - row["lower"])
    assert up > down


def test_arima_fits_on_logs_and_converges(series):
    """Подгонка идёт по логарифму цены, и она обязана СОЙТИСЬ.

    Масштаб здесь не косметика: по голому логарифму оптимизатор обрывается на
    первой итерации с converged=False, не бросая исключения, — то есть
    ``forecast(method="auto")`` этого не замечает и возвращает числа, похожие
    на нормальный ответ. Поэтому и сходимость едет в params, и тест на неё есть.
    """
    result = arima_forecast(series, horizon=30)
    assert result.params["fitted_on_logs"] is True
    assert result.params["converged"] is True


def test_arima_falls_back_to_levels_on_non_positive_prices():
    """Логарифм неопределён на нуле — подгонка честно уходит в уровни."""
    index = pd.date_range("2026-01-01", periods=40, freq="D", name="date")
    values = pd.Series(np.linspace(-5.0, 5.0, 40), index=index)
    result = arima_forecast(values, horizon=5)
    assert result.params["fitted_on_logs"] is False
    assert np.isfinite(result.frame.to_numpy()).all()


def test_residual_sigma_stays_in_dollars(series):
    """Сигма печатается пользователю с единицей измерения — она не логарифм.

    Логарифмическая сигма этого ряда около 0.02; долларовая — единицы. Тест
    ловит подмену, которая в тексте ответа выглядела бы просто маленьким числом.
    """
    result = arima_forecast(series, horizon=30)
    assert result.residual_sigma > 0.5
    assert "долл./барр." in result.interpretation()


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


def test_scenario_follows_the_constant_elasticity_curve(series):
    """Заявленная кривая должна быть той, которую код действительно применяет."""
    base = simple_exponential_smoothing(series, horizon=5)
    change = -1.02  # ровно 1% мирового предложения при 102 млн барр./сут
    cut = apply_supply_scenario(base, supply_change_mb_d=change, horizon_days=5)

    share = change / settings.global_supply_mb_d
    expected = price_multiplier(share, elasticity_for_horizon(5))
    ratio = float(cut.frame["forecast"].iloc[0] / base.frame["forecast"].iloc[0])
    assert ratio == pytest.approx(expected, rel=1e-9)


def test_the_two_forms_agree_on_small_shocks_at_the_same_elasticity():
    """Замена ФОРМЫ не должна была сама по себе менять ответ.

    Линейная форма давала множитель ``1 − share/|ε|``. На малом шоке степенная
    обязана совпасть с ней в пределах процента — при ОДНОЙ И ТОЙ ЖЕ
    эластичности. Сравнение идёт напрямую через ``price_multiplier``, а не через
    сценарий, именно поэтому: сценарий теперь берёт эластичность из измерения на
    корпусе, и разница в ответе там объясняется другим числом, а не другой
    формулой. Смешивать эти две причины в одном тесте — значит не проверить ни
    одну.
    """
    elasticity = -0.10
    share = 0.01  # 1% мирового предложения
    linear = 1.0 - share / abs(elasticity)
    assert price_multiplier(share, elasticity) == pytest.approx(linear, rel=0.01)


def test_cut_moves_price_more_than_an_equal_increase(series):
    """Асимметрия — свойство рынка, и она должна возникать из формы кривой.

    Заменить нефть немедленно нечем, а избыток упирается в стоимость хранения,
    поэтому сокращение бьёт сильнее наращивания того же объёма. Линейная форма
    этого не умела: она двигала цену одинаково в обе стороны.
    """
    base = simple_exponential_smoothing(series, horizon=5)
    start = float(base.frame["forecast"].iloc[0])

    cut = apply_supply_scenario(base, supply_change_mb_d=-2.0, horizon_days=5)
    glut = apply_supply_scenario(base, supply_change_mb_d=+2.0, horizon_days=5)

    up = float(cut.frame["forecast"].iloc[0]) - start
    down = start - float(glut.frame["forecast"].iloc[0])
    assert up > down


def test_large_shock_is_computed_rather_than_refused(series):
    """Шок в 10% предложения раньше ронял расчёт — теперь считается.

    Линейный множитель ``1 − 0.1·10`` обращался в ноль, и код отказывался
    отвечать ровно на самый интересный вопрос.
    """
    base = simple_exponential_smoothing(series, horizon=5)
    glut = apply_supply_scenario(base, supply_change_mb_d=10.2, horizon_days=5)
    ratio = float(glut.frame["forecast"].iloc[0] / base.frame["forecast"].iloc[0])
    assert 0.0 < ratio < 1.0


def test_only_total_loss_of_supply_is_refused(series):
    """Отказ остаётся, но его граница теперь физическая, а не артефакт формы."""
    base = simple_exponential_smoothing(series, horizon=5)
    with pytest.raises(ValueError):
        apply_supply_scenario(base, supply_change_mb_d=-settings.global_supply_mb_d)


@pytest.fixture
def literature_mode(monkeypatch):
    """Литературные числа вместо измеренных, для тестов о самой интерполяции."""
    import dataclasses

    import neftegaz.tools.forecast_tool as forecast_tool

    patched = dataclasses.replace(
        settings,
        elasticity_source="literature",
        demand_elasticity_short=-0.10,
        demand_elasticity_long=-0.30,
    )
    monkeypatch.setattr(forecast_tool, "settings", patched)
    forecast_tool.measured_elasticity.cache_clear()
    yield patched
    forecast_tool.measured_elasticity.cache_clear()


def test_longer_horizon_dampens_the_price_move(series, literature_mode):  # noqa: ARG001
    """Когда |ε| растёт с горизонтом, тот же шок двигает цену слабее.

    ⚠Проверяется на ЛИТЕРАТУРНЫХ числах, а не на измеренных, и это не уловка.
    Измерение дало короткий конец −0.31 против литературного длинного −0.30 —
    то есть на наших данных горизонты почти НЕ РАЗЛИЧАЮТСЯ, и утверждать
    обратное было бы выдачей желаемого за измеренное. Здесь проверяется, что
    интерполяция работает так, как задумана, при заданных ей числах.
    """
    base = simple_exponential_smoothing(series, horizon=5)
    near = apply_supply_scenario(base, supply_change_mb_d=-2.0, horizon_days=30)
    far = apply_supply_scenario(base, supply_change_mb_d=-2.0, horizon_days=1825)
    assert float(near.frame["forecast"].iloc[0]) > float(far.frame["forecast"].iloc[0])


def test_elasticity_saturates_outside_the_interpolation_range(literature_mode):
    """За пределами заданных горизонтов эластичность не экстраполируется."""
    assert elasticity_for_horizon(1) == literature_mode.demand_elasticity_short
    assert elasticity_for_horizon(10_000) == literature_mode.demand_elasticity_long
    middle = elasticity_for_horizon(
        (literature_mode.elasticity_short_days + literature_mode.elasticity_long_days) // 2
    )
    assert literature_mode.demand_elasticity_long < middle < literature_mode.demand_elasticity_short


def test_measured_elasticity_replaces_the_literature_value_on_the_short_end():
    """Ради этого задача и делалась: короткий конец больше не литературный.

    На корпусе оценка выходит около −0.31 против литературных −0.10, потому что
    формуле нужна эластичность рыночного клиринга, а не спроса: разрыв
    закрывается ещё и расходом запасов.
    """
    from neftegaz.tools.forecast_tool import measured_elasticity

    estimate = measured_elasticity()
    if estimate is None:
        pytest.skip("корпус отчётов недоступен — измерять нечего")

    assert estimate.usable
    assert elasticity_for_horizon(1) == estimate.value
    # Клиринг по модулю заметно больше спроса: если это перестанет выполняться,
    # значит оценивается уже не та величина.
    assert abs(estimate.value) > 0.15


def test_scenario_widens_the_band_relative_to_the_line(series):
    """Сценарий добавляет допущение, и коридор обязан это отразить.

    Прежняя версия умножала весь кадр на одно число: относительная ширина
    коридора не менялась, то есть код утверждал, что добавленная гипотеза об
    эластичности ничего не стоит.
    """
    base = simple_exponential_smoothing(series, horizon=5)
    cut = apply_supply_scenario(base, supply_change_mb_d=-2.0, horizon_days=5)

    def relative_width(result):
        row = result.frame.iloc[0]
        return float((row["upper"] - row["lower"]) / row["forecast"])

    assert relative_width(cut) > relative_width(base)


def test_band_order_is_preserved_after_the_scenario(series):
    """lower ≤ forecast ≤ upper — инвариант, который расширение не должно ломать."""
    base = simple_exponential_smoothing(series, horizon=5)
    for change in (-5.0, -1.0, 1.0, 5.0):
        moved = apply_supply_scenario(base, supply_change_mb_d=change, horizon_days=5)
        assert (moved.frame["lower"] <= moved.frame["forecast"]).all()
        assert (moved.frame["forecast"] <= moved.frame["upper"]).all()


def test_positive_elasticity_is_rejected():
    """Положительная эластичность означала бы, что рост цены поднимает спрос."""
    with pytest.raises(ValueError):
        price_multiplier(0.01, 0.10)


def test_zero_scenario_is_identity(series):
    base = simple_exponential_smoothing(series, horizon=5)
    assert apply_supply_scenario(base, 0.0) is base


# ── подача сценария: множитель и результат его применения ──────────────────
#
# Дефект найден не тестом и не чтением кода, а прогоном демо: модель прочитала
# «⇒ цена ×1.049» рядом с уже сдвинутым прогнозом 97.21, приняла множитель за
# неприменённый и напечатала 101.98. Расчёт был верен, ошиблась ПОДАЧА — число
# и результат его применения стояли рядом без указания порядка.


def test_baseline_point_survives_the_scenario(series):
    """Точка ДО сдвига обязана остаться доступной.

    Без неё в отчёте есть только результат и множитель, а обратное деление
    читатель делать не станет — он умножит ещё раз.
    """
    base = simple_exponential_smoothing(series, horizon=5)
    cut = apply_supply_scenario(base, supply_change_mb_d=-2.0, horizon_days=5)
    baseline = cut.params["baseline_point"]
    assert baseline == pytest.approx(float(base.frame["forecast"].iloc[-1]))
    assert float(cut.frame["forecast"].iloc[-1]) == pytest.approx(
        baseline * cut.params["price_multiplier"]
    )


def test_report_says_the_multiplier_is_already_applied(series, tmp_path):
    """Отчёт обязан сказать это словами, а не оставить на догадку читателя."""
    import neftegaz.tools.forecast_tool as tool

    frame = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"), "close": series.to_numpy()})
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)

    with_scenario = tool.run_forecast(
        horizon_days=30, method="ses", supply_change_mb_d=-1.5, prices_csv=str(path)
    )
    without = tool.run_forecast(horizon_days=30, method="ses", prices_csv=str(path))
    text = with_scenario.as_text()

    assert "УЖЕ ПРИМЕНЁН" in text
    assert "ещё раз НЕ НУЖНО" in text
    # ★Базовая цифра НАЗВАНА в тексте — та самая, что даёт прогноз без сценария.
    # Иначе читателю пришлось бы восстанавливать её делением, а он вместо этого
    # умножает.
    assert f"{without.point:.2f} долл./барр." in text


def test_scenario_numbers_are_marked_at_the_line_itself(series, tmp_path):
    """Пометка стоит у самой цифры: абзацем ниже её связывают не всегда."""
    import neftegaz.tools.forecast_tool as tool

    frame = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"), "close": series.to_numpy()})
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)

    with_scenario = tool.run_forecast(
        horizon_days=30, method="ses", supply_change_mb_d=-1.5, prices_csv=str(path)
    ).as_text()
    without = tool.run_forecast(horizon_days=30, method="ses", prices_csv=str(path)).as_text()

    assert "множитель уже учтён" in with_scenario
    # Без сценария пометки быть не должно: она сообщала бы о том, чего не было.
    assert "множитель уже учтён" not in without


# ── язык вывода: текст для человека идёт к человеку напрямую ────────────────


def test_the_interpretation_is_in_russian_and_keeps_only_proper_names():
    """★Единственное место расчётного блока, где текст шёл пересказом.

    Абзац интерпретации был английским, и до читателя он доходил только через
    языковую модель — то есть через того самого посредника, от которого расчёт
    отгорожен ровно затем, чтобы числа не пересказывались. Латиница в абзаце
    допустима только в именах: инструмент, название метода-аббревиатуры,
    название организации.
    """
    import re

    import pandas as pd

    from neftegaz.forecast.models import forecast

    index = pd.date_range("2026-01-01", periods=120, freq="D")
    series = pd.Series([70.0 + (i % 7) * 0.4 for i in range(120)], index=index)

    text = forecast(series, horizon=30, method="ses").interpretation()
    latin = set(re.findall(r"[A-Za-z][A-Za-z/]{1,}", text))
    assert latin <= {"Brent"}, f"в абзаце осталась латиница: {sorted(latin)}"
    assert "прогноз" in text and "долл./барр." in text
    # Плоская линия названа вслух: у модели уровня её нет по устройству, и
    # читатель, взявший точечную оценку на 30 дней, обязан это знать.
    assert "Линия прогноза плоская" in text


def test_arima_says_nothing_about_a_flat_line():
    """Отрицательный контроль: оговорка про плоскую линию не универсальна.

    Если бы она печаталась всегда, она была бы не сведением о методе, а
    украшением — и на методе с трендом прямо вводила бы в заблуждение.
    """
    import pandas as pd

    from neftegaz.forecast.models import arima_forecast

    index = pd.date_range("2026-01-01", periods=120, freq="D")
    series = pd.Series([70.0 + i * 0.1 for i in range(120)], index=index)
    assert "Линия прогноза плоская" not in arima_forecast(series, horizon=10).interpretation()


def test_a_scenario_keeps_the_nature_of_the_method_it_shifts():
    """★Пересборка результата не имеет права терять поля молча.

    Сценарий сдвигает числа. Если бы при этом терялась природа метода,
    факторный прогноз стал бы спрашивать второе мнение у самого себя, а плоская
    линия перестала бы называться плоской — и ни одно из двух не сказало бы о
    себе ни слова.
    """
    import pandas as pd

    from neftegaz.forecast.models import forecast
    from neftegaz.tools.forecast_tool import apply_supply_scenario

    index = pd.date_range("2026-01-01", periods=120, freq="D")
    series = pd.Series([70.0 + (i % 5) * 0.3 for i in range(120)], index=index)
    base = forecast(series, horizon=30, method="ses")
    shifted = apply_supply_scenario(base, supply_change_mb_d=-1.5, horizon_days=30)

    assert shifted.kind == base.kind
    assert shifted.flat_point_forecast == base.flat_point_forecast
    assert "Линия прогноза плоская" in shifted.interpretation()


# ── откат с ARIMA на сглаживание: читатель узнаёт ПРИЧИНУ (Р-072) ──────────


def _steady_series():
    import pandas as pd

    index = pd.date_range("2026-01-01", periods=150, freq="D", name="date")
    return pd.Series([70.0 + (i % 7) * 0.4 for i in range(150)], index=index, name="close")


def test_a_normal_run_says_nothing_about_a_fallback():
    """★ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ ИДЁТ ПЕРВЫМ И ВАЖНЕЕ ПОЛОЖИТЕЛЬНОГО.

    Оговорка, звучащая при каждом прогнозе, перестаёт что-либо значить к тому
    дню, когда повод для неё появится. Поэтому в обычном случае в тексте не
    должно быть ни причины, ни знака внимания.
    """
    from neftegaz.forecast.models import forecast

    result = forecast(_steady_series(), horizon=30, method="auto")
    assert result.fallback_reason == ""
    assert "запасной метод" not in result.interpretation()


def test_a_failed_fit_names_the_reason_in_the_text(monkeypatch):
    """Откат меняет ПРИРОДУ ответа: у метода уровня нет слагаемого тренда.

    До этого читатель видел имя отработавшего метода и не мог узнать, почему
    получил его вместо заказанного автоматического выбора.
    """
    import neftegaz.forecast.models as models

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("подгонка не сошлась")

    monkeypatch.setattr(models, "arima_forecast", _refuse, raising=True)
    result = models.forecast(_steady_series(), horizon=30, method="auto")

    assert result.fallback_reason == "RuntimeError: подгонка не сошлась"
    text = result.interpretation()
    assert "запасной метод" in text
    assert "подгонка не сошлась" in text, "причина названа коду, но не читателю"


def test_the_reason_carries_no_traceback(monkeypatch):
    """В текст ответа идут тип и сообщение, а не трассировка.

    Трассировка ничего не говорит читателю ответа и выносит наружу пути
    файловой системы — то есть сведения о машине, на которой всё это крутится.
    """
    import neftegaz.forecast.models as models

    def _refuse(*_args, **_kwargs):
        raise ValueError("первая строка\nвторая строка с /путь/к/файлу.py")

    monkeypatch.setattr(models, "arima_forecast", _refuse, raising=True)
    result = models.forecast(_steady_series(), horizon=30, method="auto")

    assert result.fallback_reason == "ValueError: первая строка"
    assert "\n" not in result.fallback_reason
    assert ".py" not in result.fallback_reason


def test_the_reason_survives_the_supply_scenario(monkeypatch):
    """★Сценарий двигает числа, но не отменяет подмены метода.

    Потеряй мы поле здесь — получили бы сценарный ответ, у которого причина
    известна коду и неизвестна читателю: ровно тот дефект, который чинится.
    """
    import neftegaz.forecast.models as models
    from neftegaz.tools.forecast_tool import apply_supply_scenario

    monkeypatch.setattr(
        models,
        "arima_forecast",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("подгонка не сошлась")),
        raising=True,
    )
    base = models.forecast(_steady_series(), horizon=30, method="auto")
    shifted = apply_supply_scenario(base, supply_change_mb_d=-1.5, horizon_days=30)

    assert shifted.fallback_reason == base.fallback_reason
    assert "запасной метод" in shifted.interpretation()


# ── несошедшаяся подгонка видна читателю, а не только коду (Р-078) ─────────


def test_a_failed_convergence_is_named_in_the_text():
    """★Худший вид отказа — тот, что молчит.

    Оптимизатор не бросает исключения: подгонка обрывается, а числа выглядят
    как обычный ответ. Признак давно ехал в параметрах результата, то есть был
    доступен разработчику и невидим читателю.

    Ряд взят НАСТОЯЩИЙ, а не подделанный: на плоской линии подгонка ARIMA
    действительно не сходится, и проверять это на выдуманном объекте значило бы
    проверять свою же заглушку.
    """
    import warnings

    import pandas as pd

    from neftegaz.forecast.models import arima_forecast

    index = pd.date_range("2026-01-01", periods=120, freq="D", name="date")
    flat = pd.Series([70.0] * 120, index=index, name="close")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = arima_forecast(flat, horizon=10)

    assert result.params["converged"] is False
    assert "НЕ СОШЛАСЬ" in result.interpretation()


def test_a_converged_fit_says_nothing_about_convergence(series):
    """★Отрицательный контроль: при обычной подгонке оговорки нет."""
    from neftegaz.forecast.models import arima_forecast

    result = arima_forecast(series, horizon=30)
    assert result.params["converged"] is True
    assert "НЕ СОШЛАСЬ" not in result.interpretation()


def test_a_method_without_convergence_says_nothing_either(series):
    """У сглаживания сходимости нет как понятия — и молчание тут верное.

    Признак читается через `params.get`, поэтому отсутствие ключа не
    превращается в тревогу. Проверяется отдельно: разница между «не сошлось» и
    «понятия не существует» видна только так.
    """
    from neftegaz.forecast.models import simple_exponential_smoothing

    result = simple_exponential_smoothing(series, horizon=30)
    assert "converged" not in result.params
    assert "НЕ СОШЛАСЬ" not in result.interpretation()


def test_the_convergence_flag_survives_the_supply_scenario():
    """Сценарий пересобирает параметры — признак обязан пережить пересборку."""
    import warnings

    import pandas as pd

    from neftegaz.forecast.models import arima_forecast
    from neftegaz.tools.forecast_tool import apply_supply_scenario

    index = pd.date_range("2026-01-01", periods=120, freq="D", name="date")
    flat = pd.Series([70.0] * 120, index=index, name="close")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = arima_forecast(flat, horizon=30)
    shifted = apply_supply_scenario(base, supply_change_mb_d=-1.5, horizon_days=30)

    assert shifted.params["converged"] is False
    assert "НЕ СОШЛАСЬ" in shifted.interpretation()
