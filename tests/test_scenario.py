"""Приёмка чтения сценария предложения из вопроса.

Приёмка объявлена ДО правки и покрывает ОБЕ стороны ошибки:

* прочитать то, что раньше молча терялось (проценты, тысячи, «б/с», mb/d,
  «миллиона баррелей»), и прочитать с верным знаком;
* НЕ выдумать сценарий там, где его нет, и не выбрать за человека, когда
  сказано неоднозначно.

★Половина этих случаев — отрицательный контроль. Проверка, которая только
подтверждает, что хорошее читается, не отличила бы новый разбор от разбора,
объявляющего сценарий в каждом вопросе.
"""

from __future__ import annotations

import pytest

from neftegaz.agent.graph import parse_supply_change
from neftegaz.agent.scenario import read_supply_scenario
from neftegaz.config import settings

# ── что раньше молчало, а теперь читается ──────────────────────────────────
#
# Именно тот замер, из-за которого модуль появился: пять из семи формулировок
# прежний разбор превращал в ноль, ещё две — в неверное число.


@pytest.mark.parametrize(
    "question,expected",
    [
        # Проценты — доля мирового предложения, приведённая к млн барр./сут.
        ("если ОПЕК+ сократит добычу на 5%", -0.05 * settings.global_supply_mb_d),
        ("если добыча вырастет на 2 процента", 0.02 * settings.global_supply_mb_d),
        # Тысячи баррелей в сутки.
        ("при сокращении добычи на 500 тыс. барр/сут", -0.5),
        # «миллиона баррелей» — та же величина другими словами.
        ("если добыча упадёт на 1,5 миллиона баррелей в сутки", -1.5),
        # Английские записи, встречающиеся в отраслевых текстах.
        ("если добыча вырастет на 2 mb/d", 2.0),
        ("при сокращении на 800 kb/d", -0.8),
        # Сокращённая русская запись.
        ("при выпадении 2 млн б/с", -2.0),
        # Формы, работавшие и раньше, — не должны сломаться.
        ("при сокращении добычи на 1.5 млн барр/сут", -1.5),
        ("если добыча вырастет на 2 млн баррелей в сутки", 2.0),
    ],
)
def test_scenario_is_read(question, expected):
    scenario = read_supply_scenario(question)
    assert scenario.stated and scenario.understood, scenario.note
    assert scenario.value_mb_d == pytest.approx(expected)


def test_negation_flips_the_sign():
    """★Знак дороже величины: «на сколько» перепроверят, направление примут на веру.

    Прежний разбор искал корень «сократ» где угодно в вопросе и выдавал
    сокращение на фразе, которая сокращение как раз ОТРИЦАЕТ.
    """
    scenario = read_supply_scenario("не сократит, а увеличит на 2 млн барр/сут")
    assert scenario.understood
    assert scenario.value_mb_d == pytest.approx(2.0)


# ── отрицательный контроль: чего разбор делать НЕ должен ───────────────────


def test_no_scenario_stays_no_scenario():
    """Вопрос без сценария обязан давать честный ноль, а не «не понял»."""
    for question in ("что с ценой Brent", "спрогнозируй Brent на 3 месяца"):
        scenario = read_supply_scenario(question)
        assert not scenario.stated
        assert scenario.understood
        assert scenario.value_mb_d == 0.0
        assert not scenario.unreadable


@pytest.mark.parametrize(
    ("question", "near", "far"),
    [
        ("сокращение с 2 до 3 млн барр/сут", -2.0, -3.0),
        ("прогноз при сокращении добычи от 2 до 3 млн барр./сут", -2.0, -3.0),
        ("сценарий: сокращение предложения на 2–3 млн барр./сут", -2.0, -3.0),
        ("что с ценой при сокращении на 1.5-2 млн барр/сут", -1.5, -2.0),
        ("если добыча вырастет с 500 до 800 тыс. барр./сут", 0.5, 0.8),
        # Вилка, записанная задом наперёд, остаётся вилкой: концы упорядочивает
        # разбор, а не человек.
        ("сокращение с 3 до 2 млн барр/сут", -2.0, -3.0),
    ],
)
def test_range_is_read_as_both_ends(question, near, far):
    """Вилка «с 2 до 3» читается диапазоном, и оба конца попадают в расчёт.

    Прежний разбор либо переспрашивал, либо — на формах через тире и «от … до»
    — молча брал ОДИН конец: замер до правки дал −3.00 на «2–3 млн барр./сут»
    и −2.00 на «1.5-2 млн барр/сут», без единого слова об этом.
    """
    scenario = read_supply_scenario(question)
    assert not scenario.unreadable
    assert scenario.is_range
    assert scenario.value_mb_d == pytest.approx(near)
    assert scenario.value_high_mb_d == pytest.approx(far)
    # Знак у концов один: вилка задаёт силу изменения, а не его сторону.
    assert scenario.value_mb_d * scenario.value_high_mb_d > 0
    # Ближний к нулю конец лежит в СТАРОМ поле — потребитель, не знающий про
    # вилку, получает осторожную оценку эффекта, а не преувеличенную.
    assert abs(scenario.value_mb_d) < abs(scenario.value_high_mb_d)


def test_range_of_percent_is_read_too():
    """Проценты — такая же единица, и вилка процентов считается так же."""
    scenario = read_supply_scenario("если предложение упадёт с 2 до 3 процентов")
    assert scenario.is_range
    step = settings.global_supply_mb_d / 100.0
    assert scenario.value_mb_d == pytest.approx(-2.0 * step)
    assert scenario.value_high_mb_d == pytest.approx(-3.0 * step)


@pytest.mark.parametrize(
    "question",
    [
        # ★Предельный случай, ради которого вилка и привязана к позиции единицы.
        # Прежнее правило искало «с N до M» по всему вопросу и объявляло вилкой
        # ВЕЛИЧИНЫ пару ГОДОВ — переспрашивало о том, что названо однозначно.
        "что с ценой с 2024 до 2026 года при сокращении на 1,5 млн барр./сут",
        "прогноз на 2024-2025 годы при сокращении добычи на 2 млн барр./сут",
        # Две цифры рядом — горизонт и величина: тоже не вилка.
        "прогноз Brent на 2 года при сокращении добычи на 1,5 млн барр./сут",
        "сокращение добычи на 500 тыс. барр./сут с 1 января",
        # Одиночное значение обязано остаться одиночным.
        "если ОПЕК+ сократит добычу на 2 млн барр./сут",
        "что с ценой при сокращении на 5%",
    ],
)
def test_a_single_value_is_not_read_as_a_range(question):
    """Отрицательный контроль: числа в другом месте фразы вилкой не становятся.

    Половина набора — не «одно число», а именно ловушки: две цифры рядом,
    из которых вторая относится к сроку, а не к добыче.
    """
    scenario = read_supply_scenario(question)
    assert not scenario.is_range
    assert scenario.value_high_mb_d is None
    assert not scenario.unreadable


def test_direction_absent_is_refused():
    """Величина без направления не домысливается ростом."""
    scenario = read_supply_scenario("сценарий: 2 млн барр/сут")
    assert scenario.unreadable
    assert "рост это или сокращение" in scenario.note


def test_unknown_unit_is_named_not_swallowed():
    """Число без опознанной единицы — повод сказать, а не считать."""
    scenario = read_supply_scenario("если ОПЕК+ сократит добычу на 2")
    assert scenario.unreadable
    assert "единица" in scenario.note


def test_two_units_at_once_is_refused():
    scenario = read_supply_scenario("если сократит на 5% или на 2 млн барр/сут")
    assert scenario.unreadable
    assert "единиц" in scenario.note


# ── совместимость и связь с расчётом ───────────────────────────────────────


def test_thin_wrapper_keeps_the_old_contract():
    """`parse_supply_change` по-прежнему отдаёт одно число — старым вызывающим."""
    assert parse_supply_change("при сокращении добычи на 1.5 млн барр/сут") == pytest.approx(-1.5)
    assert parse_supply_change("что с ценой Brent") == 0.0


def test_unreadable_scenario_is_announced_in_the_forecast(monkeypatch):
    """Непрочитанный сценарий обязан прозвучать В ОТВЕТЕ, а не остаться нулём.

    ★Это и есть весь смысл правки: раньше человек получал базовый прогноз и
    читал его как ответ на свой сценарный вопрос. Проверяется не разбор, а
    то, что его отказ доходит до текста.
    """
    from neftegaz.agent import graph as graph_module

    class _Report:
        def as_text(self):
            return "Инструмент: Brent"

    def _fake_run_forecast(**kwargs):
        assert kwargs["supply_change_mb_d"] == 0.0
        return _Report()

    monkeypatch.setattr(
        "neftegaz.tools.forecast_tool.run_forecast", _fake_run_forecast, raising=True
    )
    # ★Пример непрочитанного сценария взят НЕ вилкой: вилка с 31.08 читается
    # обоими концами (см. `test_range_is_read_as_both_ends`). Проверяемый здесь
    # механизм — «отказ разбора доходит до текста ответа» — от этого не
    # изменился, изменился лишь список случаев, при которых разбор отказывает.
    state = graph_module.node_forecast({"question": "а если ОПЕК+ сократит добычу?"})
    text = state["forecast_text"]
    assert "не прочитан" in text
    assert "БАЗОВЫЙ" in text


def test_readable_scenario_is_not_announced(monkeypatch):
    """Отрицательный контроль к предыдущему: на понятном вопросе оговорки нет."""
    from neftegaz.agent import graph as graph_module

    class _Report:
        def as_text(self):
            return "Инструмент: Brent"

    def _fake_run_forecast(**kwargs):
        assert kwargs["supply_change_mb_d"] == pytest.approx(-2.0)
        return _Report()

    monkeypatch.setattr(
        "neftegaz.tools.forecast_tool.run_forecast", _fake_run_forecast, raising=True
    )
    state = graph_module.node_forecast({"question": "если ОПЕК+ сократит добычу на 2 млн барр/сут"})
    assert "не прочитан" not in state["forecast_text"]


# ── цифра горизонта — не величина сценария ─────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "спрогнозируй Brent на 3 месяца, если ОПЕК+ сократит добычу",
        "спрогнозируй Brent на 30 дней, если ОПЕК+ увеличит добычу",
        "спрогнозируй Brent на 2027 год, если ОПЕК+ сократит добычу",
        "оцени диапазон цен на 2 квартала, если добыча упадёт",
    ],
)
def test_a_horizon_digit_is_not_a_scenario_magnitude(question):
    """★Переспрос с НЕВЕРНОЙ причиной хуже переспроса без причины.

    Признаком «величина названа» служила любая цифра в вопросе, а в вопросе про
    прогноз цифра почти всегда есть — и почти всегда это горизонт. Человек
    читал «величина названа, но единица измерения не опознана», шёл искать у
    себя величину и не находил её. Маршрут при этом верен, ответ вежлив, и
    счётчик маршрутов такого не видит: 5 из 5 стои́т и с дефектом, и без него.
    """
    scenario = read_supply_scenario(question)
    assert scenario.unreadable, question
    assert "не названа величина" in scenario.note, scenario.note


def test_a_real_magnitude_without_a_unit_is_still_reported_as_such():
    """Отрицательный контроль: вычеркнув горизонт, нельзя ослепнуть к величине.

    Правка вычёркивает цифры горизонта из счёта. Вычеркни она лишнее — разбор
    начал бы отвечать «величины нет» и там, где величина названа, то есть
    сменил бы одну неверную причину на другую.
    """
    scenario = read_supply_scenario(
        "спрогнозируй Brent на 3 месяца, если ОПЕК+ сократит добычу на 500"
    )
    assert scenario.unreadable
    assert "единица измерения не опознана" in scenario.note, scenario.note


def test_the_two_horizon_vocabularies_do_not_drift_apart():
    """★Словари единиц времени в двух модулях РАЗНЫЕ, и это осознанно.

    В `requested_horizon_days` словарь несёт множители и живёт в слое графа; в
    `scenario` он нужен лишь затем, чтобы не спутать срок с величиной. Общего
    механизма у них нет — зато есть общая проверка: всё, что первый разбор
    признаёт горизонтом, записанным цифрой, второй обязан вычеркнуть.

    Именно так расхождение и обнаружится — здесь, а не в ответе пользователю.
    """
    import re

    from neftegaz.agent.graph import requested_horizon_days
    from neftegaz.agent.scenario import _HORIZON

    phrasings = [
        "на 3 месяца",
        "на 30 дней",
        "на 2027 год",
        "на 2 квартала",
        "на 6 недель",
        "на 5 лет",
        "на 90 дн",
        "на 12 мес",
    ]
    for phrase in phrasings:
        assert requested_horizon_days(phrase) is not None, phrase
        assert not re.search(r"\d", _HORIZON.sub(" ", phrase)), phrase
