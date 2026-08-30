"""Приёмка уточняющего вопроса.

★Уточнение — самая лёгкая вещь, которую легко сделать хуже, чем её отсутствие.
Агент, переспрашивающий по ощущению «вопрос какой-то расплывчатый», тратит ход
человека и ничего за это не даёт; агент, приклеивающий к висящему вопросу любую
следующую реплику, портит ответ на новую тему. Поэтому приёмка объявлена так,
чтобы держать ОБЕ стороны:

* спрашивает — только на измеренном условии (сценарий назван, но не прочитан);
* НЕ спрашивает — на обычном вопросе, на понятном сценарии, на отраслевом
  вопросе без всякого сценария;
* склейка принимается — когда реплика действительно доводит вопрос до читаемого
  сценария, и отвергается — когда человек сменил тему;
* из уточнения есть выход: «без сценария» даёт базовый расчёт, а не тупик.
"""

from __future__ import annotations

import pytest

from neftegaz.agent import graph as graph_module
from neftegaz.agent.graph import _after_route, node_clarify, resolve_pending

CUT_WITHOUT_UNIT = "а если ОПЕК+ сократит добычу"
VILKA = "сокращение с 2 до 3 млн барр/сут"


# ── когда спрашивать, а когда молчать ──────────────────────────────────────


@pytest.mark.parametrize("question", [VILKA, "сценарий: 2 млн барр/сут", CUT_WITHOUT_UNIT])
def test_unreadable_scenario_goes_to_clarify(question):
    assert _after_route({"route": "forecast", "question": question}) == "clarify"


@pytest.mark.parametrize(
    "question",
    [
        "если ОПЕК+ сократит добычу на 2 млн барр/сут",  # сценарий прочитан
        "спрогнозируй Brent на 3 месяца",  # сценария нет вовсе
    ],
)
def test_readable_or_absent_scenario_goes_straight_to_forecast(question):
    """★Отрицательный контроль. Без него проверка не отличила бы новую ветку от
    ветки, спрашивающей на каждом вопросе."""
    assert _after_route({"route": "forecast", "question": question}) == "forecast"


def test_industry_question_is_never_clarified():
    """Сценарий не влияет на отраслевой вопрос — спрашивать там не о чем."""
    assert _after_route({"route": "industry", "question": VILKA}) == "retrieve"


def test_waiver_skips_clarification():
    assert (
        _after_route({"route": "forecast", "question": VILKA, "scenario_waived": True})
        == "forecast"
    )


# ── что именно спрашивается ────────────────────────────────────────────────


def test_clarify_names_the_defect_and_the_expected_form():
    state = node_clarify({"question": VILKA})
    assert "вилка" in state["answer"]
    assert "млн барр./сут" in state["answer"]
    # Вопрос повисает: без него ответ «на 2 млн барр./сут» не к чему отнести.
    assert state["pending_question"] == VILKA
    # И становится ходом разговора — иначе следующее «а почему?» повиснет.
    assert state["history"] == [{"question": VILKA, "answer": state["answer"]}]
    assert state["used_reports"] is False and state["used_web"] is False


# ── склейка ответа с висящим вопросом ──────────────────────────────────────


def test_reply_completes_the_pending_question():
    merged = resolve_pending(CUT_WITHOUT_UNIT, "на 2 млн барр/сут")
    assert merged is not None
    from neftegaz.agent.scenario import read_supply_scenario

    assert read_supply_scenario(merged).value_mb_d == pytest.approx(-2.0)


def test_a_new_topic_is_not_glued_to_the_pending_question():
    """★Человек вправе сменить тему, и приклеенный старый вопрос испортил бы ответ."""
    assert resolve_pending(CUT_WITHOUT_UNIT, "какая сейчас цена Brent") is None


def test_a_self_sufficient_scenario_is_not_glued_either():
    """Реплика, читаемая сама по себе, — новый вопрос, а не половина старого."""
    assert resolve_pending(CUT_WITHOUT_UNIT, "если добыча вырастет на 3 млн барр/сут") is None


# ── поведение узла маршрутизации на висящем вопросе ────────────────────────


def test_route_glues_and_sends_to_forecast_without_asking_the_classifier(monkeypatch):
    """Склейка удалась ⇒ ветка известна по построению, лишний заход к модели не нужен."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("классификатор не должен вызываться на удавшейся склейке")

    monkeypatch.setattr(graph_module, "ask", _explode)
    out = graph_module.node_route(
        {"question": "на 2 млн барр/сут", "pending_question": CUT_WITHOUT_UNIT}
    )
    assert out["route"] == "forecast"
    assert out["question"] == f"{CUT_WITHOUT_UNIT} на 2 млн барр/сут"
    assert out["pending_question"] == ""


def test_waiver_returns_the_original_question_as_a_plain_forecast(monkeypatch):
    monkeypatch.setattr(graph_module, "ask", lambda *a, **k: "forecast")
    out = graph_module.node_route({"question": "без сценария", "pending_question": VILKA})
    assert out["route"] == "forecast"
    assert out["question"] == VILKA
    assert out["scenario_waived"] is True
    assert out["pending_question"] == ""


def test_pending_is_dropped_even_when_the_glue_fails(monkeypatch):
    """★Уточнение задаётся один раз. Переживи оно неудачу — следующая посторонняя
    реплика приклеилась бы к тому же вопросу."""
    monkeypatch.setattr(graph_module, "ask", lambda *a, **k: "industry")
    out = graph_module.node_route(
        {"question": "какая сейчас цена Brent", "pending_question": CUT_WITHOUT_UNIT}
    )
    assert out["pending_question"] == ""
    assert "question" not in out  # вопрос не подменён
    assert out["scenario_waived"] is False


def test_flags_are_cleared_on_an_ordinary_turn(monkeypatch):
    """Оба поля пишутся каждый ход: иначе они протекли бы в следующие вопросы."""
    monkeypatch.setattr(graph_module, "ask", lambda *a, **k: "industry")
    out = graph_module.node_route({"question": "что с ценой Brent"})
    assert out["pending_question"] == ""
    assert out["scenario_waived"] is False


# ── оговорка в самом расчёте ───────────────────────────────────────────────


def test_waived_scenario_is_still_announced_but_differently(monkeypatch):
    """Отказ от сценария не отменяет оговорку — меняет её тон."""

    class _Report:
        def as_text(self):
            return "Инструмент: Brent"

    monkeypatch.setattr(
        "neftegaz.tools.forecast_tool.run_forecast", lambda **k: _Report(), raising=True
    )
    state = graph_module.node_forecast({"question": VILKA, "scenario_waived": True})
    assert "по твоей просьбе" in state["forecast_text"]
    assert "не прочитан" not in state["forecast_text"]
