"""Тесты памяти диалога: накопление ходов, бюджет знаков, изоляция разговоров.

Проверяются свойства, ради которых память и заводилась, а не факт её наличия:
продолжение вопроса («а на пять лет?») должно иметь к чему относиться, бюджет
контекста обязан быть жёстким, а два разговора не должны видеть друг друга.

Языковая модель здесь не участвует: узлы, которые в неё ходят, подменяются.
Тест на память, зависящий от живого endpoint'а, проверял бы доступность
endpoint'а.
"""

from __future__ import annotations

import pytest

from neftegaz.agent import graph as graph_module
from neftegaz.agent import prompts
from neftegaz.agent.graph import (
    build_checkpointer,
    build_graph,
    format_history,
    merge_history,
    new_thread_id,
)
from neftegaz.config import settings


def use_settings(monkeypatch, **overrides):
    """Подменить настройки на время теста.

    ``Settings`` — frozen dataclass, присвоить полю нельзя (и правильно: это
    единственный источник конфигурации, и менять его на ходу в продакшене
    незачем). Поэтому собирается копия с изменёнными полями, и она
    подставляется тому модулю, который её читает.
    """
    import dataclasses

    patched = dataclasses.replace(settings, **overrides)
    monkeypatch.setattr(graph_module, "settings", patched)
    return patched


# ── reducer истории ────────────────────────────────────────────────────────


def test_history_accumulates_across_updates():
    first = merge_history([], [{"question": "q1", "answer": "a1"}])
    second = merge_history(first, [{"question": "q2", "answer": "a2"}])
    assert [turn["question"] for turn in second] == ["q1", "q2"]


def test_budget_drops_the_oldest_turns_first(monkeypatch):
    """Свежие ходы нужнее: «а на пять лет?» относится к последнему вопросу."""
    use_settings(monkeypatch, history_budget_chars=40)
    turns = [{"question": f"вопрос {i}", "answer": "x" * 10} for i in range(10)]
    kept = merge_history([], turns)

    assert len(kept) < len(turns)
    assert kept[-1] == turns[-1]
    spent = sum(len(t["question"]) + len(t["answer"]) for t in kept)
    assert spent <= 40


def test_one_turn_larger_than_the_budget_still_survives(monkeypatch):
    """Пустая история хуже урезанной: ответ терял бы предмет разговора целиком."""
    use_settings(monkeypatch, history_budget_chars=10)
    kept = merge_history([], [{"question": "q", "answer": "a" * 500}])
    assert len(kept) == 1


def test_zero_budget_disables_history(monkeypatch):
    """Явный ноль — это выключатель, а не вырожденный случай."""
    use_settings(monkeypatch, history_budget_chars=0)
    assert merge_history([], [{"question": "q", "answer": "a"}]) == []


def test_a_real_sized_answer_does_not_erase_the_history(monkeypatch):
    """Ход подрезается ПРИ ЗАПИСИ, иначе память лжёт о том, что она есть.

    Настоящий ответ аналитика — три-пять тысяч знаков. Пока подрезка стояла
    только в сборке промпта, первый же такой ответ выбирал весь бюджет, и
    история схлопывалась до единственного последнего хода: снаружи память
    выглядела работающей, а не помнила ничего. Поймано сквозным прогоном.
    """
    use_settings(monkeypatch, history_budget_chars=4000, history_turn_cap_chars=1200)
    history: list[dict] = []
    for i in range(3):
        history = merge_history(history, [{"question": f"вопрос {i}", "answer": "д" * 4000}])
    assert len(history) == 3


def test_long_answer_is_capped_per_turn(monkeypatch):
    """Один длинный ответ не должен вытеснять остальную историю из промпта."""
    use_settings(monkeypatch, history_turn_cap_chars=50)
    text = format_history([{"question": "q", "answer": "a" * 5000}])
    assert len(text) < 500
    assert "обрезан по бюджету контекста" in text


# ── промпты ────────────────────────────────────────────────────────────────


def test_router_prompt_carries_history_when_there_is_one():
    with_history = prompts.build_router_prompt("а на пять лет?", "Вопрос: прогноз Brent\nОтвет: 93")
    assert "прогноз Brent" in with_history
    assert "а на пять лет?" in with_history


def test_router_prompt_stays_clean_without_history():
    """Пустой раздел в промпте классификатора — шум, сбивающий короткий ответ."""
    plain = prompts.build_router_prompt("спрогнозируй Brent")
    assert plain == prompts.ROUTER_PROMPT.format(question="спрогнозируй Brent")


def test_history_is_marked_as_not_a_source():
    """История объясняет вопрос, но ссылаться на неё нельзя.

    Иначе модель процитирует собственный прошлый ответ как проверяемый
    источник — то есть выдаст пересказ за факт.
    """
    prompt = prompts.build_answer_prompt(
        "а на пять лет?", "фрагменты", "", "", "Вопрос: q\nОтвет: a"
    )
    assert "НЕ источник" in prompt
    assert prompt.index("ПРЕДЫДУЩИЕ ХОДЫ") < prompt.index("ИСТОЧНИК 1")


# ── сквозь граф ────────────────────────────────────────────────────────────


@pytest.fixture
def offline_graph(monkeypatch):
    """Граф с подменёнными узлами, ходящими наружу."""
    monkeypatch.setattr(graph_module, "node_route", lambda state: {"route": "industry"})
    monkeypatch.setattr(
        graph_module, "node_retrieve", lambda state: {"report_hits": [], "used_reports": False}
    )
    monkeypatch.setattr(
        graph_module, "node_web", lambda state: {"web_hits": [], "used_web": False}
    )

    def fake_answer(state):
        seen = [turn["question"] for turn in state.get("history") or []]
        answer = f"ответ на «{state['question']}»; до этого: {seen}"
        return {"answer": answer, "history": [{"question": state["question"], "answer": answer}]}

    monkeypatch.setattr(graph_module, "node_answer", fake_answer)
    return build_graph(build_checkpointer())


def test_second_question_sees_the_first(offline_graph):
    thread = {"configurable": {"thread_id": new_thread_id()}}
    offline_graph.invoke({"question": "спрогнозируй Brent на 3 месяца"}, config=thread)
    second = offline_graph.invoke({"question": "а на пять лет?"}, config=thread)

    assert "спрогнозируй Brent на 3 месяца" in second["answer"]
    assert len(second["history"]) == 2


def test_threads_do_not_leak_into_each_other(offline_graph):
    """Два окна браузера — два разговора; чужие ходы не должны просачиваться."""
    first = {"configurable": {"thread_id": new_thread_id()}}
    second = {"configurable": {"thread_id": new_thread_id()}}

    offline_graph.invoke({"question": "вопрос про ОПЕК"}, config=first)
    other = offline_graph.invoke({"question": "вопрос про СПГ"}, config=second)

    assert "ОПЕК" not in other["answer"]
    assert len(other["history"]) == 1


def test_memory_off_starts_every_question_clean(monkeypatch, offline_graph):  # noqa: ARG001
    use_settings(monkeypatch, conversation_memory="off")
    assert build_checkpointer() is None

    plain = build_graph(None)
    thread = {"configurable": {"thread_id": new_thread_id()}}
    plain.invoke({"question": "первый"}, config=thread)
    second = plain.invoke({"question": "второй"}, config=thread)
    assert len(second["history"]) == 1


def test_unknown_memory_mode_is_refused(monkeypatch):
    """Опечатка в настройке не должна выглядеть как исправно пропавшая память."""
    use_settings(monkeypatch, conversation_memory="sqllite")
    with pytest.raises(ValueError, match="CONVERSATION_MEMORY"):
        build_checkpointer()


def test_sqlite_memory_survives_a_restart(monkeypatch, tmp_path):
    """Ради этого режим и существует: разговор переживает перезапуск процесса."""
    use_settings(
        monkeypatch,
        conversation_memory="sqlite",
        checkpoint_db=str(tmp_path / "conv.sqlite"),
    )

    monkeypatch.setattr(graph_module, "node_route", lambda state: {"route": "industry"})
    monkeypatch.setattr(
        graph_module, "node_retrieve", lambda state: {"report_hits": [], "used_reports": False}
    )
    monkeypatch.setattr(
        graph_module, "node_web", lambda state: {"web_hits": [], "used_web": False}
    )
    monkeypatch.setattr(
        graph_module,
        "node_answer",
        lambda state: {
            "answer": "ok",
            "history": [{"question": state["question"], "answer": "ok"}],
        },
    )

    thread = {"configurable": {"thread_id": new_thread_id()}}
    first_run = build_graph(build_checkpointer())
    first_run.invoke({"question": "запомни это"}, config=thread)

    # Новый чекпоинтер и новый скомпилированный граф — то же, что новый процесс.
    second_run = build_graph(build_checkpointer())
    resumed = second_run.invoke({"question": "и это"}, config=thread)

    assert [turn["question"] for turn in resumed["history"]] == ["запомни это", "и это"]


@pytest.mark.parametrize(
    "question",
    [
        "а если ОПЕК+ сократит добычу на 2 млн барр./сут?",
        "что если добыча упадёт на 2 млн барр./сут",
        "сценарий: снизят предложение на 2 млн барр./сут",
        "оцени цену при сокращении на 2 млн барр./сут",
    ],
)
def test_a_cut_is_read_as_a_cut_whatever_the_word_form(question):
    """Знак сценария дороже его величины: направление читатель примет на веру.

    Список корней «сокращ, снижен, …» не содержал корня «сократ», и вопрос
    «а если ОПЕК+ СОКРАТИТ добычу на 2 млн барр./сут?» считался как сценарий
    УВЕЛИЧЕНИЯ: система отвечала падением цены на вопрос о её росте. Поймано
    сквозным прогоном через живую модель — она же и заметила расхождение.
    """
    assert graph_module.parse_supply_change(question) == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "question",
    [
        "а если ОПЕК+ нарастит добычу на 2 млн барр./сут?",
        "что будет при росте предложения на 2 млн барр./сут",
    ],
)
def test_an_increase_is_still_read_as_an_increase(question):
    """Обратная сторона: расширенный список корней не должен красить всё в минус."""
    assert graph_module.parse_supply_change(question) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("а на пять лет?", 730),          # потолок: дальше истории — арифметика
        ("а на год?", 365),               # единица времени без числа значит одну
        ("на квартал", 90),
        ("на две недели", 14),
        ("спрогнозируй на 3 месяца", 90),  # цифры работали и раньше
    ],
)
def test_horizon_is_read_from_words_not_only_digits(question, expected):
    """В продолжении разговора человек почти не пишет цифру.

    «а на пять лет?» после прогноза на три месяца считался на прежние 90 дней:
    парсер знал только ``\\d+``, и ответ выглядел как отказ считать. Поймано
    сквозным прогоном.
    """
    assert graph_module.parse_horizon_days(question) == expected


def test_clamped_horizon_is_announced_not_applied_silently(monkeypatch):
    """Спросив пять лет и получив «горизонт 730 дн.», человек ошибётся в 2.5 раза."""

    class FakeReport:
        def as_text(self) -> str:
            return "Горизонт: 730 дн."

    import neftegaz.tools.forecast_tool as forecast_tool

    monkeypatch.setattr(forecast_tool, "run_forecast", lambda **kw: FakeReport())
    result = graph_module.node_forecast({"question": "а на пять лет?"})

    assert "1825" in result["forecast_text"]
    assert "730" in result["forecast_text"]


def test_an_unparsed_horizon_falls_back_instead_of_guessing():
    """«Полтора года» не округляется до года: умолчание честнее подмены."""
    assert graph_module.parse_horizon_days("а на полтора года?", default=90) == 90


def test_forecast_horizon_carries_over_but_the_scenario_does_not(monkeypatch):
    """Горизонт наследуется, величина шока — нет.

    Наследуемый шок стал бы липким: однажды названное допущение «при сокращении
    на 1.5» тихо красило бы все последующие ответы.
    """
    calls = []

    def fake_run_forecast(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("данных нет, важны сами аргументы")

    import neftegaz.tools.forecast_tool as forecast_tool

    monkeypatch.setattr(forecast_tool, "run_forecast", fake_run_forecast)

    graph_module.node_forecast({"question": "спрогнозируй Brent на 2 года при сокращении на 1.5 млн барр./сут"})
    assert calls[-1]["horizon_days"] == 730
    assert calls[-1]["supply_change_mb_d"] == pytest.approx(-1.5)

    graph_module.node_forecast({"question": "а что будет с ценой?", "last_horizon_days": 730})
    assert calls[-1]["horizon_days"] == 730
    assert calls[-1]["supply_change_mb_d"] == 0.0
