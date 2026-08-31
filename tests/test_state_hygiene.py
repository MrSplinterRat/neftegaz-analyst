"""Поля-однодневки сбрасываются каждый ход — и это проверяет машина (Р-067).

★ЗАЧЕМ ЭТОТ ФАЙЛ СУЩЕСТВУЕТ. Правило «поле, которое пишется не на всякой ветке,
обязано сбрасываться каждый ход» было записано словами рядом с кодом — и не
сработало ТРИЖДЫ подряд, каждый раз одинаково: поле добавляли, сброс дописать
забывали, а узнавали об этом двумя ходами разговора и сильно позже.

  1. `scenario_waived` — из него правило и родилось;
  2. `reports_status` / `web_status` — отказ веба на вопросе «какая сейчас цена
     Brent» приезжал оговоркой в ответ на СЛЕДУЮЩИЙ вопрос, который в веб не
     ходил вовсе;
  3. `confidence_marks` / `context_used` — панель показывала бы разметку ссылок
     и наполнение контекста ПРОШЛОГО хода как нынешние.

Третий случай доказал то, что должно было быть ясно после второго: правило,
исполняемое памятью разработчика, не исполняется. Поэтому поля разделены на три
класса в самом модуле, а здесь проверяется, что деление накрывает состояние
целиком и что сброс действительно происходит.

★Тест устроен так, чтобы падать ИМЕНЕМ ПОЛЯ: снятие одного поля из перечня
красит одну строку с этим именем, а не «что-то в графе сломалось».

★И одна тонкость, которую нельзя оставлять неназванной. Проверка «маршрутизатор
сбрасывает поле» ПАРАМЕТРИЗОВАНА ПО САМОМУ ПЕРЕЧНЮ, то есть делит с ним
механизм: убери запись из перечня — исчезнет и её проверка. Одна она была бы
самообманом. Ловит такое снятие ДРУГАЯ проверка, у которой источник свой —
объявление `AgentState`: поле, выпавшее из перечня, перестаёт быть отнесённым к
классу, и покрытие краснеет. Диверсия прогнана: снятие `confidence_marks` красит
покрытие, размер перечня и сквозной двухходовый тест — три проверки из трёх
источников.
"""

from __future__ import annotations

import typing

import pytest

from neftegaz.agent import graph as graph_module

FIELDS = list(typing.get_type_hints(graph_module.AgentState).keys())
ONE_TURN = graph_module.ONE_TURN_FIELDS
INPUT = graph_module.INPUT_FIELDS
LASTING = graph_module.LASTING_FIELDS


# ── деление обязано накрывать состояние целиком ────────────────────────────


def test_every_state_field_belongs_to_exactly_one_class():
    """★Новое поле, не отнесённое ни к одному классу, красит приёмку сразу.

    Это и есть перевод правила из головы в машину: раньше такое поле молча
    протекало в следующий ход, и цена ошибки платилась у пользователя.
    """
    classified = set(ONE_TURN) | set(INPUT) | set(LASTING)
    assert set(FIELDS) - classified == set(), (
        "поля состояния не отнесены ни к одному классу: "
        "однодневка (сбрасывается), вход (пишется каждый ход) или долгоживущее"
    )
    assert classified - set(FIELDS) == set(), "в классах есть поля, которых нет в AgentState"
    assert set(ONE_TURN) & INPUT == set()
    assert set(ONE_TURN) & LASTING == set()
    assert not INPUT & LASTING


def test_the_one_turn_list_is_not_trivially_short():
    """Содержательность закреплена числом: перечень, съёжившийся до пары полей,
    выглядел бы работающим и не проверял бы ничего."""
    assert len(ONE_TURN) >= 15, f"однодневок осталось {len(ONE_TURN)} — перечень выродился"


# ── сброс действительно происходит, поле за полем ──────────────────────────


@pytest.mark.parametrize("field", sorted(ONE_TURN), ids=sorted(ONE_TURN))
def test_the_router_resets_each_one_turn_field(field):
    """`node_route` идёт первым на каждом ходу — сброс живёт там.

    В состояние кладётся заведомо грязное значение (как будто оставшееся от
    прошлого вопроса), и проверяется, что маршрутизатор вернул поле сброшенным.
    """
    dirty = dict.fromkeys(FIELDS, "ОСТАЛОСЬ-ОТ-ПРОШЛОГО-ХОДА")
    dirty["question"] = "спрогнозируй Brent на 30 дней"
    produced = graph_module.node_route(dirty)
    assert field in produced, f"{field}: маршрутизатор не сбросил поле — оно протечёт"
    if field != "route":  # маршрут заполняется тут же, и это его работа
        assert produced[field] == ONE_TURN[field], (
            f"{field}: сброшено не в то значение, что объявлено в перечне"
        )


def test_the_reset_hands_out_copies_not_the_registry_itself():
    """★Перечень — общий объект модуля; узлы дополняют списки и словари.

    Отдай мы сам перечень, первый же ход накопил бы в нём хиты, а следующий
    получил бы их как «сброшенное состояние» — протечка через ту самую дверь,
    которую мы закрываем.
    """
    first = graph_module.fresh_turn_fields()
    first["report_hits"].append("хит прошлого хода")
    first["context_used"]["reports"] = 999
    second = graph_module.fresh_turn_fields()
    assert second["report_hits"] == []
    assert second["context_used"] == {}
    assert ONE_TURN["report_hits"] == []
    assert ONE_TURN["context_used"] == {}


# ── класс «пишется входом» — тоже утверждение о поведении ──────────────────


def test_input_fields_are_actually_written_by_the_entry_point(monkeypatch):
    """Их нельзя сбрасывать в маршрутизаторе: он идёт ПОСЛЕ входа и затёр бы
    только что положенное. Значит вход обязан писать их сам — каждый ход."""
    monkeypatch.setattr(graph_module, "_borrowed_chunk_ids", lambda _thread: ["ссылка"])
    produced = graph_module.initial_state("какая сейчас цена Brent", "поток-1")
    assert set(produced) == set(INPUT), (
        "вход пишет не тот набор полей, который объявлен классом INPUT_FIELDS"
    )


# ── и сквозная проверка двумя ходами, на новых полях ───────────────────────


def test_the_marks_of_one_turn_do_not_reach_the_next(monkeypatch):
    """★Проверяется НАСТОЯЩИМ разговором из двух ходов, а не вызовом узла.

    Поштучная проверка узла показывает, что сброс написан; протечку показывает
    только второй ход, потому что живёт она в чекпоинтере между ходами.
    """
    from neftegaz.agent.graph import build_checkpointer, build_graph, new_thread_id
    from neftegaz.rag.store import Hit

    hit = Hit(
        text="Brent Spot Average ..... 91.2 92.7 93.4 92.1",
        score=0.75,
        source_name="EIA STEO",
        date="2026-07",
        page=34,
        page_end=34,
        context="Table 2. Energy Prices\nQ1 Q2 Q3 2026",
    )

    class _Store:
        def search(self, *_args, **_kwargs):
            return [hit]

        def count(self):
            return 9450

    monkeypatch.setattr("neftegaz.rag.store.get_store", lambda: _Store(), raising=True)
    monkeypatch.setattr(
        graph_module, "node_forecast", lambda state: {"forecast_text": ""}, raising=True
    )

    # Классификатор отвечает по вопросу: первый ход — отраслевой, второй — вне
    # компетенции. Подменяется он же, что и в работе, поэтому маршрут выбирает
    # настоящий `node_route`, а не тест за него.
    def _ask(_role, prompt, *_args, **_kwargs):
        if "борщ" in prompt:
            return "other"
        return "Цена около 92.7."

    monkeypatch.setattr(graph_module, "ask", _ask, raising=True)

    graph = build_graph(build_checkpointer())
    config = {"configurable": {"thread_id": new_thread_id()}}

    first = graph.invoke({"question": "что с добычей нефти в США"}, config=config)
    assert first["context_used"], "первый ход не заполнил контекст — проверять нечего"

    # Второй вопрос уходит в отказ: узел ответа не выполняется вовсе.
    second = graph.invoke({"question": "как приготовить борщ"}, config=config)
    assert "компетен" in second["answer"] or "не отвечаю" in second["answer"].lower(), (
        f"второй ход не ушёл в отказ, проверять протечку не на чем: {second['answer'][:120]}"
    )
    assert second.get("context_used") == {}, second.get("context_used")
    assert second.get("confidence_marks") == {}, second.get("confidence_marks")
    assert second.get("report_hits") == [], "фрагменты прошлого вопроса дожили до следующего"
    assert second.get("used_reports") is False
