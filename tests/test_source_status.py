"""Приёмка: отказ поиска по отчётам не выдаётся за отсутствие данных.

★Дефект, ради которого написан файл. `node_retrieve` ловил любое исключение и
отдавал пустой список, а дальше в промпт уходило «поиск по базе отчётов
выполнен, релевантных фрагментов не найдено» — УТВЕРЖДЕНИЕ, что поиск состоялся
и корпус промолчал. Модель, честно следуя правилу «нет в источниках — так и
скажи», писала человеку «в отчётах этого нет». Отказ прибора выдавался за
свойство мира, ответ приходил из веба и выглядел совершенно обычным.

★Половина проверок здесь — отрицательный контроль, и без него мера была бы
вечно-зелёной: разбор, который оговаривается на каждом ответе, назвал бы все три
состояния и стал бы шумом, который перестают читать.
"""

from __future__ import annotations

import pytest

from neftegaz.agent import graph as graph_module
from neftegaz.agent import prompts

QUESTION = "какова добыча сырой нефти в США"


class _Broken:
    def search(self, *_args, **_kwargs):
        raise RuntimeError("коллекция повреждена")

    def count(self):
        raise RuntimeError("коллекция повреждена")


class _NotBuilt:
    def search(self, *_args, **_kwargs):
        return []

    def count(self):
        return 0


class _BuiltButSilent:
    """Индекс есть, по этому вопросу ничего — законный исход, оговорки не требует."""

    def search(self, *_args, **_kwargs):
        return []

    def count(self):
        return 9450


class _Hit:
    text = "U.S. total crude oil production ..... 13.28 13.51"
    context = ""
    score = 0.73
    source_name = "EIA STEO"
    date = "2026-07"
    page = 36
    page_end = 36


class _Working:
    def search(self, *_args, **_kwargs):
        return [_Hit()]

    def count(self):
        return 9450


def _retrieve(monkeypatch, store) -> dict:
    monkeypatch.setattr("neftegaz.rag.store.get_store", lambda: store, raising=True)
    return graph_module.node_retrieve({"question": QUESTION})


def test_a_failed_search_is_reported_as_a_failure(monkeypatch):
    state = _retrieve(monkeypatch, _Broken())
    assert state["reports_status"].startswith("failed"), state
    assert "коллекция повреждена" in state["reports_status"]


def test_an_unbuilt_index_is_reported_as_unbuilt(monkeypatch):
    assert _retrieve(monkeypatch, _NotBuilt())["reports_status"] == "empty"


def test_a_built_index_with_no_match_is_not_reported_as_broken(monkeypatch):
    """★Отрицательный контроль, и он несущий.

    Законный «корпус этого не содержит» обязан остаться законным. Пометь его
    отказом — и агент начнёт извиняться за исправную работу, а настоящий отказ
    утонет среди извинений.
    """
    assert _retrieve(monkeypatch, _BuiltButSilent())["reports_status"] == ""


def test_a_successful_search_carries_no_status(monkeypatch):
    state = _retrieve(monkeypatch, _Working())
    assert state["reports_status"] == ""
    assert state["used_reports"] is True


@pytest.mark.parametrize(
    ("status", "must_contain"),
    [
        ("failed: коллекция повреждена", "НЕ ВЫПОЛНЕН"),
        ("empty", "ПУСТА"),
    ],
)
def test_the_prompt_tells_the_model_the_search_did_not_happen(status, must_contain):
    """Модели говорится, что поиска НЕ БЫЛО, — иначе она честно скажет «данных нет»."""
    prompt = prompts.build_answer_prompt(QUESTION, "", "", "", "", status)
    assert must_contain in prompt, prompt
    assert "поиск по базе отчётов выполнен" not in prompt


def test_the_prompt_keeps_the_honest_empty_line_when_the_search_did_happen():
    """Отрицательный контроль к предыдущему."""
    prompt = prompts.build_answer_prompt(QUESTION, "", "", "", "", "")
    assert "поиск по базе отчётов выполнен, релевантных фрагментов не найдено" in prompt
    assert "НЕ ВЫПОЛНЕН" not in prompt


@pytest.mark.parametrize("status", ["failed: коллекция повреждена", "empty"])
def test_the_human_is_told_in_the_answer_itself(monkeypatch, status):
    """★Оговорку дописывает КОД, а не модель.

    Сказанное модели доходит до человека только пересказом, а пересказчик вправе
    счесть оговорку неважной. Каким источником получен ответ — проверяющий
    обязан увидеть буквально. Модель здесь подменена нарочно: она отвечает
    текстом, в котором про отказ нет ни слова, и оговорка обязана появиться
    всё равно.
    """
    monkeypatch.setattr(graph_module, "ask", lambda *a, **k: "Добыча растёт.", raising=True)
    state = graph_module.node_answer({"question": QUESTION, "reports_status": status})
    assert "Добыча растёт." in state["answer"]
    assert "★База отраслевых отчётов" in state["answer"], state["answer"]


def test_the_human_is_not_bothered_when_everything_worked(monkeypatch):
    """Отрицательный контроль: оговорка на каждом ответе — это шум."""
    monkeypatch.setattr(graph_module, "ask", lambda *a, **k: "Добыча растёт.", raising=True)
    state = graph_module.node_answer({"question": QUESTION, "reports_status": ""})
    assert state["answer"] == "Добыча растёт."


# ── та же болезнь на веб-стороне ───────────────────────────────────────────
#
# `node_web` не ловил исключений ВОВСЕ — отказ поиска валил весь прогон графа, —
# а пустая выдача означала три разных вещи сразу, и в промпт уходила одна строка
# на все три: «(веб-поиск не выполнялся или ничего не вернул)». Союз «или» делает
# её правдивой и потому бесполезной.


def _web(monkeypatch, replacement) -> dict:
    monkeypatch.setattr(
        "neftegaz.tools.web.search_web_with_status", replacement, raising=True
    )
    return graph_module.node_web({"question": "какая сейчас цена Brent"})


def test_a_failed_web_search_is_reported_as_a_failure(monkeypatch):
    state = _web(monkeypatch, lambda *a, **k: ([], "failed: сеть недоступна"))
    assert state["web_status"].startswith("failed")
    assert "сеть недоступна" in state["web_status"]


def test_a_missing_search_library_is_reported_as_unavailable(monkeypatch):
    state = _web(monkeypatch, lambda *a, **k: ([], "unavailable: ddgs не установлен"))
    assert state["web_status"].startswith("unavailable")


def test_an_empty_but_working_web_search_carries_no_status(monkeypatch):
    """Отрицательный контроль: «сходили и не нашли» — законный исход."""
    assert _web(monkeypatch, lambda *a, **k: ([], ""))["web_status"] == ""


def test_the_web_node_survives_an_exception(monkeypatch):
    """★Дополнительный источник не вправе уносить с собой уже собранный ответ.

    У соседних узлов охрана есть, у этого её не было: отказ вылетал наружу и
    валил прогон графа целиком. Падение вместо ответа хуже молчаливого нуля.
    """

    def _boom(*_args, **_kwargs):
        raise RuntimeError("совсем неожиданный отказ")

    state = _web(monkeypatch, _boom)
    assert state["web_status"].startswith("failed")
    assert state["used_web"] is False


def test_search_web_still_returns_a_plain_list():
    """Тонкая обёртка обязана остаться списком: на неё смотрят прежние вызывающие."""
    from neftegaz.tools import web as web_module

    original = web_module.search_web_with_status
    try:
        web_module.search_web_with_status = lambda *a, **k: ([], "failed: сеть недоступна")
        assert web_module.search_web("Brent") == []
    finally:
        web_module.search_web_with_status = original


# ── когда молчат ОБА источника ─────────────────────────────────────────────


def test_both_sources_down_says_the_answer_is_unverifiable():
    """★Главное предложение всей правки, и произнести его может только составная.

    Каждая половина говорила правду о себе, а вместе они лгали: отчётная
    сообщала, что ответ «опирается на остальные источники», веб-овая — что
    «только на периодические отчёты». Обе ссылались на источник, которого в этот
    момент не было. Про то, что ответ стои́т на общих знаниях модели и проверке
    не подлежит, не сказала бы ни одна.
    """
    note = prompts.sources_status_note("failed: коллекция повреждена", "failed: сеть недоступна")
    assert "НИ ОДИН ВНЕШНИЙ ИСТОЧНИК НЕ ОТВЕТИЛ" in note
    assert "коллекция повреждена" in note
    assert "сеть недоступна" in note
    # ★И ни одной ссылки на источник, которого нет.
    assert "опирается только на остальные источники" not in note
    assert "только на периодические отчёты" not in note


def test_one_source_down_keeps_its_own_note():
    """Отрицательный контроль: составная не проглатывает одиночный отказ."""
    only_reports = prompts.sources_status_note("failed: коллекция повреждена", "")
    assert "База отраслевых отчётов" in only_reports
    assert "НИ ОДИН ВНЕШНИЙ ИСТОЧНИК" not in only_reports

    only_web = prompts.sources_status_note("", "failed: сеть недоступна")
    assert "Веб-поиск НЕ ВЫПОЛНЕН" in only_web
    assert "НИ ОДИН ВНЕШНИЙ ИСТОЧНИК" not in only_web


def test_no_note_at_all_when_both_sources_worked():
    """Отрицательный контроль: оговорка на каждом ответе — шум, который не читают."""
    assert prompts.sources_status_note("", "") == ""


def test_the_note_does_not_claim_content_that_is_not_there(monkeypatch):
    """★Третья ось: отказала не только оба источника, но и языковая модель.

    Оговорка про молчащие источники говорит «всё, что написано выше». Когда
    модель тоже отказала, выше не написано НИЧЕГО — и фраза утверждает про
    содержимое, которого нет. Это тот же дефект, что был у двух половин
    оговорки; он пережил их починку, потому что тогда перебирались сочетания
    ИСТОЧНИКОВ, а отказ модели в перебор не входил.

    Перечень покрывает лишь те оси, которые в него заложены.
    """

    def _boom(*_args, **_kwargs):
        raise RuntimeError("Connection error.")

    monkeypatch.setattr(graph_module, "ask", _boom, raising=True)
    state = graph_module.node_answer(
        {
            "question": QUESTION,
            "reports_status": "failed: коллекция повреждена",
            "web_status": "failed: сеть недоступна",
        }
    )
    assert "НИ ОДИН ВНЕШНИЙ ИСТОЧНИК НЕ ОТВЕТИЛ" in state["answer"]
    assert "Ответить нечем" in state["answer"]
    assert "написано выше" not in state["answer"], state["answer"]


def test_the_note_still_claims_content_when_the_model_did_answer(monkeypatch):
    """Отрицательный контроль: с ответом модели фраза про «выше» законна и нужна.

    Ослабь её всегда — и пропадёт единственное предложение, говорящее
    проверяющему, что ответ ничем не подтверждён.
    """
    monkeypatch.setattr(graph_module, "ask", lambda *a, **k: "Цена около 92.", raising=True)
    state = graph_module.node_answer(
        {
            "question": QUESTION,
            "reports_status": "failed: коллекция повреждена",
            "web_status": "failed: сеть недоступна",
        }
    )
    assert "Всё, что написано выше" in state["answer"]
    assert "Ответить нечем" not in state["answer"]


def test_the_fallback_keeps_the_material_when_only_the_model_failed(monkeypatch):
    """Отказала только модель — материал обязан дойти до человека читаемым."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("Connection error.")

    monkeypatch.setattr(graph_module, "ask", _boom, raising=True)
    state = graph_module.node_answer({"question": QUESTION, "report_hits": [_Hit()]})
    assert "Языковая модель недоступна" in state["answer"]
    assert "с. 36" in state["answer"], state["answer"]
    assert "13.28" in state["answer"]
    assert "Ответить нечем" not in state["answer"]


# ── состояние источников не переживает свой ход ────────────────────────────


def test_a_source_failure_does_not_leak_into_the_next_turn(monkeypatch):
    """★Поле-однодневка, не сброшенное явно, живёт в чекпоинтере вечно.

    `node_web` выполняется только на веб-ветке, а `node_answer` читает
    `web_status` всегда. Отказ веба, случившийся на вопросе «какая сейчас цена
    Brent», приезжал оговоркой в ответ на СЛЕДУЮЩИЙ вопрос, который в веб не
    ходил вовсе: человеку сообщалось об отказе, которого в этом ходу не было.

    Правило записано в самом графе рядом с `scenario_waived` — «оба поля
    пишутся КАЖДЫЙ ход, а не только когда меняются». Новые поля под него не
    попали. Проверяется он единственным способом, каким такое проверяется, —
    ДВУМЯ ходами одного разговора.
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
            return [hit, hit, hit]

        def count(self):
            return 9450

    monkeypatch.setattr("neftegaz.rag.store.get_store", lambda: _Store(), raising=True)
    monkeypatch.setattr(
        "neftegaz.tools.web.search_web_with_status",
        lambda *a, **k: ([], "failed: сеть недоступна"),
        raising=True,
    )
    monkeypatch.setattr(graph_module, "ask", lambda *a, **k: "Цена около 92.7.", raising=True)
    monkeypatch.setattr(
        graph_module, "node_forecast", lambda state: {"forecast_text": ""}, raising=True
    )

    graph = build_graph(build_checkpointer())
    config = {"configurable": {"thread_id": new_thread_id()}}

    # Ход первый: «сейчас» уводит в веб, веб отказывает — оговорка законна.
    first = graph.invoke({"question": "какая сейчас цена Brent"}, config=config)
    assert "Веб-поиск НЕ ВЫПОЛНЕН" in first["answer"], first["answer"]

    # Ход второй: в веб не ходили вовсе — оговорки быть не должно.
    second = graph.invoke({"question": "спрогнозируй Brent на 30 дней"}, config=config)
    assert second.get("web_status", "") == "", second.get("web_status")
    assert "Веб-поиск НЕ ВЫПОЛНЕН" not in second["answer"], second["answer"]
