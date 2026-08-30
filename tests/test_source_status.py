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
