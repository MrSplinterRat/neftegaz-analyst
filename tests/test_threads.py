"""Реестр разговоров: заголовки, переименование, удаление, переживание рестарта.

★Каждая проверка, которая обещает «переживает перезапуск», ЗАКРЫВАЕТ реестр и
открывает его заново по тому же пути. Проверка на живом объекте подтверждала бы
сама себя: она читала бы ту же память, в которую только что писала, и прошла бы
даже если на диск не ушло ничего.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from neftegaz.agent import threads
from neftegaz.agent.threads import (
    TITLE_CAP,
    ThreadRegistry,
    default_title,
    registry_unavailable_reason,
)


@pytest.fixture
def registry(tmp_path):
    reg = ThreadRegistry(tmp_path / "conv.sqlite")
    yield reg
    reg.close()


def reopen(reg: ThreadRegistry) -> ThreadRegistry:
    """Тот же файл, новый объект — то же, что перезапуск процесса."""
    path = reg.path
    reg.close()
    return ThreadRegistry(path)


# ── заголовок по умолчанию ────────────────────────────────────────────────


def test_a_short_question_becomes_the_title_verbatim():
    assert default_title("Что с ценой Brent?") == "Что с ценой Brent?"


def test_a_long_question_is_cut_at_a_word_boundary():
    question = (
        "Спрогнозируй цену Brent на три месяца при сокращении добычи ОПЕК+ "
        "на полтора миллиона баррелей в сутки"
    )
    title = default_title(question)
    assert len(title) <= TITLE_CAP + 1  # +1 — многоточие
    assert title.endswith("…")
    # ★Обрыв ровно по границе слова: последнее слово либо целое, либо его нет.
    assert title[:-1].strip().split()[-1] in question.split()


def test_an_empty_question_still_gets_a_name():
    assert default_title("   ") == "Без названия"


# ── запись ходов ──────────────────────────────────────────────────────────


def test_the_first_turn_creates_the_thread_and_names_it(registry):
    registry.record_turn("t1", "Какой прогноз EIA по добыче?", "Ответ.")
    listed = registry.list_threads()
    assert [x.thread_id for x in listed] == ["t1"]
    assert listed[0].title == "Какой прогноз EIA по добыче?"
    assert listed[0].turns == 1
    assert listed[0].renamed is False


def test_a_second_turn_counts_up_and_leaves_the_title_alone(registry):
    registry.record_turn("t1", "Первый вопрос", "Ответ.")
    registry.record_turn("t1", "Совсем про другое", "Ответ.")
    info = registry.get("t1")
    assert info.turns == 2
    assert info.title == "Первый вопрос"
    assert [t["question"] for t in registry.turns("t1")] == ["Первый вопрос", "Совсем про другое"]


def test_threads_are_listed_newest_first(registry):
    registry.record_turn("t1", "Первый разговор", "Ответ.")
    registry.record_turn("t2", "Второй разговор", "Ответ.")
    registry.record_turn("t1", "Возвращаемся к первому", "Ответ.")
    assert [x.thread_id for x in registry.list_threads()] == ["t1", "t2"]


def test_turns_survive_a_restart(registry):
    registry.record_turn("t1", "Вопрос", "Ответ.")
    fresh = reopen(registry)
    try:
        assert [x.turns for x in fresh.list_threads()] == [1]
        assert fresh.turns("t1")[0]["answer"] == "Ответ."
    finally:
        fresh.close()


# ── переименование ────────────────────────────────────────────────────────


def test_a_rename_survives_a_restart(registry):
    registry.record_turn("t1", "Какой прогноз EIA по добыче?", "Ответ.")
    assert registry.rename("t1", "ОПЕК+ и Urals") is True
    fresh = reopen(registry)
    try:
        assert fresh.get("t1").title == "ОПЕК+ и Urals"
        assert fresh.get("t1").renamed is True
    finally:
        fresh.close()


def test_a_later_turn_does_not_overwrite_a_manual_title(registry):
    registry.record_turn("t1", "Первый вопрос", "Ответ.")
    registry.rename("t1", "Своё имя")
    registry.record_turn("t1", "Ещё вопрос", "Ответ.")
    assert registry.get("t1").title == "Своё имя"


def test_an_empty_name_is_refused(registry):
    registry.record_turn("t1", "Первый вопрос", "Ответ.")
    assert registry.rename("t1", "   ") is False
    assert registry.get("t1").title == "Первый вопрос"


def test_renaming_a_thread_that_is_not_there_reports_failure(registry):
    assert registry.rename("нет-такой", "Имя") is False


# ── удаление ──────────────────────────────────────────────────────────────


def test_deleting_a_thread_takes_its_turns_with_it(registry):
    registry.record_turn("t1", "Вопрос", "Ответ.")
    registry.record_turn("t2", "Другой", "Ответ.")
    assert registry.delete("t1") is True
    fresh = reopen(registry)
    try:
        assert [x.thread_id for x in fresh.list_threads()] == ["t2"]
        assert fresh.turns("t1") == []
    finally:
        fresh.close()


def test_deleting_a_thread_takes_its_checkpoints_too(registry):
    """★Память о ходах уходит вместе с записью о них, а не переживает её."""
    db = sqlite3.connect(str(registry.path))
    db.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)")
    db.execute("CREATE TABLE writes (thread_id TEXT, task_id TEXT)")
    db.execute("INSERT INTO checkpoints VALUES ('t1', 'c1'), ('t2', 'c2')")
    db.execute("INSERT INTO writes VALUES ('t1', 'w1')")
    db.commit()
    db.close()

    registry.record_turn("t1", "Вопрос", "Ответ.")
    registry.delete("t1")

    fresh = reopen(registry)
    try:
        left = fresh._db.execute("SELECT thread_id FROM checkpoints").fetchall()
        assert [r[0] for r in left] == ["t2"]
        assert fresh._db.execute("SELECT count(*) FROM writes").fetchone()[0] == 0
    finally:
        fresh.close()


def test_deleting_a_thread_that_is_not_there_reports_failure(registry):
    assert registry.delete("нет-такой") is False


def test_a_missing_checkpoint_table_is_not_an_error(registry):
    """При CONVERSATION_MEMORY=memory таблиц чекпойнтера в файле нет вовсе."""
    registry.record_turn("t1", "Вопрос", "Ответ.")
    assert registry.delete("t1") is True


# ── выключенный реестр объясняет себя ─────────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "must_contain"),
    [
        ("memory", "CONVERSATION_MEMORY=sqlite"),
        ("off", "с чистого листа"),
        ("невнятное", "Неизвестное значение"),
    ],
)
def test_a_disabled_registry_says_why(monkeypatch, mode, must_contain):
    # Настройки — замороженный dataclass, поэтому подменяется ССЫЛКА в модуле,
    # а не поле в объекте: замороженный объект правку молча не принял бы.
    monkeypatch.setattr(threads, "settings", replace(threads.settings, conversation_memory=mode))
    reason = registry_unavailable_reason()
    assert reason, "выключенный реестр обязан объяснять себя, а не молчать"
    assert must_contain in reason


def test_an_enabled_registry_gives_no_reason(monkeypatch):
    monkeypatch.setattr(
        threads, "settings", replace(threads.settings, conversation_memory="sqlite")
    )
    assert registry_unavailable_reason() == ""
