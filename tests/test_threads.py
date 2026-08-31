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
    HISTORY_CAP,
    TITLE_CAP,
    ThreadRegistry,
    default_title,
    fts_query,
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


# ── сквозной поиск по разговорам (Ш5) ─────────────────────────────────────


ANSWER_US = (
    "По июльскому отчёту EIA добычи нефти в США вырастет до 13.6 млн барр./сут "
    "в 2027 году, при этом сокращения ОПЕК+ удержат котировки Brent выше 70."
)
ANSWER_GAS = "Запасы газа в Европе заполнены на 90%, спрос на СПГ снижается второй квартал."


@pytest.fixture
def filled(registry):
    registry.record_turn("t-us", "Что с добычей в США?", ANSWER_US)
    registry.record_turn("t-gas", "А газ?", ANSWER_GAS)
    return registry


def test_the_query_builder_refuses_what_it_cannot_search():
    assert fts_query("  ") == ""
    assert fts_query("на") == "", "слово короче трёх букв триграммам не с чем сопоставлять"


def test_the_query_builder_quotes_every_word_separately():
    assert fts_query("прогноз добычи") == '"прогн" "добы"'


def test_a_word_from_the_middle_of_the_answer_is_found(filled):
    found = filled.search_turns("котировки")
    assert [h.thread_id for h in found] == ["t-us"]
    assert found[0].question == "Что с добычей в США?"


@pytest.mark.parametrize(
    ("query", "form_in_text"),
    [
        ("добыча", "добычи"),
        ("сокращение", "сокращения"),
        ("запас", "Запасы"),
        ("кварталы", "квартал"),
    ],
)
def test_a_different_russian_word_form_still_finds_the_turn(filled, query, form_in_text):
    """★Ровно то, ради чего взят trigram И срезка хвоста.

    Голый триграммный MATCH ищет ПОДСТРОКУ, поэтому «добыча» не нашло бы
    «добычи»: одна форма не входит в другую. Проверка идёт по обеим половинам —
    и токенизатору, и построителю запроса.
    """
    assert any(form_in_text in (h.answer + h.question) for h in filled.search_turns(query))


def test_the_known_limit_of_tail_trimming_is_stated_and_not_pretended_away(filled):
    """★ГРАНИЦА ПРИЁМА, записанная тестом, а не умолчанием.

    Срезка хвоста ловит склонение («добыча» → «добычи»), но не спряжение с
    основой, которой в тексте нет: у «снижаться» после срезки остаётся
    «снижат», а в «снижается» стои́т «снижае». Настоящий стеммер это взял бы,
    и он же стои́т отдельной зависимостью ради одного поля.

    Тест закрепляет ИЗВЕСТНЫЙ промах, чтобы он не выглядел неизвестным. Если
    когда-нибудь появится морфология, этот тест упадёт — и это будет верный
    сигнал, а не поломка.
    """
    assert filled.search_turns("снижаться") == []
    assert "снижается" in ANSWER_GAS, "слово в тексте есть — не находит именно приём"


def test_a_query_that_is_not_there_finds_nothing(filled):
    """Отрицательный контроль: пусто, а не «что-нибудь похожее»."""
    assert filled.search_turns("дивиденды Роснефти") == []
    assert filled.search_turns("борщ") == []


def test_search_spans_every_conversation(filled):
    assert {h.thread_id for h in filled.search_turns("а")} == set(), "слишком короткий запрос"
    both = filled.search_turns("нефт")
    assert {h.thread_id for h in both} == {"t-us"}


def test_a_hit_carries_the_conversation_it_came_from(filled):
    filled.rename("t-us", "США и ОПЕК+")
    hit = filled.search_turns("котировки")[0]
    assert hit.thread_title == "США и ОПЕК+"
    assert hit.ordinal == 1


def test_deleting_a_thread_takes_it_out_of_search_too(filled):
    """«Удалено» обязано значить удалено и для поиска, а не «скрыто из списка»."""
    filled.delete("t-us")
    assert filled.search_turns("котировки") == []


def test_search_survives_a_restart(filled):
    fresh = reopen(filled)
    try:
        assert len(fresh.search_turns("котировки")) == 1
    finally:
        fresh.close()


def test_an_index_that_lost_its_rows_is_rebuilt_on_open(filled):
    """Индекс, разошедшийся с ходами, не падает — он молча ничего не находит.

    Поэтому расхождение ловится счётчиком при открытии и лечится пересборкой.
    """
    filled._db.execute("INSERT INTO turns_fts (turns_fts) VALUES ('delete-all')")
    filled._db.commit()
    assert filled.search_turns("котировки") == [], "подготовка: индекс действительно опустошён"
    fresh = reopen(filled)
    try:
        assert len(fresh.search_turns("котировки")) == 1
    finally:
        fresh.close()


# ── история поиска (Ш6) ───────────────────────────────────────────────────


def test_a_repeated_query_lifts_the_existing_row(registry):
    registry.record_search("добыча США", 3)
    registry.record_search("газ", 1)
    registry.record_search("добыча США", 5)
    history = registry.list_searches()
    assert [r.query for r in history] == ["добыча США", "газ"]
    assert history[0].hits == 5, "число найденного обновляется, а не остаётся прежним"


def test_history_survives_a_restart(registry):
    registry.record_search("добыча США", 3)
    fresh = reopen(registry)
    try:
        assert [r.query for r in fresh.list_searches()] == ["добыча США"]
    finally:
        fresh.close()


def test_deleting_one_history_item_really_removes_the_row(registry):
    registry.record_search("добыча США", 3)
    registry.record_search("газ", 1)
    assert registry.forget_search("газ") is True
    fresh = reopen(registry)
    try:
        left = fresh._db.execute("SELECT count(*) FROM search_history WHERE query = 'газ'")
        assert left.fetchone()[0] == 0, "строка обязана уйти из базы, а не пометиться"
        assert [r.query for r in fresh.list_searches()] == ["добыча США"]
    finally:
        fresh.close()


def test_forgetting_a_query_that_is_not_there_reports_failure(registry):
    assert registry.forget_search("такого не искали") is False


def test_clearing_history_empties_it(registry):
    registry.record_search("один", 1)
    registry.record_search("два", 2)
    assert registry.clear_searches() == 2
    assert registry.list_searches() == []


def test_history_is_capped_and_evicts_the_oldest(registry):
    for i in range(HISTORY_CAP + 5):
        registry._db.execute(
            "INSERT INTO search_history (query, last_run, hits) VALUES (?, ?, 1)",
            (f"запрос {i}", f"2026-08-31T00:{i // 60:02d}:{i % 60:02d}+00:00"),
        )
    registry._db.commit()
    registry.record_search("самый свежий", 1)
    history = registry.list_searches()
    assert len(history) == HISTORY_CAP
    assert history[0].query == "самый свежий"
    assert "запрос 0" not in {r.query for r in history}, "вытесняется самое старое"


def test_an_empty_query_is_not_remembered(registry):
    registry.record_search("   ", 0)
    assert registry.list_searches() == []


# ── след находок (Ш2) ─────────────────────────────────────────────────────


def test_a_turn_remembers_which_chunks_were_fed(registry):
    registry.record_turn("t1", "Вопрос", "Ответ.", chunk_ids=["aaa", "bbb"])
    assert registry.chunk_trail("t1") == ["aaa", "bbb"]


def test_the_trail_keeps_only_identifiers_and_never_text(registry):
    """★Из подключаемого разговора обязаны ехать ССЫЛКИ, а не утверждения.

    Проверка структурная, а не по договорённости: в таблице следа есть колонка
    под идентификатор и нет ни одной под текст, поэтому круговая ссылка
    («модель цитирует саму себя») невозможна по устройству, а не по дисциплине.
    """
    columns = {row[1] for row in registry._db.execute("PRAGMA table_info(turn_chunks)")}
    assert columns == {"turn_id", "chunk_id", "position"}


def test_the_trail_survives_a_restart(registry):
    registry.record_turn("t1", "Вопрос", "Ответ.", chunk_ids=["aaa"])
    fresh = reopen(registry)
    try:
        assert fresh.chunk_trail("t1") == ["aaa"]
    finally:
        fresh.close()


def test_the_freshest_turn_leads_the_trail(registry):
    registry.record_turn("t1", "Первый", "Ответ.", chunk_ids=["старое"])
    registry.record_turn("t1", "Второй", "Ответ.", chunk_ids=["свежее"])
    assert registry.chunk_trail("t1") == ["свежее", "старое"]


def test_a_chunk_seen_twice_is_listed_once(registry):
    registry.record_turn("t1", "Первый", "Ответ.", chunk_ids=["aaa", "bbb"])
    registry.record_turn("t1", "Второй", "Ответ.", chunk_ids=["bbb", "ccc"])
    assert registry.chunk_trail("t1") == ["bbb", "ccc", "aaa"]


def test_a_turn_without_a_trail_is_normal(registry):
    """Ход без найденных фрагментов — обычное дело: вопрос вне области, отказ
    поиска, расчёт без отчётов. Пустой след не должен ломать запись хода."""
    registry.record_turn("t1", "Посоветуй рецепт борща", "Это вне моей области.")
    assert registry.chunk_trail("t1") == []
    assert registry.get("t1").turns == 1


def test_deleting_a_thread_takes_its_trail_with_it(registry):
    registry.record_turn("t1", "Вопрос", "Ответ.", chunk_ids=["aaa"])
    registry.record_turn("t2", "Другой", "Ответ.", chunk_ids=["bbb"])
    registry.delete("t1")
    fresh = reopen(registry)
    try:
        left = fresh._db.execute("SELECT chunk_id FROM turn_chunks").fetchall()
        assert [r[0] for r in left] == ["bbb"], "след удалённой нити обязан уйти вместе с ней"
    finally:
        fresh.close()


def test_trails_of_different_threads_do_not_mix(registry):
    registry.record_turn("t1", "Вопрос", "Ответ.", chunk_ids=["aaa"])
    registry.record_turn("t2", "Другой", "Ответ.", chunk_ids=["bbb"])
    assert registry.chunk_trail("t1") == ["aaa"]
    assert registry.chunk_trail("t2") == ["bbb"]
