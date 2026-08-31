"""Реестр разговоров: список нитей, их заголовки, ходы и удаление.

★ЗАЧЕМ ОТДЕЛЬНАЯ ТАБЛИЦА, ЕСЛИ ЧЕКПОЙНТЕР УЖЕ ХРАНИТ НИТИ.

Чекпойнтер LangGraph действительно помнит каждый ход и переживает перезапуск,
но помнит он их ПО ИДЕНТИФИКАТОРУ, который знает только тот, кто его назвал.
Интерфейс держал идентификатор в сессии браузера, а кнопка «начать заново»
чеканила новый и забывала прежний. Нити оставались в базе — просто становились
недостижимыми.

Замер на рабочей базе 31.08.2026: 96 нитей, 770 чекпоинтов, и ни одной длиннее
двух ходов — то есть весь этот объём накоплен прогонами тестов и проб, а не
разговорами. Отсюда два решения, и оба намеренные:

* **Реестр не заполняется задним числом.** Список, собранный из чекпойнтера,
  открылся бы девяноста шестью безымянными строками, из которых все девяносто
  шесть — мусор. Реестр ведётся с момента появления.
* **Ход записывает интерфейс, а не** :func:`answer_question`. Реестр — это
  разговоры пользователя, а не журнал вызовов функции. Запись внутри агента
  снова завела бы «разговор» на каждый тест и каждый пакетный прогон.

★ПОЧЕМУ РЕЕСТР ВЫКЛЮЧАЕТСЯ ВМЕСТЕ С ``CONVERSATION_MEMORY``. Настройка решает
один вопрос: попадает ли переписка с аналитиком на диск. Это данные заказчика,
и решает его регламент, а не наше удобство. Реестр, который пишет заголовки и
тексты ходов в файл при настройке «в памяти процесса», обошёл бы это решение
молча. Поэтому при ``memory`` и ``off`` реестра нет, и интерфейс говорит,
почему.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from neftegaz.config import settings

__all__ = [
    "HISTORY_CAP",
    "MIN_QUERY_CHARS",
    "SearchRecord",
    "ThreadInfo",
    "ThreadRegistry",
    "TITLE_CAP",
    "TurnHit",
    "default_title",
    "fts_query",
    "get_registry",
    "registry_unavailable_reason",
    "reset_registry",
]

# Заголовок по умолчанию — первый вопрос, обрезанный до этой длины. Число
# видно здесь, а не растворено в вызове: в списке нитей оно задаёт ширину.
TITLE_CAP = 60

# ★МИНИМАЛЬНАЯ ДЛИНА ЗАПРОСА, и она названа вслух в интерфейсе. Триграммный
# токенизатор режет текст на тройки символов, поэтому запрос короче трёх букв
# сопоставлять не с чем: он не «ничего не нашёл», он в принципе не может искать.
MIN_QUERY_CHARS = 3

# Потолок истории поиска. Вытеснение по времени последнего исполнения.
HISTORY_CAP = 200


def _now() -> str:
    """Отметка времени в UTC, пригодная и для сортировки, и для показа."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def default_title(question: str) -> str:
    """Заголовок нити по её первому вопросу.

    Обрезается по границе слова, а не по букве: «Спрогнозируй цену Brent на
    3 мес…» читается, «Спрогнозируй цену Brent на 3 ме…» — спотыкается.
    """
    text = re.sub(r"\s+", " ", question).strip()
    if not text:
        return "Без названия"
    if len(text) <= TITLE_CAP:
        return text
    cut = text[:TITLE_CAP]
    space = cut.rfind(" ")
    if space >= TITLE_CAP // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:—-") + "…"


def _stem(word: str) -> str:
    """Срезать окончание, чтобы словоформа нашла словоформу.

    ★ЗАЧЕМ ЭТО ВООБЩЕ НУЖНО, ЕСЛИ ТОКЕНИЗАТОР УЖЕ ТРИГРАММНЫЙ. Триграммы дают
    поиск по ПОДСТРОКЕ, и этого достаточно, чтобы «добыч» нашло «добычи», — но
    недостаточно для того, чего требует приёмка: «добыча» подстрокой в «добычи»
    не входит, и запрос целым словом не находит ничего. Замер 31.08 на живой
    FTS5: «добыча» → 0 совпадений, «сокращение» → 0, «прогнозы» → 0.

    Морфологии в FTS5 нет и не появится, полноценный стеммер — отдельная
    зависимость ради одного поля. Срезка хвоста — грубая замена, и цена у неё
    честная: запрос становится короче, а значит шире, и ложных срабатываний
    больше. Замер той же меры: шесть словоформ из шести нашли свои пары, четыре
    посторонних слова не нашли ничего.

    Длины хвоста выбраны по длине слова: русское окончание — одна-три буквы, и
    срезать три буквы у короткого слова значило бы искать по огрызку.
    """
    n = len(word)
    if n >= 8:
        return word[:-3]
    if n >= 6:
        return word[:-2]
    if n >= 4:
        return word[:-1]
    return word


def fts_query(text: str) -> str:
    """Запрос пользователя → выражение FTS5. Пустая строка, если искать нечего.

    Каждое слово идёт отдельной фразой в кавычках: так знаки препинания и
    операторы FTS5 (``NOT``, ``*``, скобки) не толкуются как синтаксис. Слова
    соединяются неявным И — «прогноз добычи» обязано найти ход, где есть оба.
    """
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) >= MIN_QUERY_CHARS]
    if not words:
        return ""
    # Двойная кавычка внутри фразы FTS5 экранируется удвоением.
    return " ".join('"{}"'.format(_stem(w).replace('"', '""')) for w in words)


@dataclass(frozen=True)
class TurnHit:
    """Один найденный ход — с разговором, из которого он взят."""

    thread_id: str
    thread_title: str
    ordinal: int
    asked_at: str
    question: str
    answer: str


@dataclass(frozen=True)
class SearchRecord:
    """Строка истории поиска."""

    query: str
    last_run: str
    hits: int


@dataclass(frozen=True)
class ThreadInfo:
    """Одна строка списка разговоров."""

    thread_id: str
    title: str
    created_at: str
    updated_at: str
    turns: int
    renamed: bool


_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id  TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    -- 1 = заголовок задан человеком. Без этого признака первый же следующий
    -- ход затирал бы переименование заголовком по умолчанию.
    renamed    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turns      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    ordinal   INTEGER NOT NULL,
    asked_at  TEXT NOT NULL,
    question  TEXT NOT NULL,
    answer    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS turns_by_thread ON turns (thread_id, ordinal);
CREATE INDEX IF NOT EXISTS threads_by_updated ON threads (updated_at DESC);

-- ★ТОКЕНИЗАТОР trigram, А НЕ unicode61. Штатный токенизатор режет текст на
-- слова и сопоставляет их целиком, а русской морфологии в FTS5 нет: запрос
-- «добыча» не нашёл бы «добычи». Триграммы ищут подстроку и от морфологии не
-- зависят. Цена названа и уплачена сознательно: индекс больше, а короткие
-- запросы дают ложные срабатывания — отсюда MIN_QUERY_CHARS.
--
-- content='turns' — индекс без второй копии текста: строки берутся из самой
-- таблицы ходов. Синхронизацию держат триггеры ниже.
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    question,
    answer,
    content='turns',
    content_rowid='turn_id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS turns_fts_insert AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts (rowid, question, answer)
    VALUES (new.turn_id, new.question, new.answer);
END;

-- Удаление хода обязано убирать его и из индекса, иначе удалённый разговор
-- продолжал бы находиться поиском — то есть «удалено» значило бы «скрыто».
CREATE TRIGGER IF NOT EXISTS turns_fts_delete AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts (turns_fts, rowid, question, answer)
    VALUES ('delete', old.turn_id, old.question, old.answer);
END;

CREATE TABLE IF NOT EXISTS search_history (
    query    TEXT PRIMARY KEY,
    last_run TEXT NOT NULL,
    hits     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS history_by_time ON search_history (last_run DESC);
"""


class ThreadRegistry:
    """Список разговоров и их ходы в той же базе, где лежит чекпойнтер.

    Одна база, а не вторая рядом: удаление нити обязано убрать и её ходы, и её
    чекпоинты, а две базы означали бы удаление в двух местах без общей
    транзакции — то есть состояние «ход удалён, память о нём осталась».
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False по той же причине, что и у чекпойнтера:
        # Streamlit обслуживает страницы в разных потоках.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # Чекпойнтер держит вторую связь с этим же файлом. Ожидание вместо
        # немедленного «database is locked» — дешевле любой блокировки в коде.
        self._db.execute("PRAGMA busy_timeout = 5000")
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()
            self._resync_fts()

    def _resync_fts(self) -> None:
        """Пересобрать поисковый индекс, если он разошёлся с таблицей ходов.

        Нужно в двух случаях: ходы записаны раньше, чем появился индекс (то
        есть при обновлении уже работающей установки), и порча содержимого.
        ★Проверка — СЧЁТЧИК, а не доверие триггерам: разошедшийся индекс не
        падает, он молча ничего не находит, и это неотличимо от «в разговорах
        такого нет».

        ⚠СЧИТАЕТСЯ ТЕНЕВАЯ ``turns_fts_docsize``, А НЕ ``turns_fts``. У таблицы
        с ``content='turns'`` запрос ``count(*) FROM turns_fts`` идёт в саму
        таблицу ходов, а не в индекс, и потому ВСЕГДА с ней согласен: замер
        31.08 — после полного опустошения индекса он по-прежнему отвечал «2»,
        тогда как ``docsize`` честно показал «0». Проверка, читающая
        проверяемое через тот же путь, проверкой не является.
        """
        turns = self._db.execute("SELECT count(*) FROM turns").fetchone()[0]
        indexed = self._db.execute("SELECT count(*) FROM turns_fts_docsize").fetchone()[0]
        if turns != indexed:
            self._db.execute("INSERT INTO turns_fts (turns_fts) VALUES ('rebuild')")
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── чтение ─────────────────────────────────────────────────────────────

    def list_threads(self) -> list[ThreadInfo]:
        """Все разговоры, свежий сверху."""
        with self._lock:
            rows = self._db.execute(
                "SELECT thread_id, title, created_at, updated_at, turns, renamed "
                "FROM threads ORDER BY updated_at DESC"
            ).fetchall()
        return [
            ThreadInfo(
                thread_id=r["thread_id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                turns=r["turns"],
                renamed=bool(r["renamed"]),
            )
            for r in rows
        ]

    def get(self, thread_id: str) -> ThreadInfo | None:
        for info in self.list_threads():
            if info.thread_id == thread_id:
                return info
        return None

    def turns(self, thread_id: str) -> list[dict]:
        """Ходы нити по порядку — тем же видом, каким их держит интерфейс."""
        with self._lock:
            rows = self._db.execute(
                "SELECT question, answer, asked_at FROM turns "
                "WHERE thread_id = ? ORDER BY ordinal",
                (thread_id,),
            ).fetchall()
        return [
            {"question": r["question"], "answer": r["answer"], "asked_at": r["asked_at"]}
            for r in rows
        ]

    # ── запись ─────────────────────────────────────────────────────────────

    def record_turn(self, thread_id: str, question: str, answer: str) -> None:
        """Записать ход и подтянуть за ним запись о самой нити.

        Заголовок ставится по ПЕРВОМУ вопросу и больше не меняется сам:
        разговор узнаётся по тому, с чего начался, а не по тому, о чём зашла
        речь на десятом ходу.
        """
        stamp = _now()
        with self._lock:
            cur = self._db.execute(
                "SELECT turns FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if cur is None:
                self._db.execute(
                    "INSERT INTO threads (thread_id, title, renamed, created_at, updated_at, "
                    "turns) VALUES (?, ?, 0, ?, ?, 0)",
                    (thread_id, default_title(question), stamp, stamp),
                )
                ordinal = 1
            else:
                ordinal = int(cur["turns"]) + 1
            self._db.execute(
                "INSERT INTO turns (thread_id, ordinal, asked_at, question, answer) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, ordinal, stamp, question, answer),
            )
            self._db.execute(
                "UPDATE threads SET turns = ?, updated_at = ? WHERE thread_id = ?",
                (ordinal, stamp, thread_id),
            )
            self._db.commit()

    def rename(self, thread_id: str, title: str) -> bool:
        """Переименовать разговор. Пустое имя не принимается.

        Возвращает False, если такой нити нет: молчаливое «переименовал» на
        отсутствующей нити выглядело бы успехом.
        """
        clean = re.sub(r"\s+", " ", title).strip()
        if not clean:
            return False
        with self._lock:
            cur = self._db.execute(
                "UPDATE threads SET title = ?, renamed = 1 WHERE thread_id = ?",
                (clean[:TITLE_CAP], thread_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    def delete(self, thread_id: str) -> bool:
        """Удалить разговор целиком: ходы, запись в реестре и его чекпоинты.

        ★Чекпоинты удаляются тоже. Идентификаторы случайны и не переиспользуются,
        так что оставленные строки никого бы не спутали, — но «удалено» обязано
        значить удалено, а не «пропало из списка».

        Таблицы чекпойнтера принадлежат LangGraph, поэтому их наличие
        проверяется, а не предполагается: при ``CONVERSATION_MEMORY=memory`` их
        в этом файле нет вовсе.
        """
        with self._lock:
            present = {
                row["name"]
                for row in self._db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            cur = self._db.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
            removed = cur.rowcount > 0
            self._db.execute("DELETE FROM turns WHERE thread_id = ?", (thread_id,))
            for table in ("checkpoints", "writes"):
                if table in present:
                    self._db.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            self._db.commit()
        return removed


    # ── сквозной поиск по разговорам ───────────────────────────────────────

    def search_turns(self, query: str, limit: int = 50) -> list[TurnHit]:
        """Найти ходы во ВСЕХ разговорах.

        Ищем по разговорам, а не по отчётам, и выдачи не смешиваем: у них
        разный провенанс и разная цена ошибки. Промах поиска по разговорам
        стои́т лишнего клика, промах по отчётам — неверного числа в ответе.
        """
        expression = fts_query(query)
        if not expression:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT t.thread_id, t.ordinal, t.asked_at, t.question, t.answer, "
                "       COALESCE(th.title, t.thread_id) AS title "
                "FROM turns_fts f "
                "JOIN turns t ON t.turn_id = f.rowid "
                "LEFT JOIN threads th ON th.thread_id = t.thread_id "
                "WHERE turns_fts MATCH ? "
                "ORDER BY rank, t.asked_at DESC LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [
            TurnHit(
                thread_id=r["thread_id"],
                thread_title=r["title"],
                ordinal=r["ordinal"],
                asked_at=r["asked_at"],
                question=r["question"],
                answer=r["answer"],
            )
            for r in rows
        ]

    # ── история поиска ─────────────────────────────────────────────────────

    def record_search(self, query: str, hits: int) -> None:
        """Запомнить запрос. Повтор ПОДНИМАЕТ строку, а не плодит новую."""
        clean = re.sub(r"\s+", " ", query).strip()
        if not clean:
            return
        stamp = _now()
        with self._lock:
            self._db.execute(
                "INSERT INTO search_history (query, last_run, hits) VALUES (?, ?, ?) "
                "ON CONFLICT(query) DO UPDATE SET last_run = excluded.last_run, "
                "hits = excluded.hits",
                (clean, stamp, hits),
            )
            # Вытеснение по времени: сверх потолка уходит самое старое.
            self._db.execute(
                "DELETE FROM search_history WHERE query NOT IN ("
                "SELECT query FROM search_history ORDER BY last_run DESC LIMIT ?)",
                (HISTORY_CAP,),
            )
            self._db.commit()

    def list_searches(self) -> list[SearchRecord]:
        with self._lock:
            rows = self._db.execute(
                "SELECT query, last_run, hits FROM search_history ORDER BY last_run DESC"
            ).fetchall()
        return [
            SearchRecord(query=r["query"], last_run=r["last_run"], hits=r["hits"]) for r in rows
        ]

    def forget_search(self, query: str) -> bool:
        """Убрать один запрос из истории.

        ★Строка УХОДИТ ИЗ БАЗЫ, а не помечается скрытой. Это история человека,
        и «удалено» обязано значить удалено, иначе мы храним то, что нас
        попросили не хранить.
        """
        with self._lock:
            cur = self._db.execute("DELETE FROM search_history WHERE query = ?", (query,))
            self._db.commit()
            return cur.rowcount > 0

    def clear_searches(self) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM search_history")
            self._db.commit()
            return cur.rowcount


_REGISTRY: ThreadRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def registry_unavailable_reason() -> str:
    """Почему реестра нет — пустой строкой, если он есть.

    Строка идёт прямо в интерфейс: выключенный реестр обязан объяснять себя,
    иначе он неотличим от сломанного.
    """
    mode = settings.conversation_memory.strip().lower()
    if mode == "sqlite":
        return ""
    if mode == "memory":
        return (
            "Разговоры хранятся в памяти процесса (CONVERSATION_MEMORY=memory), "
            "поэтому списка разговоров нет: перезапуск стёр бы его вместе с ними. "
            "Поставьте CONVERSATION_MEMORY=sqlite, чтобы включить."
        )
    if mode == "off":
        return (
            "Память диалога выключена (CONVERSATION_MEMORY=off): каждый вопрос "
            "отвечается с чистого листа, и разговоров, которые можно было бы "
            "перечислить, не существует."
        )
    return f"Неизвестное значение CONVERSATION_MEMORY={settings.conversation_memory!r}."


def get_registry() -> ThreadRegistry | None:
    """Общий реестр процесса — или None, если настройка его не разрешает."""
    global _REGISTRY
    if registry_unavailable_reason():
        return None
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = ThreadRegistry(settings.checkpoint_db)
        return _REGISTRY


def reset_registry() -> None:
    """Забыть общий реестр — нужно тестам и смене настройки на лету."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None:
            _REGISTRY.close()
        _REGISTRY = None
