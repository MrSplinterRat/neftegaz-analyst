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
    "ThreadInfo",
    "ThreadRegistry",
    "TITLE_CAP",
    "default_title",
    "get_registry",
    "registry_unavailable_reason",
    "reset_registry",
]

# Заголовок по умолчанию — первый вопрос, обрезанный до этой длины. Число
# видно здесь, а не растворено в вызове: в списке нитей оно задаёт ширину.
TITLE_CAP = 60


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
